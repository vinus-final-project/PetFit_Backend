"""LLM 응답 검증 (파이프라인 12단계).

생성형 모델의 출력은 신뢰할 수 없다. 스키마를 벗어나거나, 탐지되지 않은 객체를
근거로 들거나, 우선순위를 중복 부여한다. 저장 전에 여기서 걸러낸다.

프롬프트 설계서의 검증 절차 8단계를 구현하며, 위반은 두 갈래로 나뉜다.

    복구 : 결과를 고쳐서 통과시킨다. 재호출하지 않는다.
    거절 : 재생성을 요청한다. 3회 실패하면 분석 실패로 처리한다.

**둘을 나누는 기준은 "모델을 다시 부르지 않고 고칠 수 있는가"다.** 우선순위 중복은
번호만 다시 매기면 되지만, 서술이 2건 미만이면 없는 내용을 지어낼 수 없다.
추론 1회가 수 초이므로 고칠 수 있는 것을 재생성으로 넘기면 분석 시간만 늘어난다.

**LLM 없이 전부 검증할 수 있다.** 입력이 문자열이고 출력이 값이므로 모든 분기를
단위 테스트로 확인한다. 모델 선정이나 API 키를 기다릴 필요가 없다.
"""

import json
import logging
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.ai.pipeline import Recommendation, RiskFactor
from app.schemas.enums import RecommendationType, RiskSource

__all__ = [
    "AnalysisOutput",
    "ValidationResult",
    "Rejection",
    "Repair",
    "validate",
    "MAX_RISK_FACTORS",
    "MIN_ANALYSIS",
    "MAX_ANALYSIS",
    "MIN_RECOMMENDATIONS",
    "MAX_RECOMMENDATIONS",
]

logger = logging.getLogger(__name__)

# --- 생성 개수 제한 ---------------------------------------------------------
# 프롬프트 설계서의 "생성 개수 제한" 표를 따른다. 성능평가로 조정되는 값이 아니라
# 프롬프트 계약이므로 core/constants.py 가 아니라 여기에 둔다.

#: 위험 요소. 위험이 없으면 빈 배열이 정상이다.
MAX_RISK_FACTORS = 5
#: 생활환경 분석 서술
MIN_ANALYSIS = 2
MAX_ANALYSIS = 4
#: 환경 개선 추천. 위험이 없어도 개선 추천은 있어야 한다.
MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 5

#: 코드펜스. 프롬프트가 금지해도 모델이 자주 붙인다.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
#: 최상위 JSON 객체. 앞뒤에 설명 문장이 붙은 경우 추출한다.
_BRACES = re.compile(r"\{.*\}", re.DOTALL)
#: 연속 공백. 위험 요소 중복 판정 시 표기 차이를 흡수한다.
_SPACES = re.compile(r"\s+")


class Rejection(str, Enum):
    """재생성을 요청해야 하는 위반. 고쳐서 통과시킬 수 없다."""

    NOT_JSON = "NOT_JSON"
    MISSING_FIELD = "MISSING_FIELD"
    BAD_SHAPE = "BAD_SHAPE"
    BAD_TYPE = "BAD_TYPE"
    BAD_SOURCE = "BAD_SOURCE"
    TOO_FEW_ANALYSIS = "TOO_FEW_ANALYSIS"
    NO_RECOMMENDATION = "NO_RECOMMENDATION"

    @property
    def instruction(self) -> str:
        """재생성 시 프롬프트에 덧붙일 지시문.

        무엇이 잘못됐는지 알려주지 않고 다시 부르면 같은 실패가 반복된다.
        """
        return _INSTRUCTIONS[self]


_INSTRUCTIONS: dict[Rejection, str] = {
    Rejection.NOT_JSON: (
        "직전 응답은 JSON으로 읽을 수 없었습니다. "
        "설명 문장과 코드펜스 없이 JSON 객체 하나만 출력하세요."
    ),
    Rejection.MISSING_FIELD: (
        "직전 응답에 필수 항목이 없었습니다. "
        "riskFactors, analysis, recommendations 세 가지를 모두 포함하세요. "
        "해당하는 내용이 없으면 빈 배열로 출력하세요."
    ),
    Rejection.BAD_SHAPE: (
        "직전 응답의 항목 구조가 형식과 달랐습니다. "
        "riskFactors는 {text, source} 객체 배열, analysis는 문자열 배열, "
        "recommendations는 {type, text, priority, source} 객체 배열이어야 합니다."
    ),
    Rejection.BAD_TYPE: (
        "직전 응답의 recommendations[].type 이 정의된 값이 아니었습니다. "
        "SAFETY, ACTIVITY, REST, ENVIRONMENT 중 하나만 사용하세요."
    ),
    Rejection.BAD_SOURCE: (
        "직전 응답의 source 가 정의된 값이 아니었습니다. "
        "DETECTED 또는 OBSERVED만 사용하세요."
    ),
    Rejection.TOO_FEW_ANALYSIS: (
        f"직전 응답의 analysis 가 {MIN_ANALYSIS}건 미만이었습니다. "
        f"생활환경 분석을 {MIN_ANALYSIS}건 이상 {MAX_ANALYSIS}건 이하로 작성하세요."
    ),
    Rejection.NO_RECOMMENDATION: (
        "직전 응답에 recommendations 가 없었습니다. "
        "위험 요소가 없더라도 환경 개선 추천을 1건 이상 작성하세요."
    ),
}


class Repair(str, Enum):
    """모델을 다시 부르지 않고 고친 항목. 성능평가의 형식 준수율 측정에 쓴다."""

    STRIPPED_FENCE = "STRIPPED_FENCE"
    EXTRACTED_OBJECT = "EXTRACTED_OBJECT"
    IGNORED_SCORE = "IGNORED_SCORE"
    DEMOTED_TO_OBSERVED = "DEMOTED_TO_OBSERVED"
    DEDUPED_RISK_FACTORS = "DEDUPED_RISK_FACTORS"
    TRUNCATED_RISK_FACTORS = "TRUNCATED_RISK_FACTORS"
    TRUNCATED_ANALYSIS = "TRUNCATED_ANALYSIS"
    TRUNCATED_RECOMMENDATIONS = "TRUNCATED_RECOMMENDATIONS"
    RENUMBERED_PRIORITY = "RENUMBERED_PRIORITY"


@dataclass(frozen=True)
class AnalysisOutput:
    """검증을 통과한 12단계 산출물.

    `PipelineResult` 의 서술 3종과 그대로 대응한다. 점수는 포함하지 않는다.
    생성형 AI는 점수를 만들지 않으며, 규칙 기반으로 산출된 값을 입력으로 받는다.
    """

    risk_factors: tuple[RiskFactor, ...] = ()
    analysis: tuple[str, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    """검증 결과.

    `ok` 가 True면 `value` 를 저장한다. False면 `rejection.instruction` 을 덧붙여
    재생성을 요청한다.
    """

    ok: bool
    value: AnalysisOutput | None = None
    rejection: Rejection | None = None
    repairs: tuple[Repair, ...] = field(default_factory=tuple)

    @classmethod
    def reject(cls, rejection: Rejection) -> "ValidationResult":
        return cls(ok=False, rejection=rejection)

    @classmethod
    def accept(
        cls, value: AnalysisOutput, repairs: Collection[Repair]
    ) -> "ValidationResult":
        # 같은 복구가 여러 항목에서 발생해도 종류 단위로 한 번만 기록한다.
        ordered = tuple(r for r in Repair if r in set(repairs))
        return cls(ok=True, value=value, repairs=ordered)


def validate(raw: str, detected_names: Collection[str]) -> ValidationResult:
    """LLM 응답을 검증하고 저장 가능한 값으로 변환한다.

    Args:
        raw: 모델이 반환한 원문.
        detected_names: 탐지 신뢰 기준을 통과한 객체의 한글 이름. `DETECTED`
            표기가 실제 탐지 결과에 근거하는지 판정하는 데 사용한다.

    Returns:
        통과 시 복구가 적용된 결과, 실패 시 거절 사유.
    """
    repairs: set[Repair] = set()

    # 1. JSON 파싱
    data = _parse(raw, repairs)
    if data is None:
        return ValidationResult.reject(Rejection.NOT_JSON)

    # 2. 필수 필드 존재
    if not {"riskFactors", "analysis", "recommendations"} <= data.keys():
        return ValidationResult.reject(Rejection.MISSING_FIELD)

    # 점수는 규칙 기반으로 산출된다. 모델이 만들어 보내도 읽지 않는다.
    if "petFitScore" in data:
        repairs.add(Repair.IGNORED_SCORE)

    if not all(
        isinstance(data[k], list)
        for k in ("riskFactors", "analysis", "recommendations")
    ):
        return ValidationResult.reject(Rejection.BAD_SHAPE)

    names = [n for n in detected_names if n]

    # 3~5. 항목별 값 검증과 근거 강등
    factors = _build_risk_factors(data["riskFactors"], names, repairs)
    if isinstance(factors, Rejection):
        return ValidationResult.reject(factors)

    analysis = _build_analysis(data["analysis"])
    if isinstance(analysis, Rejection):
        return ValidationResult.reject(analysis)

    recommendations = _build_recommendations(data["recommendations"], repairs)
    if isinstance(recommendations, Rejection):
        return ValidationResult.reject(recommendations)

    # 6~8. 개수·중복·우선순위 정리
    factors = _normalize_risk_factors(factors, repairs)

    if len(analysis) < MIN_ANALYSIS:
        return ValidationResult.reject(Rejection.TOO_FEW_ANALYSIS)
    if len(analysis) > MAX_ANALYSIS:
        analysis = analysis[:MAX_ANALYSIS]
        repairs.add(Repair.TRUNCATED_ANALYSIS)

    if len(recommendations) < MIN_RECOMMENDATIONS:
        return ValidationResult.reject(Rejection.NO_RECOMMENDATION)
    recommendations = _normalize_recommendations(recommendations, repairs)

    return ValidationResult.accept(
        AnalysisOutput(
            risk_factors=tuple(factors),
            analysis=tuple(analysis),
            recommendations=tuple(recommendations),
        ),
        repairs,
    )


# =============================================================================
# 1단계 — 파싱
# =============================================================================


def _parse(raw: str, repairs: set[Repair]) -> dict[str, Any] | None:
    """응답에서 JSON 객체를 꺼낸다.

    코드펜스와 앞뒤 설명 문장은 **형식 오류로 보지 않고 벗겨낸다.** 내용이 아니라
    표기 문제이고, 재생성은 추론 1회를 다시 쓰기 때문이다. 벗겨낸 사실은
    `Repair` 로 남겨 성능평가의 형식 준수율 측정에 쓴다.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    text = raw.strip()

    parsed = _loads(text)
    if parsed is not None:
        return parsed

    fenced = _FENCE.match(text)
    if fenced:
        parsed = _loads(fenced.group(1))
        if parsed is not None:
            repairs.add(Repair.STRIPPED_FENCE)
            return parsed

    braced = _BRACES.search(text)
    if braced:
        parsed = _loads(braced.group(0))
        if parsed is not None:
            repairs.add(Repair.EXTRACTED_OBJECT)
            return parsed

    return None


def _loads(text: str) -> dict[str, Any] | None:
    """JSON 객체로 읽는다. 배열·문자열 등 객체가 아니면 None."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


# =============================================================================
# 3~5단계 — 항목 변환
# =============================================================================


def _build_risk_factors(
    items: Sequence[Any], names: Sequence[str], repairs: set[Repair]
) -> list[RiskFactor] | Rejection:
    """위험 요소를 변환한다. 근거가 확인되지 않는 DETECTED는 OBSERVED로 낮춘다."""
    result: list[RiskFactor] = []
    for item in items:
        if not isinstance(item, dict):
            return Rejection.BAD_SHAPE

        text = _clean_text(item.get("text"))
        if text is None:
            return Rejection.BAD_SHAPE

        source = _to_source(item.get("source"))
        if source is None:
            return Rejection.BAD_SOURCE

        source = _verify_detected(source, text, names, repairs)
        result.append(RiskFactor(text=text, source=source))
    return result


def _build_analysis(items: Sequence[Any]) -> list[str] | Rejection:
    """분석 서술을 변환한다. 문자열 배열이어야 한다."""
    result: list[str] = []
    for item in items:
        text = _clean_text(item)
        if text is None:
            return Rejection.BAD_SHAPE
        result.append(text)
    return result


def _build_recommendations(
    items: Sequence[Any], repairs: set[Repair]
) -> list[Recommendation] | Rejection:
    """개선 추천을 변환한다.

    **추천의 `source` 는 탐지 목록과 대조하지 않는다.** 위험 요소와 달리 추천은
    "없는 것을 갖추라"는 내용이 정상이기 때문이다. 급수기가 탐지되지 않아
    생활환경 점수가 깎였다면 "급수기를 설치해주세요" 가 나오는데, 이 문장이
    언급하는 급수기는 탐지 목록에 **없는 것이 당연하다.** 대조하면 부재형 항목에
    근거한 추천이 모두 근거 없음으로 잘못 판정된다.

    명세도 같은 구분을 둔다. 위험 요소의 `DETECTED` 는 "입력에 존재하는 객체에만
    사용"이지만, 추천의 `source` 는 "근거가 된 위험 요소의 source 와 일치"다.

    `priority` 는 뒤에서 다시 매기므로, 값이 없거나 정수가 아니면 생성 순서를
    임시 순위로 쓴다. 이것 때문에 재생성을 요청하지 않는다.
    """
    result: list[Recommendation] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            return Rejection.BAD_SHAPE

        text = _clean_text(item.get("text"))
        if text is None:
            return Rejection.BAD_SHAPE

        try:
            rec_type = RecommendationType(item.get("type"))
        except ValueError:
            return Rejection.BAD_TYPE

        source = _to_source(item.get("source"))
        if source is None:
            return Rejection.BAD_SOURCE

        priority = item.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool) or priority < 1:
            priority = index
            repairs.add(Repair.RENUMBERED_PRIORITY)

        result.append(
            Recommendation(type=rec_type, text=text, priority=priority, source=source)
        )
    return result


def _verify_detected(
    source: RiskSource, text: str, names: Sequence[str], repairs: set[Repair]
) -> RiskSource:
    """`DETECTED` 표기가 실제 탐지 결과에 근거하는지 확인한다.

    서술에 탐지된 객체 이름이 하나도 없으면 `OBSERVED` 로 낮춘다.
    **삭제하지 않는다.** 탐지 목록에는 없어도 이미지에서 확인했을 수 있으므로
    폐기하면 실제 위험 요소를 잃는다. `OBSERVED` 는 점수에 영향을 주지 않으니
    잘못 낮춰도 Pet Fit Score의 재현성은 유지된다.

    객체 이름이 서술에 등장하는지만 본다. 서술이 옳은지는 판정할 수 없다.
    문장의 의미 검증은 성능평가에서 사람이 수행한다.
    """
    if source is not RiskSource.DETECTED:
        return source
    if any(name in text for name in names):
        return source

    repairs.add(Repair.DEMOTED_TO_OBSERVED)
    return RiskSource.OBSERVED


# =============================================================================
# 6~8단계 — 개수·중복·우선순위
# =============================================================================


def _normalize_risk_factors(
    factors: Sequence[RiskFactor], repairs: set[Repair]
) -> list[RiskFactor]:
    """중복을 제거한 뒤 개수를 제한한다.

    **중복 제거가 먼저다.** 잘라낸 뒤 중복을 지우면 살릴 수 있었던 항목까지 잃는다.
    같은 객체가 여러 개 탐지되어도 위험 요소는 하나로 묶어 서술해야 한다.
    """
    seen: set[str] = set()
    unique: list[RiskFactor] = []
    for factor in factors:
        key = _SPACES.sub(" ", factor.text).strip()
        if key in seen:
            repairs.add(Repair.DEDUPED_RISK_FACTORS)
            continue
        seen.add(key)
        unique.append(factor)

    if len(unique) > MAX_RISK_FACTORS:
        unique = unique[:MAX_RISK_FACTORS]
        repairs.add(Repair.TRUNCATED_RISK_FACTORS)
    return unique


def _normalize_recommendations(
    items: Sequence[Recommendation], repairs: set[Repair]
) -> list[Recommendation]:
    """우선순위 순으로 잘라내고 1부터 다시 매긴다.

    `UNIQUE (analysis_id, priority)` 제약이 있어 중복된 우선순위는 저장에 실패한다.
    모델은 `[1, 1, 2]` 같은 값을 흔히 만들므로 항상 다시 매긴다.

    정렬은 안정 정렬이라 같은 우선순위 안에서는 생성 순서가 보존된다.
    """
    ordered = sorted(items, key=lambda r: r.priority)

    if len(ordered) > MAX_RECOMMENDATIONS:
        ordered = ordered[:MAX_RECOMMENDATIONS]
        repairs.add(Repair.TRUNCATED_RECOMMENDATIONS)

    if [r.priority for r in ordered] != list(range(1, len(ordered) + 1)):
        repairs.add(Repair.RENUMBERED_PRIORITY)

    return [
        Recommendation(type=r.type, text=r.text, priority=i, source=r.source)
        for i, r in enumerate(ordered, start=1)
    ]


# =============================================================================
# 공통
# =============================================================================


def _clean_text(value: Any) -> str | None:
    """문자열을 다듬는다. 문자열이 아니거나 비어 있으면 None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _to_source(value: Any) -> RiskSource | None:
    """근거 값을 Enum으로 바꾼다. 정의되지 않은 값이면 None."""
    try:
        return RiskSource(value)
    except ValueError:
        return None
