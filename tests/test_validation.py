"""LLM 응답 검증 검증.

프롬프트 설계서의 검증 절차 8단계가 실제로 동작하는지 확인한다.

**LLM을 호출하지 않는다.** 모델이 낼 법한 잘못된 응답을 문자열로 넣고 결과를
확인한다. 모델 선정·API 키·가중치 다운로드 없이 모든 분기를 검증할 수 있다.

확인 대상은 세 가지다.

    거절  : 재생성을 요청해야 하는 위반을 정확한 사유로 잡는가
    복구  : 고칠 수 있는 위반을 재호출 없이 고치는가
    보존  : 고치는 과정에서 살릴 수 있는 내용을 잃지 않는가
"""

import json

import pytest

from app.ai.validation import (
    MAX_ANALYSIS,
    MAX_RECOMMENDATIONS,
    MAX_RISK_FACTORS,
    MIN_ANALYSIS,
    Rejection,
    Repair,
    validate,
)
from app.schemas.enums import RecommendationType, RiskSource

#: 탐지 신뢰 기준을 통과한 객체. AI 분석 정의서의 계산 예시와 같은 구성이다.
DETECTED = ("전선", "창문", "카펫", "소파", "급수기")


def payload(**overrides) -> str:
    """유효한 응답을 만들고 일부만 바꾼다.

    테스트마다 전체 JSON을 적으면 무엇을 검증하는지 묻힌다.
    """
    body = {
        "riskFactors": [
            {"text": "전선이 바닥에 엉킨 채 노출되어 있습니다.", "source": "DETECTED"},
        ],
        "analysis": [
            "활동 공간은 충분하지만 전선이 위험 요소입니다.",
            "휴식 공간은 소파로 일부 확보되어 있습니다.",
        ],
        "recommendations": [
            {
                "type": "SAFETY",
                "text": "전선을 벽면으로 정리해주세요.",
                "priority": 1,
                "source": "DETECTED",
            },
        ],
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def run(**overrides):
    return validate(payload(**overrides), DETECTED)


# =============================================================================
# 1단계 — JSON 파싱
# =============================================================================


class TestParsing:
    async def test_valid_json_passes(self) -> None:
        result = run()
        assert result.ok
        assert result.repairs == ()

    @pytest.mark.parametrize(
        "raw", ["", "   ", "설명만 있고 JSON이 없습니다", "{망가진 json", "null"]
    )
    async def test_unparseable_is_rejected(self, raw) -> None:
        result = validate(raw, DETECTED)
        assert not result.ok
        assert result.rejection is Rejection.NOT_JSON

    async def test_array_wrapped_object_is_extracted(self) -> None:
        """배열로 감싸 보내면 안쪽 객체를 건져낸다. 표기 문제이므로 재생성하지 않는다."""
        result = validate(f"[{payload()}]", DETECTED)
        assert result.ok
        assert Repair.EXTRACTED_OBJECT in result.repairs

    @pytest.mark.parametrize("raw", ["[1, 2, 3]", "[]", '"문자열"', "42"])
    async def test_unsalvageable_non_object_is_rejected(self, raw) -> None:
        """건져낼 객체가 없으면 형식 오류다."""
        assert validate(raw, DETECTED).rejection is Rejection.NOT_JSON

    async def test_code_fence_is_stripped_not_rejected(self) -> None:
        """코드펜스는 표기 문제다. 재생성에 추론 1회를 더 쓰지 않는다."""
        result = validate(f"```json\n{payload()}\n```", DETECTED)
        assert result.ok
        assert Repair.STRIPPED_FENCE in result.repairs

    async def test_bare_fence_is_stripped(self) -> None:
        result = validate(f"```\n{payload()}\n```", DETECTED)
        assert result.ok
        assert Repair.STRIPPED_FENCE in result.repairs

    async def test_surrounding_prose_is_extracted(self) -> None:
        result = validate(f"분석 결과입니다.\n{payload()}\n감사합니다.", DETECTED)
        assert result.ok
        assert Repair.EXTRACTED_OBJECT in result.repairs

    async def test_clean_json_records_no_repair(self) -> None:
        """정상 응답에 복구 기록이 남으면 형식 준수율 측정이 오염된다."""
        assert run().repairs == ()


# =============================================================================
# 2단계 — 필수 필드
# =============================================================================


class TestRequiredFields:
    @pytest.mark.parametrize("missing", ["riskFactors", "analysis", "recommendations"])
    async def test_missing_field_is_rejected(self, missing) -> None:
        body = json.loads(payload())
        del body[missing]
        result = validate(json.dumps(body, ensure_ascii=False), DETECTED)
        assert result.rejection is Rejection.MISSING_FIELD

    async def test_empty_risk_factors_is_valid(self) -> None:
        """위험 요소가 없으면 빈 배열이 정상이다. 없는 위험을 만들지 않는다."""
        result = run(riskFactors=[])
        assert result.ok
        assert result.value.risk_factors == ()

    @pytest.mark.parametrize("field", ["riskFactors", "analysis", "recommendations"])
    async def test_non_array_field_is_rejected(self, field) -> None:
        result = run(**{field: "배열이 아님"})
        assert result.rejection is Rejection.BAD_SHAPE

    async def test_score_is_ignored_not_rejected(self) -> None:
        """점수는 규칙 기반으로 산출된다. 모델이 보내도 읽지 않는다."""
        result = run(petFitScore={"total": 99})
        assert result.ok
        assert Repair.IGNORED_SCORE in result.repairs
        assert not hasattr(result.value, "total")


# =============================================================================
# 3단계 — recommendations[].type
# =============================================================================


class TestRecommendationType:
    @pytest.mark.parametrize(
        "value", ["safety", "URGENT", "", None, 1, "SAFETY_IMPROVEMENT"]
    )
    async def test_undefined_type_is_rejected(self, value) -> None:
        result = run(
            recommendations=[
                {"type": value, "text": "전선 정리", "priority": 1, "source": "DETECTED"}
            ]
        )
        assert result.rejection is Rejection.BAD_TYPE

    @pytest.mark.parametrize("value", list(RecommendationType))
    async def test_all_defined_types_pass(self, value) -> None:
        result = run(
            recommendations=[
                {
                    "type": value.value,
                    "text": "전선을 정리해주세요.",
                    "priority": 1,
                    "source": "DETECTED",
                }
            ]
        )
        assert result.ok
        assert result.value.recommendations[0].type is value


# =============================================================================
# 4단계 — source
# =============================================================================


class TestSource:
    @pytest.mark.parametrize("value", ["detected", "GUESSED", "", None, 0])
    async def test_undefined_source_in_risk_factor_is_rejected(self, value) -> None:
        result = run(riskFactors=[{"text": "전선이 노출되어 있습니다.", "source": value}])
        assert result.rejection is Rejection.BAD_SOURCE

    @pytest.mark.parametrize("value", ["detected", "GUESSED", None])
    async def test_undefined_source_in_recommendation_is_rejected(self, value) -> None:
        result = run(
            recommendations=[
                {"type": "SAFETY", "text": "전선 정리", "priority": 1, "source": value}
            ]
        )
        assert result.rejection is Rejection.BAD_SOURCE

    async def test_observed_is_preserved(self) -> None:
        result = run(
            riskFactors=[{"text": "창가에 화분이 놓여 있습니다.", "source": "OBSERVED"}]
        )
        assert result.ok
        assert result.value.risk_factors[0].source is RiskSource.OBSERVED


# =============================================================================
# 5단계 — DETECTED 근거 확인
# =============================================================================


class TestDetectedVerification:
    async def test_detected_mentioning_known_object_is_kept(self) -> None:
        result = run(
            riskFactors=[{"text": "전선이 바닥에 노출되어 있습니다.", "source": "DETECTED"}]
        )
        assert result.ok
        assert result.value.risk_factors[0].source is RiskSource.DETECTED
        assert Repair.DEMOTED_TO_OBSERVED not in result.repairs

    async def test_unfounded_detected_is_demoted_not_removed(self) -> None:
        """탐지 목록에 없는 근거는 버리지 않고 낮춘다.

        이미지에서 확인했을 수 있으므로 폐기하면 실제 위험 요소를 잃는다.
        """
        result = run(
            riskFactors=[{"text": "약병이 낮은 선반에 있습니다.", "source": "DETECTED"}]
        )
        assert result.ok
        assert len(result.value.risk_factors) == 1
        assert result.value.risk_factors[0].source is RiskSource.OBSERVED
        assert Repair.DEMOTED_TO_OBSERVED in result.repairs

    async def test_demotion_preserves_text(self) -> None:
        text = "약병이 낮은 선반에 있습니다."
        result = run(riskFactors=[{"text": text, "source": "DETECTED"}])
        assert result.value.risk_factors[0].text == text

    async def test_recommendation_source_is_not_object_verified(self) -> None:
        """추천은 "없는 것을 갖추라"는 내용이 정상이다.

        급수기가 탐지되지 않아 점수가 깎였다면 "급수기를 설치해주세요" 가 나오는데,
        이 문장이 말하는 급수기는 탐지 목록에 없는 것이 당연하다. 대조하면 부재형
        항목에 근거한 추천이 전부 근거 없음으로 잘못 판정된다.
        """
        result = run(
            recommendations=[
                {
                    "type": "ENVIRONMENT",
                    "text": "급식기를 설치하여 정해진 자리에서 먹게 해주세요.",
                    "priority": 1,
                    "source": "DETECTED",
                }
            ]
        )
        assert result.ok
        assert result.value.recommendations[0].source is RiskSource.DETECTED
        assert Repair.DEMOTED_TO_OBSERVED not in result.repairs

    async def test_absent_object_recommendation_keeps_detected(self) -> None:
        """반려동물 침대가 없어 휴식환경이 깎인 경우다. 근거는 탐지 결과다."""
        result = run(
            recommendations=[
                {
                    "type": "REST",
                    "text": "조용한 곳에 반려동물 전용 침대를 놓아주세요.",
                    "priority": 1,
                    "source": "DETECTED",
                }
            ]
        )
        assert result.value.recommendations[0].source is RiskSource.DETECTED

    async def test_observed_is_never_demoted(self) -> None:
        """OBSERVED는 근거가 이미지다. 탐지 목록과 대조할 대상이 아니다."""
        result = run(
            riskFactors=[{"text": "약병이 놓여 있습니다.", "source": "OBSERVED"}]
        )
        assert Repair.DEMOTED_TO_OBSERVED not in result.repairs

    async def test_empty_detection_demotes_everything(self) -> None:
        """탐지 결과가 없으면 어떤 DETECTED도 근거를 가질 수 없다."""
        result = validate(payload(), [])
        assert result.ok
        assert all(f.source is RiskSource.OBSERVED for f in result.value.risk_factors)


# =============================================================================
# 6단계 — 생성 개수
# =============================================================================


class TestCounts:
    async def test_excess_risk_factors_are_truncated(self) -> None:
        items = [
            {"text": f"전선 문제 {i}가 있습니다.", "source": "DETECTED"}
            for i in range(MAX_RISK_FACTORS + 3)
        ]
        result = run(riskFactors=items)
        assert result.ok
        assert len(result.value.risk_factors) == MAX_RISK_FACTORS
        assert Repair.TRUNCATED_RISK_FACTORS in result.repairs

    async def test_too_few_analysis_is_rejected(self) -> None:
        """서술이 모자라면 지어낼 수 없다. 재생성해야 한다."""
        result = run(analysis=["한 건뿐입니다."])
        assert result.rejection is Rejection.TOO_FEW_ANALYSIS

    async def test_empty_analysis_is_rejected(self) -> None:
        assert run(analysis=[]).rejection is Rejection.TOO_FEW_ANALYSIS

    async def test_excess_analysis_is_truncated(self) -> None:
        result = run(analysis=[f"서술 {i}입니다." for i in range(MAX_ANALYSIS + 2)])
        assert result.ok
        assert len(result.value.analysis) == MAX_ANALYSIS
        assert Repair.TRUNCATED_ANALYSIS in result.repairs

    async def test_minimum_analysis_passes(self) -> None:
        result = run(analysis=[f"서술 {i}입니다." for i in range(MIN_ANALYSIS)])
        assert result.ok

    async def test_no_recommendation_is_rejected(self) -> None:
        """위험이 없어도 개선 추천은 있어야 한다."""
        assert run(recommendations=[]).rejection is Rejection.NO_RECOMMENDATION

    async def test_excess_recommendations_are_truncated_by_priority(self) -> None:
        items = [
            {
                "type": "SAFETY",
                "text": f"조치 {i}를 해주세요.",
                "priority": i,
                "source": "OBSERVED",
            }
            for i in range(1, MAX_RECOMMENDATIONS + 4)
        ]
        result = run(recommendations=items)
        assert result.ok
        assert len(result.value.recommendations) == MAX_RECOMMENDATIONS
        # 우선순위 상위만 남는다. 뒤쪽 항목이 잘린다.
        assert result.value.recommendations[0].text == "조치 1를 해주세요."
        assert Repair.TRUNCATED_RECOMMENDATIONS in result.repairs


# =============================================================================
# 7단계 — priority
# =============================================================================


class TestPriority:
    async def test_duplicate_priority_is_renumbered(self) -> None:
        """UNIQUE (analysis_id, priority) 제약이 있어 중복은 저장에 실패한다."""
        result = run(
            recommendations=[
                {"type": "SAFETY", "text": "가", "priority": 1, "source": "OBSERVED"},
                {"type": "REST", "text": "나", "priority": 1, "source": "OBSERVED"},
                {"type": "ACTIVITY", "text": "다", "priority": 2, "source": "OBSERVED"},
            ]
        )
        assert result.ok
        assert [r.priority for r in result.value.recommendations] == [1, 2, 3]
        assert Repair.RENUMBERED_PRIORITY in result.repairs

    async def test_gapped_priority_is_renumbered(self) -> None:
        result = run(
            recommendations=[
                {"type": "SAFETY", "text": "가", "priority": 3, "source": "OBSERVED"},
                {"type": "REST", "text": "나", "priority": 7, "source": "OBSERVED"},
            ]
        )
        assert [r.priority for r in result.value.recommendations] == [1, 2]
        assert Repair.RENUMBERED_PRIORITY in result.repairs

    async def test_relative_order_is_preserved(self) -> None:
        result = run(
            recommendations=[
                {"type": "REST", "text": "나중", "priority": 9, "source": "OBSERVED"},
                {"type": "SAFETY", "text": "먼저", "priority": 2, "source": "OBSERVED"},
            ]
        )
        assert [r.text for r in result.value.recommendations] == ["먼저", "나중"]

    async def test_equal_priority_keeps_generation_order(self) -> None:
        """안정 정렬이므로 같은 순위 안에서는 생성 순서가 보존된다."""
        result = run(
            recommendations=[
                {"type": "SAFETY", "text": "첫째", "priority": 1, "source": "OBSERVED"},
                {"type": "REST", "text": "둘째", "priority": 1, "source": "OBSERVED"},
            ]
        )
        assert [r.text for r in result.value.recommendations] == ["첫째", "둘째"]

    @pytest.mark.parametrize("value", [None, 0, -1, "1", 1.5, True])
    async def test_unusable_priority_falls_back_to_order(self, value) -> None:
        """어차피 다시 매기므로 재생성을 요청하지 않는다."""
        result = run(
            recommendations=[
                {"type": "SAFETY", "text": "가", "priority": value, "source": "OBSERVED"}
            ]
        )
        assert result.ok
        assert result.value.recommendations[0].priority == 1
        assert Repair.RENUMBERED_PRIORITY in result.repairs

    async def test_correct_priority_records_no_repair(self) -> None:
        result = run(
            recommendations=[
                {"type": "SAFETY", "text": "가", "priority": 1, "source": "DETECTED"},
                {"type": "REST", "text": "나", "priority": 2, "source": "DETECTED"},
            ]
        )
        assert Repair.RENUMBERED_PRIORITY not in result.repairs


# =============================================================================
# 8단계 — riskFactors 중복
# =============================================================================


class TestDeduplication:
    async def test_identical_text_is_deduped(self) -> None:
        text = "전선이 바닥에 노출되어 있습니다."
        result = run(
            riskFactors=[
                {"text": text, "source": "DETECTED"},
                {"text": text, "source": "DETECTED"},
            ]
        )
        assert result.ok
        assert len(result.value.risk_factors) == 1
        assert Repair.DEDUPED_RISK_FACTORS in result.repairs

    async def test_whitespace_difference_is_still_duplicate(self) -> None:
        result = run(
            riskFactors=[
                {"text": "전선이 바닥에 노출되어 있습니다.", "source": "DETECTED"},
                {"text": "전선이  바닥에\n노출되어 있습니다.", "source": "DETECTED"},
            ]
        )
        assert len(result.value.risk_factors) == 1

    async def test_different_text_is_kept(self) -> None:
        result = run(
            riskFactors=[
                {"text": "전선이 바닥에 노출되어 있습니다.", "source": "DETECTED"},
                {"text": "창문에 안전장치가 없습니다.", "source": "DETECTED"},
            ]
        )
        assert len(result.value.risk_factors) == 2
        assert Repair.DEDUPED_RISK_FACTORS not in result.repairs

    async def test_dedupe_runs_before_truncation(self) -> None:
        """잘라낸 뒤 중복을 지우면 살릴 수 있었던 항목을 잃는다.

        중복 3건 + 고유 4건을 넣으면, 중복 제거가 먼저일 때만 고유 4건이 남는다.
        """
        items = [{"text": "같은 위험입니다.", "source": "OBSERVED"} for _ in range(3)]
        items += [
            {"text": f"고유 위험 {i}입니다.", "source": "OBSERVED"} for i in range(4)
        ]
        result = run(riskFactors=items)

        texts = [f.text for f in result.value.risk_factors]
        assert len(texts) == 5
        assert len([t for t in texts if t.startswith("고유")]) == 4


# =============================================================================
# 항목 구조
# =============================================================================


class TestShape:
    @pytest.mark.parametrize("item", ["문자열", 1, None, []])
    async def test_non_object_risk_factor_is_rejected(self, item) -> None:
        assert run(riskFactors=[item]).rejection is Rejection.BAD_SHAPE

    @pytest.mark.parametrize("item", [1, None, {"text": "가"}])
    async def test_non_string_analysis_is_rejected(self, item) -> None:
        assert run(analysis=[item, "정상 서술입니다."]).rejection is Rejection.BAD_SHAPE

    @pytest.mark.parametrize("value", ["", "   ", None, 1])
    async def test_empty_text_is_rejected(self, value) -> None:
        assert run(riskFactors=[{"text": value, "source": "OBSERVED"}]).rejection is (
            Rejection.BAD_SHAPE
        )

    async def test_text_is_trimmed(self) -> None:
        result = run(
            riskFactors=[{"text": "  전선이 노출되어 있습니다.  ", "source": "DETECTED"}]
        )
        assert result.value.risk_factors[0].text == "전선이 노출되어 있습니다."


# =============================================================================
# 재생성 지시문
# =============================================================================


class TestInstructions:
    @pytest.mark.parametrize("rejection", list(Rejection))
    async def test_every_rejection_has_instruction(self, rejection) -> None:
        """사유만 알리고 다시 부르면 같은 실패가 반복된다."""
        assert rejection.instruction
        assert len(rejection.instruction) > 10

    async def test_rejected_result_carries_no_value(self) -> None:
        result = run(analysis=[])
        assert not result.ok
        assert result.value is None
