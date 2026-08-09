"""환경 분석 프롬프트 조립 (파이프라인 12단계).

「프롬프트 설계서 v2.1」의 System Prompt·그룹별 지시문·Few-shot 3종을 코드로 옮긴다.
문서가 정본이므로 문구를 임의로 고치지 않는다. 바꿔야 하면 문서를 먼저 고친다.

**제공자에 중립적인 값만 만든다.** GPT-4o와 Qwen2.5-VL은 메시지 형식과 이미지
전달 방식이 다르다. 여기서는 문자열과 프레임 번호까지만 만들고, 제공자별 요청
본문 조립은 `app/ai/llm/` 의 어댑터가 담당한다. 모델 선정이 바뀌어도 이 파일은
수정하지 않는다.

생성 개수 제한은 `validation` 의 상수를 참조한다. 프롬프트와 검증이 서로 다른
숫자를 들고 있으면 통과할 수 없는 응답을 계속 요구하게 된다.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.pipeline import DetectedObject, select_analysis_frames
from app.ai.score_generator import PetFitScore
from app.ai.validation import (
    MAX_ANALYSIS,
    MAX_RECOMMENDATIONS,
    MAX_RISK_FACTORS,
    MIN_ANALYSIS,
    MIN_RECOMMENDATIONS,
    Rejection,
)
from app.schemas.enums import AnimalGroup, SpaceType
from app.utils.rounding import normalize

__all__ = [
    "AnalysisContext",
    "Prompt",
    "build",
    "select_image_frames",
    "SYSTEM_PROMPT",
    "GROUP_INSTRUCTIONS",
    "NO_IMAGE_RULE",
    "FEW_SHOTS",
]


# =============================================================================
# System Prompt — 프롬프트 설계서 v2.1
# =============================================================================

SYSTEM_PROMPT = """당신은 반려동물 생활환경 분석 전문가입니다.

## 입력

1. detectedObjects : 객체 탐지로 확인된 객체 목록
2. petFitScore     : 규칙 기반으로 이미 산출된 환경 점수
3. animalGroup     : 반려동물 그룹
4. spaceType       : 촬영한 공간 종류 (거실·침실·주방·베란다)
5. occupancyRatio  : 공간 점유율 (0.0 ~ 1.0). 값이 클수록 가구가 공간을 많이 차지합니다.
6. 실내 공간 사진  : 최대 4장

## 수행 작업

1. 사진을 살펴 detectedObjects에 없는 위험 요소를 찾습니다.
2. 위험 요소 식별 (riskFactors)
3. 생활환경 분석 (analysis)
4. 환경 개선 추천 (recommendations)

객체 탐지는 정해진 12종만 인식합니다.
약병, 화분, 비닐봉지처럼 목록에 없는 위험 요소는 사진에서만 확인할 수 있으므로,
반드시 사진을 함께 살펴본 뒤 분석합니다.

## 위험 요소의 두 가지 근거

위험 요소는 근거에 따라 source를 구분하여 출력합니다.

- DETECTED : 입력된 detectedObjects에 있는 객체에 근거한 위험 요소
- OBSERVED : 이미지에서 직접 확인한, detectedObjects에 없는 위험 요소

OBSERVED는 이미지에 **명확히 보이는 것만** 작성합니다.
가려져 있거나 판단이 어려우면 작성하지 않습니다.

OBSERVED 예시
- 약병, 화분, 비닐봉지, 쓰레기통 등 삼킬 수 있는 물건
- 전선이 바닥에 엉켜 있는 상태
- 창문이 열려 있는 상태
- 열린 서랍, 뾰족한 모서리

## 절대 규칙

- 점수(petFitScore)를 생성하지 않습니다. 점수는 이미 산출되어 입력으로 주어집니다.
- DETECTED는 detectedObjects에 있는 객체에만 사용합니다.
  사진에 보이지 않아도 detectedObjects에 있으면 DETECTED로 작성할 수 있습니다.
- OBSERVED는 사진에 명확히 보이는 것만 작성합니다. 추측하지 않습니다.
- 입력된 점수와 상충하는 서술을 하지 않습니다.
  점수가 높은 항목을 위험하다고 서술하지 않습니다.
- JSON 객체 하나만 출력합니다. 설명 문장이나 코드펜스를 포함하지 않습니다.
- 모든 텍스트는 한국어로 작성합니다.

## 서술 방식

- 보호자에게 설명하듯 존댓말로 작성합니다.
- 한 항목은 1~2문장으로 작성합니다.
- 불안을 과장하지 않고 사실과 조치 중심으로 서술합니다.
- 수의학적 진단이나 질병을 단정하지 않습니다.
- 추천은 보호자가 바로 실행할 수 있는 구체적 행동으로 작성합니다.

## 출력 형식

{
  "riskFactors": [
    { "text": "...", "source": "DETECTED|OBSERVED" }
  ],
  "analysis": ["..."],
  "recommendations": [
    {
      "type": "SAFETY|ACTIVITY|REST|ENVIRONMENT",
      "text": "...",
      "priority": 1,
      "source": "DETECTED|OBSERVED"
    }
  ]
}

## 공간 종류 반영

spaceType에 해당하지 않는 항목은 점수에서 제외되어 있습니다.
제외된 항목을 부족하다고 서술하지 않습니다.

- 침실  : 급식·급수 환경을 언급하지 않습니다.
- 주방  : 휴식 공간, 수직 공간, 미끄럼 위험, 활동 공간을 언급하지 않습니다.
- 베란다 : 전선·계단·창문 외의 항목을 언급하지 않습니다.

침실에 급수기가 없는 것은 정상입니다. 이를 개선 사항으로 제시하지 않습니다.

## 서술 일관성 규칙

- analysis에는 근거 구분을 표기하지 않습니다. 종합 서술이기 때문입니다.
- 다만 이미지에서만 확인한 내용을 analysis에 단독으로 쓰지 않습니다.
  반드시 riskFactors에 OBSERVED로 먼저 등재한 뒤 analysis에서 언급합니다.
- 동일한 위험 요소를 riskFactors에 두 번 이상 쓰지 않습니다.
  같은 종류의 객체가 여러 개 탐지되어도 위험 요소는 하나로 묶어 서술합니다."""


#: 생성 개수 안내.
#:
#: 설계서의 System Prompt에는 없고 "생성 개수 제한" 표에만 있는 내용이다.
#: 모델에게 알려주지 않으면 범위를 벗어난 응답이 나와 재생성이 늘어난다.
#: 숫자는 `validation` 의 상수를 그대로 쓴다. 두 곳이 어긋나면 통과할 수 없는
#: 응답을 계속 요구하게 된다.
COUNT_RULES = f"""

## 생성 개수

- riskFactors     : {MAX_RISK_FACTORS}건 이하. 위험이 없으면 빈 배열로 출력합니다.
- analysis        : {MIN_ANALYSIS}건 이상 {MAX_ANALYSIS}건 이하
- recommendations : {MIN_RECOMMENDATIONS}건 이상 {MAX_RECOMMENDATIONS}건 이하
- priority        : 1부터 시작하는 연속된 정수를 중복 없이 부여합니다."""


# =============================================================================
# 반려동물 그룹별 지시문 — 프롬프트 설계서 v2.1
# =============================================================================

_SMALL_DOG = """소형견 기준으로 분석합니다.

우선 확인 항목
- 전선 노출 (매우 중요) : 체구가 작아 바닥 전선에 접근하기 쉽습니다.
- 계단 위험 : 관절이 약해 계단 오르내림이 부담이 됩니다.
- 미끄럼 위험 : 미끄러운 바닥은 슬개골에 부담을 줍니다.
- 활동 공간 : 가구가 공간을 많이 차지하면 움직임이 제한됩니다.
- 휴식 공간 : 조용하고 아늑한 휴식처가 필요합니다.
- 급식·급수 환경 : 급식기와 급수기가 모두 갖춰져야 합니다.

창문과 수직 공간은 소형견에게 중요도가 낮으므로 강조하지 않습니다."""

_LARGE_DOG = """중·대형견 기준으로 분석합니다.

우선 확인 항목
- 전선 노출 (매우 중요)
- 활동 공간 (매우 중요) : 체구가 커서 충분한 이동 공간이 필요합니다.
- 미끄럼 위험 (매우 중요) : 체중이 실려 관절 부담과 낙상 위험이 큽니다.
- 계단 위험 (매우 중요)
- 휴식 공간 : 몸을 완전히 뻗을 수 있는 크기가 필요합니다.
- 급식·급수 환경 : 급식기와 급수기가 모두 갖춰져야 합니다.

창문과 수직 공간은 중·대형견에게 중요도가 낮으므로 강조하지 않습니다."""

_CAT = """고양이 기준으로 분석합니다.

우선 확인 항목
- 창문 안전 (매우 중요) : 추락 위험이 있어 안전망이 필요합니다.
- 수직 공간 (매우 중요) : 높은 곳에 오르는 습성이 있습니다. 캣타워로 판정합니다.
- 숨을 공간 (매우 중요) : 스트레스 완화를 위해 은신처가 필요합니다.
  은신처는 객체 탐지 대상이 아닙니다.
  사진에서 박스·터널·펫하우스가 보이면 OBSERVED로 언급하고,
  보이지 않으면 언급하지 않습니다.
- 전선 노출 : 물어뜯을 수 있습니다.
- 활동 공간 : 바닥 공간에 여유가 있어야 합니다.
- 휴식 공간 : 조용한 휴식처가 필요합니다.
- 급식·급수 환경 : 급식기와 급수기가 모두 갖춰져야 합니다.

계단과 미끄럼은 고양이에게 중요도가 낮으므로 강조하지 않습니다."""

#: 이미지 없이 호출할 때 덧붙이는 규칙.
#:
#: 프레임을 읽지 못해도 분석은 진행한다. 다만 관찰 근거가 없으므로 `OBSERVED` 를
#: 만들면 전부 환각이다. 점수에는 영향이 없지만 사용자에게 없는 위험을 알리게 된다.
NO_IMAGE_RULE = """## 사진 없음

이번 요청에는 사진이 포함되지 않았습니다.
OBSERVED 위험 요소와 추천을 작성하지 마세요. detectedObjects만 근거로 사용합니다."""


#: 그룹별 지시문. 확장 그룹은 분석 기준이 정의되지 않아 포함하지 않는다.
GROUP_INSTRUCTIONS: dict[AnimalGroup, str] = {
    AnimalGroup.SMALL_DOG: _SMALL_DOG,
    AnimalGroup.LARGE_DOG: _LARGE_DOG,
    AnimalGroup.CAT: _CAT,
}


# =============================================================================
# Few-shot — 프롬프트 설계서 v2.1
# =============================================================================


@dataclass(frozen=True)
class FewShot:
    """예시 1건. 입력과 출력을 JSON 문자열로 보관한다.

    실제 호출에는 이미지가 함께 전달되지만 예시는 텍스트만 쓴다. 예시용 이미지를
    저장소에 두면 관리 대상이 늘고, 형식 학습에는 텍스트만으로 충분하다.
    """

    user: str
    assistant: str


def _shot(user: dict, assistant: dict) -> FewShot:
    return FewShot(
        user=json.dumps(user, ensure_ascii=False, indent=2),
        assistant=json.dumps(assistant, ensure_ascii=False, indent=2),
    )


#: Few-shot 3종. 그룹별 차이와 "위험 없음" 경우를 함께 보여준다.
#: 점수는 AI 분석 정의서의 계산 규칙으로 검산된 값이다. 임의로 바꾸면
#: 모델에게 점수와 서술이 어긋난 예시를 학습시키게 된다.
FEW_SHOTS: tuple[FewShot, ...] = (
    # 고양이 — 창문 위험, OBSERVED 발견
    _shot(
        {
            "animalGroup": "cat",
            "spaceType": "living_room",
            "representativeFrame": {"frameNumber": 112},
            "detectedObjects": [
                {"name": "창문", "risk": "HIGH"},
                {"name": "캣타워", "risk": "SAFE"},
                {"name": "소파", "risk": "SAFE"},
                {"name": "카펫", "risk": "SAFE"},
                {"name": "급식기", "risk": "SAFE"},
                {"name": "급수기", "risk": "SAFE"},
            ],
            "occupancyRatio": 0.35,
            "petFitScore": {
                "total": 75,
                "safety": 57,
                "activity": 100,
                "rest": 60,
                "environment": 100,
            },
        },
        {
            "riskFactors": [
                {
                    "text": "창문에 안전장치가 없어 추락 위험이 있습니다.",
                    "source": "DETECTED",
                },
                {
                    "text": "창가 선반에 화분이 놓여 있어 흙을 파헤치거나 삼킬 수 있습니다.",
                    "source": "OBSERVED",
                },
            ],
            "analysis": [
                "캣타워가 있어 수직 활동 공간은 우수하며 급식·급수 환경도 갖춰져 있습니다.",
                "창문 주변 안전 조치가 필요하여 안전성 점수가 낮게 산출되었습니다.",
                "고양이 전용 휴식처가 없어 소파를 함께 사용하고 있는 것으로 보입니다.",
            ],
            "recommendations": [
                {
                    "type": "SAFETY",
                    "text": "창문 안전망을 설치하는 것을 권장합니다.",
                    "priority": 1,
                    "source": "DETECTED",
                },
                {
                    "type": "SAFETY",
                    "text": "화분을 고양이가 닿지 않는 곳으로 옮겨주세요.",
                    "priority": 2,
                    "source": "OBSERVED",
                },
                {
                    "type": "REST",
                    "text": "조용한 곳에 고양이 전용 방석을 놓아주세요.",
                    "priority": 3,
                    "source": "DETECTED",
                },
            ],
        },
    ),
    # 소형견 — 전선 위험
    _shot(
        {
            "animalGroup": "small_dog",
            "spaceType": "living_room",
            "representativeFrame": {"frameNumber": 18},
            "detectedObjects": [
                {"name": "전선", "risk": "HIGH"},
                {"name": "소파", "risk": "SAFE"},
                {"name": "급수기", "risk": "SAFE"},
            ],
            "occupancyRatio": 0.45,
            "petFitScore": {
                "total": 50,
                "safety": 50,
                "activity": 40,
                "rest": 60,
                "environment": 50,
            },
        },
        {
            "riskFactors": [
                {
                    "text": "전선이 바닥에 엉킨 채 노출되어 있습니다.",
                    "source": "DETECTED",
                },
                {
                    "text": "낮은 선반에 약병이 놓여 있어 반려동물이 닿을 수 있습니다.",
                    "source": "OBSERVED",
                },
            ],
            "analysis": [
                "바닥에 미끄럼 방지 매트가 없어 관절에 부담이 될 수 있습니다.",
                "휴식 공간은 소파뿐이며 반려동물 전용 자리가 마련되어 있지 않습니다.",
            ],
            "recommendations": [
                {
                    "type": "SAFETY",
                    "text": "전선을 벽면으로 정리하거나 전선 커버를 씌워주세요.",
                    "priority": 1,
                    "source": "DETECTED",
                },
                {
                    "type": "SAFETY",
                    "text": "약병을 서랍이나 높은 곳으로 옮겨주세요.",
                    "priority": 2,
                    "source": "OBSERVED",
                },
                {
                    "type": "REST",
                    "text": "조용한 곳에 반려동물 전용 침대를 놓아주세요.",
                    "priority": 3,
                    "source": "DETECTED",
                },
            ],
        },
    ),
    # 중·대형견 — 위험 요소 없음
    _shot(
        {
            "animalGroup": "large_dog",
            "spaceType": "living_room",
            "representativeFrame": {"frameNumber": 22},
            "detectedObjects": [
                {"name": "소파", "risk": "SAFE"},
                {"name": "카펫", "risk": "SAFE"},
                {"name": "급식기", "risk": "SAFE"},
            ],
            "occupancyRatio": 0.30,
            "petFitScore": {
                "total": 78,
                "safety": 100,
                "activity": 75,
                "rest": 60,
                "environment": 50,
            },
        },
        {
            "riskFactors": [],
            "analysis": [
                "위험 요소가 확인되지 않은 안전한 공간입니다.",
                "카펫이 깔려 있어 미끄럼 위험이 낮습니다.",
                "급수기가 확인되지 않아 생활환경 점수가 낮게 산출되었습니다.",
            ],
            "recommendations": [
                {
                    "type": "ENVIRONMENT",
                    "text": "급수기를 설치하여 언제든 물을 마실 수 있도록 해주세요.",
                    "priority": 1,
                    "source": "DETECTED",
                },
                {
                    "type": "REST",
                    "text": "체구에 맞는 넓은 반려동물 침대를 마련해주세요.",
                    "priority": 2,
                    "source": "DETECTED",
                },
            ],
        },
    ),
)


# =============================================================================
# 조립
# =============================================================================


@dataclass(frozen=True)
class AnalysisContext:
    """12단계 입력.

    앞 단계(1~11)의 산출물을 모은 것이다. 점수는 여기서 만들지 않고 이미 산출된
    값을 받는다. 서술이 점수와 어긋나지 않게 하려면 모델이 점수를 알아야 한다.
    """

    group: AnimalGroup
    space: SpaceType
    objects: Sequence[DetectedObject]
    occupancy_ratio: float
    score: PetFitScore
    #: 분석 대표 프레임 번호. 썸네일로 쓰이는 프레임이다.
    thumbnail_frame: int


@dataclass(frozen=True)
class Prompt:
    """제공자에 중립적인 프롬프트.

    어댑터가 이 값을 각 제공자의 요청 본문으로 옮긴다.
    """

    system: str
    few_shots: tuple[FewShot, ...]
    user: str
    #: 함께 보낼 원본 프레임 번호. 마킹하지 않은 이미지를 보낸다.
    image_frames: tuple[int, ...]


def build(
    context: AnalysisContext,
    *,
    retry: Rejection | None = None,
    include_few_shots: bool = True,
    has_images: bool = True,
) -> Prompt:
    """프롬프트를 조립한다.

    Args:
        context: 12단계 입력.
        retry: 직전 응답이 거절된 사유. 지정하면 재생성 지시문을 덧붙인다.
            사유를 알리지 않고 다시 부르면 같은 실패가 반복된다.
        include_few_shots: 예시 포함 여부. 토큰 사용량과 형식 준수율의
            균형은 성능평가에서 측정한다.
        has_images: 이미지를 함께 보내는지 여부. 보내지 못하면 관찰 근거가
            없으므로 `OBSERVED` 를 만들지 말라고 알린다.

    Returns:
        조립된 프롬프트.

    Raises:
        KeyError: 분석 기준이 정의되지 않은 반려동물 그룹인 경우.
    """
    system = "\n\n".join(
        (SYSTEM_PROMPT, COUNT_RULES.strip(), GROUP_INSTRUCTIONS[context.group])
    )

    user = json.dumps(_user_payload(context), ensure_ascii=False, indent=2)
    if not has_images:
        user = f"{user}\n\n{NO_IMAGE_RULE}"
    if retry is not None:
        user = f"{user}\n\n## 재생성 요청\n\n{retry.instruction}"

    return Prompt(
        system=system,
        few_shots=FEW_SHOTS if include_few_shots else (),
        user=user,
        image_frames=select_image_frames(context),
    )


def _user_payload(context: AnalysisContext) -> dict:
    """모델에 전달할 입력 JSON.

    Few-shot 예시와 동일한 구조여야 한다. 예시와 실제 입력의 모양이 다르면
    예시가 형식을 알려주는 역할을 하지 못한다.

    `occupancyRatio` 는 활동성 점수의 산출 근거다. 이 값이 없으면 모델은
    "활동 공간이 부족합니다" 라고 쓸 근거를 갖지 못한다. 탐지 목록에는 가구의
    개수만 있고 차지하는 면적은 없다.
    """
    return {
        "animalGroup": context.group.value,
        "spaceType": context.space.value,
        "representativeFrame": {"frameNumber": context.thumbnail_frame},
        "detectedObjects": [
            {"name": o.name, "risk": o.risk.value} for o in _ordered(context.objects)
        ],
        # 저장 자릿수(NUMERIC(5,4))와 맞춘다. 같은 입력에 같은 프롬프트가 나와야 한다.
        "occupancyRatio": normalize(context.occupancy_ratio),
        "petFitScore": context.score.as_dict(),
    }


def _ordered(objects: Sequence[DetectedObject]) -> list[DetectedObject]:
    """탐지 객체를 결정적 순서로 정렬한다.

    입력 순서가 흔들리면 같은 분석에 서로 다른 프롬프트가 만들어져, 생성 결과의
    차이가 모델 때문인지 입력 때문인지 구분할 수 없게 된다.

    위험한 것을 앞에 두어 모델이 먼저 읽게 한다.
    """
    return sorted(
        objects, key=lambda o: (-o.risk.rank, -o.confidence, o.name, o.frame_number)
    )


def select_image_frames(context: AnalysisContext) -> tuple[int, ...]:
    """함께 보낼 원본 프레임 번호를 고른다.

    **선정 규칙은 `app.ai.pipeline` 에 있다.** Vision 이 같은 규칙으로 프레임을
    남기므로, 여기서 따로 구현하면 한쪽만 고쳐졌을 때 요청한 프레임이 없어
    이미지가 조용히 빠진다.

    **마킹하지 않은 원본을 보낸다.** Bounding Box를 그린 이미지를 보내면 모델이
    표시된 객체에만 주목하여 그 외의 위험 요소를 놓친다. 탐지 대상 12종 밖의
    위험을 찾는 것이 이미지를 보내는 이유이므로 목적과 어긋난다.

    Returns:
        프레임 번호. 최대 `LLM_MAX_IMAGES` 개이며 중복이 없다.
    """
    return select_analysis_frames(context.thumbnail_frame, context.objects)
