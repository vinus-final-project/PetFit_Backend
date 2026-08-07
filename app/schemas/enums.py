"""API 공통 열거형.

API 명세서의 Enum 정의와 1:1로 대응한다.
값을 추가하거나 변경할 때는 DB CHECK 제약과 함께 갱신해야 한다.
"""

from enum import Enum

__all__ = [
    "AnimalGroup",
    "SpaceType",
    "RiskLevel",
    "AnalysisStatus",
    "AnalysisStage",
    "RiskSource",
    "RecommendationType",
    "AnalysisItem",
    "ScoreCategory",
    "progress_for",
]


class AnimalGroup(str, Enum):
    """반려동물 그룹.

    확장 그룹(SMALL_ANIMAL, BIRD, REPTILE)은 코드만 예약하며
    분석 기준이 정의되지 않아 요청 시 400을 반환한다.
    """

    SMALL_DOG = "small_dog"
    LARGE_DOG = "large_dog"
    CAT = "cat"

    SMALL_ANIMAL = "small_animal"
    BIRD = "bird"
    REPTILE = "reptile"

    @classmethod
    def analyzable(cls) -> tuple["AnimalGroup", ...]:
        """분석 기준이 정의된 그룹만 반환한다."""
        return (cls.SMALL_DOG, cls.LARGE_DOG, cls.CAT)

    @property
    def label(self) -> str:
        return _ANIMAL_LABELS[self]


_ANIMAL_LABELS = {
    AnimalGroup.SMALL_DOG: "소형견",
    AnimalGroup.LARGE_DOG: "중·대형견",
    AnimalGroup.CAT: "고양이",
    AnimalGroup.SMALL_ANIMAL: "소동물",
    AnimalGroup.BIRD: "조류",
    AnimalGroup.REPTILE: "파충류",
}


class SpaceType(str, Enum):
    """분석 대상 공간 종류.

    공간에 따라 평가하는 분석 항목이 달라진다.
    서로 다른 공간의 점수는 직접 비교할 수 없다.
    """

    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BALCONY = "balcony"

    @property
    def label(self) -> str:
        return _SPACE_LABELS[self]


_SPACE_LABELS = {
    SpaceType.LIVING_ROOM: "거실",
    SpaceType.BEDROOM: "침실",
    SpaceType.KITCHEN: "주방",
    SpaceType.BALCONY: "베란다",
}


class RiskLevel(str, Enum):
    """위험 수준. 마킹 색상 표시 전용이며 점수 산출에는 사용하지 않는다."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SAFE = "SAFE"

    @property
    def rank(self) -> int:
        """정렬용 순위. 값이 클수록 위험하다."""
        return _RISK_RANK[self]

    @property
    def marking_color(self) -> str | None:
        """마킹 Bounding Box 색상. SAFE는 마킹하지 않으므로 None."""
        return _RISK_COLORS[self]


_RISK_RANK = {RiskLevel.HIGH: 3, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 1, RiskLevel.SAFE: 0}
_RISK_COLORS = {
    RiskLevel.HIGH: "red",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.LOW: "green",
    RiskLevel.SAFE: None,
}


class AnalysisStatus(str, Enum):
    """분석 상태. 부분 성공은 정의하지 않는다."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AnalysisStage(str, Enum):
    """사용자에게 표시하는 진행 단계.

    AI 파이프라인 12단계와 1:1로 대응하지 않는다.
    밀리초 단위로 끝나는 후처리는 하나의 단계로 묶는다.
    """

    FRAME_EXTRACTION = "FRAME_EXTRACTION"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    OBJECT_TRACKING = "OBJECT_TRACKING"
    FRAME_SELECTION = "FRAME_SELECTION"
    RISK_MARKING = "RISK_MARKING"
    SCORE_CALCULATION = "SCORE_CALCULATION"
    ENVIRONMENT_ANALYSIS = "ENVIRONMENT_ANALYSIS"

    @property
    def progress(self) -> int:
        """단계 진입 시점의 진행률. 구간 내에서 보간하지 않는다."""
        return _STAGE_PROGRESS[self]


_STAGE_PROGRESS = {
    AnalysisStage.FRAME_EXTRACTION: 10,
    AnalysisStage.OBJECT_DETECTION: 20,
    AnalysisStage.OBJECT_TRACKING: 55,
    AnalysisStage.FRAME_SELECTION: 65,
    AnalysisStage.RISK_MARKING: 72,
    AnalysisStage.SCORE_CALCULATION: 78,
    AnalysisStage.ENVIRONMENT_ANALYSIS: 82,
}


def progress_for(status: "AnalysisStatus", stage: "AnalysisStage | None") -> int:
    """상태와 단계로 진행률을 산출한다.

    ``progress`` 는 ``stage`` 에서 파생되는 값이므로 컬럼에 직접 대입하지 않는다.
    저장·응답 양쪽이 이 함수를 거쳐야 두 값이 어긋나지 않는다.

    Args:
        status: 분석 상태.
        stage: 현재 단계. PENDING·COMPLETED 에서는 None.

    Returns:
        0 이상 100 이하의 진행률.

    Raises:
        ValueError: PROCESSING 인데 단계가 없는 경우.
    """
    if status is AnalysisStatus.COMPLETED:
        return 100
    if status is AnalysisStatus.PENDING:
        return 0
    if status is AnalysisStatus.PROCESSING and stage is None:
        raise ValueError("PROCESSING 상태에는 stage가 있어야 한다")
    # FAILED는 파이프라인 진입 전 실패 시 stage가 없을 수 있다.
    return stage.progress if stage else 0


class RiskSource(str, Enum):
    """위험 요소의 판단 근거.

    OBSERVED는 객체 탐지 대상 밖의 위험 요소를 다루며 점수에 영향을 주지 않는다.
    """

    DETECTED = "DETECTED"
    OBSERVED = "OBSERVED"


class RecommendationType(str, Enum):
    """환경 개선 추천 유형. 평가 항목과 1:1로 대응한다."""

    SAFETY = "SAFETY"
    ACTIVITY = "ACTIVITY"
    REST = "REST"
    ENVIRONMENT = "ENVIRONMENT"


class ScoreCategory(str, Enum):
    """Pet Fit Score 평가 항목."""

    SAFETY = "safety"
    ACTIVITY = "activity"
    REST = "rest"
    ENVIRONMENT = "environment"

    @property
    def weight(self) -> float:
        """종합 점수 산출 비중."""
        return _CATEGORY_WEIGHTS[self]


_CATEGORY_WEIGHTS = {
    ScoreCategory.SAFETY: 0.40,
    ScoreCategory.ACTIVITY: 0.25,
    ScoreCategory.REST: 0.20,
    ScoreCategory.ENVIRONMENT: 0.15,
}


class AnalysisItem(str, Enum):
    """분석 항목 9종.

    HIDING_SPACE는 판정할 탐지 객체가 정의되지 않아 현재 산출할 수 없다.
    """

    CABLE_EXPOSURE = "전선 노출"
    ACTIVITY_SPACE = "활동 공간"
    SLIP_RISK = "미끄럼 위험"
    REST_SPACE = "휴식 공간"
    WINDOW_SAFETY = "창문 안전"
    VERTICAL_SPACE = "수직 공간"
    STAIRS_RISK = "계단 위험"
    HIDING_SPACE = "숨을 공간"
    FEEDING_ENV = "급식·급수 환경"

    @property
    def category(self) -> ScoreCategory:
        """해당 항목이 속한 평가 항목."""
        return _ITEM_CATEGORY[self]


_ITEM_CATEGORY = {
    AnalysisItem.CABLE_EXPOSURE: ScoreCategory.SAFETY,
    AnalysisItem.SLIP_RISK: ScoreCategory.SAFETY,
    AnalysisItem.STAIRS_RISK: ScoreCategory.SAFETY,
    AnalysisItem.WINDOW_SAFETY: ScoreCategory.SAFETY,
    AnalysisItem.ACTIVITY_SPACE: ScoreCategory.ACTIVITY,
    AnalysisItem.VERTICAL_SPACE: ScoreCategory.ACTIVITY,
    AnalysisItem.REST_SPACE: ScoreCategory.REST,
    AnalysisItem.HIDING_SPACE: ScoreCategory.REST,
    AnalysisItem.FEEDING_ENV: ScoreCategory.ENVIRONMENT,
}
