"""AI 분석 파이프라인 인터페이스.

**서비스 계층과 AI 계층의 경계다.** 서비스 계층은 이 계약만 알고, AI 계층은
이 계약만 지킨다. 양쪽이 서로의 내부를 몰라도 병렬로 개발할 수 있다.

파이프라인 12단계의 내부 구성은 AI 설계서를 따른다. 여기서는 **입력과 출력,
그리고 진행 상황을 알리는 방법**만 정의한다.

구현체는 두 가지다.

    StubPipeline  : 고정된 탐지 결과를 반환한다. 프론트 연동과 통합 검증용
    (미구현)      : 실제 YOLO·VLM 파이프라인

교체 시 서비스 계층은 수정하지 않는다.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.constants import LLM_MAX_IMAGES
from app.schemas.enums import (
    AnalysisStage,
    AnimalGroup,
    RecommendationType,
    RiskLevel,
    RiskSource,
    SpaceType,
)

__all__ = [
    "DetectedObject",
    "RiskFactor",
    "Recommendation",
    "PipelineResult",
    "StageCallback",
    "Pipeline",
    "PipelineError",
    "select_analysis_frames",
]


class PipelineError(Exception):
    """파이프라인 실행 실패.

    실패한 단계를 함께 전달한다. 클라이언트가 재촬영과 재시도 중 무엇을
    안내할지 이 값으로 분기하므로, 단계를 잃으면 안내가 불가능해진다.

    Args:
        message: 사용자에게 표시할 실패 사유. 내부 오류 메시지를 그대로 넣지 않는다.
        stage: 실패한 단계. 파이프라인 진입 전 실패면 None.
    """

    def __init__(self, message: str, stage: AnalysisStage | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage


@dataclass(frozen=True)
class DetectedObject:
    """탐지된 객체 인스턴스 1개.

    동일한 ``name`` 이 여러 번 나타날 수 있다. 전선이 두 곳에 있으면 2건이다.

    좌표는 ``frame_number`` 프레임 기준 정규화 값(0.0~1.0)이다.
    """

    name: str
    risk: RiskLevel
    confidence: float
    detection_frame_count: int
    frame_number: int
    x: float
    y: float
    width: float
    height: float
    #: 마킹 이미지 경로. SAFE는 생성하지 않으므로 None.
    marked_image_path: str | None = None


@dataclass(frozen=True)
class RiskFactor:
    """위험 요소 서술 1건."""

    text: str
    source: RiskSource


@dataclass(frozen=True)
class Recommendation:
    """환경 개선 추천 1건. ``priority`` 는 1부터 시작하며 분석 내에서 중복되지 않는다."""

    type: RecommendationType
    text: str
    priority: int
    source: RiskSource


@dataclass(frozen=True)
class PipelineResult:
    """파이프라인 산출물 전체.

    부분 성공은 정의하지 않는다. 이 객체가 반환되면 모든 단계가 성공한 것이다.
    """

    #: 영상 길이(초). 3.0 이상 30.0 이하.
    capture_duration: float
    #: 추출한 프레임 수. 15 이상 30 이하.
    frame_count: int
    #: 분석 대표 프레임 경로. 목록·썸네일용
    thumbnail_path: str
    #: 프레임별 활동 공간 점유율의 중앙값. 0.0 이상 1.0 이하.
    occupancy_ratio: float

    detected_objects: Sequence[DetectedObject] = field(default_factory=tuple)
    risk_factors: Sequence[RiskFactor] = field(default_factory=tuple)
    #: 생활환경 분석 서술 2~4건
    analysis: Sequence[str] = field(default_factory=tuple)
    recommendations: Sequence[Recommendation] = field(default_factory=tuple)

    @property
    def object_names(self) -> set[str]:
        """점수 산출에 넘길 객체 이름 집합.

        Score Generator는 인스턴스 수가 아니라 **존재 여부**로 감점을 판정한다.
        """
        return {o.name for o in self.detected_objects}

    @property
    def marked_image_paths(self) -> list[str]:
        """생성된 마킹 이미지 경로. 삭제·재시도 시 정리 대상이다."""
        return [o.marked_image_path for o in self.detected_objects if o.marked_image_path]


def select_analysis_frames(
    thumbnail_frame: int,
    objects: Sequence[DetectedObject],
    limit: int = LLM_MAX_IMAGES,
) -> tuple[int, ...]:
    """환경 분석에 함께 보낼 원본 프레임 번호를 고른다.

    **이 규칙을 쓰는 곳이 둘이다.** Vision(9·10단계)은 고른 프레임을 실제로
    들고 있어야 하고, 환경 분석(12단계)은 그중 무엇을 보낼지 정해야 한다.
    각자 구현하면 한쪽만 고쳐졌을 때 요청한 프레임이 없어 이미지가 조용히 빠진다.
    양쪽이 이미 이 모듈에 의존하므로 여기에 둔다.

    분석 대표 프레임 1장에 위험 객체의 대표 프레임을 위험도 순으로 채운다.
    ``SAFE`` 는 넣지 않는다. 탐지 대상 12종 밖의 위험 요소를 찾는 것이 이미지를
    보내는 이유이므로, 위험 판정이 붙은 장면을 우선 보낸다.

    같은 프레임을 두 번 넣지 않는다. 중복으로 상한을 채우면 그만큼 다른 장면을
    보지 못한다.

    Args:
        thumbnail_frame: 분석 대표 프레임 번호. 항상 첫 번째로 들어간다.
        objects: 채택된 객체. 위험도와 confidence 가 채워져 있어야 한다.
        limit: 최대 장수.

    Returns:
        프레임 번호. 중복이 없으며 ``limit`` 을 넘지 않는다.
    """
    risky = sorted(
        (o for o in objects if o.risk is not RiskLevel.SAFE),
        key=lambda o: (-o.risk.rank, -o.confidence, o.frame_number),
    )

    numbers: list[int] = []
    for number in (thumbnail_frame, *(o.frame_number for o in risky)):
        if number not in numbers:
            numbers.append(number)
        if len(numbers) == limit:
            break
    return tuple(numbers)


#: 단계 진입을 알리는 콜백.
#:
#: 서비스 계층이 이 콜백에서 ``progress_for()`` 로 진행률을 계산해 DB에 기록한다.
#: 파이프라인은 진행률 수치를 알 필요가 없다.
StageCallback = Callable[[AnalysisStage], Awaitable[None]]


@runtime_checkable
class Pipeline(Protocol):
    """AI 분석 파이프라인.

    구현체는 12단계를 순서대로 수행하고, 각 단계 진입 시 ``on_stage`` 를 호출한다.
    """

    async def run(
        self,
        video_path: Path,
        group: AnimalGroup,
        space: SpaceType,
        on_stage: StageCallback,
    ) -> PipelineResult:
        """분석을 수행한다.

        Args:
            video_path: 업로드된 영상 파일 경로.
            group: 반려동물 그룹. 위험도 판정과 가중치에 사용한다.
            space: 촬영한 공간 종류. 평가 항목 적용 범위를 결정한다.
            on_stage: 단계 진입 시 호출할 콜백. 반드시 await 한다.

        Returns:
            파이프라인 산출물.

        Raises:
            PipelineError: 어느 단계에서든 실패한 경우. 실패한 단계를 포함한다.
        """
        ...
