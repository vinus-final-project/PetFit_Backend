"""Vision 파이프라인 조립 (1~10단계).

여섯 부품을 순서대로 엮고 단계마다 진행 상황을 알린다.

    FRAME_EXTRACTION  10%   frames.extract
    OBJECT_DETECTION  20%   detector.detect          <- 가장 오래 걸린다
    OBJECT_TRACKING   55%   occupancy + tracking
    FRAME_SELECTION   65%   imaging.select_analysis_frame
    RISK_MARKING      72%   imaging.build

**무거운 호출을 스레드로 내보낸다.** 영상 디코딩과 모델 추론은 CPU를 점유하는
동기 작업이라 이벤트 루프에서 그대로 실행하면 수십 초 동안 서버 전체가 멈춘다.
동시 처리 2건이 사실상 1건이 되고, 진행률 폴링도 응답하지 않아 프론트에는
멈춘 것처럼 보인다.

12단계(환경 분석)는 여기 없다. 생성형 AI는 별도 담당이며, 두 산출물을 합쳐
``app.ai.pipeline.Pipeline`` 계약을 만족시키는 조립은 상위에서 한다.
"""

import asyncio
import logging
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from app.ai.pipeline import PipelineError, StageCallback
from app.ai.vision import frames as frame_module
from app.ai.vision import imaging
from app.ai.vision.detector import Detector
from app.ai.vision.occupancy import occupancy_ratio
from app.ai.vision.tracking import IouTracker, Tracker, adopt
from app.ai.vision.types import Detection, ImageSink, VisionResult
from app.core.constants import FRAME_MAX_EDGE
from app.schemas.enums import AnalysisStage, AnimalGroup

__all__ = ["VisionPipeline", "STAGE_MESSAGES"]

logger = logging.getLogger(__name__)

#: 단계별 실패 사유. 사용자에게 그대로 표시되므로 내부 오류를 넣지 않는다.
#:
#: 프론트는 실패 단계로 재촬영과 재시도 중 무엇을 안내할지 나눈다.
#: 앞 단계는 영상 품질 문제일 가능성이 높아 재촬영, 뒤 단계는 재시도다.
STAGE_MESSAGES: dict[AnalysisStage, str] = {
    AnalysisStage.OBJECT_DETECTION: "영상에서 객체를 인식하지 못했습니다. 다시 촬영해주세요.",
    AnalysisStage.OBJECT_TRACKING: "탐지 결과를 정리하지 못했습니다. 다시 시도해주세요.",
    AnalysisStage.FRAME_SELECTION: "대표 장면을 선택하지 못했습니다. 다시 시도해주세요.",
    AnalysisStage.RISK_MARKING: "결과 이미지를 만들지 못했습니다. 다시 시도해주세요.",
}


class VisionPipeline:
    """영상에서 객체 목록과 시각화까지 만든다.

    탐지기와 추적기를 주입받는다. 성능평가로 구현이 바뀌어도 이 파일은
    수정하지 않는다.
    """

    def __init__(
        self,
        detector: Detector,
        storage: ImageSink,
        tracker: Tracker | None = None,
        max_edge: int = FRAME_MAX_EDGE,
    ) -> None:
        """
        Args:
            detector: 객체 탐지기.
            storage: 이미지 저장소.
            tracker: 객체 추적기. 기본값은 IoU 병합이다.
            max_edge: 보관할 프레임의 긴 변 픽셀 상한.
        """
        self._detector = detector
        self._storage = storage
        self._tracker = tracker or IouTracker()
        self._max_edge = max_edge

    async def run(
        self,
        video_path: Path,
        group: AnimalGroup,
        on_stage: StageCallback,
    ) -> VisionResult:
        """1~10단계를 수행한다.

        Args:
            video_path: 업로드된 영상 파일 경로.
            group: 반려동물 그룹. 위험도 판정에 쓴다.
            on_stage: 단계 진입 시 호출할 콜백.

        Returns:
            객체 목록·점유율·시각화 산출물.

        Raises:
            PipelineError: 어느 단계에서든 실패한 경우. 실패한 단계를 포함한다.
        """
        await on_stage(AnalysisStage.FRAME_EXTRACTION)
        # extract 는 자체적으로 FRAME_EXTRACTION 단계의 PipelineError 를 낸다.
        video = await asyncio.to_thread(
            frame_module.extract, video_path, self._max_edge
        )

        await on_stage(AnalysisStage.OBJECT_DETECTION)
        with _guard(AnalysisStage.OBJECT_DETECTION):
            detections = await asyncio.to_thread(self._detector.detect, video.frames)
        _check_alignment(detections, len(video.frames))

        await on_stage(AnalysisStage.OBJECT_TRACKING)
        with _guard(AnalysisStage.OBJECT_TRACKING):
            # 점유율을 추적보다 먼저 구한다. 통합 후에는 프레임별 정보가 사라진다.
            #
            # 스레드로 내보내지 않는다. 좌표 산술만 하므로 30프레임 기준
            # 밀리초 단위이고, 스레드 전환 비용이 계산보다 크다.
            occupancy = occupancy_ratio(detections)
            objects = adopt(self._tracker.track(detections))

        await on_stage(AnalysisStage.FRAME_SELECTION)
        with _guard(AnalysisStage.FRAME_SELECTION):
            analysis_frame = imaging.select_analysis_frame(
                video.frames, detections, {o.class_code for o in objects}
            )

        await on_stage(AnalysisStage.RISK_MARKING)
        with _guard(AnalysisStage.RISK_MARKING):
            visuals = await asyncio.to_thread(
                imaging.build,
                video.frames,
                analysis_frame,
                objects,
                group,
                self._storage,
            )

        logger.info(
            "Vision 완료: %s장 / 객체 %s건 / 점유율 %s / 마킹 %s장",
            video.count,
            len(visuals.detected_objects),
            occupancy,
            sum(1 for o in visuals.detected_objects if o.marked_image_path),
        )

        return VisionResult(
            capture_duration=video.duration,
            frame_count=video.count,
            occupancy_ratio=occupancy,
            thumbnail_path=visuals.thumbnail_path,
            detected_objects=tuple(visuals.detected_objects),
            analysis_frames=tuple(visuals.analysis_frames),
        )


@contextmanager
def _guard(stage: AnalysisStage):
    """단계에서 발생한 예외를 실패 단계가 붙은 오류로 바꾼다.

    단계를 잃으면 재촬영과 재시도 중 무엇을 안내할지 분기할 수 없다.
    라이브러리 예외 메시지는 사용자에게 노출하지 않고 로그에만 남긴다.
    """
    try:
        yield
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s 단계 실패", stage.value)
        raise PipelineError(STAGE_MESSAGES[stage], stage) from exc


def _check_alignment(detections: Sequence[Sequence[Detection]], expected: int) -> None:
    """탐지 결과가 프레임과 1:1로 대응하는지 확인한다.

    어긋나면 점유율이 엉뚱한 프레임 수로 계산되고 대표 프레임 선정도 밀린다.
    증상이 뒤 단계에서 나타나 원인을 탐지기까지 되짚기 어려우므로 여기서 막는다.
    """
    if len(detections) != expected:
        raise PipelineError(
            STAGE_MESSAGES[AnalysisStage.OBJECT_DETECTION],
            AnalysisStage.OBJECT_DETECTION,
        )
