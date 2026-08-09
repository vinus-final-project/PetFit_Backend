"""분석 파이프라인 조립 (1~12단계).

Vision(1~10)과 환경 분석(12)을 엮어 `app.ai.pipeline.Pipeline` 계약을 만족시킨다.
그 사이에 점수 산출(11)이 들어간다.

    VisionPipeline      1~10  객체 목록 · 점유율 · 대표 프레임 · 마킹 이미지
    Score Generator     11    규칙 기반 점수
    EnvironmentAnalyzer 12    위험 요소 · 서술 · 개선 추천

**11단계가 12단계보다 먼저다.** 산출된 점수를 모델 입력으로 넘겨야 서술이 점수와
어긋나지 않는다. "안전성 100점"인데 "전선이 위험합니다"가 함께 나오면 사용자는
결과를 신뢰하지 않는다.

여기서 만든 점수는 `AnalysisService.mark_completed` 가 저장 시점에 다시 산출한다.
같은 입력에 같은 값을 내는 규칙 기반 함수이므로 두 값은 항상 일치한다. 서비스
계층이 파이프라인 구현을 믿지 않고 스스로 계산하는 편이 안전하다.
"""

import asyncio
import io
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.ai.environment_analysis import EnvironmentAnalyzer
from app.ai.pipeline import PipelineResult, StageCallback
from app.ai.prompts import AnalysisContext
from app.ai.score_generator import generate
from app.ai.vision.pipeline import VisionPipeline
from app.ai.vision.types import Frame, ImageStore, VisionResult
from app.schemas.enums import AnalysisStage, AnimalGroup, SpaceType

__all__ = ["AnalysisPipeline", "encode_frames", "JPEG_QUALITY"]

logger = logging.getLogger(__name__)

#: 모델에 보낼 프레임의 JPEG 품질.
#:
#: 원본은 이미 긴 변 1280으로 줄여 둔 것이다. 여기서 더 낮추면 바닥에 놓인
#: 약병이나 가는 전선처럼 작은 물체가 뭉개져, 이미지를 보내는 이유인
#: `OBSERVED` 발견율이 떨어진다.
JPEG_QUALITY = 85


def encode_frames(frames: Sequence[Frame], quality: int = JPEG_QUALITY) -> dict[int, bytes]:
    """프레임을 모델에 보낼 JPEG 바이트로 바꾼다.

    `VisionLLM` 규약이 바이트를 받는 이유는 제공자마다 요구가 다르기 때문이다.
    MLX는 경로를, 외부 API는 base64를 받는다. 바이트로 넘기고 변환은 어댑터가 한다.

    한 장이 실패해도 나머지는 보낸다. 이미지 하나 때문에 분석 전체를 실패로 만들
    이유가 없다. 전부 실패하면 탐지 결과만으로 서술한다.

    Args:
        frames: 마킹하지 않은 원본 프레임.
        quality: JPEG 품질.

    Returns:
        프레임 번호 → JPEG 바이트.
    """
    encoded: dict[int, bytes] = {}
    for frame in frames:
        try:
            image = frame.image
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            encoded[frame.number] = buffer.getvalue()
        except Exception as exc:  # noqa: BLE001
            logger.info("프레임 %s 변환 실패: %s", frame.number, type(exc).__name__)
    return encoded


class AnalysisPipeline:
    """1~12단계 전체를 수행한다.

    Args:
        vision: 1~10단계.
        analyzer: 12단계.
        store: 이미지 저장소. 12단계에서 실패했을 때 Vision이 만든 파일을
            지우는 데 쓴다. 생략하면 정리하지 않는다.
        jpeg_quality: 모델에 보낼 프레임의 품질.
    """

    def __init__(
        self,
        vision: VisionPipeline,
        analyzer: EnvironmentAnalyzer,
        store: ImageStore | None = None,
        *,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        self._vision = vision
        self._analyzer = analyzer
        self._store = store
        self._quality = jpeg_quality

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
            group: 반려동물 그룹.
            space: 촬영한 공간 종류. 평가 항목 적용 범위를 결정한다.
            on_stage: 단계 진입 시 호출할 콜백.

        Returns:
            파이프라인 산출물.

        Raises:
            PipelineError: 어느 단계에서든 실패한 경우. 실패한 단계를 포함한다.
        """
        vision = await self._vision.run(video_path, group, on_stage)

        try:
            await on_stage(AnalysisStage.SCORE_CALCULATION)
            score = generate(group, space, vision.object_names, vision.occupancy_ratio)

            await on_stage(AnalysisStage.ENVIRONMENT_ANALYSIS)
            report = await self._analyzer.analyze(
                AnalysisContext(
                    group=group,
                    space=space,
                    objects=vision.detected_objects,
                    occupancy_ratio=vision.occupancy_ratio,
                    score=score,
                    thumbnail_frame=vision.thumbnail_frame,
                ),
                await self._images(vision.analysis_frames),
            )
        except Exception:
            # Vision이 만든 이미지는 이 시점까지 DB에 기록되지 않았다. 여기서
            # 지우지 않으면 참조 없이 디스크에 남고, 경로를 잃어 이후 추적할 수 없다.
            self._discard(vision)
            raise

        logger.info(
            "환경 분석 완료: 재생성 %d회, 이미지 %d장",
            report.regenerations,
            report.images_sent,
        )

        return PipelineResult(
            capture_duration=vision.capture_duration,
            frame_count=vision.frame_count,
            thumbnail_path=vision.thumbnail_path,
            occupancy_ratio=vision.occupancy_ratio,
            detected_objects=vision.detected_objects,
            risk_factors=report.output.risk_factors,
            analysis=report.output.analysis,
            recommendations=report.output.recommendations,
        )

    async def _images(self, frames: Sequence[Frame]):
        """프레임을 JPEG로 바꿔 번호로 찾을 수 있게 만든다.

        인코딩은 CPU를 쓰는 동기 작업이라 스레드로 내보낸다. 이벤트 루프에서
        돌리면 그동안 진행 상태 조회가 응답하지 않는다.
        """
        encoded: Mapping[int, bytes] = await asyncio.to_thread(
            encode_frames, frames, self._quality
        )
        return encoded.get

    def _discard(self, vision: VisionResult) -> None:
        """Vision이 만든 이미지를 지운다. 실패해도 예외를 내지 않는다."""
        if self._store is None:
            return
        paths = vision.marked_image_paths
        if not paths:
            return
        try:
            removed = self._store.delete(*paths)
            logger.info("환경 분석 실패로 이미지 %d장을 정리했다", removed)
        except Exception:  # noqa: BLE001
            # 정리 실패가 원래의 실패 사유를 덮으면 안 된다.
            logger.exception("이미지 정리 실패")
