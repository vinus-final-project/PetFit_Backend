"""실제 분석 파이프라인 조립 (1~12단계).

Vision(1~10) · Score Generator(11) · 환경 분석(12) 을 엮어 ``Pipeline`` 계약을
만족시킨다. **세 담당의 산출물이 만나는 유일한 지점이다.**

    VisionPipeline      -> VisionResult          AI C
    score_generator     -> PetFitScore           완성
    EnvironmentAnalyzer -> AnalysisReport        AI D
                        -> PipelineResult

교체는 여기 한 곳에서 일어난다. ``app/main.py`` 의 ``StubPipeline`` 자리에 이
객체를 넣으면 API·DB·프론트는 수정 없이 실제 결과를 받는다.

**점수를 두 번 산출한다.** 여기서 한 번(12단계 입력용), 서비스 계층이 저장할 때
한 번이다. 규칙 기반이라 같은 입력에 같은 값이 나오므로 어긋나지 않는다. 서술이
참조한 점수와 화면에 표시되는 점수가 다르면 안 되므로, 이 성질은 테스트로 고정한다.
"""

import asyncio
import io
import logging
from collections.abc import Sequence
from pathlib import Path

from app.ai.environment_analysis import EnvironmentAnalyzer
from app.ai.pipeline import PipelineError, PipelineResult, StageCallback
from app.ai.prompts import AnalysisContext
from app.ai.score_generator import generate
from app.ai.vision.pipeline import VisionPipeline
from app.ai.vision.types import Frame, ImageStore, VisionResult
from app.schemas.enums import AnalysisStage, AnimalGroup, SpaceType

__all__ = ["AnalysisPipeline", "SCORE_FAILURE_MESSAGE", "LLM_IMAGE_QUALITY"]

logger = logging.getLogger(__name__)

#: 12단계에 보내는 이미지 품질.
#:
#: 모델이 어차피 자체 해상도로 줄이므로 원본 품질이 필요하지 않다. 4장을 한 번에
#: 보내며, 로컬 실행이라도 인코딩과 전처리 시간이 응답 속도 측정에 섞인다.
LLM_IMAGE_QUALITY = 85

#: 점수 산출 실패 사유. 정의되지 않은 그룹처럼 설정 문제일 때만 발생한다.
SCORE_FAILURE_MESSAGE = "점수를 산출하지 못했습니다. 다시 시도해주세요."


class AnalysisPipeline:
    """Vision 과 환경 분석을 엮은 실제 파이프라인.

    Args:
        vision: 영상 처리·탐지·시각화 담당.
        analyzer: 환경 분석 담당.
        storage: 이미지 저장소. 실패 시 만들어 둔 이미지를 지우는 데 쓴다.
    """

    def __init__(
        self,
        vision: VisionPipeline,
        analyzer: EnvironmentAnalyzer,
        storage: ImageStore,
    ) -> None:
        self._vision = vision
        self._analyzer = analyzer
        self._storage = storage

    async def run(
        self,
        video_path: Path,
        group: AnimalGroup,
        space: SpaceType,
        on_stage: StageCallback,
    ) -> PipelineResult:
        """12단계를 모두 수행한다.

        Args:
            video_path: 업로드된 영상 파일 경로.
            group: 반려동물 그룹.
            space: 촬영한 공간 종류.
            on_stage: 단계 진입 시 호출할 콜백.

        Returns:
            서비스 계층이 저장할 산출물.

        Raises:
            PipelineError: 어느 단계에서든 실패한 경우.
        """
        vision = await self._vision.run(video_path, group, on_stage)

        # 여기서부터 실패하면 Vision 이 만든 이미지가 DB 참조 없이 남는다.
        # 분석 1건당 최대 12장이므로 재시도까지 겹치면 저장소가 빠르게 쌓인다.
        try:
            await on_stage(AnalysisStage.SCORE_CALCULATION)
            score = self._score(vision, group, space)

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
                load_image=await self._image_loader(vision.analysis_frames),
            )
        except BaseException:
            # **Exception 이 아니라 BaseException 이다.** 처리 제한 시간을 넘기면
            # asyncio 가 CancelledError 를 던지는데, 이것은 BaseException 직속이라
            # `except Exception` 으로는 잡히지 않는다. 그대로 두면 타임아웃 때마다
            # Vision 이 만든 이미지가 DB 참조 없이 남아 영구 잔류한다.
            self._discard(vision)
            raise

        logger.info(
            "분석 완료: 종합 %s점 / 재생성 %s회 / 이미지 %s장",
            score.total,
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

    def _discard(self, vision: VisionResult) -> None:
        """만들어 둔 이미지를 지운다. 실패해도 예외를 밖으로 내지 않는다.

        **정리 실패가 원래의 실패 사유를 덮으면 안 된다.** 여기서 예외가 나가면
        타임아웃이나 환경 분석 실패가 저장소 오류로 둔갑해, 클라이언트가 재촬영과
        재시도 중 무엇을 안내할지 정할 근거를 잃는다. 원인 추적도 막힌다.

        지우지 못한 파일은 앱 시작 시 `cleanup_orphan_images()` 가 회수한다.
        """
        try:
            self._storage.delete(*vision.marked_image_paths)
        except Exception:  # noqa: BLE001
            logger.exception("실패 정리 중 이미지 삭제 실패")

    def _score(self, vision: VisionResult, group: AnimalGroup, space: SpaceType):
        """11단계. 규칙 기반이라 실패할 일이 거의 없다.

        그래도 감싸는 이유는, 여기서 예외가 나면 단계 없는 오류가 되어 클라이언트가
        재촬영과 재시도를 구분할 수 없기 때문이다.
        """
        try:
            return generate(group, space, vision.object_names, vision.occupancy_ratio)
        except Exception as exc:  # noqa: BLE001
            logger.exception("점수 산출 실패: %s / %s", group, space)
            raise PipelineError(
                SCORE_FAILURE_MESSAGE, AnalysisStage.SCORE_CALCULATION
            ) from exc

    async def _image_loader(self, frames: Sequence[Frame]):
        """프레임 번호를 원본 JPEG 바이트로 바꾸는 함수를 만든다.

        미리 인코딩해 사전으로 만든다. 12단계가 콜백을 동기로 호출하므로,
        그 안에서 인코딩하면 이벤트 루프에서 이미지 4장을 압축하게 된다.

        **마킹하지 않은 원본이다.** Bounding Box 를 그린 이미지를 보내면 모델이
        표시된 객체에만 주목해, 탐지 대상 밖의 위험 요소를 찾는다는 목적이 사라진다.
        """
        encoded = await asyncio.to_thread(_encode, frames)

        def load(number: int) -> bytes | None:
            data = encoded.get(number)
            if data is None:
                # 12단계의 프레임 선정 규칙과 Vision 이 남긴 프레임이 어긋난 경우다.
                # 두 곳이 같은 규칙을 각자 구현하고 있어 한쪽만 바뀌면 발생한다.
                logger.warning(
                    "환경 분석이 요청한 프레임 %s 를 Vision 이 남기지 않았다", number
                )
            return data

        return load


def _encode(frames: Sequence[Frame]) -> dict[int, bytes]:
    """프레임을 JPEG 바이트로 바꾼다. 블로킹 호출이다.

    **한 장이 실패해도 나머지는 보낸다.** 12단계는 이미지가 하나도 없어도 탐지
    결과만으로 서술하도록 만들어져 있다. 여기서 예외를 그대로 올리면 그 대비까지
    가지 못하고 분석 전체가 실패한다. 이미지 한 장 때문에 잃기에는 앞 단계의
    영상 처리와 추론이 아깝다.
    """
    encoded: dict[int, bytes] = {}
    for frame in frames:
        try:
            buffer = io.BytesIO()
            image = (
                frame.image if frame.image.mode == "RGB" else frame.image.convert("RGB")
            )
            image.save(buffer, "JPEG", quality=LLM_IMAGE_QUALITY, optimize=True)
            encoded[frame.number] = buffer.getvalue()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "프레임 %s 변환 실패: %s", frame.number, type(exc).__name__
            )
    return encoded
