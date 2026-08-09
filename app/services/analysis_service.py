"""분석 서비스.

분석의 생명주기를 관리한다. 상태 전이·소유권 검사·재시도·삭제 규칙이 모두 여기에 있다.

**상태를 바꾸는 경로를 이 파일로 단일화한다.** 여러 곳에서 status를 직접 대입하면
`stage` · `progress` · `completed_at` · `error_message` 의 정합성이 깨지고, DB CHECK
제약에 걸려 저장이 실패한다.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.pipeline import PipelineResult
from app.ai.score_generator import generate
from app.core.constants import MAX_RETRY_COUNT, PROCESSING_TIMEOUT_SECONDS
from app.core.exceptions import ErrorCode, PetFitError
from app.models import Analysis, DetectedObject, Recommendation
from app.schemas.enums import (
    AnalysisStage,
    AnalysisStatus,
    AnimalGroup,
    SpaceType,
    progress_for,
)
from app.services.repository import AnalysisRepository
from app.services.storage import Storage
from app.utils.timeutil import now_naive

__all__ = ["AnalysisService", "AnalysisDetail", "RESTART_MESSAGE", "TIMEOUT_MESSAGE"]

logger = logging.getLogger(__name__)

#: 서버 재시작으로 중단된 분석에 남기는 사유.
RESTART_MESSAGE = "서버가 재시작되어 분석이 중단되었습니다. 다시 시도해주세요."

#: 처리 제한 시간을 초과한 분석에 남기는 사유.
TIMEOUT_MESSAGE = "분석 시간이 초과되었습니다. 영상을 다시 촬영해주세요."

#: 고아 이미지로 판정하기까지 기다리는 시간(초).
#:
#: 마킹 이미지는 DB에 기록되기 전에 먼저 디스크에 쓰인다. 이 간격이 없으면
#: 진행 중인 분석이 방금 만든 파일을 지우게 된다. 처리 제한 시간에 여유를 더한다.
ORPHAN_IMAGE_MIN_AGE = PROCESSING_TIMEOUT_SECONDS * 2


@dataclass(frozen=True)
class AnalysisDetail:
    """상세 조회 결과. 응답 DTO 변환에 필요한 것을 함께 담는다."""

    analysis: Analysis
    objects: Sequence[DetectedObject]
    recommendations: Sequence[Recommendation]


class AnalysisService:
    """분석 생명주기 관리.

    Args:
        session: DB 세션.
        storage: 파일 저장소. 삭제·재시도 시 파일 정리에 사용한다.
    """

    def __init__(self, session: AsyncSession, storage: Storage) -> None:
        self._session = session
        self._storage = storage
        self._repo = AnalysisRepository(session)

    # --- 조회 -------------------------------------------------------------

    async def get_owned(self, analysis_id: int, device_id: str) -> Analysis:
        """소유한 분석을 조회한다.

        Raises:
            PetFitError: 존재하지 않거나 다른 기기의 분석인 경우 404.
        """
        row = await self._repo.get(analysis_id, device_id)
        if row is None:
            raise PetFitError(ErrorCode.ANALYSIS_NOT_FOUND)
        return row

    async def get_internal(self, analysis_id: int) -> Analysis | None:
        """소유권 검사 없이 조회한다.

        **백그라운드 워커 전용이다.** 워커는 요청 컨텍스트가 없어 `device_id` 를
        모른다. API 경로에서는 반드시 `get_owned()` 를 쓴다.
        """
        return await self._repo.get_by_id(analysis_id)

    async def get_detail(self, analysis_id: int, device_id: str) -> AnalysisDetail:
        """완료된 분석의 상세를 조회한다.

        Raises:
            PetFitError: 소유하지 않으면 404, 완료 전이면 409.
        """
        row = await self.get_owned(analysis_id, device_id)
        if row.status != AnalysisStatus.COMPLETED.value:
            raise PetFitError(ErrorCode.ANALYSIS_NOT_COMPLETED, status=row.status)

        return AnalysisDetail(
            analysis=row,
            objects=await self._repo.get_objects(analysis_id),
            recommendations=await self._repo.get_recommendations(analysis_id),
        )

    async def list_history(
        self, device_id: str, page: int, size: int, status: AnalysisStatus | None
    ) -> tuple[Sequence[Analysis], int, set[int]]:
        """이력을 조회한다.

        Returns:
            (항목, 전체 건수, 재시도 가능한 분석 ID 집합).
        """
        rows, total = await self._repo.list_page(device_id, page, size, status)
        has_active = await self._repo.has_active(device_id)
        retryable = {r.analysis_id for r in rows if self._can_retry(r, has_active)}
        return rows, total, retryable

    async def can_retry(self, row: Analysis) -> bool:
        """재시도 가능 여부를 판정한다. API-006 응답의 `canRetry` 에 사용한다."""
        return self._can_retry(row, await self._repo.has_active(row.device_id))

    @staticmethod
    def _can_retry(row: Analysis, device_has_active: bool) -> bool:
        """재시도 세 조건을 종합한다.

            canRetry = FAILED  AND  retry_count < 3  AND  기기에 진행 중인 분석 없음

        클라이언트가 각각을 판단하기 어려우므로 서버가 불리언으로 제공한다.
        이 값이 없으면 클라이언트는 재시도를 눌러 409를 받아야 불가능함을 알게 된다.

        자기 자신을 진행 중 판정에서 제외할 필요는 없다. FAILED 행은 애초에
        진행 중 상태에 포함되지 않는다.
        """
        return (
            row.status == AnalysisStatus.FAILED.value
            and row.retry_count < MAX_RETRY_COUNT
            and not device_has_active
        )

    # --- 생성 -------------------------------------------------------------

    async def create(
        self,
        device_id: str,
        group: AnimalGroup,
        space: SpaceType,
        video_relative_path: str,
    ) -> Analysis:
        """분석을 접수한다.

        Raises:
            PetFitError: 해당 기기에 진행 중인 분석이 있으면 409.
        """
        if await self._repo.has_active(device_id):
            raise PetFitError(
                ErrorCode.ANALYSIS_IN_PROGRESS, status=AnalysisStatus.PROCESSING.value
            )

        row = Analysis(
            device_id=device_id,
            animal_group=group.value,
            space_type=space.value,
            status=AnalysisStatus.PENDING.value,
            stage=None,
            progress=0,
            video_path=video_relative_path,
            created_at=now_naive(),
            # JSON 컬럼을 DB 기본값에 맡기지 않고 명시한다.
            # 서버 기본값에 의존하면 flush 직후 ORM 객체가 None을 들고 있어,
            # 새로고침 전에 접근하는 코드가 조용히 깨진다.
            risk_factors=[],
            analysis_result=[],
        )
        self._repo.add(row)
        await self._session.flush()  # analysis_id 확보
        return row

    # --- 상태 전이 ---------------------------------------------------------

    async def mark_stage(self, row: Analysis, stage: AnalysisStage) -> None:
        """단계 진입을 기록한다.

        `progress` 를 직접 대입하지 않고 `progress_for()` 로 산출한다.
        두 값이 어긋나면 DB CHECK 제약에 걸린다.
        """
        row.status = AnalysisStatus.PROCESSING.value
        row.stage = stage.value
        row.progress = progress_for(AnalysisStatus.PROCESSING, stage)
        row.error_message = None
        await self._session.flush()

    async def mark_completed(self, row: Analysis, result: PipelineResult) -> None:
        """분석 결과를 저장하고 완료로 전이한다.

        점수는 **규칙 기반으로 여기서 산출한다.** 파이프라인이 점수를 만들지 않으므로,
        AI 구현이 바뀌어도 같은 탐지 결과에 같은 점수가 나온다.
        """
        score = generate(
            AnimalGroup(row.animal_group),
            SpaceType(row.space_type),
            result.object_names,
            result.occupancy_ratio,
        )

        row.status = AnalysisStatus.COMPLETED.value
        row.stage = None
        row.progress = progress_for(AnalysisStatus.COMPLETED, None)
        row.error_message = None
        row.completed_at = now_naive()

        row.capture_duration = result.capture_duration
        row.frame_count = result.frame_count
        row.thumbnail_path = result.thumbnail_path
        row.occupancy_ratio = result.occupancy_ratio

        row.total_score = score.total
        row.safety_score = score.safety
        row.activity_score = score.activity
        row.rest_score = score.rest
        row.environment_score = score.environment

        row.risk_factors = [
            {"text": f.text, "source": f.source.value} for f in result.risk_factors
        ]
        row.analysis_result = list(result.analysis)

        for o in result.detected_objects:
            self._session.add(
                DetectedObject(
                    analysis_id=row.analysis_id,
                    object_name=o.name,
                    confidence=o.confidence,
                    detection_frame_count=o.detection_frame_count,
                    risk_level=o.risk.value,
                    frame_number=o.frame_number,
                    marked_image_path=o.marked_image_path,
                    x=o.x,
                    y=o.y,
                    width=o.width,
                    height=o.height,
                )
            )

        for r in result.recommendations:
            self._session.add(
                Recommendation(
                    analysis_id=row.analysis_id,
                    recommendation_type=r.type.value,
                    recommendation_text=r.text,
                    priority=r.priority,
                    source=r.source.value,
                )
            )

        await self._session.flush()

    async def mark_failed(
        self, row: Analysis, message: str, stage: AnalysisStage | None = None
    ) -> None:
        """분석을 실패로 전이한다.

        `stage` 와 `progress` 를 **0으로 되돌리지 않는다.** 10에서 실패한 것과
        82까지 진행하다 실패한 것이 구분되어야 클라이언트가 재촬영과 재시도 중
        무엇을 안내할지 정할 수 있다.
        """
        row.status = AnalysisStatus.FAILED.value
        row.error_message = message
        # completed_at 은 COMPLETED 에서만 값을 가진다. 남겨두면 CHECK 제약에 걸린다.
        row.completed_at = None
        if stage is not None:
            row.stage = stage.value
            row.progress = progress_for(AnalysisStatus.FAILED, stage)
        await self._session.flush()

    # --- 재시도 -----------------------------------------------------------

    async def prepare_retry(self, analysis_id: int, device_id: str) -> Analysis:
        """재시도를 준비한다. 기존 결과와 이미지 파일을 정리하고 PENDING으로 되돌린다.

        **새 분석을 만들지 않는다.** 동일 영상의 실패 기록이 이력에 중복되지 않게 한다.
        업로드된 영상은 유지하고 재분석 입력으로 쓴다.

        Raises:
            PetFitError: 실패 상태가 아니면 409, 횟수 초과면 409, 진행 중이 있으면 409.
        """
        row = await self.get_owned(analysis_id, device_id)

        if row.status != AnalysisStatus.FAILED.value:
            raise PetFitError(ErrorCode.ANALYSIS_NOT_RETRYABLE, status=row.status)
        if row.retry_count >= MAX_RETRY_COUNT:
            raise PetFitError(ErrorCode.RETRY_LIMIT_EXCEEDED, status=row.status)
        if await self._repo.has_active(device_id, exclude_id=analysis_id):
            raise PetFitError(
                ErrorCode.ANALYSIS_IN_PROGRESS, status=AnalysisStatus.PROCESSING.value
            )

        # 경로를 먼저 모은 뒤 파일을 지운다. 행을 먼저 지우면 경로를 잃는다.
        objects = await self._repo.get_objects(analysis_id)
        paths = [o.marked_image_path for o in objects]
        paths.append(row.thumbnail_path)
        self._storage.delete(*paths)

        await self._repo.clear_children(analysis_id)

        row.status = AnalysisStatus.PENDING.value
        row.stage = None
        row.progress = 0
        row.error_message = None
        row.retry_count += 1
        row.completed_at = None
        row.thumbnail_path = None
        row.occupancy_ratio = 0
        row.total_score = 0
        row.safety_score = 0
        row.activity_score = 0
        row.rest_score = 0
        row.environment_score = 0
        row.risk_factors = []
        row.analysis_result = []
        await self._session.flush()
        return row

    # --- 삭제 -------------------------------------------------------------

    async def delete(self, analysis_id: int, device_id: str) -> None:
        """분석과 관련 파일을 삭제한다.

        파일을 먼저 지우고 행을 나중에 지운다. 행을 먼저 지우면 파일 경로를 잃어
        디스크에 영구히 잔류한다.

        Raises:
            PetFitError: 소유하지 않으면 404, 진행 중이면 409.
        """
        row = await self.get_owned(analysis_id, device_id)

        if row.status in (AnalysisStatus.PENDING.value, AnalysisStatus.PROCESSING.value):
            raise PetFitError(ErrorCode.ANALYSIS_NOT_DELETABLE, status=row.status)

        objects = await self._repo.get_objects(analysis_id)
        paths = [o.marked_image_path for o in objects]
        paths.extend([row.thumbnail_path, row.video_path])
        removed = self._storage.delete(*paths)
        logger.info("분석 %s 삭제: 파일 %d개 정리", analysis_id, removed)

        await self._repo.delete(row)
        await self._session.flush()

    # --- 재시작 정리 -------------------------------------------------------

    async def cleanup_interrupted(self) -> int:
        """서버 재시작으로 중단된 분석을 실패로 정리한다.

        앱 시작 시 1회 호출한다. 정리하지 않으면 해당 기기는 새 분석도, 삭제도,
        재시도도 할 수 없는 상태로 영구히 잠긴다.

        Returns:
            정리한 건수.
        """
        count = await self._repo.fail_all_active(RESTART_MESSAGE)
        if count:
            logger.warning("재시작으로 중단된 분석 %d건을 실패 처리했다", count)
        return count

    async def cleanup_orphan_images(
        self, min_age_seconds: float = ORPHAN_IMAGE_MIN_AGE
    ) -> int:
        """DB가 참조하지 않는 이미지를 삭제한다.

        앱 시작 시 1회 호출한다.

        마킹 이미지는 **DB에 기록되기 전에 먼저 디스크에 쓰인다.** 그 사이에
        분석이 실패하거나 처리 제한 시간을 넘겨 취소되면 파일만 남는다. 경로가
        어디에도 기록되지 않아 재시도·삭제로도 정리되지 않는다.

        파이썬은 스레드를 강제 종료할 수 없어, 취소된 작업이 이미지를 마저 쓰는
        경우도 있다. 이때는 파이프라인이 지울 대상을 알 수 없으므로 이 정리가
        유일한 회수 경로다.

        Args:
            min_age_seconds: 이 시간보다 오래된 파일만 대상으로 한다.
                진행 중인 분석이 방금 만든 파일을 지우지 않기 위해 필요하다.

        Returns:
            삭제한 파일 수.
        """
        candidates = self._storage.list_images(min_age_seconds)
        if not candidates:
            return 0

        referenced = await self._repo.referenced_image_paths()
        orphans = [path for path in candidates if path not in referenced]
        if not orphans:
            return 0

        removed = self._storage.delete(*orphans)
        logger.warning("참조 없는 이미지 %d개를 삭제했다", removed)
        return removed
