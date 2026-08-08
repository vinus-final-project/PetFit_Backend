"""분석 저장소.

SQL 질의를 한곳에 모은다. 서비스 계층은 비즈니스 규칙만 다루고 질의를 직접
작성하지 않는다. 질의가 여러 곳에 흩어지면 인덱스를 타지 않는 조회가 섞여 들어가고,
소유권 조건을 빠뜨린 질의가 생긴다.

**모든 단건 조회는 `device_id` 를 조건에 포함한다.** 조회 후 애플리케이션에서
비교하는 방식은 빠뜨리기 쉽다. 질의 자체에 넣으면 구조적으로 막힌다.
"""

from collections.abc import Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Analysis, DetectedObject, Recommendation
from app.schemas.enums import AnalysisStatus

__all__ = ["AnalysisRepository"]

#: 진행 중으로 간주하는 상태. 기기당 동시 분석 제한과 재시작 정리의 대상이다.
ACTIVE_STATUSES = (AnalysisStatus.PENDING.value, AnalysisStatus.PROCESSING.value)


class AnalysisRepository:
    """analysis 및 자식 테이블 질의."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- 조회 -------------------------------------------------------------

    async def get(self, analysis_id: int, device_id: str) -> Analysis | None:
        """소유한 분석 1건을 조회한다.

        다른 기기의 분석이면 None을 반환한다. 호출자는 404로 변환한다.
        403을 쓰면 해당 ID의 분석이 존재한다는 사실이 노출된다.
        """
        stmt = select(Analysis).where(
            Analysis.analysis_id == analysis_id,
            Analysis.device_id == device_id,
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_id(self, analysis_id: int) -> Analysis | None:
        """소유권 검사 없이 조회한다.

        **내부 전용이다.** 백그라운드 워커는 요청 컨텍스트가 없어 `device_id` 를
        모른다. API 경로에서는 반드시 `get()` 을 쓴다.
        """
        return await self._session.get(Analysis, analysis_id)

    async def get_objects(self, analysis_id: int) -> Sequence[DetectedObject]:
        """탐지 객체를 조회한다. 정렬은 응답 계층이 담당한다."""
        stmt = select(DetectedObject).where(DetectedObject.analysis_id == analysis_id)
        return (await self._session.execute(stmt)).scalars().all()

    async def get_recommendations(self, analysis_id: int) -> Sequence[Recommendation]:
        """추천을 우선순위 오름차순으로 조회한다."""
        stmt = (
            select(Recommendation)
            .where(Recommendation.analysis_id == analysis_id)
            .order_by(Recommendation.priority)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def list_page(
        self,
        device_id: str,
        page: int,
        size: int,
        status: AnalysisStatus | None = None,
    ) -> tuple[Sequence[Analysis], int]:
        """이력을 페이지 단위로 조회한다.

        정렬은 `created_at` 내림차순으로 고정한다. 인덱스
        `ix_analysis_device_created` 와 `ix_analysis_device_status_created` 를 탄다.

        Returns:
            (해당 페이지 항목, 전체 건수).
        """
        conditions = [Analysis.device_id == device_id]
        if status is not None:
            conditions.append(Analysis.status == status.value)

        total = (
            await self._session.execute(
                select(func.count()).select_from(Analysis).where(*conditions)
            )
        ).scalar_one()

        stmt = (
            select(Analysis)
            .where(*conditions)
            .order_by(Analysis.created_at.desc(), Analysis.analysis_id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return rows, total

    async def has_active(self, device_id: str, exclude_id: int | None = None) -> bool:
        """해당 기기에 진행 중인 분석이 있는지 확인한다.

        Args:
            exclude_id: 검사에서 제외할 분석 ID. 재시도 시 자기 자신을 제외한다.
        """
        conditions = [
            Analysis.device_id == device_id,
            Analysis.status.in_(ACTIVE_STATUSES),
        ]
        if exclude_id is not None:
            conditions.append(Analysis.analysis_id != exclude_id)

        stmt = select(Analysis.analysis_id).where(*conditions).limit(1)
        return (await self._session.execute(stmt)).first() is not None

    # --- 생성 · 삭제 -------------------------------------------------------

    def add(self, row: Analysis) -> Analysis:
        """새 분석을 세션에 추가한다. 커밋은 호출자가 한다."""
        self._session.add(row)
        return row

    async def delete(self, row: Analysis) -> None:
        """분석 행을 삭제한다. 자식 행은 ON DELETE CASCADE로 함께 지워진다."""
        await self._session.delete(row)

    async def clear_children(self, analysis_id: int) -> None:
        """탐지 객체와 추천을 삭제한다. 재시도 시 재생성을 위해 비운다."""
        await self._session.execute(
            delete(DetectedObject).where(DetectedObject.analysis_id == analysis_id)
        )
        await self._session.execute(
            delete(Recommendation).where(Recommendation.analysis_id == analysis_id)
        )

    # --- 일괄 갱신 ---------------------------------------------------------

    async def fail_all_active(self, message: str) -> int:
        """진행 중인 모든 분석을 실패로 정리한다.

        서버 재시작 시 호출한다. 인프로세스 큐는 재시작으로 사라지지만 DB 행은
        남는다. 정리하지 않으면 해당 기기는 **새 분석도, 삭제도, 재시도도**
        할 수 없는 상태로 영구히 잠긴다.

        `stage` 는 지우지 않는다. 어느 단계에서 중단됐는지가 남아야 한다.

        Returns:
            정리한 행 수.
        """
        stmt = (
            update(Analysis)
            .where(Analysis.status.in_(ACTIVE_STATUSES))
            .values(status=AnalysisStatus.FAILED.value, error_message=message)
        )
        return (await self._session.execute(stmt)).rowcount or 0

    async def list_stale_processing(self, before) -> Sequence[Analysis]:
        """지정 시각 이전에 시작된 진행 중 분석을 조회한다. 타임아웃 감시에 사용한다."""
        stmt = select(Analysis).where(
            Analysis.status == AnalysisStatus.PROCESSING.value,
            Analysis.created_at < before,
        )
        return (await self._session.execute(stmt)).scalars().all()
