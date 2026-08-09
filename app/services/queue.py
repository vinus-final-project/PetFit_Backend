"""분석 작업 큐.

**인프로세스 큐다.** Mac Studio 단일 장비에서 동시 2건을 처리하는 MVP 규모라
외부 브로커(Redis·Celery)를 두지 않는다.

    동시 처리 2건   : 세마포어
    대기열 10건     : 세마포어 대기 + 카운터
    기기당 1건      : 서비스 계층의 has_active 검사
    제한 시간 180초 : asyncio.wait_for

인프로세스이므로 **재시작하면 큐가 사라진다.** DB에 남은 PENDING·PROCESSING 행은
앱 시작 시 `AnalysisService.cleanup_interrupted()` 가 실패로 정리한다. 정리하지 않으면
해당 기기는 새 분석도, 삭제도, 재시도도 할 수 없는 상태로 잠긴다.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

from app.ai.pipeline import Pipeline, PipelineError
from app.core.constants import (
    MAX_CONCURRENT_ANALYSIS,
    MAX_QUEUE_SIZE,
    PROCESSING_TIMEOUT_SECONDS,
)
from app.core.exceptions import ErrorCode, PetFitError
from app.db.session import session_scope
from app.schemas.enums import AnalysisStage, AnimalGroup, SpaceType
from app.services.analysis_service import TIMEOUT_MESSAGE, AnalysisService
from app.services.storage import Storage

__all__ = ["AnalysisQueue"]

logger = logging.getLogger(__name__)

#: 파이프라인 진입 전 실패에 사용하는 사유.
INTERNAL_MESSAGE = "분석 중 오류가 발생했습니다. 다시 시도해주세요."


class AnalysisQueue:
    """분석 작업을 백그라운드에서 실행한다.

    Args:
        pipeline: 실행할 파이프라인. 스텁과 실제 구현을 교체할 수 있다.
        storage: 파일 저장소.
        concurrency: 동시 처리 수.
        capacity: 대기 가능한 최대 작업 수. 처리 중인 것을 포함한다.
        timeout: 작업 1건의 제한 시간(초).
        session_factory: DB 세션 컨텍스트를 만드는 호출 가능 객체.
            기본값은 운영 세션이며, 테스트는 인메모리 세션을 주입한다.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        storage: Storage,
        concurrency: int = MAX_CONCURRENT_ANALYSIS,
        capacity: int = MAX_QUEUE_SIZE,
        timeout: float = PROCESSING_TIMEOUT_SECONDS,
        session_factory: Callable[[], AbstractAsyncContextManager] = session_scope,
    ) -> None:
        self._pipeline = pipeline
        self._storage = storage
        self._semaphore = asyncio.Semaphore(concurrency)
        self._capacity = capacity
        self._timeout = timeout
        # 세션 생성을 주입받는다. 고정하면 MySQL 서버 없이 테스트할 수 없다.
        self._session_factory = session_factory
        self._pending = 0
        self._tasks: set[asyncio.Task] = set()

    @property
    def size(self) -> int:
        """대기·처리 중인 작업 수."""
        return self._pending

    @property
    def is_full(self) -> bool:
        """새 작업을 받을 수 없는 상태인지.

        용량 판정을 큐 안에 둔다. 호출자가 `size` 와 상수를 직접 비교하면
        인스턴스마다 다른 `capacity` 를 무시하게 되어, 사전 검사와 `submit()` 의
        판정이 어긋난다.
        """
        return self._pending >= self._capacity

    def submit(
        self,
        analysis_id: int,
        video_path: Path,
        group: AnimalGroup,
        space: SpaceType,
    ) -> None:
        """작업을 등록한다. 즉시 반환하며 실행은 백그라운드에서 이뤄진다.

        Raises:
            PetFitError: 대기열이 가득 찬 경우 503.
        """
        if self.is_full:
            raise PetFitError(ErrorCode.QUEUE_FULL)

        self._pending += 1
        task = asyncio.create_task(self._run(analysis_id, video_path, group, space))

        # 참조를 보관하지 않으면 GC가 실행 중인 태스크를 수거할 수 있다.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(
        self,
        analysis_id: int,
        video_path: Path,
        group: AnimalGroup,
        space: SpaceType,
    ) -> None:
        """작업 1건을 실행한다. 예외를 밖으로 내보내지 않는다."""
        try:
            async with self._semaphore:
                await self._execute(analysis_id, video_path, group, space)
        except Exception:  # noqa: BLE001
            # 여기까지 온 예외는 상태 기록조차 실패한 경우다. 로그만 남긴다.
            logger.exception("분석 %s 처리 중 복구 불가 오류", analysis_id)
        finally:
            self._pending -= 1

    async def _execute(
        self,
        analysis_id: int,
        video_path: Path,
        group: AnimalGroup,
        space: SpaceType,
    ) -> None:
        """파이프라인을 실행하고 결과를 저장한다.

        단계 갱신과 최종 결과 저장에 **각각 별도 세션**을 쓴다. 한 세션을 오래 열어두면
        수십 초 동안 커넥션을 점유하고, 진행률 갱신이 커밋되지 않아 폴링에 보이지 않는다.
        """
        failure: tuple[str, AnalysisStage | None] | None = None

        async def on_stage(stage: AnalysisStage) -> None:
            async with self._session_factory() as session:
                service = AnalysisService(session, self._storage)
                row = await service.get_internal(analysis_id)
                if row is not None:
                    await service.mark_stage(row, stage)

        try:
            # asyncio.timeout() 은 Python 3.11 이상 전용이다. 3.10을 지원하므로 wait_for를 쓴다.
            result = await asyncio.wait_for(
                self._pipeline.run(video_path, group, space, on_stage), self._timeout
            )
        except asyncio.TimeoutError:
            # 서버 타임아웃이 없으면 중단된 분석이 PROCESSING으로 영구히 남아
            # 삭제조차 할 수 없게 된다.
            logger.warning("분석 %s 제한 시간(%s초) 초과", analysis_id, self._timeout)
            failure = (TIMEOUT_MESSAGE, None)
        except PipelineError as exc:
            logger.info("분석 %s 실패: %s (%s)", analysis_id, exc.message, exc.stage)
            failure = (exc.message, exc.stage)
        except Exception:  # noqa: BLE001
            logger.exception("분석 %s 파이프라인 예외", analysis_id)
            failure = (INTERNAL_MESSAGE, None)

        async with self._session_factory() as session:
            service = AnalysisService(session, self._storage)
            row = await service.get_internal(analysis_id)
            if row is None:
                logger.warning("분석 %s 행이 사라졌다. 결과를 버린다", analysis_id)
                return

            if failure is not None:
                await service.mark_failed(row, failure[0], failure[1])
            else:
                await service.mark_completed(row, result)

    async def drain(self, timeout: float = 5.0) -> None:
        """실행 중인 작업이 끝나기를 기다린다. 종료 시 호출한다."""
        if not self._tasks:
            return
        await asyncio.wait(set(self._tasks), timeout=timeout)
