"""분석 작업 큐 검증.

동시 처리 제한 · 대기열 상한 · 타임아웃 · 실패 처리를 실제로 실행해 확인한다.
스텁 파이프라인을 쓰므로 AI 모델 없이 돌아간다.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ai.pipeline import PipelineError
from app.ai.stub import StubPipeline
from app.core.exceptions import ErrorCode, PetFitError
from app.schemas.enums import AnalysisStage, AnalysisStatus, AnimalGroup, SpaceType
from app.services.analysis_service import TIMEOUT_MESSAGE, AnalysisService
from app.services.queue import INTERNAL_MESSAGE, AnalysisQueue
from tests.conftest import sqlite_metadata

DEVICE = "3f2b8c10-9d7e-4a51-8f6c-2e4b7a9d0c35"
VIDEO = Path("a.mp4")


@pytest_asyncio.fixture
async def sessionmaker_():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(sqlite_metadata().create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def session_factory(sessionmaker_):
    """큐에 주입할 세션 컨텍스트. 운영의 session_scope 와 같은 동작을 한다."""

    @asynccontextmanager
    async def factory():
        async with sessionmaker_() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    return factory


@pytest_asyncio.fixture
async def make_analysis(sessionmaker_, storage):
    """분석 행을 만들고 ID를 돌려준다."""

    async def _make(device=DEVICE, group=AnimalGroup.SMALL_DOG, space=SpaceType.LIVING_ROOM):
        async with sessionmaker_() as session:
            service = AnalysisService(session, storage)
            row = await service.create(device, group, space, "/videos/a.mp4")
            await session.commit()
            return row.analysis_id

    return _make


@pytest_asyncio.fixture
async def fetch(sessionmaker_, storage):
    """분석 행을 다시 읽는다."""

    async def _fetch(analysis_id):
        async with sessionmaker_() as session:
            return await AnalysisService(session, storage).get_internal(analysis_id)

    return _fetch


def build(storage, session_factory, pipeline=None, **kwargs):
    return AnalysisQueue(
        pipeline=pipeline or StubPipeline(speed=0),
        storage=storage,
        session_factory=session_factory,
        **kwargs,
    )


class TestSubmit:
    async def test_returns_immediately(self, storage, session_factory, make_analysis) -> None:
        """submit 은 즉시 반환한다. 실행은 백그라운드다."""
        queue = build(storage, session_factory)
        aid = await make_analysis()

        queue.submit(aid, VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        assert queue.size == 1

        await queue.drain()
        assert queue.size == 0

    async def test_completes_analysis(
        self, storage, session_factory, make_analysis, fetch
    ) -> None:
        queue = build(storage, session_factory)
        aid = await make_analysis()

        queue.submit(aid, VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        await queue.drain()

        row = await fetch(aid)
        assert row.status == AnalysisStatus.COMPLETED.value
        assert row.progress == 100
        assert row.total_score == 56

    async def test_missing_row_is_tolerated(self, storage, session_factory) -> None:
        """삭제된 분석의 결과는 버린다. 예외를 밖으로 내지 않는다."""
        queue = build(storage, session_factory)
        queue.submit(99999, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)
        await queue.drain()
        assert queue.size == 0


class TestCapacity:
    async def test_queue_full_raises_503(self, storage, session_factory) -> None:
        """대기열이 가득 차면 QUEUE_FULL 이다."""
        queue = build(storage, session_factory, concurrency=1, capacity=2,
                      pipeline=StubPipeline(speed=0.02))

        for i in range(2):
            queue.submit(i + 1, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)

        with pytest.raises(PetFitError) as e:
            queue.submit(99, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)
        assert e.value.code is ErrorCode.QUEUE_FULL

        await queue.drain()

    async def test_is_full_follows_instance_capacity(
        self, storage, session_factory
    ) -> None:
        """용량 판정은 상수가 아니라 인스턴스의 capacity 를 따라야 한다.

        라우터가 영상을 저장하기 전에 이 값을 보고 거절한다. 상수와 어긋나면
        사전 검사를 통과한 요청이 submit 에서 503으로 떨어지고, 그때는 이미
        100MB를 디스크에 쓴 뒤다.
        """
        queue = build(storage, session_factory, concurrency=1, capacity=2,
                      pipeline=StubPipeline(speed=0.02))
        assert queue.is_full is False

        queue.submit(1, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)
        assert queue.is_full is False

        queue.submit(2, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)
        assert queue.is_full is True

        await queue.drain()
        assert queue.is_full is False

    async def test_slot_frees_after_completion(self, storage, session_factory) -> None:
        queue = build(storage, session_factory, capacity=1)
        queue.submit(1, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)
        await queue.drain()

        # 자리가 비었으므로 다시 넣을 수 있다.
        queue.submit(2, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)
        await queue.drain()
        assert queue.size == 0


class TestConcurrency:
    async def test_limits_simultaneous_runs(self, storage, session_factory) -> None:
        """동시 처리 수를 초과해 실행되지 않아야 한다.

        Mac Studio 한 대에서 VLM까지 함께 돌리므로 초과 실행은 메모리 압박으로 이어진다.
        """
        running = 0
        peak = 0

        class Counting:
            async def run(self, video_path, group, space, on_stage):
                nonlocal running, peak
                running += 1
                peak = max(peak, running)
                try:
                    await asyncio.sleep(0.05)
                    return await StubPipeline(speed=0).run(
                        video_path, group, space, on_stage
                    )
                finally:
                    running -= 1

        queue = build(storage, session_factory, pipeline=Counting(), concurrency=2, capacity=10)
        for i in range(5):
            queue.submit(i + 1, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)

        await queue.drain(timeout=10)
        assert peak <= 2

    async def test_all_tasks_eventually_run(self, storage, session_factory) -> None:
        started = []

        class Recording:
            async def run(self, video_path, group, space, on_stage):
                started.append(1)
                return await StubPipeline(speed=0).run(video_path, group, space, on_stage)

        queue = build(storage, session_factory, pipeline=Recording(), concurrency=2, capacity=10)
        for i in range(5):
            queue.submit(i + 1, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)

        await queue.drain(timeout=10)
        assert len(started) == 5


class TestTimeout:
    async def test_timeout_marks_failed(
        self, storage, session_factory, make_analysis, fetch
    ) -> None:
        """제한 시간을 넘기면 FAILED 로 전환한다.

        전환하지 않으면 PROCESSING 이 영구히 남아 삭제조차 할 수 없다.
        """

        class Hanging:
            async def run(self, video_path, group, space, on_stage):
                await asyncio.sleep(10)

        queue = build(storage, session_factory, pipeline=Hanging(), timeout=0.05)
        aid = await make_analysis()

        queue.submit(aid, VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        await queue.drain(timeout=5)

        row = await fetch(aid)
        assert row.status == AnalysisStatus.FAILED.value
        assert row.error_message == TIMEOUT_MESSAGE

    async def test_timeout_frees_slot(self, storage, session_factory, make_analysis) -> None:
        class Hanging:
            async def run(self, video_path, group, space, on_stage):
                await asyncio.sleep(10)

        queue = build(storage, session_factory, pipeline=Hanging(), timeout=0.05)
        queue.submit(await make_analysis(), VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        await queue.drain(timeout=5)
        assert queue.size == 0


class TestFailure:
    async def test_pipeline_error_preserves_stage(
        self, storage, session_factory, make_analysis, fetch
    ) -> None:
        """실패 지점을 보존해야 재촬영과 재시도를 구분해 안내할 수 있다."""
        queue = build(
            storage,
            session_factory,
            pipeline=StubPipeline(speed=0, fail_at=AnalysisStage.OBJECT_DETECTION),
        )
        aid = await make_analysis()

        queue.submit(aid, VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        await queue.drain(timeout=5)

        row = await fetch(aid)
        assert row.status == AnalysisStatus.FAILED.value
        assert row.stage == AnalysisStage.OBJECT_DETECTION.value
        assert row.progress == 20

    async def test_unexpected_error_uses_generic_message(
        self, storage, session_factory, make_analysis, fetch
    ) -> None:
        """내부 오류 메시지를 사용자에게 그대로 노출하지 않는다."""

        class Broken:
            async def run(self, video_path, group, space, on_stage):
                raise RuntimeError("connection string: user:password@host")

        queue = build(storage, session_factory, pipeline=Broken())
        aid = await make_analysis()

        queue.submit(aid, VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        await queue.drain(timeout=5)

        row = await fetch(aid)
        assert row.status == AnalysisStatus.FAILED.value
        assert row.error_message == INTERNAL_MESSAGE
        assert "password" not in row.error_message

    async def test_failure_does_not_stop_queue(self, storage, session_factory, fetch) -> None:
        """한 작업이 실패해도 다른 작업은 계속 처리된다."""
        calls = []

        class SometimesBroken:
            async def run(self, video_path, group, space, on_stage):
                calls.append(1)
                if len(calls) == 1:
                    raise PipelineError("실패", AnalysisStage.OBJECT_DETECTION)
                return await StubPipeline(speed=0).run(video_path, group, space, on_stage)

        queue = build(storage, session_factory, pipeline=SometimesBroken())
        for i in range(3):
            queue.submit(i + 1, VIDEO, AnimalGroup.CAT, SpaceType.BALCONY)

        await queue.drain(timeout=5)
        assert len(calls) == 3
        assert queue.size == 0


class TestStageProgress:
    async def test_stages_are_recorded_in_order(
        self, storage, session_factory, make_analysis, fetch
    ) -> None:
        """단계 진입이 DB에 기록되어야 폴링으로 진행률이 보인다."""
        seen = []

        class Watching:
            async def run(self, video_path, group, space, on_stage):
                async def spy(stage):
                    await on_stage(stage)
                    row = await fetch(1)
                    seen.append((stage, row.progress if row else None))

                return await StubPipeline(speed=0).run(video_path, group, space, spy)

        queue = build(storage, session_factory, pipeline=Watching())
        aid = await make_analysis()
        assert aid == 1

        queue.submit(aid, VIDEO, AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM)
        await queue.drain(timeout=5)

        assert [s for s, _ in seen] == list(AnalysisStage)
        assert [p for _, p in seen] == [s.progress for s in AnalysisStage]
