"""분석 서비스 검증.

상태 전이·소유권·재시도·삭제 규칙을 실제 DB에 걸어 확인한다.
모델의 CHECK 제약이 그대로 적용되므로, 정합성이 깨진 전이는 저장 단계에서 걸린다.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ai.stub import StubPipeline
from app.core.exceptions import ErrorCode, PetFitError
from app.schemas.enums import (
    AnalysisStage,
    AnalysisStatus,
    AnimalGroup,
    SpaceType,
)
from app.services.analysis_service import RESTART_MESSAGE, AnalysisService
from tests.conftest import sqlite_metadata

DEVICE = "3f2b8c10-9d7e-4a51-8f6c-2e4b7a9d0c35"
OTHER_DEVICE = "00000000-0000-4000-8000-000000000000"


@pytest_asyncio.fixture
async def sessionmaker_():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(sqlite_metadata().create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest_asyncio.fixture
async def service(sessionmaker_, storage):
    async with sessionmaker_() as session:
        yield AnalysisService(session, storage)
        await session.commit()


async def _new(service, device=DEVICE, group=AnimalGroup.SMALL_DOG, space=SpaceType.LIVING_ROOM):
    row = await service.create(device, group, space, "/videos/a.mp4")
    await service._session.commit()
    return row


async def _run_stub(service, row, **kwargs):
    """스텁 파이프라인을 끝까지 돌리고 완료 처리한다."""
    async def on_stage(stage):
        await service.mark_stage(row, stage)

    result = await StubPipeline(speed=0, **kwargs).run(
        "a.mp4", AnimalGroup(row.animal_group), SpaceType(row.space_type), on_stage
    )
    await service.mark_completed(row, result)
    await service._session.commit()
    return result


class TestLifecycle:
    async def test_create_starts_pending(self, service) -> None:
        row = await _new(service)
        assert row.status == AnalysisStatus.PENDING.value
        assert row.progress == 0
        assert row.stage is None
        assert row.risk_factors == []

    async def test_stage_updates_progress(self, service) -> None:
        """progress는 stage에서 파생된다. 직접 대입하지 않는다."""
        row = await _new(service)
        seen = []

        async def on_stage(stage):
            await service.mark_stage(row, stage)
            seen.append((stage, row.progress))

        await StubPipeline(speed=0).run(
            "a.mp4", AnimalGroup.SMALL_DOG, SpaceType.LIVING_ROOM, on_stage
        )
        assert seen == [(s, s.progress) for s in AnalysisStage]
        assert [p for _, p in seen] == sorted(p for _, p in seen)

    async def test_completed_matches_document_example(self, service) -> None:
        """스텁 탐지 결과가 문서 계산 예시(거실 56)를 그대로 재현해야 한다."""
        row = await _new(service)
        await _run_stub(service, row)

        assert row.status == AnalysisStatus.COMPLETED.value
        assert row.progress == 100
        assert row.stage is None
        assert row.completed_at is not None
        assert (row.total_score, row.safety_score) == (56, 50)

    async def test_children_persisted(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)

        detail = await service.get_detail(row.analysis_id, DEVICE)
        assert len(detail.objects) == 5
        assert len(detail.recommendations) == 3
        assert [r.priority for r in detail.recommendations] == [1, 2, 3]

    async def test_safe_objects_have_no_marking(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)

        detail = await service.get_detail(row.analysis_id, DEVICE)
        for obj in detail.objects:
            if obj.risk_level == "SAFE":
                assert obj.marked_image_path is None
            else:
                assert obj.marked_image_path is not None


class TestOwnership:
    async def test_other_device_gets_not_found(self, service) -> None:
        """404를 쓴다. 403은 해당 ID의 분석이 존재한다는 사실을 노출한다."""
        row = await _new(service)
        with pytest.raises(PetFitError) as e:
            await service.get_owned(row.analysis_id, OTHER_DEVICE)
        assert e.value.code is ErrorCode.ANALYSIS_NOT_FOUND

    async def test_missing_id_gets_not_found(self, service) -> None:
        with pytest.raises(PetFitError) as e:
            await service.get_owned(99999, DEVICE)
        assert e.value.code is ErrorCode.ANALYSIS_NOT_FOUND

    async def test_detail_requires_completed(self, service) -> None:
        row = await _new(service)
        with pytest.raises(PetFitError) as e:
            await service.get_detail(row.analysis_id, DEVICE)
        assert e.value.code is ErrorCode.ANALYSIS_NOT_COMPLETED


class TestConcurrencyLimit:
    async def test_one_per_device(self, service) -> None:
        await _new(service)
        with pytest.raises(PetFitError) as e:
            await service.create(
                DEVICE, AnimalGroup.CAT, SpaceType.BEDROOM, "/videos/b.mp4"
            )
        assert e.value.code is ErrorCode.ANALYSIS_IN_PROGRESS

    async def test_other_device_unaffected(self, service) -> None:
        await _new(service)
        row = await service.create(
            OTHER_DEVICE, AnimalGroup.CAT, SpaceType.BEDROOM, "/videos/b.mp4"
        )
        assert row.analysis_id is not None

    async def test_completed_does_not_block(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        assert await service.create(
            DEVICE, AnimalGroup.CAT, SpaceType.BEDROOM, "/videos/b.mp4"
        )


class TestRestartCleanup:
    async def test_interrupted_becomes_failed(self, service) -> None:
        """재시작으로 중단된 분석을 정리하지 않으면 기기가 영구히 잠긴다.

        새 분석(409) · 삭제(409) · 재시도(409)가 모두 막힌다.
        """
        row = await _new(service)
        count = await service.cleanup_interrupted()
        await service._session.commit()

        assert count == 1
        refreshed = await service.get_owned(row.analysis_id, DEVICE)
        assert refreshed.status == AnalysisStatus.FAILED.value
        assert refreshed.error_message == RESTART_MESSAGE

    async def test_device_unlocked_after_cleanup(self, service) -> None:
        await _new(service)
        await service.cleanup_interrupted()
        await service._session.commit()
        assert await service.create(
            DEVICE, AnimalGroup.CAT, SpaceType.BEDROOM, "/videos/b.mp4"
        )

    async def test_completed_untouched(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        assert await service.cleanup_interrupted() == 0


class TestOrphanImages:
    """DB가 참조하지 않는 이미지 회수.

    마킹 이미지는 **DB에 기록되기 전에 먼저 디스크에 쓰인다.** 그 사이에 분석이
    취소되면 경로가 어디에도 남지 않아 재시도·삭제로 정리되지 않는다. 파이썬은
    스레드를 강제 종료할 수 없어 취소된 작업이 이미지를 마저 쓰기도 한다.
    시작할 때 회수하는 것이 유일한 경로다.
    """

    def _age(self, storage, path, seconds=3600):
        import os
        import time

        target = storage.image_dir / path.split("/")[-1]
        old = time.time() - seconds
        os.utime(target, (old, old))

    def _make_image(self, storage):
        from PIL import Image

        return storage.save_image(Image.new("RGB", (10, 10)))

    async def test_removes_unreferenced_image(self, service, storage) -> None:
        path = self._make_image(storage)
        self._age(storage, path)

        assert await service.cleanup_orphan_images() == 1
        assert not (storage.image_dir / path.split("/")[-1]).exists()

    async def test_keeps_referenced_thumbnail(self, service, storage) -> None:
        row = await _new(service)
        path = self._make_image(storage)
        self._age(storage, path)

        row.thumbnail_path = path
        await service._session.commit()

        assert await service.cleanup_orphan_images() == 0
        assert (storage.image_dir / path.split("/")[-1]).is_file()

    async def test_keeps_referenced_marked_image(self, service, storage) -> None:
        row = await _new(service)
        await _run_stub(service, row)

        marked = [
            o.marked_image_path
            for o in await service._repo.get_objects(row.analysis_id)
            if o.marked_image_path
        ]
        assert marked

        # 스텁이 만든 경로에 실제 파일을 만들어 둔다.
        for path in marked:
            (storage.image_dir / path.split("/")[-1]).write_bytes(b"x")
            self._age(storage, path)

        assert await service.cleanup_orphan_images() == 0
        for path in marked:
            assert (storage.image_dir / path.split("/")[-1]).is_file()

    async def test_keeps_recent_files(self, service, storage) -> None:
        """진행 중인 분석이 방금 만든 파일을 지우면 안 된다."""
        path = self._make_image(storage)

        assert await service.cleanup_orphan_images() == 0
        assert (storage.image_dir / path.split("/")[-1]).is_file()

    async def test_mixed(self, service, storage) -> None:
        row = await _new(service)
        kept = self._make_image(storage)
        orphan_a = self._make_image(storage)
        orphan_b = self._make_image(storage)

        for path in (kept, orphan_a, orphan_b):
            self._age(storage, path)

        row.thumbnail_path = kept
        await service._session.commit()

        assert await service.cleanup_orphan_images() == 2
        assert (storage.image_dir / kept.split("/")[-1]).is_file()

    async def test_no_images(self, service) -> None:
        assert await service.cleanup_orphan_images() == 0

    async def test_default_age_covers_the_timeout(self) -> None:
        """기본 대기 시간이 처리 제한 시간보다 길어야 한다.

        짧으면 진행 중인 분석의 파일을 지운다.
        """
        from app.core.constants import PROCESSING_TIMEOUT_SECONDS
        from app.services.analysis_service import ORPHAN_IMAGE_MIN_AGE

        assert ORPHAN_IMAGE_MIN_AGE > PROCESSING_TIMEOUT_SECONDS


class TestRetry:
    async def test_retry_resets_state(self, service) -> None:
        row = await _new(service)
        await service.mark_failed(row, "실패", AnalysisStage.OBJECT_DETECTION)
        await service._session.commit()

        retried = await service.prepare_retry(row.analysis_id, DEVICE)
        assert retried.status == AnalysisStatus.PENDING.value
        assert retried.retry_count == 1
        assert retried.progress == 0
        assert retried.stage is None
        assert retried.error_message is None
        assert retried.total_score == 0

    async def test_video_is_kept(self, service) -> None:
        """영상은 재분석 입력이므로 유지한다. 재전송을 요구하지 않는다."""
        row = await _new(service)
        await service.mark_failed(row, "실패")
        await service._session.commit()

        retried = await service.prepare_retry(row.analysis_id, DEVICE)
        assert retried.video_path == "/videos/a.mp4"

    async def test_same_id_reused(self, service) -> None:
        """새 분석을 만들지 않는다. 동일 영상의 실패가 이력에 중복되지 않게 한다."""
        row = await _new(service)
        await service.mark_failed(row, "실패")
        await service._session.commit()

        retried = await service.prepare_retry(row.analysis_id, DEVICE)
        assert retried.analysis_id == row.analysis_id

    async def test_children_cleared(self, service) -> None:
        """재시도 시 기존 결과를 지운다.

        현재 구현에서 자식 행은 완료 시점에만 생성되므로 FAILED 행에는 없다.
        다만 명세가 정리를 요구하고, 부분 저장으로 바뀔 수 있으므로 방어적으로 검증한다.
        """
        from app.models import DetectedObject, Recommendation

        row = await _new(service)
        await service.mark_failed(row, "실패")
        service._session.add(
            DetectedObject(
                analysis_id=row.analysis_id,
                object_name="전선",
                confidence=0.9,
                detection_frame_count=3,
                risk_level="HIGH",
                frame_number=1,
                x=0.1,
                y=0.1,
                width=0.2,
                height=0.2,
            )
        )
        service._session.add(
            Recommendation(
                analysis_id=row.analysis_id,
                recommendation_type="SAFETY",
                recommendation_text="전선을 정리해주세요.",
                priority=1,
                source="DETECTED",
            )
        )
        await service._session.commit()

        await service.prepare_retry(row.analysis_id, DEVICE)
        assert await service._repo.get_objects(row.analysis_id) == []
        assert await service._repo.get_recommendations(row.analysis_id) == []

    async def test_limit_is_three(self, service) -> None:
        row = await _new(service)
        for _ in range(3):
            await service.mark_failed(row, "실패")
            await service._session.commit()
            await service.prepare_retry(row.analysis_id, DEVICE)
            await service._session.commit()

        await service.mark_failed(row, "실패")
        await service._session.commit()
        with pytest.raises(PetFitError) as e:
            await service.prepare_retry(row.analysis_id, DEVICE)
        assert e.value.code is ErrorCode.RETRY_LIMIT_EXCEEDED
        assert row.retry_count == 3

    async def test_only_failed_is_retryable(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        with pytest.raises(PetFitError) as e:
            await service.prepare_retry(row.analysis_id, DEVICE)
        assert e.value.code is ErrorCode.ANALYSIS_NOT_RETRYABLE


class TestCanRetry:
    async def test_true_when_failed_and_idle(self, service) -> None:
        row = await _new(service)
        await service.mark_failed(row, "실패")
        await service._session.commit()
        assert await service.can_retry(row) is True

    async def test_false_when_another_is_active(self, service) -> None:
        """다른 분석이 진행 중이면 재시도할 수 없다.

        이 값이 틀리면 클라이언트가 재시도를 눌러 409를 받고서야 알게 된다.
        """
        failed = await _new(service)
        await service.mark_failed(failed, "실패")
        await service._session.commit()

        await service.create(DEVICE, AnimalGroup.CAT, SpaceType.BEDROOM, "/videos/b.mp4")
        await service._session.commit()

        assert await service.can_retry(failed) is False

    async def test_false_when_limit_reached(self, service) -> None:
        row = await _new(service)
        row.retry_count = 3
        await service.mark_failed(row, "실패")
        await service._session.commit()
        assert await service.can_retry(row) is False

    async def test_list_respects_active_analysis(self, service) -> None:
        failed = await _new(service)
        await service.mark_failed(failed, "실패")
        await service._session.commit()
        await service.create(DEVICE, AnimalGroup.CAT, SpaceType.BEDROOM, "/videos/b.mp4")
        await service._session.commit()

        _, _, retryable = await service.list_history(DEVICE, 1, 20, None)
        assert retryable == set()


class TestFailure:
    async def test_stage_and_progress_preserved(self, service) -> None:
        """10에서 실패한 것과 82까지 가다 실패한 것이 구분되어야 한다."""
        row = await _new(service)
        await service.mark_stage(row, AnalysisStage.OBJECT_DETECTION)
        await service.mark_failed(row, "객체 탐지에 실패했습니다.", AnalysisStage.OBJECT_DETECTION)
        await service._session.commit()

        assert row.stage == AnalysisStage.OBJECT_DETECTION.value
        assert row.progress == 20

    async def test_failure_before_pipeline_has_no_stage(self, service) -> None:
        row = await _new(service)
        await service.mark_failed(row, "대기열이 가득 찼습니다.")
        await service._session.commit()
        assert row.stage is None
        assert row.progress == 0

    async def test_stub_can_simulate_failure(self, service) -> None:
        """스텁으로 실패 흐름을 재현할 수 있어야 프론트가 실패 화면을 개발한다."""
        from app.ai.pipeline import PipelineError

        row = await _new(service)
        with pytest.raises(PipelineError) as e:
            await _run_stub(service, row, fail_at=AnalysisStage.OBJECT_DETECTION)
        assert e.value.stage is AnalysisStage.OBJECT_DETECTION


class TestDelete:
    @pytest.mark.parametrize("status", ["PENDING", "PROCESSING"])
    async def test_active_cannot_be_deleted(self, service, status) -> None:
        """진행 중 삭제를 허용하면 워커가 사라진 행을 갱신하려 시도한다."""
        row = await _new(service)
        if status == "PROCESSING":
            await service.mark_stage(row, AnalysisStage.OBJECT_DETECTION)
            await service._session.commit()

        with pytest.raises(PetFitError) as e:
            await service.delete(row.analysis_id, DEVICE)
        assert e.value.code is ErrorCode.ANALYSIS_NOT_DELETABLE

    async def test_completed_can_be_deleted(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        await service.delete(row.analysis_id, DEVICE)
        await service._session.commit()
        assert await service.get_internal(row.analysis_id) is None

    async def test_children_cascade(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        analysis_id = row.analysis_id
        await service.delete(analysis_id, DEVICE)
        await service._session.commit()

        assert await service._repo.get_objects(analysis_id) == []
        assert await service._repo.get_recommendations(analysis_id) == []

    async def test_other_device_cannot_delete(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        with pytest.raises(PetFitError) as e:
            await service.delete(row.analysis_id, OTHER_DEVICE)
        assert e.value.code is ErrorCode.ANALYSIS_NOT_FOUND


class TestHistory:
    async def test_ordered_by_created_at_desc(self, service) -> None:
        for i in range(3):
            row = await service.create(
                DEVICE, AnimalGroup.CAT, SpaceType.BALCONY, f"/videos/{i}.mp4"
            )
            await service.mark_failed(row, "실패")
            await service._session.commit()

        rows, total, _ = await service.list_history(DEVICE, 1, 20, None)
        assert total == 3
        assert all(
            rows[i].created_at >= rows[i + 1].created_at for i in range(len(rows) - 1)
        )

    async def test_status_filter(self, service) -> None:
        row = await _new(service)
        await _run_stub(service, row)
        await service.create(DEVICE, AnimalGroup.CAT, SpaceType.BALCONY, "/videos/b.mp4")
        await service._session.commit()

        _, total, _ = await service.list_history(
            DEVICE, 1, 20, AnalysisStatus.COMPLETED
        )
        assert total == 1

    async def test_pagination(self, service) -> None:
        for i in range(5):
            row = await service.create(
                DEVICE, AnimalGroup.CAT, SpaceType.BALCONY, f"/videos/{i}.mp4"
            )
            await service.mark_failed(row, "실패")
            await service._session.commit()

        page1, total, _ = await service.list_history(DEVICE, 1, 2, None)
        page2, _, _ = await service.list_history(DEVICE, 2, 2, None)
        assert total == 5
        assert len(page1) == len(page2) == 2
        assert {r.analysis_id for r in page1}.isdisjoint({r.analysis_id for r in page2})

    async def test_device_isolation(self, service) -> None:
        await _new(service)
        await service.create(
            OTHER_DEVICE, AnimalGroup.CAT, SpaceType.BALCONY, "/videos/b.mp4"
        )
        await service._session.commit()

        _, total, _ = await service.list_history(DEVICE, 1, 20, None)
        assert total == 1
