"""API 계층 검증.

라우터가 명세대로 응답하는지 확인한다. 확인 대상은 세 가지다.

    오류 코드   : 상황마다 명세에 정의된 code와 HTTP 상태가 나가는가
    응답 형식   : camelCase 필드명, 목록 래핑, null 처리
    처리 순서   : 검증·저장·접수의 순서와 실패 시 되돌리기

AI 파이프라인과 MySQL은 쓰지 않는다. 큐와 저장소는 가짜로 바꾸고 DB는 SQLite에
같은 CHECK 제약을 걸어 띄운다. 서비스 계층의 규칙은 `test_analysis_service.py` 가
이미 검증하므로 여기서는 **라우터가 그것을 올바르게 엮는지**만 본다.
"""

import os
import tempfile
import uuid

# 앱을 만들 때 저장소 디렉터리가 생성된다. 설정을 읽기 전에 임시 경로로 돌려
# 저장소 루트가 작업 디렉터리에 만들어지지 않게 한다.
os.environ["STORAGE_ROOT"] = tempfile.mkdtemp(prefix="petfit-test-")

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.ai.stub import StubPipeline  # noqa: E402
from app.api.deps import get_db  # noqa: E402
from app.core.constants import MAX_QUEUE_SIZE  # noqa: E402
from app.core.exceptions import ErrorCode, PetFitError  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas.enums import AnalysisStatus, AnimalGroup, SpaceType  # noqa: E402
from app.services.analysis_service import AnalysisService  # noqa: E402
from app.services.storage import Storage, VideoInfo  # noqa: E402
from tests.conftest import sqlite_metadata  # noqa: E402

DEVICE = "3f2b8c10-9d7e-4a51-8f6c-2e4b7a9d0c35"
OTHER_DEVICE = "00000000-0000-4000-8000-000000000000"
HEADERS = {"X-Device-Id": DEVICE}

VIDEO = {"video": ("room.mp4", b"fake-mp4-bytes", "video/mp4")}
FORM = {"animalGroup": "small_dog", "spaceType": "living_room"}


class FakeStorage(Storage):
    """영상 검증을 건너뛰는 저장소.

    실제 검증은 PyAV로 컨테이너를 열어야 하므로 유효한 H.264 파일이 필요하다.
    여기서 확인하려는 것은 라우터가 저장 실패와 성공을 어떻게 다루는지이므로,
    판별만 대신하고 파일 삭제는 실제 동작을 그대로 쓴다.
    """

    def __init__(self, root) -> None:
        super().__init__(root)
        self.deleted: list[str] = []
        self.error: Exception | None = None

    async def save_video(self, upload) -> VideoInfo:
        if self.error is not None:
            raise self.error
        # 첨부 누락 판정은 실제 저장소와 동일하게 유지한다. 여기를 빠뜨리면
        # 라우터가 아니라 이 가짜 구현을 시험하게 된다.
        if upload is None or not getattr(upload, "filename", None):
            raise PetFitError(ErrorCode.VIDEO_REQUIRED, field="video")
        path = self.video_dir / f"{uuid.uuid4()}.mp4"
        path.write_bytes(b"fake")
        return VideoInfo(path=path, duration=8.4, width=1920, height=1080, codec="h264")

    def delete(self, *relative_paths) -> int:
        self.deleted.extend(p for p in relative_paths if p)
        return super().delete(*relative_paths)


class FakeQueue:
    """작업을 실행하지 않고 접수 기록만 남기는 큐."""

    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.full = False
        self.fail_submit = False

    @property
    def size(self) -> int:
        return MAX_QUEUE_SIZE if self.full else 0

    def submit(self, analysis_id, video_path, group, space) -> None:
        if self.full or self.fail_submit:
            raise PetFitError(ErrorCode.QUEUE_FULL)
        self.submitted.append((analysis_id, video_path, group, space))

    async def drain(self, timeout: float = 5.0) -> None:
        return None


@pytest_asyncio.fixture
async def sessionmaker_():
    """CHECK 제약을 그대로 유지한 SQLite. MySQL 서버 없이 돌린다."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(sqlite_metadata().create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def app(sessionmaker_, tmp_path):
    """가짜 저장소·큐를 끼운 앱. 수명주기는 실행하지 않는다."""
    application = create_app()
    application.state.storage = FakeStorage(tmp_path / "storage")
    application.state.queue = FakeQueue()

    async def _db():
        async with sessionmaker_() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db] = _db
    return application


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- 시드 헬퍼 -------------------------------------------------------------


async def _seed(
    sessionmaker_,
    storage,
    *,
    device=DEVICE,
    group=AnimalGroup.SMALL_DOG,
    space=SpaceType.LIVING_ROOM,
    complete=False,
    fail=False,
    retry_count=0,
):
    """분석 행을 하나 만든다. 필요하면 완료·실패 상태까지 진행시킨다."""
    async with sessionmaker_() as session:
        service = AnalysisService(session, storage)
        row = await service.create(device, group, space, "/videos/seed.mp4")

        if complete:
            async def on_stage(stage):
                await service.mark_stage(row, stage)

            result = await StubPipeline(speed=0).run("seed.mp4", group, space, on_stage)
            await service.mark_completed(row, result)
        elif fail:
            await service.mark_failed(row, "객체 탐지에 실패했습니다.")
            row.retry_count = retry_count

        await session.commit()
        return row.analysis_id


# =============================================================================
# 메타 API
# =============================================================================


class TestMeta:
    async def test_animals_returns_only_analyzable(self, client) -> None:
        res = await client.get("/animals")
        assert res.status_code == 200

        codes = [a["code"] for a in res.json()["animals"]]
        assert codes == ["small_dog", "large_dog", "cat"]

    async def test_animals_hides_reserved_groups(self, client) -> None:
        """확장 그룹은 Enum에 예약만 되어 있고 노출하지 않는다."""
        codes = {a["code"] for a in (await client.get("/animals")).json()["animals"]}
        assert not codes & {"small_animal", "bird", "reptile"}

    async def test_spaces(self, client) -> None:
        res = await client.get("/spaces")
        codes = [s["code"] for s in res.json()["spaces"]]
        assert codes == ["living_room", "bedroom", "kitchen", "balcony"]

    async def test_meta_needs_no_device_id(self, client) -> None:
        """그룹·공간 목록은 기기 식별자 없이 조회할 수 있어야 한다."""
        assert (await client.get("/animals")).status_code == 200


# =============================================================================
# 기기 식별자
# =============================================================================


class TestDeviceId:
    async def test_missing_header(self, client) -> None:
        res = await client.get("/analysis")
        assert res.status_code == 400
        assert res.json()["code"] == "DEVICE_ID_REQUIRED"

    async def test_not_uuid(self, client) -> None:
        res = await client.get("/analysis", headers={"X-Device-Id": "not-a-uuid"})
        assert res.status_code == 400
        assert res.json()["code"] == "DEVICE_ID_INVALID"

    async def test_uuid_but_not_v4(self, client) -> None:
        """v1 UUID는 시각·MAC 기반이라 추측 가능하다. v4만 받는다."""
        res = await client.get(
            "/analysis", headers={"X-Device-Id": "3f2b8c10-9d7e-11a1-8f6c-2e4b7a9d0c35"}
        )
        assert res.status_code == 400
        assert res.json()["code"] == "DEVICE_ID_INVALID"

    async def test_error_body_has_standard_shape(self, client) -> None:
        body = (await client.get("/analysis")).json()
        assert set(body) == {"code", "message", "field", "status"}


# =============================================================================
# API-002 분석 요청
# =============================================================================


class TestCreate:
    async def test_accepts_and_returns_id(self, client, app) -> None:
        res = await client.post("/analysis", headers=HEADERS, files=VIDEO, data=FORM)

        assert res.status_code == 202
        body = res.json()
        assert body["status"] == "PENDING"
        assert body["analysisId"] >= 1
        assert len(app.state.queue.submitted) == 1

    async def test_submits_with_group_and_space(self, client, app) -> None:
        await client.post(
            "/analysis",
            headers=HEADERS,
            files=VIDEO,
            data={"animalGroup": "cat", "spaceType": "balcony"},
        )
        _, _, group, space = app.state.queue.submitted[0]
        assert group is AnimalGroup.CAT
        assert space is SpaceType.BALCONY

    async def test_missing_animal_group(self, client) -> None:
        res = await client.post(
            "/analysis", headers=HEADERS, files=VIDEO, data={"spaceType": "living_room"}
        )
        assert res.status_code == 400
        assert res.json()["code"] == "ANIMAL_GROUP_REQUIRED"

    async def test_reserved_animal_group(self, client) -> None:
        res = await client.post(
            "/analysis",
            headers=HEADERS,
            files=VIDEO,
            data={"animalGroup": "reptile", "spaceType": "living_room"},
        )
        assert res.status_code == 400
        assert res.json()["code"] == "ANIMAL_GROUP_UNSUPPORTED"

    async def test_missing_space_type(self, client) -> None:
        res = await client.post(
            "/analysis", headers=HEADERS, files=VIDEO, data={"animalGroup": "small_dog"}
        )
        assert res.status_code == 400
        assert res.json()["code"] == "SPACE_TYPE_REQUIRED"

    async def test_missing_video(self, client) -> None:
        res = await client.post("/analysis", headers=HEADERS, data=FORM)
        assert res.status_code == 400
        assert res.json()["code"] == "VIDEO_REQUIRED"

    async def test_form_validated_before_video_is_saved(self, client, app) -> None:
        """폼이 잘못되면 영상을 읽지 않는다. 어차피 거절할 요청에 100MB를 쓰지 않는다."""
        app.state.storage.error = AssertionError("영상을 저장하면 안 된다")

        res = await client.post(
            "/analysis", headers=HEADERS, files=VIDEO, data={"spaceType": "living_room"}
        )
        assert res.status_code == 400

    async def test_second_request_conflicts(self, client, sessionmaker_, app) -> None:
        await _seed(sessionmaker_, app.state.storage)

        res = await client.post("/analysis", headers=HEADERS, files=VIDEO, data=FORM)
        assert res.status_code == 409
        body = res.json()
        assert body["code"] == "ANALYSIS_IN_PROGRESS"
        assert body["status"] == "PROCESSING"

    async def test_conflict_removes_uploaded_video(
        self, client, sessionmaker_, app
    ) -> None:
        """접수에 실패하면 이미 저장한 영상을 지운다. 참조 없는 파일은 추적할 수 없다."""
        await _seed(sessionmaker_, app.state.storage)

        await client.post("/analysis", headers=HEADERS, files=VIDEO, data=FORM)

        storage = app.state.storage
        assert len(storage.deleted) == 1
        assert not (storage.video_dir / storage.deleted[0].split("/")[-1]).exists()

    async def test_other_device_can_start(self, client, sessionmaker_, app) -> None:
        """동시 1건 제한은 기기 단위다. 다른 기기는 막히지 않는다."""
        await _seed(sessionmaker_, app.state.storage)

        res = await client.post(
            "/analysis",
            headers={"X-Device-Id": OTHER_DEVICE},
            files=VIDEO,
            data=FORM,
        )
        assert res.status_code == 202

    async def test_queue_full_rejects_before_saving(self, client, app) -> None:
        app.state.queue.full = True
        app.state.storage.error = AssertionError("영상을 저장하면 안 된다")

        res = await client.post("/analysis", headers=HEADERS, files=VIDEO, data=FORM)
        assert res.status_code == 503
        assert res.json()["code"] == "QUEUE_FULL"

    async def test_queue_full_creates_no_row(self, client, app) -> None:
        app.state.queue.full = True
        await client.post("/analysis", headers=HEADERS, files=VIDEO, data=FORM)

        listed = (await client.get("/analysis", headers=HEADERS)).json()
        assert listed["pagination"]["totalCount"] == 0

    async def test_submit_race_leaves_retryable_failure(self, client, app) -> None:
        """사전 검사를 통과한 뒤 대기열이 차면, 행을 지우지 않고 실패로 남긴다.

        사용자는 여유가 생겼을 때 영상을 다시 올리지 않고 재시도할 수 있다.
        """
        app.state.queue.fail_submit = True

        res = await client.post("/analysis", headers=HEADERS, files=VIDEO, data=FORM)
        assert res.status_code == 503

        item = (await client.get("/analysis", headers=HEADERS)).json()["analyses"][0]
        assert item["status"] == "FAILED"
        assert item["canRetry"] is True


# =============================================================================
# API-003 이력 조회
# =============================================================================


class TestList:
    async def test_empty_returns_array_not_null(self, client) -> None:
        body = (await client.get("/analysis", headers=HEADERS)).json()
        assert body["analyses"] == []
        assert body["pagination"]["totalPages"] == 0

    async def test_only_own_device(self, client, sessionmaker_, app) -> None:
        await _seed(sessionmaker_, app.state.storage, device=OTHER_DEVICE)

        body = (await client.get("/analysis", headers=HEADERS)).json()
        assert body["pagination"]["totalCount"] == 0

    async def test_pagination_shape(self, client, sessionmaker_, app) -> None:
        for _ in range(3):
            analysis_id = await _seed(
                sessionmaker_, app.state.storage, complete=True
            )
            assert analysis_id

        body = (await client.get("/analysis?page=1&size=2", headers=HEADERS)).json()
        assert len(body["analyses"]) == 2
        assert body["pagination"] == {
            "page": 1,
            "size": 2,
            "totalCount": 3,
            "totalPages": 2,
            "hasNext": True,
        }

    async def test_incomplete_has_null_score_and_thumbnail(
        self, client, sessionmaker_, app
    ) -> None:
        await _seed(sessionmaker_, app.state.storage)

        item = (await client.get("/analysis", headers=HEADERS)).json()["analyses"][0]
        assert item["petFitScore"] is None
        assert item["thumbnailImage"] is None
        assert item["canRetry"] is False

    async def test_completed_carries_score(self, client, sessionmaker_, app) -> None:
        await _seed(sessionmaker_, app.state.storage, complete=True)

        item = (await client.get("/analysis", headers=HEADERS)).json()["analyses"][0]
        # 스텁은 문서의 계산 예시와 같은 탐지 결과를 쓴다. 소형견·거실이면 56점이다.
        assert item["petFitScore"]["total"] == 56
        assert item["spaceType"] == "living_room"

    async def test_status_filter(self, client, sessionmaker_, app) -> None:
        await _seed(sessionmaker_, app.state.storage, complete=True)

        body = (
            await client.get("/analysis?status=FAILED", headers=HEADERS)
        ).json()
        assert body["pagination"]["totalCount"] == 0

    @pytest.mark.parametrize(
        "query,code",
        [
            ("page=0", "PAGE_INVALID"),
            ("size=0", "SIZE_INVALID"),
            ("size=51", "SIZE_INVALID"),
            ("status=BAD", "STATUS_INVALID"),
        ],
    )
    async def test_out_of_range(self, client, query, code) -> None:
        res = await client.get(f"/analysis?{query}", headers=HEADERS)
        assert res.status_code == 400
        assert res.json()["code"] == code

    @pytest.mark.parametrize("query,code", [("page=abc", "PAGE_INVALID"), ("size=x", "SIZE_INVALID")])
    async def test_wrong_type_maps_to_spec_code(self, client, query, code) -> None:
        """FastAPI 기본 422 본문은 code가 없어 클라이언트가 분기할 수 없다."""
        res = await client.get(f"/analysis?{query}", headers=HEADERS)
        assert res.status_code == 400
        assert res.json()["code"] == code


# =============================================================================
# API-004 상세 조회
# =============================================================================


class TestDetail:
    async def test_not_completed(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage)

        res = await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        assert res.status_code == 409
        assert res.json()["code"] == "ANALYSIS_NOT_COMPLETED"
        assert res.json()["status"] == "PENDING"

    async def test_other_device_gets_404_not_403(
        self, client, sessionmaker_, app
    ) -> None:
        """403은 해당 ID의 분석이 존재한다는 사실을 노출한다."""
        analysis_id = await _seed(
            sessionmaker_, app.state.storage, device=OTHER_DEVICE, complete=True
        )

        res = await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        assert res.status_code == 404
        assert res.json()["code"] == "ANALYSIS_NOT_FOUND"

    async def test_unknown_id(self, client) -> None:
        res = await client.get("/analysis/9999", headers=HEADERS)
        assert res.status_code == 404

    async def test_non_numeric_id(self, client) -> None:
        """경로 변수가 정수가 아니어도 404로 통일한다."""
        res = await client.get("/analysis/abc", headers=HEADERS)
        assert res.status_code == 404
        assert res.json()["code"] == "ANALYSIS_NOT_FOUND"

    async def test_detail_shape(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        body = (await client.get(f"/analysis/{analysis_id}", headers=HEADERS)).json()

        assert body["status"] == "COMPLETED"
        assert body["petFitScore"]["total"] == 56
        assert body["thumbnailImage"].startswith("/images/")
        assert body["analysis"]
        assert body["completedAt"].endswith("+09:00")

    async def test_objects_sorted_by_risk(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        objects = (
            await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        ).json()["detectedObjects"]

        ranks = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "SAFE": 0}
        assert [ranks[o["risk"]] for o in objects] == sorted(
            [ranks[o["risk"]] for o in objects], reverse=True
        )
        assert objects[0]["name"] == "전선"

    async def test_safe_objects_have_no_marked_image(
        self, client, sessionmaker_, app
    ) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        objects = (
            await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        ).json()["detectedObjects"]

        for obj in objects:
            if obj["risk"] == "SAFE":
                assert obj["markedImage"] is None
            else:
                assert obj["markedImage"]

    async def test_risk_factors_detected_first(
        self, client, sessionmaker_, app
    ) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        factors = (
            await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        ).json()["riskFactors"]

        sources = [f["source"] for f in factors]
        assert sources == sorted(sources, key=lambda s: s != "DETECTED")

    async def test_recommendations_sorted_by_priority(
        self, client, sessionmaker_, app
    ) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        items = (
            await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        ).json()["recommendations"]

        assert [r["priority"] for r in items] == sorted(r["priority"] for r in items)


# =============================================================================
# API-006 진행 상태
# =============================================================================


class TestStatus:
    async def test_pending(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage)

        body = (
            await client.get(f"/analysis/{analysis_id}/status", headers=HEADERS)
        ).json()

        assert body["status"] == "PENDING"
        assert body["progress"] == 0
        assert body["stage"] is None
        assert body["canRetry"] is False
        assert body["message"] is None

    async def test_completed(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        body = (
            await client.get(f"/analysis/{analysis_id}/status", headers=HEADERS)
        ).json()

        assert body["progress"] == 100
        assert body["stage"] is None

    async def test_failed_keeps_stage(self, client, sessionmaker_, app) -> None:
        """실패 지점이 남아야 재촬영과 재시도 중 무엇을 안내할지 분기할 수 있다."""
        analysis_id = await _seed(sessionmaker_, app.state.storage, fail=True)

        body = (
            await client.get(f"/analysis/{analysis_id}/status", headers=HEADERS)
        ).json()

        assert body["status"] == "FAILED"
        assert body["canRetry"] is True
        assert body["message"]

    async def test_retry_limit_disables_retry(
        self, client, sessionmaker_, app
    ) -> None:
        analysis_id = await _seed(
            sessionmaker_, app.state.storage, fail=True, retry_count=3
        )

        body = (
            await client.get(f"/analysis/{analysis_id}/status", headers=HEADERS)
        ).json()

        assert body["retryCount"] == 3
        assert body["canRetry"] is False

    async def test_other_device(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(
            sessionmaker_, app.state.storage, device=OTHER_DEVICE
        )

        res = await client.get(f"/analysis/{analysis_id}/status", headers=HEADERS)
        assert res.status_code == 404


# =============================================================================
# API-007 재시도
# =============================================================================


class TestRetry:
    async def test_failed_analysis_is_resubmitted(
        self, client, sessionmaker_, app
    ) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, fail=True)

        res = await client.post(f"/analysis/{analysis_id}/retry", headers=HEADERS)

        assert res.status_code == 202
        assert res.json() == {
            "analysisId": analysis_id,
            "status": "PENDING",
            "retryCount": 1,
        }
        assert len(app.state.queue.submitted) == 1

    async def test_reuses_uploaded_video(self, client, sessionmaker_, app) -> None:
        """영상을 재전송하지 않는다. 저장된 파일을 그대로 다시 쓴다."""
        analysis_id = await _seed(sessionmaker_, app.state.storage, fail=True)

        await client.post(f"/analysis/{analysis_id}/retry", headers=HEADERS)

        _, video_path, _, _ = app.state.queue.submitted[0]
        assert video_path == app.state.storage.video_dir / "seed.mp4"

    async def test_completed_cannot_retry(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        res = await client.post(f"/analysis/{analysis_id}/retry", headers=HEADERS)
        assert res.status_code == 409
        assert res.json()["code"] == "ANALYSIS_NOT_RETRYABLE"
        assert res.json()["status"] == "COMPLETED"

    async def test_limit_exceeded(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(
            sessionmaker_, app.state.storage, fail=True, retry_count=3
        )

        res = await client.post(f"/analysis/{analysis_id}/retry", headers=HEADERS)
        assert res.status_code == 409
        assert res.json()["code"] == "RETRY_LIMIT_EXCEEDED"

    async def test_other_device(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(
            sessionmaker_, app.state.storage, device=OTHER_DEVICE, fail=True
        )

        res = await client.post(f"/analysis/{analysis_id}/retry", headers=HEADERS)
        assert res.status_code == 404


# =============================================================================
# API-005 삭제
# =============================================================================


class TestDelete:
    async def test_completed(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        res = await client.delete(f"/analysis/{analysis_id}", headers=HEADERS)
        assert res.status_code == 200
        assert res.json() == {"message": "분석 결과가 삭제되었습니다."}

        assert (
            await client.get(f"/analysis/{analysis_id}", headers=HEADERS)
        ).status_code == 404

    async def test_removes_video_and_images(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, complete=True)

        await client.delete(f"/analysis/{analysis_id}", headers=HEADERS)

        deleted = app.state.storage.deleted
        assert "/videos/seed.mp4" in deleted
        assert any(p.startswith("/images/") for p in deleted)

    async def test_pending_cannot_be_deleted(
        self, client, sessionmaker_, app
    ) -> None:
        """진행 중인 분석을 지우면 백그라운드 작업이 사라진 행을 갱신하려 한다."""
        analysis_id = await _seed(sessionmaker_, app.state.storage)

        res = await client.delete(f"/analysis/{analysis_id}", headers=HEADERS)
        assert res.status_code == 409
        assert res.json()["code"] == "ANALYSIS_NOT_DELETABLE"

    async def test_failed_can_be_deleted(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(sessionmaker_, app.state.storage, fail=True)

        res = await client.delete(f"/analysis/{analysis_id}", headers=HEADERS)
        assert res.status_code == 200

    async def test_other_device(self, client, sessionmaker_, app) -> None:
        analysis_id = await _seed(
            sessionmaker_, app.state.storage, device=OTHER_DEVICE, complete=True
        )

        res = await client.delete(f"/analysis/{analysis_id}", headers=HEADERS)
        assert res.status_code == 404


# =============================================================================
# 라우팅
# =============================================================================


class TestRouting:
    async def test_unknown_path_uses_standard_error_shape(self, client) -> None:
        body = (await client.get("/nope")).json()
        assert set(body) == {"code", "message", "field", "status"}

    async def test_openapi_is_served(self, client) -> None:
        """프론트는 /docs 로 계약을 확인한다."""
        res = await client.get("/openapi.json")
        assert res.status_code == 200
        assert "/analysis/{analysis_id}/status" in res.json()["paths"]

    async def test_status_route_is_not_shadowed(self, client) -> None:
        """`/analysis/{id}/status` 가 `/analysis/{id}` 에 먹히지 않아야 한다."""
        res = await client.get("/analysis/1/status", headers=HEADERS)
        assert res.json()["code"] == "ANALYSIS_NOT_FOUND"
