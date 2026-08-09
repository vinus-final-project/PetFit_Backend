"""파이프라인 산출물이 실제로 저장되는지 검증.

**이 파일이 없어서 실제 분석이 한 건도 저장되지 않는 상태를 놓쳤다.**

각 계층은 잘 검증되고 있었다. 파이프라인은 결과를 만들었고, 서비스 계층은
결과를 저장했다. 그런데 서비스 계층 테스트는 `StubPipeline` 의 고정 결과만
써서, 실제 Vision 이 만든 값이 DB 제약을 통과하는지는 아무도 확인하지 않았다.

    Vision 이 매긴 프레임 번호   0부터
    ck_detected_object_frame_number   1 이상

전 계층 테스트가 통과하는데도 저장이 실패했다.

여기서는 **진짜 영상 파일에서 시작해 DB 행이 남을 때까지** 통과시킨다.
제약을 그대로 유지한 SQLite 를 쓰므로 MySQL 없이 돌아간다.
"""

import json

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ai.analysis_pipeline import AnalysisPipeline
from app.ai.environment_analysis import EnvironmentAnalyzer
from app.ai.llm.fake import FakeLLM
from app.ai.vision.detector import StubDetector
from app.ai.vision.pipeline import VisionPipeline
from app.schemas.enums import AnalysisStatus, AnimalGroup, RiskLevel, SpaceType
from app.services.analysis_service import AnalysisService
from tests.conftest import sqlite_metadata

DEVICE = "3f2b8c10-9d7e-4a51-8f6c-2e4b7a9d0c35"
GROUP = AnimalGroup.SMALL_DOG
SPACE = SpaceType.LIVING_ROOM

RESPONSE = json.dumps(
    {
        "riskFactors": [
            {"text": "전선이 바닥에 노출되어 있습니다.", "source": "DETECTED"},
            {"text": "창가에 화분이 놓여 있습니다.", "source": "OBSERVED"},
        ],
        "analysis": [
            "활동 공간은 충분하지만 전선이 위험 요소입니다.",
            "휴식 공간은 소파로 일부 확보되어 있습니다.",
        ],
        "recommendations": [
            {
                "type": "SAFETY",
                "text": "전선을 벽면으로 정리해주세요.",
                "priority": 1,
                "source": "DETECTED",
            },
            {
                "type": "ENVIRONMENT",
                "text": "화분을 손이 닿지 않는 곳으로 옮겨주세요.",
                "priority": 2,
                "source": "OBSERVED",
            },
        ],
    },
    ensure_ascii=False,
)


@pytest_asyncio.fixture
async def sessionmaker_():
    """CHECK 제약을 그대로 유지한 SQLite. 제약을 빼면 이 파일의 의미가 사라진다."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(sqlite_metadata().create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def scene(make_video):
    return make_video(seconds=8.4, fps=30, width=640, height=360)


@pytest_asyncio.fixture
async def saved(sessionmaker_, storage, scene):
    """영상을 분석하고 결과를 저장한 뒤 행을 돌려준다."""
    async def noop(stage):
        return None

    pipeline = AnalysisPipeline(
        VisionPipeline(StubDetector(), storage),
        EnvironmentAnalyzer(FakeLLM(RESPONSE)),
        storage,
    )
    result = await pipeline.run(scene, GROUP, SPACE, noop)

    async with sessionmaker_() as session:
        service = AnalysisService(session, storage)
        row = await service.create(DEVICE, GROUP, SPACE, "/videos/a.mp4")
        await session.commit()

        await service.mark_completed(row, result)
        await session.commit()

        yield row, result, service


class TestPersistence:
    async def test_saves_without_constraint_violation(self, saved) -> None:
        """가장 중요한 검증이다. 여기서 실패하면 분석이 한 건도 저장되지 않는다."""
        row, _, _ = saved
        assert row.status == AnalysisStatus.COMPLETED.value

    async def test_frame_numbers_satisfy_the_constraint(self, saved) -> None:
        """ck_detected_object_frame_number 는 1 이상을 요구한다."""
        _, result, _ = saved
        assert all(o.frame_number >= 1 for o in result.detected_objects)

    async def test_measurements_are_stored(self, saved) -> None:
        row, result, _ = saved

        assert row.frame_count == result.frame_count
        assert row.thumbnail_path == result.thumbnail_path
        assert float(row.occupancy_ratio) == pytest.approx(result.occupancy_ratio)

    async def test_score_is_recomputed_on_save(self, saved) -> None:
        """서비스 계층이 파이프라인을 믿지 않고 스스로 산출한다."""
        row, _, _ = saved
        assert row.total_score == 56

    async def test_detected_objects_are_stored(self, saved) -> None:
        row, result, service = saved
        rows = await service._repo.get_objects(row.analysis_id)

        assert len(rows) == len(result.detected_objects)
        assert {r.object_name for r in rows} == result.object_names

    async def test_recommendations_are_stored(self, saved) -> None:
        row, _, service = saved
        rows = await service._repo.get_recommendations(row.analysis_id)

        assert len(rows) == 2
        assert sorted(r.priority for r in rows) == [1, 2]

    async def test_risk_factors_are_stored(self, saved) -> None:
        row, _, _ = saved

        assert len(row.risk_factors) == 2
        assert {f["source"] for f in row.risk_factors} == {"DETECTED", "OBSERVED"}

    async def test_analysis_text_is_stored(self, saved) -> None:
        row, _, _ = saved
        assert len(row.analysis_result) == 2


class TestMarkingConstraint:
    """ck_detected_object_marking — SAFE 는 마킹 경로를 가질 수 없다."""

    async def test_safe_objects_have_no_marked_path(self, saved) -> None:
        row, _, service = saved
        rows = await service._repo.get_objects(row.analysis_id)

        for r in rows:
            if r.risk_level == RiskLevel.SAFE.value:
                assert r.marked_image_path is None

    async def test_risky_objects_have_a_marked_path(self, saved) -> None:
        row, _, service = saved
        rows = await service._repo.get_objects(row.analysis_id)

        risky = [r for r in rows if r.risk_level != RiskLevel.SAFE.value]
        assert risky
        assert all(r.marked_image_path for r in risky)


class TestRetrievable:
    """저장한 결과를 다시 읽을 수 있어야 프론트가 화면을 그린다."""

    async def test_detail_round_trip(self, saved) -> None:
        row, _, service = saved
        detail = await service.get_detail(row.analysis_id, DEVICE)

        assert detail.analysis.total_score == 56
        assert len(detail.objects) == 5
        assert len(detail.recommendations) == 2

    async def test_response_dto_builds(self, saved) -> None:
        """DTO 변환까지 통과해야 API 가 응답할 수 있다."""
        from app.schemas.analysis import AnalysisDetailOut

        row, _, service = saved
        detail = await service.get_detail(row.analysis_id, DEVICE)
        out = AnalysisDetailOut.from_model(
            detail.analysis, detail.objects, detail.recommendations
        )

        body = out.model_dump(by_alias=True)
        assert body["petFitScore"]["total"] == 56
        assert len(body["detectedObjects"]) == 5

        # 프레임 번호는 응답에 포함되지 않는다. 마킹 배경을 고르는 내부 값이라
        # 클라이언트가 쓸 곳이 없다. 대신 그 결과인 마킹 경로가 나간다.
        assert "frameNumber" not in body["detectedObjects"][0]
        assert any(o["markedImage"] for o in body["detectedObjects"])


class TestGroups:
    """그룹마다 위험도가 달라 마킹 대상이 바뀐다. 제약도 함께 바뀐다."""

    @pytest.mark.parametrize(
        "group", [AnimalGroup.SMALL_DOG, AnimalGroup.LARGE_DOG, AnimalGroup.CAT]
    )
    async def test_every_group_saves(
        self, sessionmaker_, storage, scene, group
    ) -> None:
        async def noop(stage):
            return None

        pipeline = AnalysisPipeline(
            VisionPipeline(StubDetector(), storage),
            EnvironmentAnalyzer(FakeLLM(RESPONSE)),
            storage,
        )
        result = await pipeline.run(scene, group, SPACE, noop)

        async with sessionmaker_() as session:
            service = AnalysisService(session, storage)
            row = await service.create(DEVICE, group, SPACE, "/videos/a.mp4")
            await session.commit()

            await service.mark_completed(row, result)
            await session.commit()

        assert row.status == AnalysisStatus.COMPLETED.value

    @pytest.mark.parametrize("space", list(SpaceType))
    async def test_every_space_saves(
        self, sessionmaker_, storage, scene, space
    ) -> None:
        async def noop(stage):
            return None

        pipeline = AnalysisPipeline(
            VisionPipeline(StubDetector(), storage),
            EnvironmentAnalyzer(FakeLLM(RESPONSE)),
            storage,
        )
        result = await pipeline.run(scene, GROUP, space, noop)

        async with sessionmaker_() as session:
            service = AnalysisService(session, storage)
            row = await service.create(DEVICE, GROUP, space, "/videos/a.mp4")
            await session.commit()

            await service.mark_completed(row, result)
            await session.commit()

        assert 0 <= row.total_score <= 100
