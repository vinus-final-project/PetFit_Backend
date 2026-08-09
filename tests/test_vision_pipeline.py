"""Vision 파이프라인 전 구간 검증.

진짜 영상 파일에서 시작해 객체 목록과 이미지까지 만든다. 부품별 검증은 각
파일에 있고, 여기서 보는 것은 **엮인 뒤에도 맞는가** 이다.

기준 장면은 ``app.ai.stub.StubPipeline`` 과 같은 5개 객체다. 소형견·거실에서
종합 56점이 나와야 하며, 이 값은 AI 분석 정의서의 계산 예시와 같다. 파이프라인
어딘가가 규칙과 어긋나면 이 숫자가 달라진다.
"""

import asyncio
import time

import pytest

from app.ai.pipeline import PipelineError
from app.ai.score_generator import generate
from app.ai.vision.detector import StubDetector
from app.ai.vision.pipeline import VisionPipeline
from app.ai.vision.types import VisionResult
from app.core.constants import FRAME_MAX, FRAME_MIN, LLM_MAX_IMAGES
from app.schemas.enums import AnalysisStage, AnimalGroup, RiskLevel, SpaceType

GROUP = AnimalGroup.SMALL_DOG

#: 8.4초 영상은 25프레임이 되어 기준 장면의 프레임 번호와 정확히 맞물린다.
SCENE_SECONDS = 8.4


@pytest.fixture
def scene(make_video):
    return make_video(seconds=SCENE_SECONDS, fps=30, width=640, height=360)


@pytest.fixture
def stages():
    """단계 진입을 기록하는 콜백."""
    seen: list[AnalysisStage] = []

    async def record(stage: AnalysisStage) -> None:
        seen.append(stage)

    record.seen = seen
    return record


def pipeline(storage, detector=None, **kwargs):
    return VisionPipeline(detector or StubDetector(), storage, **kwargs)


class TestRun:
    async def test_returns_vision_result(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert isinstance(result, VisionResult)

    async def test_frame_count_is_within_contract(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert FRAME_MIN <= result.frame_count <= FRAME_MAX

    async def test_reports_duration(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert result.capture_duration == pytest.approx(SCENE_SECONDS, abs=0.2)

    async def test_detects_the_scene_objects(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert result.object_names == {"전선", "창문", "카펫", "소파", "급수기"}

    async def test_no_duplicate_objects(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert len(result.detected_objects) == 5

    async def test_occupancy_is_within_range(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert 0.0 < result.occupancy_ratio <= 1.0

    async def test_creates_thumbnail_file(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        name = result.thumbnail_path.split("/")[-1]
        assert (storage.image_dir / name).is_file()

    async def test_marks_only_risky_objects(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)

        for obj in result.detected_objects:
            if obj.risk is RiskLevel.SAFE:
                assert obj.marked_image_path is None
            else:
                assert obj.marked_image_path is not None

    async def test_marked_files_exist(self, storage, scene, stages) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)

        for path in result.marked_image_paths:
            assert (storage.image_dir / path.split("/")[-1]).is_file()

    async def test_analysis_frames_respect_the_limit(
        self, storage, scene, stages
    ) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        assert 1 <= len(result.analysis_frames) <= LLM_MAX_IMAGES

    async def test_is_deterministic(self, storage, scene, stages) -> None:
        """같은 영상은 같은 결과를 내야 한다. 점수의 재현성이 여기에 달렸다."""
        first = await pipeline(storage).run(scene, GROUP, stages)
        second = await pipeline(storage).run(scene, GROUP, stages)

        assert first.occupancy_ratio == second.occupancy_ratio
        assert first.object_names == second.object_names
        assert [o.confidence for o in first.detected_objects] == [
            o.confidence for o in second.detected_objects
        ]


class TestScoreReproduction:
    """문서의 계산 예시를 재현하는지 확인한다."""

    async def test_small_dog_living_room_scores_56(
        self, storage, scene, stages
    ) -> None:
        result = await pipeline(storage).run(scene, GROUP, stages)
        score = generate(
            GROUP, SpaceType.LIVING_ROOM, result.object_names, result.occupancy_ratio
        )
        assert score.total == 56

    async def test_risk_depends_on_the_group(self, storage, scene, stages) -> None:
        """창문은 소형견에게 LOW, 고양이에게 HIGH 다."""
        dog = await pipeline(storage).run(scene, AnimalGroup.SMALL_DOG, stages)
        cat = await pipeline(storage).run(scene, AnimalGroup.CAT, stages)

        def risk_of(result, name):
            return next(o.risk for o in result.detected_objects if o.name == name)

        assert risk_of(dog, "창문") is RiskLevel.LOW
        assert risk_of(cat, "창문") is RiskLevel.HIGH

    async def test_cable_is_high_for_every_group(self, storage, scene, stages) -> None:
        """감전 위험은 종에 무관하다. 최소 위험도 하한이 적용된다."""
        for group in AnimalGroup:
            if group not in (AnimalGroup.SMALL_DOG, AnimalGroup.LARGE_DOG,
                             AnimalGroup.CAT):
                continue
            result = await pipeline(storage).run(scene, group, stages)
            cable = next(o for o in result.detected_objects if o.name == "전선")
            assert cable.risk is RiskLevel.HIGH


class TestStages:
    async def test_reports_every_stage(self, storage, scene, stages) -> None:
        await pipeline(storage).run(scene, GROUP, stages)
        assert stages.seen == [
            AnalysisStage.FRAME_EXTRACTION,
            AnalysisStage.OBJECT_DETECTION,
            AnalysisStage.OBJECT_TRACKING,
            AnalysisStage.FRAME_SELECTION,
            AnalysisStage.RISK_MARKING,
        ]

    async def test_progress_increases(self, storage, scene, stages) -> None:
        await pipeline(storage).run(scene, GROUP, stages)
        progress = [s.progress for s in stages.seen]
        assert progress == sorted(progress)

    async def test_stops_at_marking(self, storage, scene, stages) -> None:
        """점수 산출과 환경 분석은 Vision 의 범위가 아니다."""
        await pipeline(storage).run(scene, GROUP, stages)
        assert AnalysisStage.SCORE_CALCULATION not in stages.seen
        assert AnalysisStage.ENVIRONMENT_ANALYSIS not in stages.seen

    async def test_stage_is_reported_before_the_work(self, storage, scene) -> None:
        """단계는 진입 시점에 알려야 폴링이 진행 중임을 보여준다."""
        marker = []

        async def record(stage):
            marker.append(("stage", stage))

        class Watching(StubDetector):
            def detect(self, frames):
                marker.append(("detect", None))
                return super().detect(frames)

        await pipeline(storage, Watching()).run(scene, GROUP, record)

        detect_at = marker.index(("detect", None))
        stage_at = marker.index(("stage", AnalysisStage.OBJECT_DETECTION))
        assert stage_at < detect_at


class TestEventLoop:
    async def test_heavy_work_does_not_block_the_loop(self, storage, scene) -> None:
        """추론과 디코딩은 동기 작업이다. 스레드로 내보내지 않으면 서버가 멈춘다.

        멈추면 동시 처리 2건이 사실상 1건이 되고, 진행률 폴링도 응답하지 않아
        프론트에는 분석이 멈춘 것처럼 보인다.
        """
        ticks = 0
        running = True

        async def tick():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.01)

        class Slow(StubDetector):
            def detect(self, frames):
                time.sleep(0.3)
                return super().detect(frames)

        async def noop(stage):
            return None

        ticker = asyncio.create_task(tick())
        await pipeline(storage, Slow()).run(scene, GROUP, noop)
        running = False
        await ticker

        # 0.3초 동안 막히지 않았다면 최소 열 번 넘게 돌았어야 한다.
        assert ticks > 10


class TestCancellation:
    """취소가 얼마나 빨리 먹히는가.

    **파이썬은 스레드를 강제 종료할 수 없다.** 처리 제한 시간을 넘겨 작업이
    취소돼도 추론 스레드는 끝까지 돈다. 좀비가 GPU를 물고 있으면 동시 처리
    제한이 무의미해지고, 기본 스레드 풀이 소진되면 정상 요청까지 멈춘다.

    없앨 수는 없고, 낭비되는 작업을 줄일 수만 있다.
    """

    async def test_detection_is_split_into_chunks(self, storage, scene) -> None:
        """전부를 한 번에 보내면 취소해도 30프레임을 끝까지 추론한다."""
        sizes = []

        class Counting(StubDetector):
            def detect(self, frames):
                sizes.append(len(frames))
                return super().detect(frames)

        async def noop(stage):
            return None

        await pipeline(storage, Counting(), detect_chunk=8).run(scene, GROUP, noop)

        assert len(sizes) > 1
        assert max(sizes) <= 8
        assert sum(sizes) == 25

    async def test_chunking_does_not_change_the_result(self, storage, scene) -> None:
        """묶음을 나눠도 탐지 결과는 같아야 한다."""
        async def noop(stage):
            return None

        whole = await pipeline(storage, detect_chunk=100).run(scene, GROUP, noop)
        split = await pipeline(storage, detect_chunk=3).run(scene, GROUP, noop)

        assert whole.object_names == split.object_names
        assert whole.occupancy_ratio == split.occupancy_ratio

    async def test_cancellation_stops_between_chunks(self, storage, scene) -> None:
        """묶음 사이가 취소 확인 지점이 된다.

        나누지 않으면 전 프레임을 추론한 뒤에야 취소가 반영된다.
        """
        seen = 0

        class Slow(StubDetector):
            def detect(self, frames):
                nonlocal seen
                seen += len(frames)
                time.sleep(0.15)
                return super().detect(frames)

        async def noop(stage):
            return None

        task = asyncio.create_task(
            pipeline(storage, Slow(), detect_chunk=5).run(scene, GROUP, noop)
        )
        await asyncio.sleep(0.5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # 25프레임 전부가 아니라 일부만 처리됐어야 한다.
        await asyncio.sleep(0.3)
        assert seen < 25


class TestFailure:
    async def test_unreadable_video(self, storage, tmp_path, stages) -> None:
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")

        with pytest.raises(PipelineError) as e:
            await pipeline(storage).run(broken, GROUP, stages)
        assert e.value.stage is AnalysisStage.FRAME_EXTRACTION

    async def test_detector_failure_keeps_the_stage(
        self, storage, scene, stages
    ) -> None:
        class Broken(StubDetector):
            def detect(self, frames):
                raise RuntimeError("model weights not found at /opt/models/yolo26m.pt")

        with pytest.raises(PipelineError) as e:
            await pipeline(storage, Broken()).run(scene, GROUP, stages)
        assert e.value.stage is AnalysisStage.OBJECT_DETECTION

    async def test_internal_message_is_not_exposed(
        self, storage, scene, stages
    ) -> None:
        """라이브러리 오류에는 경로나 설정이 섞여 나온다."""
        class Broken(StubDetector):
            def detect(self, frames):
                raise RuntimeError("connect failed: user:secret@10.0.0.1")

        with pytest.raises(PipelineError) as e:
            await pipeline(storage, Broken()).run(scene, GROUP, stages)
        assert "secret" not in e.value.message
        assert "10.0.0.1" not in e.value.message

    async def test_misaligned_detector_output_is_rejected(
        self, storage, scene, stages
    ) -> None:
        """프레임과 1:1로 대응하지 않으면 점유율이 엉뚱한 수로 계산된다.

        증상이 뒤 단계에서 나타나 원인을 탐지기까지 되짚기 어렵다.
        """
        class Short(StubDetector):
            def detect(self, frames):
                return super().detect(frames)[:3]

        with pytest.raises(PipelineError) as e:
            await pipeline(storage, Short()).run(scene, GROUP, stages)
        assert e.value.stage is AnalysisStage.OBJECT_DETECTION

    async def test_marking_failure_keeps_the_stage(
        self, storage, scene, stages
    ) -> None:
        class BrokenStorage:
            image_dir = storage.image_dir

            def save_image(self, image):
                raise OSError("disk full")

        broken = VisionPipeline(StubDetector(), BrokenStorage())

        with pytest.raises(PipelineError) as e:
            await broken.run(scene, GROUP, stages)
        assert e.value.stage is AnalysisStage.RISK_MARKING

    async def test_stage_is_recorded_before_failure(
        self, storage, scene, stages
    ) -> None:
        """실패해도 어디까지 갔는지 남아야 재촬영과 재시도를 구분할 수 있다."""
        class Broken(StubDetector):
            def detect(self, frames):
                raise RuntimeError("boom")

        with pytest.raises(PipelineError):
            await pipeline(storage, Broken()).run(scene, GROUP, stages)

        assert stages.seen[-1] is AnalysisStage.OBJECT_DETECTION


class TestEmptyScene:
    async def test_video_with_no_objects(self, storage, scene, stages) -> None:
        """아무것도 탐지되지 않아도 분석은 완료된다. 빈 방일 수 있다."""
        result = await pipeline(storage, StubDetector([])).run(scene, GROUP, stages)

        assert result.detected_objects == ()
        assert result.occupancy_ratio == 0.0
        assert result.thumbnail_path.startswith("/images/")
        assert len(result.analysis_frames) == 1
