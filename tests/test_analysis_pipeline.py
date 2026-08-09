"""1~12단계 조립 검증.

Vision과 환경 분석을 엮는 지점만 본다. 각 단계의 내부 동작은
`test_vision_pipeline.py` 와 `test_environment_analysis.py` 가 이미 검증한다.

여기서 확인하는 것은 **경계에서 값이 제대로 넘어가는가**다. 점수가 12단계 입력으로
들어가는지, 단계 콜백이 빠짐없이 불리는지, 실패했을 때 만든 이미지를 정리하는지.
경계는 각자의 단위 테스트가 보지 못하는 자리라 버그가 남기 쉽다.
"""

import json

import pytest
from PIL import Image

from app.ai.analysis_pipeline import AnalysisPipeline, encode_frames
from app.ai.environment_analysis import EnvironmentAnalyzer
from app.ai.llm.fake import FakeLLM
from app.ai.pipeline import DetectedObject, PipelineError
from app.ai.score_generator import generate
from app.ai.vision.types import Frame, VisionResult
from app.schemas.enums import AnalysisStage, AnimalGroup, RiskLevel, SpaceType

GROUP = AnimalGroup.SMALL_DOG
SPACE = SpaceType.LIVING_ROOM


def frame(number: int, size=(64, 48)) -> Frame:
    return Frame(number=number, timestamp=number / 3, image=Image.new("RGB", size, (120, 60, 30)))


def detected(name, risk=RiskLevel.SAFE, frame_number=1, marked=None) -> DetectedObject:
    return DetectedObject(
        name=name,
        risk=risk,
        confidence=0.9,
        detection_frame_count=5,
        frame_number=frame_number,
        x=0.1,
        y=0.1,
        width=0.2,
        height=0.2,
        marked_image_path=marked,
    )


def vision_result(**overrides) -> VisionResult:
    values = {
        "capture_duration": 8.4,
        "frame_count": 25,
        "occupancy_ratio": 0.35,
        "thumbnail_path": "/images/thumb.jpg",
        "thumbnail_frame": 7,
        "detected_objects": (
            detected("전선", RiskLevel.HIGH, 3, "/images/mark-1.jpg"),
            detected("소파", RiskLevel.SAFE, 5),
        ),
        "analysis_frames": (frame(7), frame(3)),
    }
    values.update(overrides)
    return VisionResult(**values)


class FakeVision:
    """1~10단계 자리. 고정된 산출물을 돌려주고 단계 콜백을 흉내 낸다."""

    def __init__(self, result: VisionResult | None = None, error: Exception | None = None):
        self._result = result if result is not None else vision_result()
        self._error = error
        self.calls: list[tuple] = []

    async def run(self, video_path, group, on_stage):
        self.calls.append((video_path, group))
        for stage in (
            AnalysisStage.FRAME_EXTRACTION,
            AnalysisStage.OBJECT_DETECTION,
            AnalysisStage.OBJECT_TRACKING,
            AnalysisStage.FRAME_SELECTION,
            AnalysisStage.RISK_MARKING,
        ):
            await on_stage(stage)
        if self._error is not None:
            raise self._error
        return self._result


class FakeStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def save_image(self, image) -> str:
        return "/images/x.jpg"

    def delete(self, *relative_paths) -> int:
        kept = [p for p in relative_paths if p]
        self.deleted.extend(kept)
        return len(kept)


def response(**overrides) -> str:
    body = {
        "riskFactors": [
            {"text": "전선이 바닥에 노출되어 있습니다.", "source": "DETECTED"},
        ],
        "analysis": ["활동 공간은 충분합니다.", "전선 정리가 필요합니다."],
        "recommendations": [
            {
                "type": "SAFETY",
                "text": "전선을 벽면으로 정리해주세요.",
                "priority": 1,
                "source": "DETECTED",
            },
        ],
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


def make(llm=None, vision=None, store=None) -> tuple:
    llm = llm if llm is not None else FakeLLM(response())
    vision = vision if vision is not None else FakeVision()
    pipeline = AnalysisPipeline(vision, EnvironmentAnalyzer(llm), store)
    return pipeline, llm, vision


async def run(pipeline, stages=None):
    async def on_stage(stage):
        if stages is not None:
            stages.append(stage)

    return await pipeline.run("clip.mp4", GROUP, SPACE, on_stage)


# =============================================================================
# 산출물 조립
# =============================================================================


class TestResult:
    async def test_carries_vision_values(self) -> None:
        result = await run(make()[0])

        assert result.capture_duration == 8.4
        assert result.frame_count == 25
        assert result.occupancy_ratio == 0.35
        assert result.thumbnail_path == "/images/thumb.jpg"

    async def test_carries_detected_objects(self) -> None:
        result = await run(make()[0])
        assert [o.name for o in result.detected_objects] == ["전선", "소파"]

    async def test_carries_generated_narrative(self) -> None:
        result = await run(make()[0])

        assert len(result.risk_factors) == 1
        assert len(result.analysis) == 2
        assert len(result.recommendations) == 1

    async def test_result_has_no_score(self) -> None:
        """점수는 서비스 계층이 저장 시점에 산출한다."""
        result = await run(make()[0])
        assert not hasattr(result, "total_score")

    async def test_object_names_feed_scoring(self) -> None:
        """서비스가 이 집합으로 점수를 다시 계산한다."""
        result = await run(make()[0])
        assert result.object_names == {"전선", "소파"}


# =============================================================================
# 단계 진행
# =============================================================================


class TestStages:
    async def test_all_stages_in_order(self) -> None:
        stages: list[AnalysisStage] = []
        await run(make()[0], stages)

        assert stages == list(AnalysisStage)

    async def test_score_stage_precedes_environment(self) -> None:
        """점수를 모델 입력으로 넘기려면 먼저 산출해야 한다."""
        stages: list[AnalysisStage] = []
        await run(make()[0], stages)

        assert stages.index(AnalysisStage.SCORE_CALCULATION) < stages.index(
            AnalysisStage.ENVIRONMENT_ANALYSIS
        )


# =============================================================================
# 11단계 → 12단계 전달
# =============================================================================


class TestScoreHandoff:
    async def test_score_reaches_the_model(self) -> None:
        pipeline, llm, _ = make()
        await run(pipeline)

        payload = json.loads(llm.prompts[0].user.split("\n\n##")[0])
        expected = generate(GROUP, SPACE, {"전선", "소파"}, 0.35)
        assert payload["petFitScore"] == expected.as_dict()

    async def test_space_reaches_the_model(self) -> None:
        """공간에 따라 평가 항목이 달라진다. 서술에도 반영되어야 한다."""
        pipeline, llm, _ = make()

        async def on_stage(stage):
            return None

        await pipeline.run("clip.mp4", GROUP, SpaceType.BALCONY, on_stage)

        payload = json.loads(llm.prompts[0].user.split("\n\n##")[0])
        assert payload["spaceType"] == "balcony"

    async def test_thumbnail_frame_reaches_the_model(self) -> None:
        pipeline, llm, _ = make()
        await run(pipeline)

        payload = json.loads(llm.prompts[0].user.split("\n\n##")[0])
        assert payload["representativeFrame"]["frameNumber"] == 7

    async def test_score_matches_what_service_will_compute(self) -> None:
        """파이프라인과 서비스가 다른 점수를 내면 서술과 점수가 어긋난다."""
        pipeline, llm, _ = make()
        result = await run(pipeline)

        sent = json.loads(llm.prompts[0].user.split("\n\n##")[0])["petFitScore"]
        recomputed = generate(GROUP, SPACE, result.object_names, result.occupancy_ratio)
        assert sent == recomputed.as_dict()


# =============================================================================
# 이미지 전달
# =============================================================================


class TestImages:
    async def test_analysis_frames_are_sent(self) -> None:
        pipeline, llm, _ = make()
        await run(pipeline)

        assert llm.image_counts[0] == 2

    async def test_no_frames_sends_nothing(self) -> None:
        pipeline, llm, _ = make(vision=FakeVision(vision_result(analysis_frames=())))
        await run(pipeline)

        assert llm.image_counts[0] == 0

    async def test_encoded_as_jpeg(self) -> None:
        encoded = encode_frames([frame(1)])
        assert encoded[1].startswith(b"\xff\xd8")

    async def test_encoding_keeps_frame_numbers(self) -> None:
        encoded = encode_frames([frame(7), frame(3)])
        assert set(encoded) == {7, 3}

    async def test_non_rgb_is_converted(self) -> None:
        """JPEG는 알파 채널을 저장하지 못한다."""
        odd = Frame(number=1, timestamp=0.0, image=Image.new("RGBA", (8, 8)))
        assert 1 in encode_frames([odd])

    async def test_broken_frame_does_not_stop_the_rest(self) -> None:
        class Broken:
            mode = "RGB"

            def save(self, *args, **kwargs):
                raise OSError("손상된 이미지")

        frames = [Frame(number=1, timestamp=0.0, image=Broken()), frame(2)]
        encoded = encode_frames(frames)

        assert set(encoded) == {2}


# =============================================================================
# 실패 처리
# =============================================================================


class TestFailure:
    async def test_vision_failure_propagates(self) -> None:
        error = PipelineError("객체 탐지에 실패했습니다.", AnalysisStage.OBJECT_DETECTION)
        pipeline, _, _ = make(vision=FakeVision(error=error))

        with pytest.raises(PipelineError) as exc:
            await run(pipeline)

        assert exc.value.stage is AnalysisStage.OBJECT_DETECTION

    async def test_environment_failure_reports_its_stage(self) -> None:
        pipeline, _, _ = make(llm=FakeLLM("깨진 응답"))

        with pytest.raises(PipelineError) as exc:
            await run(pipeline)

        assert exc.value.stage is AnalysisStage.ENVIRONMENT_ANALYSIS

    async def test_environment_failure_discards_images(self) -> None:
        """DB에 기록되기 전이라 지우지 않으면 참조 없이 디스크에 남는다."""
        store = FakeStore()
        pipeline, _, _ = make(llm=FakeLLM("깨진 응답"), store=store)

        with pytest.raises(PipelineError):
            await run(pipeline)

        assert set(store.deleted) == {"/images/thumb.jpg", "/images/mark-1.jpg"}

    async def test_success_keeps_images(self) -> None:
        store = FakeStore()
        pipeline, _, _ = make(store=store)
        await run(pipeline)

        assert store.deleted == []

    async def test_vision_failure_needs_no_cleanup(self) -> None:
        """Vision이 실패하면 만든 이미지도 없다."""
        store = FakeStore()
        error = PipelineError("추출 실패", AnalysisStage.FRAME_EXTRACTION)
        pipeline, _, _ = make(vision=FakeVision(error=error), store=store)

        with pytest.raises(PipelineError):
            await run(pipeline)

        assert store.deleted == []

    async def test_cleanup_failure_does_not_mask_the_cause(self) -> None:
        """정리에 실패해도 원래 실패 사유가 살아 있어야 한다."""

        class BrokenStore(FakeStore):
            def delete(self, *paths):
                raise OSError("디스크 오류")

        pipeline, _, _ = make(llm=FakeLLM("깨진 응답"), store=BrokenStore())

        with pytest.raises(PipelineError) as exc:
            await run(pipeline)

        assert exc.value.stage is AnalysisStage.ENVIRONMENT_ANALYSIS

    async def test_no_store_is_allowed(self) -> None:
        pipeline, _, _ = make(llm=FakeLLM("깨진 응답"), store=None)

        with pytest.raises(PipelineError):
            await run(pipeline)


# =============================================================================
# 계약
# =============================================================================


class TestContract:
    async def test_satisfies_pipeline_protocol(self) -> None:
        """서비스 계층이 StubPipeline 자리에 그대로 끼울 수 있어야 한다."""
        from app.ai.pipeline import Pipeline

        pipeline, _, _ = make()
        assert isinstance(pipeline, Pipeline)

    async def test_video_path_reaches_vision(self) -> None:
        pipeline, _, vision = make()
        await run(pipeline)

        assert vision.calls[0][0] == "clip.mp4"
        assert vision.calls[0][1] is GROUP
