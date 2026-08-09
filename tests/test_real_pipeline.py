"""전체 파이프라인 조립 검증.

영상 파일에서 시작해 ``PipelineResult`` 까지 간다. 부품별 검증은 각 담당의
테스트에 있고, 여기서 보는 것은 **세 산출물이 만나는 지점**이다.

경계에서만 드러나는 문제를 본다.

    점수 이중 산출  : 서술이 참조한 점수와 저장될 점수가 같은가
    프레임 선정 규칙 : Vision 과 12단계가 각자 구현한 규칙이 일치하는가
    실패 시 정리     : 뒤 단계가 실패하면 만들어 둔 이미지를 지우는가
"""

import json

import pytest

from app.ai.environment_analysis import EnvironmentAnalyzer
from app.ai.llm.base import LLMError
from app.ai.llm.fake import FakeLLM
from app.ai.pipeline import Pipeline, PipelineError, PipelineResult
from app.ai.prompts import AnalysisContext, select_image_frames
from app.ai.real_pipeline import RealPipeline
from app.ai.score_generator import generate
from app.ai.vision.detector import StubDetector
from app.ai.vision.pipeline import VisionPipeline
from app.schemas.enums import (
    AnalysisStage,
    AnimalGroup,
    RecommendationType,
    RiskLevel,
    RiskSource,
    SpaceType,
)

GROUP = AnimalGroup.SMALL_DOG
SPACE = SpaceType.LIVING_ROOM
SCENE_SECONDS = 8.4


def response(**overrides) -> str:
    """검증을 통과하는 응답."""
    body = {
        "riskFactors": [
            {"text": "전선이 바닥에 노출되어 있습니다.", "source": "DETECTED"},
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
        ],
    }
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


@pytest.fixture
def scene(make_video):
    return make_video(seconds=SCENE_SECONDS, fps=30, width=640, height=360)


@pytest.fixture
def stages():
    seen: list[AnalysisStage] = []

    async def record(stage: AnalysisStage) -> None:
        seen.append(stage)

    record.seen = seen
    return record


def build(storage, llm=None, detector=None) -> RealPipeline:
    return RealPipeline(
        vision=VisionPipeline(detector or StubDetector(), storage),
        analyzer=EnvironmentAnalyzer(llm or FakeLLM(response())),
        storage=storage,
    )


class TestContract:
    async def test_satisfies_the_pipeline_protocol(self, storage) -> None:
        assert isinstance(build(storage), Pipeline)

    async def test_returns_pipeline_result(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)
        assert isinstance(result, PipelineResult)

    async def test_reports_every_stage_in_order(self, storage, scene, stages) -> None:
        await build(storage).run(scene, GROUP, SPACE, stages)
        assert stages.seen == list(AnalysisStage)

    async def test_progress_never_goes_backwards(
        self, storage, scene, stages
    ) -> None:
        await build(storage).run(scene, GROUP, SPACE, stages)
        progress = [s.progress for s in stages.seen]
        assert progress == sorted(progress)

    async def test_can_replace_the_stub(self, storage, scene, stages) -> None:
        """StubPipeline 과 같은 자리에 들어가므로 필드 구성이 같아야 한다."""
        from app.ai.stub import StubPipeline

        real = await build(storage).run(scene, GROUP, SPACE, stages)
        stub = await StubPipeline(speed=0).run(scene, GROUP, SPACE, stages)

        assert set(vars(real)) == set(vars(stub))


class TestVisionOutput:
    async def test_carries_detection_results(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)
        assert result.object_names == {"전선", "창문", "카펫", "소파", "급수기"}

    async def test_carries_measurements(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)

        assert result.capture_duration == pytest.approx(SCENE_SECONDS, abs=0.2)
        assert result.frame_count == 25
        assert 0.0 < result.occupancy_ratio < 1.0
        assert result.thumbnail_path.startswith("/images/")

    async def test_marked_images_exist(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)

        assert result.marked_image_paths
        for path in result.marked_image_paths:
            assert (storage.image_dir / path.split("/")[-1]).is_file()


class TestScoreConsistency:
    """서술이 참조한 점수와 화면에 표시될 점수는 같아야 한다."""

    async def test_reproduces_the_documented_score(
        self, storage, scene, stages
    ) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)
        score = generate(GROUP, SPACE, result.object_names, result.occupancy_ratio)
        assert score.total == 56

    async def test_llm_receives_the_stored_score(
        self, storage, scene, stages
    ) -> None:
        """서비스 계층이 저장할 점수를 12단계가 그대로 받아야 한다.

        어긋나면 "안전합니다" 라는 서술 옆에 낮은 점수가 표시된다.
        """
        llm = FakeLLM(response())
        result = await build(storage, llm).run(scene, GROUP, SPACE, stages)

        stored = generate(GROUP, SPACE, result.object_names, result.occupancy_ratio)
        sent = json.loads(_payload_of(llm.prompts[0]))["petFitScore"]

        assert sent["total"] == stored.total

    async def test_score_changes_with_the_space(self, storage, scene, stages) -> None:
        """공간이 다르면 평가 항목이 달라 점수가 달라진다."""
        result = await build(storage).run(scene, GROUP, SPACE, stages)

        living = generate(GROUP, SpaceType.LIVING_ROOM, result.object_names,
                          result.occupancy_ratio)
        balcony = generate(GROUP, SpaceType.BALCONY, result.object_names,
                           result.occupancy_ratio)
        assert living.total != balcony.total


class TestImages:
    async def test_sends_images_to_the_model(self, storage, scene, stages) -> None:
        """이미지가 없으면 탐지 대상 밖의 위험 요소를 찾을 수 없다."""
        llm = FakeLLM(response())
        await build(storage, llm).run(scene, GROUP, SPACE, stages)

        assert llm.image_counts[0] > 0

    async def test_respects_the_image_limit(self, storage, scene, stages) -> None:
        from app.core.constants import LLM_MAX_IMAGES

        llm = FakeLLM(response())
        await build(storage, llm).run(scene, GROUP, SPACE, stages)

        assert llm.image_counts[0] <= LLM_MAX_IMAGES

    async def test_sends_unmarked_originals(self, storage, scene, stages) -> None:
        """박스를 그린 이미지를 보내면 모델이 표시된 객체만 다시 서술한다."""
        from io import BytesIO

        from PIL import Image

        sent: list[bytes] = []

        class Capturing(FakeLLM):
            async def complete(self, prompt, images):
                sent.extend(images)
                return await super().complete(prompt, images)

        await build(storage, Capturing(response())).run(scene, GROUP, SPACE, stages)

        for data in sent:
            with Image.open(BytesIO(data)) as image:
                colors = {c for _, c in image.convert("RGB").getcolors(1 << 20)}
            assert (255, 0, 0) not in colors

    async def test_frame_selection_rules_agree(self, storage, scene, stages) -> None:
        """Vision 이 남긴 프레임과 12단계가 요청하는 프레임이 같아야 한다.

        규칙은 ``pipeline.select_analysis_frames()`` 하나로 통합되어 있으므로
        지금은 갈라질 수 없다. 어느 한쪽이 다시 직접 구현하면 이 테스트가 잡는다.
        요청한 프레임이 없으면 이미지가 조용히 빠져 관찰 근거가 사라진다.
        """
        vision = VisionPipeline(StubDetector(), storage)
        result = await vision.run(scene, GROUP, stages)

        score = generate(GROUP, SPACE, result.object_names, result.occupancy_ratio)
        context = AnalysisContext(
            group=GROUP,
            space=SPACE,
            objects=result.detected_objects,
            occupancy_ratio=result.occupancy_ratio,
            score=score,
            thumbnail_frame=result.thumbnail_frame,
        )

        requested = set(select_image_frames(context))
        available = {f.number for f in result.analysis_frames}
        assert requested == available

    async def test_thumbnail_frame_matches_the_first_analysis_frame(
        self, storage, scene, stages
    ) -> None:
        result = await VisionPipeline(StubDetector(), storage).run(
            scene, GROUP, stages
        )
        assert result.thumbnail_frame == result.analysis_frames[0].number


class TestAnalyzerOutput:
    async def test_carries_risk_factors(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)

        assert len(result.risk_factors) == 1
        assert result.risk_factors[0].source is RiskSource.DETECTED

    async def test_carries_analysis(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)
        assert len(result.analysis) == 2

    async def test_carries_recommendations(self, storage, scene, stages) -> None:
        result = await build(storage).run(scene, GROUP, SPACE, stages)

        assert len(result.recommendations) == 1
        assert result.recommendations[0].type is RecommendationType.SAFETY
        assert result.recommendations[0].priority == 1

    async def test_regeneration_still_produces_a_result(
        self, storage, scene, stages
    ) -> None:
        """첫 응답이 거절돼도 재생성으로 통과하면 분석은 성공한다."""
        llm = FakeLLM("설명 문장만 있고 JSON이 없습니다", response())
        result = await build(storage, llm).run(scene, GROUP, SPACE, stages)

        assert llm.call_count == 2
        assert len(result.analysis) == 2


class TestFailure:
    async def test_vision_failure_propagates(self, storage, tmp_path, stages) -> None:
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")

        with pytest.raises(PipelineError) as e:
            await build(storage).run(broken, GROUP, SPACE, stages)
        assert e.value.stage is AnalysisStage.FRAME_EXTRACTION

    async def test_analysis_failure_keeps_the_stage(
        self, storage, scene, stages
    ) -> None:
        llm = FakeLLM(LLMError("모델 응답 없음"))

        with pytest.raises(PipelineError) as e:
            await build(storage, llm).run(scene, GROUP, SPACE, stages)
        assert e.value.stage is AnalysisStage.ENVIRONMENT_ANALYSIS

    async def test_analysis_failure_removes_created_images(
        self, storage, scene, stages
    ) -> None:
        """Vision 이 만든 이미지가 DB 참조 없이 남으면 추적할 수 없다.

        재시도까지 겹치면 분석 1건에 최대 12장이 쌓인다.
        """
        llm = FakeLLM(LLMError("모델 응답 없음"))

        with pytest.raises(PipelineError):
            await build(storage, llm).run(scene, GROUP, SPACE, stages)

        assert list(storage.image_dir.glob("*.jpg")) == []

    async def test_successful_run_keeps_images(self, storage, scene, stages) -> None:
        """성공하면 지우지 않는다. DB가 참조한다."""
        await build(storage).run(scene, GROUP, SPACE, stages)
        assert list(storage.image_dir.glob("*.jpg"))

    async def test_error_message_hides_internals(
        self, storage, scene, stages
    ) -> None:
        llm = FakeLLM(LLMError("connect failed: user:secret@10.0.0.1"))

        with pytest.raises(PipelineError) as e:
            await build(storage, llm).run(scene, GROUP, SPACE, stages)
        assert "secret" not in e.value.message


class TestGroups:
    async def test_risk_depends_on_the_group(self, storage, scene, stages) -> None:
        dog = await build(storage).run(scene, AnimalGroup.SMALL_DOG, SPACE, stages)
        cat = await build(storage).run(scene, AnimalGroup.CAT, SPACE, stages)

        def risk_of(result, name):
            return next(o.risk for o in result.detected_objects if o.name == name)

        assert risk_of(dog, "창문") is RiskLevel.LOW
        assert risk_of(cat, "창문") is RiskLevel.HIGH

    async def test_group_reaches_the_prompt(self, storage, scene, stages) -> None:
        llm = FakeLLM(response())
        await build(storage, llm).run(scene, AnimalGroup.CAT, SPACE, stages)

        payload = json.loads(_payload_of(llm.prompts[0]))
        assert payload["animalGroup"] == AnimalGroup.CAT.value

    async def test_space_reaches_the_prompt(self, storage, scene, stages) -> None:
        llm = FakeLLM(response())
        await build(storage, llm).run(scene, GROUP, SpaceType.BEDROOM, stages)

        payload = json.loads(_payload_of(llm.prompts[0]))
        assert payload["spaceType"] == SpaceType.BEDROOM.value


def _payload_of(prompt) -> str:
    """user 메시지에서 입력 JSON 부분만 꺼낸다."""
    text = prompt.user
    end = text.rfind("}")
    return text[: end + 1]
