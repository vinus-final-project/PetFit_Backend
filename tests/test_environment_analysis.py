"""환경 분석 검증 (파이프라인 12단계).

조립 → 호출 → 검증 → 재생성 흐름이 명세대로 도는지 확인한다.

**실제 모델을 쓰지 않는다.** `FakeLLM` 에 응답을 순서대로 지정하면 "두 번 거절 후
통과", "끝까지 실패" 같은 경로를 결정적으로 재현할 수 있다. 실제 모델로는 이런
순서를 만들 수 없어 재생성 로직이 검증되지 않은 채 남는다.
"""

import json

import pytest

from app.ai.environment_analysis import (
    FAILURE_MESSAGE,
    EnvironmentAnalyzer,
)
from app.ai.llm.base import LLMError
from app.ai.llm.fake import FakeLLM, always_fails
from app.ai.pipeline import DetectedObject, PipelineError
from app.ai.prompts import AnalysisContext, NO_IMAGE_RULE
from app.ai.score_generator import generate
from app.ai.validation import Rejection, Repair
from app.schemas.enums import (
    AnalysisStage,
    AnimalGroup,
    RiskLevel,
    RiskSource,
    SpaceType,
)


def obj(name, risk=RiskLevel.SAFE, confidence=0.9, frame=1) -> DetectedObject:
    return DetectedObject(
        name=name,
        risk=risk,
        confidence=confidence,
        detection_frame_count=5,
        frame_number=frame,
        x=0.1,
        y=0.1,
        width=0.2,
        height=0.2,
    )


OBJECTS = [
    obj("전선", RiskLevel.HIGH, 0.94, 3),
    obj("창문", RiskLevel.LOW, 0.91, 18),
    obj("소파", RiskLevel.SAFE, 0.98, 5),
]


def context(objects=OBJECTS, group=AnimalGroup.SMALL_DOG) -> AnalysisContext:
    space = SpaceType.LIVING_ROOM
    names = {o.name for o in objects}
    return AnalysisContext(
        group=group,
        space=space,
        objects=objects,
        occupancy_ratio=0.35,
        score=generate(group, space, names, 0.35),
        thumbnail_frame=7,
    )


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


def payload_of(prompt) -> dict:
    """user 메시지에서 입력 JSON만 꺼낸다.

    뒤에 "사진 없음"·"재생성 요청" 같은 규칙 문단이 붙을 수 있어 통째로는
    파싱되지 않는다.
    """
    text = prompt.user
    end = text.find("\n\n##")
    return json.loads(text if end == -1 else text[:end])


def loader(*frames_with_data: int):
    """지정한 프레임만 읽히는 이미지 로더."""
    allowed = set(frames_with_data)

    def load(number: int) -> bytes | None:
        return f"frame-{number}".encode() if number in allowed else None

    return load


# =============================================================================
# 정상 경로
# =============================================================================


class TestSuccess:
    async def test_returns_output(self) -> None:
        report = await EnvironmentAnalyzer(FakeLLM(response())).analyze(context())

        assert len(report.output.risk_factors) == 1
        assert len(report.output.analysis) == 2
        assert len(report.output.recommendations) == 1

    async def test_single_call_when_valid(self) -> None:
        llm = FakeLLM(response())
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert llm.call_count == 1
        assert report.regenerations == 0
        assert report.attempts[0].ok

    async def test_never_produces_score(self) -> None:
        """점수는 규칙 기반으로 산출된다. 12단계 산출물에 점수가 있으면 안 된다."""
        report = await EnvironmentAnalyzer(FakeLLM(response())).analyze(context())
        assert not hasattr(report.output, "total")

    async def test_prompt_carries_context(self) -> None:
        llm = FakeLLM(response())
        ctx = context()
        await EnvironmentAnalyzer(llm).analyze(ctx)

        payload = payload_of(llm.prompts[0])
        assert payload["petFitScore"] == ctx.score.as_dict()
        assert payload["animalGroup"] == "small_dog"

    async def test_repairs_are_recorded(self) -> None:
        """성능평가의 형식 준수율 측정에 쓴다."""
        llm = FakeLLM(f"```json\n{response()}\n```")
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert report.attempts[0].repairs == (Repair.STRIPPED_FENCE,)
        assert llm.call_count == 1


# =============================================================================
# 재생성
# =============================================================================


class TestRegeneration:
    async def test_retries_after_rejection(self) -> None:
        llm = FakeLLM("JSON이 아닙니다", response())
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert llm.call_count == 2
        assert report.regenerations == 1
        assert report.attempts[0].rejection is Rejection.NOT_JSON
        assert report.attempts[1].ok

    async def test_retry_prompt_carries_reason(self) -> None:
        """사유를 알리지 않고 다시 부르면 같은 실패가 반복된다."""
        llm = FakeLLM("JSON이 아닙니다", response())
        await EnvironmentAnalyzer(llm).analyze(context())

        assert "재생성 요청" not in llm.prompts[0].user
        assert Rejection.NOT_JSON.instruction in llm.prompts[1].user

    async def test_reason_matches_actual_rejection(self) -> None:
        llm = FakeLLM(response(analysis=["한 건뿐입니다."]), response())
        await EnvironmentAnalyzer(llm).analyze(context())

        assert Rejection.TOO_FEW_ANALYSIS.instruction in llm.prompts[1].user

    async def test_succeeds_on_last_allowed_attempt(self) -> None:
        llm = FakeLLM("깨짐", "깨짐", "깨짐", response())
        report = await EnvironmentAnalyzer(llm, max_regenerations=3).analyze(context())

        assert llm.call_count == 4
        assert report.regenerations == 3

    async def test_exhausted_raises_pipeline_error(self) -> None:
        llm = FakeLLM("계속 깨짐")
        with pytest.raises(PipelineError) as exc:
            await EnvironmentAnalyzer(llm, max_regenerations=2).analyze(context())

        assert llm.call_count == 3
        assert exc.value.message == FAILURE_MESSAGE

    async def test_failure_reports_stage(self) -> None:
        """실패 단계가 남아야 클라이언트가 재시도를 안내할 수 있다."""
        with pytest.raises(PipelineError) as exc:
            await EnvironmentAnalyzer(FakeLLM("깨짐"), max_regenerations=0).analyze(
                context()
            )

        assert exc.value.stage is AnalysisStage.ENVIRONMENT_ANALYSIS

    async def test_internal_message_is_not_leaked(self) -> None:
        """모델의 오류 문구를 사용자에게 그대로 보여주지 않는다."""
        llm = always_fails("CUDA out of memory at 0x7f")
        with pytest.raises(PipelineError) as exc:
            await EnvironmentAnalyzer(llm, max_regenerations=0).analyze(context())

        assert "CUDA" not in exc.value.message


# =============================================================================
# 호출 실패
# =============================================================================


class TestCallFailure:
    async def test_llm_error_is_retried(self) -> None:
        llm = FakeLLM(LLMError("일시적 오류"), response())
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert llm.call_count == 2
        assert report.attempts[0].error == "LLMError"
        assert report.attempts[1].ok

    async def test_unexpected_exception_is_retried(self) -> None:
        """모델은 다양한 예외를 낸다. 어떤 것이든 한 번의 실패로 센다."""
        llm = FakeLLM(RuntimeError("메모리 부족"), response())
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert report.attempts[0].error == "RuntimeError"
        assert llm.call_count == 2

    async def test_call_failure_adds_no_retry_instruction(self) -> None:
        """호출이 실패한 것은 응답의 문제가 아니다. 잘못을 지적할 대상이 없다."""
        llm = FakeLLM(LLMError("일시적 오류"), response())
        await EnvironmentAnalyzer(llm).analyze(context())

        assert "재생성 요청" not in llm.prompts[1].user

    async def test_persistent_failure_raises(self) -> None:
        with pytest.raises(PipelineError):
            await EnvironmentAnalyzer(always_fails(), max_regenerations=1).analyze(
                context()
            )


# =============================================================================
# 이미지
# =============================================================================


class TestImages:
    async def test_sends_selected_frames(self) -> None:
        llm = FakeLLM(response())
        report = await EnvironmentAnalyzer(llm).analyze(
            context(), loader(7, 3, 18)
        )

        assert report.images_sent == 3
        assert llm.image_counts[0] == 3

    async def test_no_loader_sends_nothing(self) -> None:
        llm = FakeLLM(response())
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert report.images_sent == 0
        assert llm.image_counts[0] == 0

    async def test_unreadable_frames_are_skipped(self) -> None:
        """이미지 하나 때문에 분석 전체를 실패시키지 않는다."""
        llm = FakeLLM(response())
        report = await EnvironmentAnalyzer(llm).analyze(context(), loader(7))

        assert report.images_sent == 1

    async def test_loader_exception_is_tolerated(self) -> None:
        def broken(number: int) -> bytes:
            raise OSError("파일이 손상되었다")

        report = await EnvironmentAnalyzer(FakeLLM(response())).analyze(
            context(), broken
        )
        assert report.images_sent == 0

    async def test_no_image_forbids_observed(self) -> None:
        """관찰 근거가 없으면 OBSERVED는 전부 환각이다."""
        llm = FakeLLM(response())
        await EnvironmentAnalyzer(llm).analyze(context())

        assert NO_IMAGE_RULE in llm.prompts[0].user

    async def test_with_image_allows_observed(self) -> None:
        llm = FakeLLM(response())
        await EnvironmentAnalyzer(llm).analyze(context(), loader(7))

        assert NO_IMAGE_RULE not in llm.prompts[0].user

    async def test_images_loaded_once_across_retries(self) -> None:
        """재생성마다 다시 읽으면 디스크를 반복해서 친다."""
        reads: list[int] = []

        def counting(number: int) -> bytes:
            reads.append(number)
            return b"data"

        llm = FakeLLM("깨짐", response())
        await EnvironmentAnalyzer(llm).analyze(context(), counting)

        assert llm.call_count == 2
        assert len(reads) == len(set(reads))


# =============================================================================
# 검증 연동
# =============================================================================


class TestValidationIntegration:
    async def test_unfounded_detected_is_demoted(self) -> None:
        llm = FakeLLM(
            response(
                riskFactors=[
                    {"text": "약병이 선반에 있습니다.", "source": "DETECTED"}
                ]
            )
        )
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert report.output.risk_factors[0].source is RiskSource.OBSERVED
        assert Repair.DEMOTED_TO_OBSERVED in report.attempts[0].repairs

    async def test_priority_is_renumbered(self) -> None:
        """UNIQUE (analysis_id, priority) 제약에 걸리면 저장이 실패한다."""
        llm = FakeLLM(
            response(
                recommendations=[
                    {"type": "SAFETY", "text": "가", "priority": 1, "source": "DETECTED"},
                    {"type": "REST", "text": "나", "priority": 1, "source": "DETECTED"},
                ]
            )
        )
        report = await EnvironmentAnalyzer(llm).analyze(context())

        assert [r.priority for r in report.output.recommendations] == [1, 2]

    async def test_output_is_storable(self) -> None:
        """산출물이 PipelineResult 의 서술 3종과 맞아야 한다."""
        report = await EnvironmentAnalyzer(FakeLLM(response())).analyze(context())

        assert all(isinstance(t, str) for t in report.output.analysis)
        assert all(r.priority >= 1 for r in report.output.recommendations)
        assert all(
            f.source in (RiskSource.DETECTED, RiskSource.OBSERVED)
            for f in report.output.risk_factors
        )


# =============================================================================
# 그룹·공간 반영
# =============================================================================


class TestContext:
    @pytest.mark.parametrize("group", list(AnimalGroup.analyzable()))
    async def test_group_instruction_is_sent(self, group) -> None:
        llm = FakeLLM(response())
        await EnvironmentAnalyzer(llm).analyze(context(group=group))

        payload = payload_of(llm.prompts[0])
        assert payload["animalGroup"] == group.value

    async def test_few_shots_can_be_disabled(self) -> None:
        llm = FakeLLM(response())
        await EnvironmentAnalyzer(llm, include_few_shots=False).analyze(context())

        assert llm.prompts[0].few_shots == ()

    async def test_few_shots_included_by_default(self) -> None:
        llm = FakeLLM(response())
        await EnvironmentAnalyzer(llm).analyze(context())

        assert len(llm.prompts[0].few_shots) == 3
