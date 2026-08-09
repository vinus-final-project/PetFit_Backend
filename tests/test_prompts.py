"""프롬프트 조립 검증.

문서(프롬프트 설계서 v2.1)의 내용이 프롬프트에 실제로 들어가는지, 같은 입력에
같은 프롬프트가 나오는지 확인한다.

**LLM을 호출하지 않는다.** 조립 결과는 문자열이므로 값으로 검증할 수 있다.

Few-shot 예시의 점수는 AI 분석 정의서의 계산 규칙으로 검산한다. 예시가 틀리면
모델에게 점수와 서술이 어긋난 사례를 학습시키게 된다.
"""

import json

import pytest

from app.ai.pipeline import DetectedObject
from app.ai.prompts import (
    FEW_SHOTS,
    GROUP_INSTRUCTIONS,
    SYSTEM_PROMPT,
    AnalysisContext,
    build,
    select_image_frames,
)
from app.ai.score_generator import generate
from app.ai.validation import (
    MAX_ANALYSIS,
    MAX_RECOMMENDATIONS,
    MAX_RISK_FACTORS,
    MIN_ANALYSIS,
    Rejection,
    validate,
)
from app.core.constants import LLM_MAX_IMAGES
from app.schemas.enums import AnimalGroup, RiskLevel, SpaceType


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


def context(
    group=AnimalGroup.SMALL_DOG,
    space=SpaceType.LIVING_ROOM,
    objects=None,
    occupancy=0.35,
    thumbnail=7,
) -> AnalysisContext:
    objects = (
        objects
        if objects is not None
        else [
            obj("전선", RiskLevel.HIGH, 0.94, 3),
            obj("창문", RiskLevel.LOW, 0.91, 18),
            obj("소파", RiskLevel.SAFE, 0.98, 5),
        ]
    )
    names = {o.name for o in objects}
    return AnalysisContext(
        group=group,
        space=space,
        objects=objects,
        occupancy_ratio=occupancy,
        score=generate(group, space, names, occupancy),
        thumbnail_frame=thumbnail,
    )


# =============================================================================
# System Prompt
# =============================================================================


class TestSystemPrompt:
    async def test_contains_absolute_rules(self) -> None:
        """점수를 만들지 않는다는 규칙은 프롬프트의 핵심이다."""
        assert "점수(petFitScore)를 생성하지 않습니다" in SYSTEM_PROMPT

    async def test_explains_both_sources(self) -> None:
        assert "DETECTED" in SYSTEM_PROMPT
        assert "OBSERVED" in SYSTEM_PROMPT

    async def test_states_space_exclusions(self) -> None:
        """공간별 미적용 항목을 부족하다고 서술하면 점수와 어긋난다."""
        for line in ("침실", "주방", "베란다"):
            assert line in SYSTEM_PROMPT

    async def test_forbids_code_fence(self) -> None:
        assert "코드펜스" in SYSTEM_PROMPT

    @pytest.mark.parametrize("group", list(GROUP_INSTRUCTIONS))
    async def test_group_instruction_is_appended(self, group) -> None:
        prompt = build(context(group=group))
        assert GROUP_INSTRUCTIONS[group] in prompt.system
        assert SYSTEM_PROMPT in prompt.system

    async def test_only_selected_group_instruction_is_included(self) -> None:
        """다른 그룹 지시문이 섞이면 중요도 판단이 흐려진다."""
        prompt = build(context(group=AnimalGroup.CAT))
        assert GROUP_INSTRUCTIONS[AnimalGroup.SMALL_DOG] not in prompt.system

    async def test_reserved_group_raises(self) -> None:
        """확장 그룹은 분석 기준이 정의되지 않았다."""
        ctx = context()
        broken = AnalysisContext(
            group=AnimalGroup.REPTILE,
            space=ctx.space,
            objects=ctx.objects,
            occupancy_ratio=ctx.occupancy_ratio,
            score=ctx.score,
            thumbnail_frame=ctx.thumbnail_frame,
        )
        with pytest.raises(KeyError):
            build(broken)

    async def test_count_limits_match_validator(self) -> None:
        """프롬프트와 검증이 다른 숫자를 들면 통과 못 할 응답을 계속 요구한다."""
        system = build(context()).system
        assert f"{MAX_RISK_FACTORS}건 이하" in system
        assert f"{MIN_ANALYSIS}건 이상 {MAX_ANALYSIS}건 이하" in system
        assert f"{MAX_RECOMMENDATIONS}건 이하" in system


# =============================================================================
# 입력 페이로드
# =============================================================================


class TestUserPayload:
    async def test_shape_matches_few_shot(self) -> None:
        """예시와 실제 입력의 모양이 다르면 예시가 형식을 알려주지 못한다."""
        actual = json.loads(build(context()).user)
        example = json.loads(FEW_SHOTS[0].user)
        assert actual.keys() == example.keys()

    async def test_carries_score(self) -> None:
        ctx = context()
        payload = json.loads(build(ctx).user)
        assert payload["petFitScore"] == ctx.score.as_dict()

    async def test_carries_occupancy_ratio(self) -> None:
        """활동성 점수의 근거다. 없으면 모델이 공간 부족을 서술할 수 없다."""
        payload = json.loads(build(context(occupancy=0.62)).user)
        assert payload["occupancyRatio"] == 0.62

    async def test_occupancy_is_normalized(self) -> None:
        """저장 자릿수와 맞춰야 같은 입력에 같은 프롬프트가 나온다."""
        payload = json.loads(build(context(occupancy=0.3500001)).user)
        assert payload["occupancyRatio"] == 0.35

    async def test_objects_carry_name_and_risk_only(self) -> None:
        payload = json.loads(build(context()).user)
        for item in payload["detectedObjects"]:
            assert set(item) == {"name", "risk"}

    async def test_objects_are_risk_ordered(self) -> None:
        """위험한 것을 앞에 두어 모델이 먼저 읽게 한다."""
        payload = json.loads(build(context()).user)
        assert payload["detectedObjects"][0]["name"] == "전선"

    async def test_duplicate_instances_are_kept(self) -> None:
        """전선이 두 곳이면 두 건이다. 묶어 서술하는 것은 모델의 몫이다."""
        objects = [
            obj("전선", RiskLevel.HIGH, 0.94, 3),
            obj("전선", RiskLevel.HIGH, 0.88, 9),
        ]
        payload = json.loads(build(context(objects=objects)).user)
        assert len(payload["detectedObjects"]) == 2

    async def test_space_is_carried(self) -> None:
        payload = json.loads(build(context(space=SpaceType.BALCONY)).user)
        assert payload["spaceType"] == "balcony"

    async def test_same_input_same_prompt(self) -> None:
        """입력 순서가 결과를 바꾸면 차이의 원인이 모델인지 입력인지 알 수 없다."""
        a = [obj("전선", RiskLevel.HIGH, 0.9, 3), obj("소파", RiskLevel.SAFE, 0.8, 5)]
        assert build(context(objects=a)).user == build(context(objects=a[::-1])).user


# =============================================================================
# 이미지 선정
# =============================================================================


class TestImageSelection:
    async def test_thumbnail_comes_first(self) -> None:
        frames = select_image_frames(context(thumbnail=7))
        assert frames[0] == 7

    async def test_risk_frames_follow_in_risk_order(self) -> None:
        objects = [
            obj("창문", RiskLevel.LOW, 0.91, 18),
            obj("전선", RiskLevel.HIGH, 0.94, 3),
            obj("계단", RiskLevel.MEDIUM, 0.80, 25),
        ]
        frames = select_image_frames(context(objects=objects, thumbnail=7))
        assert frames == (7, 3, 25, 18)

    async def test_safe_objects_are_excluded(self) -> None:
        """SAFE는 위험 요소가 아니다. 자리를 차지하면 위험 장면이 밀린다."""
        objects = [
            obj("소파", RiskLevel.SAFE, 0.98, 40),
            obj("전선", RiskLevel.HIGH, 0.94, 3),
        ]
        assert 40 not in select_image_frames(context(objects=objects, thumbnail=7))

    async def test_duplicate_frames_sent_once(self) -> None:
        objects = [
            obj("전선", RiskLevel.HIGH, 0.94, 3),
            obj("창문", RiskLevel.HIGH, 0.90, 3),
        ]
        frames = select_image_frames(context(objects=objects, thumbnail=3))
        assert frames == (3,)

    async def test_never_exceeds_limit(self) -> None:
        objects = [obj(f"위험{i}", RiskLevel.HIGH, 0.9, i) for i in range(10, 20)]
        frames = select_image_frames(context(objects=objects, thumbnail=7))
        assert len(frames) == LLM_MAX_IMAGES

    async def test_no_risky_objects_sends_thumbnail_only(self) -> None:
        objects = [obj("소파", RiskLevel.SAFE, 0.98, 5)]
        assert select_image_frames(context(objects=objects, thumbnail=7)) == (7,)

    async def test_prompt_exposes_frames(self) -> None:
        ctx = context()
        assert build(ctx).image_frames == select_image_frames(ctx)


# =============================================================================
# 재생성
# =============================================================================


class TestRetry:
    async def test_no_retry_adds_nothing(self) -> None:
        assert "재생성 요청" not in build(context()).user

    @pytest.mark.parametrize("rejection", list(Rejection))
    async def test_instruction_is_appended(self, rejection) -> None:
        user = build(context(), retry=rejection).user
        assert "재생성 요청" in user
        assert rejection.instruction in user

    async def test_input_is_preserved_on_retry(self) -> None:
        """재생성해도 분석 대상은 같다. 입력이 빠지면 다른 분석이 된다."""
        ctx = context()
        user = build(ctx, retry=Rejection.NOT_JSON).user
        assert json.dumps(ctx.score.as_dict(), ensure_ascii=False)[1:20] in user or (
            str(ctx.score.total) in user
        )


# =============================================================================
# Few-shot
# =============================================================================


class TestFewShots:
    async def test_three_examples(self) -> None:
        assert len(FEW_SHOTS) == 3

    async def test_can_be_omitted(self) -> None:
        assert build(context(), include_few_shots=False).few_shots == ()

    @pytest.mark.parametrize("shot", FEW_SHOTS)
    async def test_output_is_valid_json(self, shot) -> None:
        assert isinstance(json.loads(shot.assistant), dict)

    @pytest.mark.parametrize("shot", FEW_SHOTS)
    async def test_output_passes_our_own_validator(self, shot) -> None:
        """예시가 검증을 통과하지 못하면 모델에게 거절당할 형식을 가르치는 셈이다."""
        detected = [o["name"] for o in json.loads(shot.user)["detectedObjects"]]
        result = validate(shot.assistant, detected)
        assert result.ok, result.rejection
        assert result.repairs == ()

    @pytest.mark.parametrize("shot", FEW_SHOTS)
    async def test_score_matches_calculation_rule(self, shot) -> None:
        """예시 점수가 틀리면 점수와 서술이 어긋난 사례를 학습시킨다."""
        payload = json.loads(shot.user)
        expected = generate(
            AnimalGroup(payload["animalGroup"]),
            SpaceType(payload["spaceType"]),
            {o["name"] for o in payload["detectedObjects"]},
            payload["occupancyRatio"],
        )
        assert payload["petFitScore"] == expected.as_dict()

    @pytest.mark.parametrize("shot", FEW_SHOTS)
    async def test_input_shape_is_consistent(self, shot) -> None:
        payload = json.loads(shot.user)
        assert set(payload) == {
            "animalGroup",
            "spaceType",
            "representativeFrame",
            "detectedObjects",
            "occupancyRatio",
            "petFitScore",
        }

    async def test_covers_no_risk_case(self) -> None:
        """위험이 없으면 빈 배열이 정상이라는 것을 예시로 보여줘야 한다."""
        assert any(
            json.loads(s.assistant)["riskFactors"] == [] for s in FEW_SHOTS
        )

    async def test_covers_observed_case(self) -> None:
        """탐지 대상 밖의 위험을 찾는 것이 이미지를 보내는 이유다."""
        sources = {
            f["source"]
            for s in FEW_SHOTS
            for f in json.loads(s.assistant)["riskFactors"]
        }
        assert "OBSERVED" in sources

    async def test_covers_every_group(self) -> None:
        groups = {json.loads(s.user)["animalGroup"] for s in FEW_SHOTS}
        assert groups == {g.value for g in AnimalGroup.analyzable()}

    @pytest.mark.parametrize("shot", FEW_SHOTS)
    async def test_never_outputs_score(self, shot) -> None:
        assert "petFitScore" not in json.loads(shot.assistant)
