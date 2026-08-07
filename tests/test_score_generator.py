"""Pet Fit Score 산출 검증.

AI 분석 정의서의 계산 예시와 공간별 점수 비교표를 기준값으로 삼는다.
문서와 구현이 어긋나면 이 테스트가 실패한다.
"""

import pytest

from app.ai.score_generator import generate
from app.schemas.enums import AnimalGroup as G, ScoreCategory as C, SpaceType as S

#: AI 분석 정의서 계산 예시의 탐지 결과
EXAMPLE_OBJECTS = frozenset({"전선", "창문", "카펫", "소파", "급수기"})
EXAMPLE_OCCUPANCY = 0.35


class TestDocumentedExample:
    """소형견 · 거실 · 문서 계산 예시 (총점 56)."""

    @pytest.fixture
    def score(self):
        return generate(G.SMALL_DOG, S.LIVING_ROOM, EXAMPLE_OBJECTS, EXAMPLE_OCCUPANCY)

    def test_safety(self, score) -> None:
        """전선 3.0 + 창문 1.0 = 4.0 / w합 8 → 50"""
        assert score.safety == 50

    def test_activity(self, score) -> None:
        """수직 공간 1.0 / w합 3 → 66.67 → 67"""
        assert score.activity == 67

    def test_rest(self, score) -> None:
        """휴식 공간 0.8 / w합 2 → 60. 숨을 공간은 제외된다."""
        assert score.rest == 60

    def test_environment(self, score) -> None:
        """급식·급수 1.0 / w합 2 → 50"""
        assert score.environment == 50

    def test_total(self, score) -> None:
        """20.00 + 16.75 + 12.00 + 7.50 = 56.25 → 56"""
        assert score.total == 56

    def test_hiding_space_excluded_from_breakdown(self, score) -> None:
        items = {r.item for rows in score.breakdown.values() for r in rows}
        assert not any(i.name == "HIDING_SPACE" for i in items)


class TestSpaceComparison:
    """공간별 점수 비교표. 동일 탐지 결과에서 공간만 바꾼다."""

    @pytest.mark.parametrize(
        "space, safety, activity, rest, environment, total",
        [
            (S.LIVING_ROOM, 50, 67, 60, 50, 56),
            (S.BEDROOM, 50, 67, 60, 100, 64),
            (S.BALCONY, 33, 100, 100, 100, 73),
        ],
    )
    def test_matches_document(
        self, space, safety, activity, rest, environment, total
    ) -> None:
        s = generate(G.SMALL_DOG, space, EXAMPLE_OBJECTS, EXAMPLE_OCCUPANCY)
        assert (s.safety, s.activity, s.rest, s.environment, s.total) == (
            safety,
            activity,
            rest,
            environment,
            total,
        )

    def test_empty_category_scores_100_not_0(self) -> None:
        """산출 가능한 항목이 없는 평가 항목은 100으로 처리한다.

        0으로 두면 평가하지 않은 항목 때문에 종합 점수가 부당하게 낮아진다.
        """
        s = generate(G.SMALL_DOG, S.BALCONY, EXAMPLE_OBJECTS, EXAMPLE_OCCUPANCY)
        assert s.activity == 100
        assert s.breakdown[C.ACTIVITY] == []


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        """규칙 기반 산출의 핵심 요구사항이다."""
        runs = {
            generate(G.CAT, S.LIVING_ROOM, EXAMPLE_OBJECTS, EXAMPLE_OCCUPANCY).total
            for _ in range(50)
        }
        assert len(runs) == 1

    def test_object_order_does_not_matter(self) -> None:
        a = generate(G.CAT, S.LIVING_ROOM, ["전선", "창문", "소파"], 0.3)
        b = generate(G.CAT, S.LIVING_ROOM, ["소파", "창문", "전선"], 0.3)
        assert a.as_dict() == b.as_dict()


class TestBounds:
    @pytest.mark.parametrize("group", list(G.analyzable()))
    @pytest.mark.parametrize("space", list(S))
    def test_perfect_environment_scores_100(self, group: G, space: S) -> None:
        """감점 요소가 없고 필요 객체가 모두 있으면 만점이다."""
        objects = {"카펫", "캣타워", "반려동물 침대", "급식기", "급수기", "소파"}
        s = generate(group, space, objects, 0.2)
        assert s.as_dict() == {
            "total": 100,
            "safety": 100,
            "activity": 100,
            "rest": 100,
            "environment": 100,
        }

    @pytest.mark.parametrize("group", list(G.analyzable()))
    def test_worst_environment_scores_0(self, group: G) -> None:
        """적용 항목이 모두 최대 감점이면 0점이다.

        베란다는 탐지형 3항목만 평가하므로 전부 p=1.0 이 될 수 있다.
        """
        s = generate(group, S.BALCONY, {"전선", "계단", "창문"}, 1.0)
        assert s.safety == 0

    @pytest.mark.parametrize("group", list(G.analyzable()))
    def test_living_room_safety_never_reaches_0(self, group: G) -> None:
        """거실 안전성은 0이 될 수 없다.

        미끄럼 위험은 카펫 미탐지 시 p=0.5 다. 카펫이 없다는 사실만으로
        바닥이 미끄럽다고 단정할 수 없으므로 최대 감점을 주지 않는다.
        """
        s = generate(group, S.LIVING_ROOM, {"전선", "계단", "창문"}, 1.0)
        assert s.safety > 0

    @pytest.mark.parametrize("group", list(G.analyzable()))
    @pytest.mark.parametrize("space", list(S))
    @pytest.mark.parametrize("occupancy", [0.0, 0.4, 0.5, 0.7, 0.8, 1.0])
    def test_always_within_range(self, group: G, space: S, occupancy: float) -> None:
        s = generate(group, space, EXAMPLE_OBJECTS, occupancy)
        for name, value in s.as_dict().items():
            assert 0 <= value <= 100, name


class TestGroupWeighting:
    def test_cat_penalized_more_for_window(self) -> None:
        """창문 안전은 고양이에게 ●● , 소형견에게 ○ 이다."""
        cat = generate(G.CAT, S.BALCONY, {"창문"}, 0.2).safety
        dog = generate(G.SMALL_DOG, S.BALCONY, {"창문"}, 0.2).safety
        assert cat < dog

    def test_cat_penalized_more_for_missing_cat_tower(self) -> None:
        """수직 공간은 고양이에게 ●● , 소형견에게 ○ 이다."""
        cat = generate(G.CAT, S.LIVING_ROOM, EXAMPLE_OBJECTS, 0.2).activity
        dog = generate(G.SMALL_DOG, S.LIVING_ROOM, EXAMPLE_OBJECTS, 0.2).activity
        assert cat < dog

    def test_large_dog_penalized_more_for_stairs(self) -> None:
        """계단 위험은 중·대형견에게 ●● , 고양이에게 ○ 이다."""
        large = generate(G.LARGE_DOG, S.BALCONY, {"계단"}, 0.2).safety
        cat = generate(G.CAT, S.BALCONY, {"계단"}, 0.2).safety
        assert large < cat

    @pytest.mark.parametrize("group", [G.SMALL_ANIMAL, G.BIRD, G.REPTILE])
    def test_unsupported_group_raises(self, group: G) -> None:
        """MVP 미지원 그룹은 중요도가 정의되지 않아 KeyError 를 낸다."""
        with pytest.raises(KeyError):
            generate(group, S.LIVING_ROOM, EXAMPLE_OBJECTS, 0.3)
