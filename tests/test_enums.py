"""Enum 정합성 검증.

Enum 값은 API 응답·DB CHECK 제약·프론트엔드가 공유하는 계약이다.
"""

import pytest

from app.rules.importance import WEIGHT
from app.schemas.enums import (
    AnalysisItem as I,
    AnalysisStage,
    AnalysisStatus,
    AnimalGroup as G,
    RiskLevel as R,
    ScoreCategory as C,
    SpaceType as S,
)


class TestAnimalGroup:
    def test_six_groups_three_analyzable(self) -> None:
        assert len(list(G)) == 6
        assert set(G.analyzable()) == {G.SMALL_DOG, G.LARGE_DOG, G.CAT}

    def test_importance_defined_only_for_analyzable(self) -> None:
        for row in WEIGHT.values():
            assert set(row) == set(G.analyzable())


class TestAnalysisItem:
    def test_nine_items(self) -> None:
        assert len(list(I)) == 9

    def test_every_item_has_category(self) -> None:
        for item in I:
            assert item.category in set(C)

    def test_every_item_has_importance(self) -> None:
        assert set(WEIGHT) == set(I)


class TestScoreCategory:
    def test_weights_sum_to_one(self) -> None:
        assert sum(c.weight for c in C) == pytest.approx(1.0)

    def test_documented_weights(self) -> None:
        assert C.SAFETY.weight == 0.40
        assert C.ACTIVITY.weight == 0.25
        assert C.REST.weight == 0.20
        assert C.ENVIRONMENT.weight == 0.15

    def test_every_category_has_items(self) -> None:
        for category in C:
            assert any(i.category is category for i in I)


class TestRiskLevel:
    def test_rank_is_ordered(self) -> None:
        assert R.SAFE.rank < R.LOW.rank < R.MEDIUM.rank < R.HIGH.rank

    def test_only_safe_has_no_color(self) -> None:
        """SAFE 만 마킹 이미지를 생성하지 않는다."""
        assert R.SAFE.marking_color is None
        for level in (R.LOW, R.MEDIUM, R.HIGH):
            assert level.marking_color is not None


class TestValuesAreStable:
    """값이 바뀌면 저장된 데이터와 프론트엔드 계약이 깨진다."""

    @pytest.mark.parametrize(
        "enum_cls, expected",
        [
            (G, {"small_dog", "large_dog", "cat", "small_animal", "bird", "reptile"}),
            (S, {"living_room", "bedroom", "kitchen", "balcony"}),
            (R, {"SAFE", "LOW", "MEDIUM", "HIGH"}),
            (AnalysisStatus, {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}),
        ],
    )
    def test_values(self, enum_cls, expected: set[str]) -> None:
        assert {m.value for m in enum_cls} == expected

    def test_stage_progress_is_strictly_increasing(self) -> None:
        """진행률은 되돌아가지 않아야 한다. 사용자가 멈춘 것으로 오해한다."""
        progresses = [s.progress for s in AnalysisStage]
        assert progresses == sorted(set(progresses))

    def test_last_stage_is_not_100(self) -> None:
        """100%는 status = COMPLETED 시점에만 표시한다.

        마지막 단계 진입만으로 100%를 표시하면 저장이 끝나기 전에
        완료로 보여 사용자가 결과 화면으로 이동했다가 빈 화면을 보게 된다.
        """
        progresses = [s.progress for s in AnalysisStage]
        assert progresses[0] > 0
        assert progresses[-1] < 100
