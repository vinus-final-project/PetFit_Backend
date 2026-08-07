"""분석 항목별 감점률 검증.

감점률은 반려동물 그룹과 무관하다. 그룹 차이는 가중치로만 반영한다.
"""

import pytest

from app.rules.penalty import occupancy_penalty, penalty_of
from app.schemas.enums import AnalysisItem as I, AnimalGroup as G


class TestOccupancyPenalty:
    @pytest.mark.parametrize(
        "ratio, expected",
        [
            (0.00, 0.0),
            (0.35, 0.0),
            (0.40, 0.0),   # 경계 포함
            (0.41, 0.4),
            (0.60, 0.4),   # 경계 포함
            (0.61, 0.7),
            (0.75, 0.7),   # 경계 포함
            (0.76, 1.0),
            (1.00, 1.0),
        ],
    )
    def test_thresholds(self, ratio: float, expected: float) -> None:
        assert occupancy_penalty(ratio) == expected

    def test_boundary_is_inclusive(self) -> None:
        """임계값 자체는 낮은 감점 구간에 속한다."""
        assert occupancy_penalty(0.40) < occupancy_penalty(0.4001)

    def test_floating_point_error_does_not_flip_threshold(self) -> None:
        """0.05 + 0.35 는 0.39999... 이지만 0.40 구간으로 판정되어야 한다."""
        assert occupancy_penalty(0.05 + 0.35) == occupancy_penalty(0.40)


class TestPenaltyOf:
    @pytest.mark.parametrize(
        "item, objects, expected",
        [
            # 탐지형 — 있으면 감점
            (I.CABLE_EXPOSURE, {"전선"}, 1.0),
            (I.CABLE_EXPOSURE, set(), 0.0),
            (I.STAIRS_RISK, {"계단"}, 1.0),
            (I.STAIRS_RISK, set(), 0.0),
            (I.WINDOW_SAFETY, {"창문"}, 1.0),
            (I.WINDOW_SAFETY, set(), 0.0),
            # 부재형 — 없으면 감점
            (I.SLIP_RISK, {"카펫"}, 0.0),
            (I.SLIP_RISK, set(), 0.5),
            (I.VERTICAL_SPACE, {"캣타워"}, 0.0),
            (I.VERTICAL_SPACE, set(), 1.0),
            # 휴식 공간 — 전용 침대 > 대체 가구 > 없음
            (I.REST_SPACE, {"반려동물 침대"}, 0.0),
            (I.REST_SPACE, {"반려동물 침대", "소파"}, 0.0),
            (I.REST_SPACE, {"소파"}, 0.4),
            (I.REST_SPACE, {"침대"}, 0.4),
            (I.REST_SPACE, set(), 1.0),
            # 급식·급수 — 개수 비례
            (I.FEEDING_ENV, {"급식기", "급수기"}, 0.0),
            (I.FEEDING_ENV, {"급수기"}, 0.5),
            (I.FEEDING_ENV, {"급식기"}, 0.5),
            (I.FEEDING_ENV, set(), 1.0),
        ],
    )
    def test_penalty(self, item: I, objects: set[str], expected: float) -> None:
        assert penalty_of(item, objects, 0.0) == expected

    def test_hiding_space_is_unmeasurable(self) -> None:
        """숨을 공간은 판정할 탐지 객체가 없어 산출 불가(None)를 반환한다.

        None 은 가중치 합에서 제외를 의미한다. 0.0(감점 없음)과 구별해야 한다.
        """
        assert penalty_of(I.HIDING_SPACE, {"소파", "침대"}, 0.5) is None

    @pytest.mark.parametrize("item", [i for i in I if i is not I.HIDING_SPACE])
    def test_all_items_measurable_except_hiding(self, item: I) -> None:
        assert penalty_of(item, set(), 0.5) is not None

    @pytest.mark.parametrize("item", list(I))
    def test_penalty_is_within_range(self, item: I) -> None:
        p = penalty_of(item, {"전선", "창문", "카펫"}, 0.5)
        assert p is None or 0.0 <= p <= 1.0

    def test_group_does_not_affect_penalty(self) -> None:
        """감점률 함수는 그룹을 인자로 받지 않는다.

        중요도가 감점률과 가중치에 이중 적용되는 것을 구조적으로 막는다.
        """
        import inspect

        params = inspect.signature(penalty_of).parameters
        assert not any(p.annotation is G for p in params.values())
