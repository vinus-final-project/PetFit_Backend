"""공간별 분석 항목 적용 규칙 검증."""

import pytest

from app.rules.space_rules import APPLICABLE, applicable_items, is_applicable
from app.schemas.enums import AnalysisItem as I, SpaceType as S


class TestApplicable:
    def test_all_spaces_defined(self) -> None:
        assert set(APPLICABLE) == set(S)

    @pytest.mark.parametrize(
        "space, count",
        [(S.LIVING_ROOM, 8), (S.BEDROOM, 7), (S.KITCHEN, 4), (S.BALCONY, 3)],
    )
    def test_item_count_per_space(self, space: S, count: int) -> None:
        assert len(applicable_items(space)) == count

    @pytest.mark.parametrize("space", list(S))
    def test_detection_items_apply_everywhere(self, space: S) -> None:
        """탐지형 항목은 없으면 감점 0이므로 모든 공간에서 평가한다."""
        for item in (I.CABLE_EXPOSURE, I.STAIRS_RISK, I.WINDOW_SAFETY):
            assert is_applicable(item, space)

    @pytest.mark.parametrize("space", list(S))
    def test_hiding_space_never_applies(self, space: S) -> None:
        """숨을 공간은 산출 불가 항목이므로 어느 공간에서도 평가하지 않는다."""
        assert not is_applicable(I.HIDING_SPACE, space)
        assert I.HIDING_SPACE not in applicable_items(space)

    def test_feeding_env_excluded_from_bedroom(self) -> None:
        """침실에 급수기가 없다고 감점하지 않는다."""
        assert not is_applicable(I.FEEDING_ENV, S.BEDROOM)
        assert is_applicable(I.FEEDING_ENV, S.KITCHEN)
        assert is_applicable(I.FEEDING_ENV, S.LIVING_ROOM)

    def test_balcony_has_only_detection_items(self) -> None:
        """베란다는 부재형·점유율형 항목을 평가하지 않는다."""
        assert applicable_items(S.BALCONY) == frozenset(
            {I.CABLE_EXPOSURE, I.STAIRS_RISK, I.WINDOW_SAFETY}
        )

    def test_rest_space_excluded_from_kitchen_and_balcony(self) -> None:
        assert not is_applicable(I.REST_SPACE, S.KITCHEN)
        assert not is_applicable(I.REST_SPACE, S.BALCONY)
