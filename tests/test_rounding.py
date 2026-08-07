"""반올림 규칙 검증.

Pet Fit Score는 동일 입력에 항상 동일한 값을 반환해야 한다.
Python 내장 round() 는 은행가 반올림이라 0.5 처리가 직관과 다르다.
"""

import pytest

from app.utils.rounding import normalize, round_half_up


class TestRoundHalfUp:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (49.5, 50),
            (0.5, 1),
            (1.5, 2),
            (2.5, 3),
            (66.67, 67),
            (56.25, 56),
            (33.333, 33),
            (0.0, 0),
            (100.0, 100),
        ],
    )
    def test_rounds_half_up(self, value: float, expected: int) -> None:
        assert round_half_up(value) == expected

    def test_differs_from_builtin_round(self) -> None:
        """내장 round() 는 은행가 반올림이므로 결과가 다르다."""
        assert round(0.5) == 0
        assert round_half_up(0.5) == 1

        assert round(2.5) == 2
        assert round_half_up(2.5) == 3


class TestNormalize:
    def test_removes_floating_point_error(self) -> None:
        """부동소수점 오차가 임계값 판정을 뒤집지 않아야 한다."""
        raw = 0.05 + 0.35
        assert raw != 0.40                     # 0.39999999999999997
        assert normalize(raw) == 0.40

    @pytest.mark.parametrize(
        "expr",
        [0.05 + 0.35, 0.1 + 0.3, 0.2 + 0.2, 0.15 + 0.25],
    )
    def test_all_paths_reach_same_threshold(self, expr: float) -> None:
        """계산 순서가 달라도 동일한 임계값 판정을 내려야 한다."""
        assert normalize(expr) <= 0.40

    def test_precision_matches_column_scale(self) -> None:
        """DECIMAL(5,4) 과 자릿수를 맞춘다."""
        assert normalize(0.123456) == 0.1235
