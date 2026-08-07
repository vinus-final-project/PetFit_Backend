"""위험도 판정 검증.

risk_level 은 마킹 색상 표시 전용이며 점수 산출에는 사용하지 않는다.
"""

import pytest

from app.rules.risk_rules import classify
from app.schemas.enums import AnimalGroup as G, RiskLevel as R


class TestClassify:
    @pytest.mark.parametrize(
        "obj, group, expected",
        [
            ("계단", G.SMALL_DOG, R.MEDIUM),
            ("계단", G.LARGE_DOG, R.HIGH),
            ("계단", G.CAT, R.LOW),
            ("창문", G.SMALL_DOG, R.LOW),
            ("창문", G.LARGE_DOG, R.LOW),
            ("창문", G.CAT, R.HIGH),
        ],
    )
    def test_group_dependent_objects(self, obj: str, group: G, expected: R) -> None:
        """그룹에 따라 위험 수준이 달라진다."""
        assert classify(obj, group) is expected

    @pytest.mark.parametrize("group", list(G.analyzable()))
    def test_cable_always_high(self, group: G) -> None:
        """전선은 최소 위험도 하한으로 모든 그룹에서 HIGH 이다.

        중요도만 따르면 고양이는 MEDIUM(노란색)이 된다. 감전 위험은 종에 무관하며,
        색상은 사용자에게 절대적 위험도로 읽히므로 하한을 적용한다.
        """
        assert classify("전선", group) is R.HIGH

    @pytest.mark.parametrize("obj", ["소파", "침대", "카펫", "급식기", "캣타워", "의자"])
    def test_non_risk_objects_are_safe(self, obj: str) -> None:
        assert classify(obj, G.CAT) is R.SAFE

    def test_unknown_object_is_safe(self) -> None:
        assert classify("존재하지않는객체", G.SMALL_DOG) is R.SAFE

    def test_safe_has_no_marking_color(self) -> None:
        """SAFE 객체는 마킹 이미지를 생성하지 않는다."""
        assert R.SAFE.marking_color is None
        assert R.HIGH.marking_color == "red"
