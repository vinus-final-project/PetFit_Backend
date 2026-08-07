"""Bounding Box 합집합 면적 검증.

좌표 압축은 근사가 아니라 정확값을 산출한다.
고정 격자 근사는 판정 방식에 따라 최대 4% 오차가 발생하며,
점유율 임계값 경계에서 감점률을 뒤집는다.
"""

import pytest

from app.utils.geometry import BoundingBox, union_area


class TestUnionArea:
    def test_empty(self) -> None:
        assert union_area([]) == 0.0

    def test_single_box(self) -> None:
        assert union_area([BoundingBox(0.1, 0.1, 0.4, 0.4)]) == pytest.approx(0.16)

    def test_fully_overlapping_boxes_counted_once(self) -> None:
        """완전히 겹친 두 박스는 하나의 면적과 같아야 한다."""
        box = BoundingBox(0.1, 0.1, 0.4, 0.4)
        assert union_area([box, box]) == pytest.approx(0.16)

    def test_disjoint_boxes_sum(self) -> None:
        """겹치지 않는 박스는 개별 면적의 단순 합과 일치해야 한다."""
        boxes = [
            BoundingBox(0.0, 0.0, 0.2, 0.2),
            BoundingBox(0.3, 0.3, 0.2, 0.2),
            BoundingBox(0.6, 0.6, 0.2, 0.2),
        ]
        assert union_area(boxes) == pytest.approx(0.12)

    def test_touching_edges_not_double_counted(self) -> None:
        """경계가 맞닿은 박스는 접합면이 중복 계산되지 않아야 한다.

        좌측·상단 경계는 포함하고 우측·하단은 제외하는 규칙을 검증한다.
        """
        boxes = [
            BoundingBox(0.0, 0.0, 0.3, 0.3),
            BoundingBox(0.3, 0.0, 0.3, 0.3),
        ]
        assert union_area(boxes) == pytest.approx(0.18)

    def test_never_exceeds_one(self) -> None:
        """겹치는 대형 박스가 많아도 점유율이 1.0을 넘지 않아야 한다.

        단순 면적 합이면 1.44가 되어 비율이 1을 초과한다.
        """
        boxes = [
            BoundingBox(0.0, 0.0, 0.6, 0.6),
            BoundingBox(0.3, 0.0, 0.6, 0.6),
            BoundingBox(0.0, 0.3, 0.6, 0.6),
            BoundingBox(0.3, 0.3, 0.6, 0.6),
        ]
        naive = sum(b.area for b in boxes)
        assert naive > 1.0
        assert union_area(boxes) <= 1.0

    def test_nested_box(self) -> None:
        """작은 박스가 큰 박스에 완전히 포함되면 큰 박스 면적과 같다."""
        outer = BoundingBox(0.1, 0.1, 0.5, 0.5)
        inner = BoundingBox(0.2, 0.2, 0.1, 0.1)
        assert union_area([outer, inner]) == pytest.approx(outer.area)

    def test_zero_size_ignored(self) -> None:
        boxes = [BoundingBox(0.1, 0.1, 0.4, 0.4), BoundingBox(0.5, 0.5, 0.0, 0.2)]
        assert union_area(boxes) == pytest.approx(0.16)
