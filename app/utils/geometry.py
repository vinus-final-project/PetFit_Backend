"""Bounding Box 기하 연산.

활동 공간 점유율은 Bounding Box의 **합집합** 면적으로 산출한다.
단순 면적 합을 사용하면 겹친 영역이 중복 계산되어 점유율이 1.0을 초과한다.

합집합은 좌표 압축(Coordinate Compression)으로 정확히 계산한다.
고정 격자 근사는 판정 방식에 따라 최대 4% 오차가 발생하며,
점유율 임계값(0.40 / 0.60 / 0.75) 경계에서 감점률을 뒤집는다.
"""

from typing import Iterable, NamedTuple

__all__ = ["BoundingBox", "union_area"]


class BoundingBox(NamedTuple):
    """정규화 좌표 기준 Bounding Box.

    좌표는 프레임의 가로·세로를 각각 1.0으로 하는 정규화 값이다.
    원점은 프레임 좌측 상단이다.
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, px: float, py: float) -> bool:
        """점 (px, py) 가 박스 내부에 있는지 판정한다.

        좌측·상단 경계는 포함하고 우측·하단 경계는 제외한다.
        인접한 박스가 맞닿았을 때 경계면이 중복 계산되는 것을 막는다.
        """
        return self.x <= px < self.right and self.y <= py < self.bottom


def union_area(boxes: Iterable[BoundingBox]) -> float:
    """Bounding Box 합집합 면적을 정확히 계산한다.

    좌표 압축으로 박스 경계에서만 평면을 분할한다. 분할된 각 셀은
    전부 덮이거나 전부 덮이지 않으므로 근사 오차가 발생하지 않는다.

    탐지 객체가 최대 12종이므로 좌표는 축당 최대 24개, 셀은 최대 529개이다.

    Args:
        boxes: 정규화 좌표 Bounding Box 목록.

    Returns:
        합집합 면적. 0.0 이상 1.0 이하.

    Examples:
        >>> a = BoundingBox(0.1, 0.1, 0.4, 0.4)
        >>> round(union_area([a, a]), 4)
        0.16
        >>> b = BoundingBox(0.5, 0.1, 0.4, 0.4)
        >>> round(union_area([a, b]), 4)
        0.32
    """
    boxes = [b for b in boxes if b.width > 0 and b.height > 0]
    if not boxes:
        return 0.0

    xs = sorted({v for b in boxes for v in (b.x, b.right)})
    ys = sorted({v for b in boxes for v in (b.y, b.bottom)})

    total = 0.0
    for i in range(len(xs) - 1):
        cx = (xs[i] + xs[i + 1]) / 2
        for j in range(len(ys) - 1):
            cy = (ys[j] + ys[j + 1]) / 2
            if any(b.contains(cx, cy) for b in boxes):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total
