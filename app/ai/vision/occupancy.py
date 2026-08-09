"""활동 공간 점유율 산출 (4단계).

    프레임별 점유율 = Bounding Box 합집합 면적 / 프레임 면적
    최종 점유율     = median(프레임별 점유율)

좌표가 정규화되어 있어 프레임 면적은 항상 1.0 이다. 합집합 면적이 곧 점유율이다.

**객체 추적보다 먼저 실행한다.** 점유율은 프레임 단위 지표이므로 통합 후에는
"3번 프레임에 무엇이 몇 개 있었는지" 가 사라져 산출할 수 없다.

면적은 단순 합이 아니라 합집합이다. 겹친 박스를 중복 계산하면 점유율이 1.0을
넘는다. 계산은 ``utils.geometry.union_area()`` 가 좌표 압축으로 수행한다.
"""

import logging
from collections.abc import Sequence
from statistics import median

from app.ai.vision.types import Detection
from app.utils.geometry import union_area
from app.utils.rounding import normalize

__all__ = ["frame_ratios", "occupancy_ratio"]

logger = logging.getLogger(__name__)


def frame_ratios(detections: Sequence[Sequence[Detection]]) -> list[float]:
    """프레임별 점유율을 구한다.

    **탐지가 없는 프레임은 0.0 이며 제외하지 않는다.** 빈 벽이나 바닥을 비춘
    구간은 실제로 비어 있는 공간이고, 그것도 활동 공간의 일부다. 제외하면
    가구가 많이 잡힌 프레임만 남아 점유율이 실제보다 높게 나온다.

    Args:
        detections: 프레임별 탐지 결과. 탐지기 출력을 그대로 받는다.

    Returns:
        프레임 순서대로의 점유율. 0.0 이상 1.0 이하.
    """
    return [_ratio_of(row) for row in detections]


def occupancy_ratio(detections: Sequence[Sequence[Detection]]) -> float:
    """활동 공간 점유율을 구한다.

    중앙값을 쓰는 이유는 카메라가 벽이나 소파를 정면으로 크게 비춘 한두
    프레임이 평균을 통째로 끌어올리기 때문이다.

    Args:
        detections: 프레임별 탐지 결과.

    Returns:
        프레임별 점유율의 중앙값. 소수 넷째 자리까지. 프레임이 없으면 0.0.

    Note:
        임계값 0.40 / 0.60 / 0.75 의 경계에서 감점률이 갈린다. 저장 컬럼이
        ``NUMERIC(5,4)`` 이므로 같은 자릿수로 맞춰, 저장 전후의 판정이
        달라지지 않게 한다.

    Examples:
        >>> occupancy_ratio([])
        0.0
    """
    ratios = frame_ratios(detections)
    if not ratios:
        return 0.0
    return normalize(median(ratios))


def _ratio_of(row: Sequence[Detection]) -> float:
    """프레임 1장의 점유율."""
    area = union_area(d.box for d in row)

    if area > 1.0:
        # 도달하면 탐지기가 좌표 규약을 어긴 것이다. 그대로 두면 DB의
        # ck_analysis_occupancy_ratio 제약에 걸려 분석 전체가 실패하고,
        # 원인이 탐지기라는 사실은 오류 메시지에 남지 않는다.
        logger.warning("점유율이 1.0을 초과했다: %s. 탐지 좌표를 확인해야 한다", area)
        return 1.0

    return area
