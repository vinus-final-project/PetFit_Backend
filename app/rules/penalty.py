"""분석 항목별 감점률 산출 규칙.

감점률은 **반려동물 그룹과 무관하게** 산출한다. 그룹별 차이는 가중치로만 반영한다.
``risk_level`` 을 감점률에 반영하면 중요도가 이중으로 적용된다.

    p = 0.0  문제 없음
    p = 1.0  최대 감점
"""

from collections.abc import Collection

from app.core.constants import OCCUPANCY_THRESHOLDS
from app.schemas.enums import AnalysisItem as I
from app.utils.rounding import normalize

__all__ = ["penalty_of", "occupancy_penalty"]


def occupancy_penalty(occupancy_ratio: float) -> float:
    """활동 공간 점유율을 감점률로 변환한다.

    부동소수점 오차가 임계값 판정을 뒤집지 않도록 비교 전 정규화한다.

    Args:
        occupancy_ratio: 프레임별 점유율의 중앙값. 0.0 이상 1.0 이하.

    Returns:
        0.0, 0.4, 0.7, 1.0 중 하나.
    """
    value = normalize(occupancy_ratio)
    for threshold, p in OCCUPANCY_THRESHOLDS:
        if value <= threshold:
            return p
    return 1.0


def penalty_of(item: I, objects: Collection[str], occupancy_ratio: float) -> float | None:
    """분석 항목의 감점률을 산출한다.

    Args:
        item: 분석 항목.
        objects: 채택된 객체의 한글 이름 집합.
        occupancy_ratio: 활동 공간 점유율.

    Returns:
        0.0 이상 1.0 이하의 감점률. 산출할 수 없는 항목은 None.
    """
    has = objects.__contains__

    if item is I.CABLE_EXPOSURE:
        return 1.0 if has("전선") else 0.0
    if item is I.STAIRS_RISK:
        return 1.0 if has("계단") else 0.0
    if item is I.WINDOW_SAFETY:
        return 1.0 if has("창문") else 0.0
    if item is I.SLIP_RISK:
        return 0.0 if has("카펫") else 0.5
    if item is I.VERTICAL_SPACE:
        return 0.0 if has("캣타워") else 1.0
    if item is I.REST_SPACE:
        if has("반려동물 침대"):
            return 0.0
        return 0.4 if (has("소파") or has("침대")) else 1.0
    if item is I.FEEDING_ENV:
        count = has("급식기") + has("급수기")
        return {2: 0.0, 1: 0.5, 0: 1.0}[count]
    if item is I.ACTIVITY_SPACE:
        return occupancy_penalty(occupancy_ratio)

    # HIDING_SPACE — 판정할 탐지 객체가 정의되지 않았다.
    return None
