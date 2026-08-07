"""위험도(risk_level) 판정 규칙.

``risk_level`` 은 YOLO가 산출하지 않는다. 탐지된 객체와 반려동물 그룹의
조합으로 백엔드가 판정한다.

    risk_level = max(중요도 유래 등급, 객체별 최소 위험도)

**마킹 색상 표시 전용이다.** Pet Fit Score 산출에는 사용하지 않는다.
중요도가 이중으로 적용되기 때문이다.
"""

from app.rules.importance import WEIGHT
from app.schemas.enums import AnalysisItem as I, AnimalGroup as G, RiskLevel as R

__all__ = ["classify", "MINIMUM_RISK", "RISK_ITEM_OF"]

#: 가중치 → 위험 등급
_WEIGHT_TO_RISK = {3: R.HIGH, 2: R.MEDIUM, 1: R.LOW}

#: 객체명 → 위험도 판정에 사용할 분석 항목
RISK_ITEM_OF: dict[str, I] = {
    "전선": I.CABLE_EXPOSURE,
    "계단": I.STAIRS_RISK,
    "창문": I.WINDOW_SAFETY,
}

#: 객체별 최소 위험도.
#: 물리적 위해가 반려동물 종류와 무관한 객체에 하한을 둔다.
#: 감전 위험은 종에 무관하며, 고양이는 씹는 습성으로 오히려 위험이 크다.
MINIMUM_RISK: dict[str, R] = {
    "전선": R.HIGH,
}


def classify(object_name: str, group: G) -> R:
    """객체와 반려동물 그룹으로 위험 수준을 판정한다.

    Args:
        object_name: 한글 객체명.
        group: 반려동물 그룹.

    Returns:
        위험 수준. 위험 판정 대상이 아닌 객체는 SAFE.

    Examples:
        >>> classify("전선", G.CAT)
        <RiskLevel.HIGH: 'HIGH'>
        >>> classify("창문", G.SMALL_DOG)
        <RiskLevel.LOW: 'LOW'>
        >>> classify("소파", G.CAT)
        <RiskLevel.SAFE: 'SAFE'>
    """
    item = RISK_ITEM_OF.get(object_name)
    if item is None:
        return R.SAFE

    derived = _WEIGHT_TO_RISK[WEIGHT[item][group]]
    floor = MINIMUM_RISK.get(object_name)
    if floor is None:
        return derived
    return derived if derived.rank >= floor.rank else floor
