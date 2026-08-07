"""공간별 분석 항목 적용 규칙.

공간에 해당하지 않는 항목은 가중치 합에서 **제외**한다. 감점 0으로 처리하지 않는다.
감점 0으로 두면 해당 항목이 만점인 것처럼 계산되어 점수가 부풀려진다.

항목 유형에 따라 공간 구분의 필요 여부가 다르다.

    탐지형  : 객체가 있으면 감점 → 없으면 감점 0이므로 전 공간 적용
    부재형  : 객체가 없으면 감점 → 정상인 공간에서 부당하게 감점되므로 구분 필요
    점유율형: 공간 특성에 따라 좁은 것이 정상일 수 있으므로 구분 필요
"""

from app.schemas.enums import AnalysisItem as I, SpaceType as S

__all__ = ["APPLICABLE", "is_applicable", "applicable_items"]

#: 공간별 적용 분석 항목
APPLICABLE: dict[S, frozenset[I]] = {
    S.LIVING_ROOM: frozenset({
        I.CABLE_EXPOSURE, I.STAIRS_RISK, I.WINDOW_SAFETY,
        I.ACTIVITY_SPACE, I.SLIP_RISK, I.REST_SPACE,
        I.VERTICAL_SPACE, I.FEEDING_ENV,
    }),
    S.BEDROOM: frozenset({
        I.CABLE_EXPOSURE, I.STAIRS_RISK, I.WINDOW_SAFETY,
        I.ACTIVITY_SPACE, I.SLIP_RISK, I.REST_SPACE,
        I.VERTICAL_SPACE,
    }),
    S.KITCHEN: frozenset({
        I.CABLE_EXPOSURE, I.STAIRS_RISK, I.WINDOW_SAFETY,
        I.FEEDING_ENV,
    }),
    S.BALCONY: frozenset({
        I.CABLE_EXPOSURE, I.STAIRS_RISK, I.WINDOW_SAFETY,
    }),
}

#: 판정할 탐지 객체가 정의되지 않아 어느 공간에서도 산출할 수 없는 항목
UNMEASURABLE: frozenset[I] = frozenset({I.HIDING_SPACE})


def is_applicable(item: I, space: S) -> bool:
    """해당 공간에서 분석 항목을 평가하는지 여부를 반환한다."""
    if item in UNMEASURABLE:
        return False
    return item in APPLICABLE[space]


def applicable_items(space: S) -> frozenset[I]:
    """해당 공간에서 평가하는 분석 항목 집합을 반환한다."""
    return APPLICABLE[space] - UNMEASURABLE
