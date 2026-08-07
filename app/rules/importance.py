"""반려동물 그룹별 분석 항목 중요도.

AI 분석 정의서의 "반려동물 그룹별 분석 기준" 표와 1:1로 대응한다.
중요도는 점수 계산식의 가중치로 변환된다.

    ●● (매우 중요) = 3
    ●  (중요)      = 2
    ○  (참고)      = 1
"""

from app.schemas.enums import AnalysisItem as I, AnimalGroup as G

__all__ = ["WEIGHT", "weight_of"]

#: 중요도 기호 → 가중치
_SYMBOL = {"●●": 3, "●": 2, "○": 1}

#: 분석 항목 × 반려동물 그룹 → 중요도 기호
_MATRIX: dict[I, dict[G, str]] = {
    I.CABLE_EXPOSURE: {G.SMALL_DOG: "●●", G.LARGE_DOG: "●●", G.CAT: "●"},
    I.ACTIVITY_SPACE: {G.SMALL_DOG: "●", G.LARGE_DOG: "●●", G.CAT: "●"},
    I.SLIP_RISK:      {G.SMALL_DOG: "●", G.LARGE_DOG: "●●", G.CAT: "○"},
    I.REST_SPACE:     {G.SMALL_DOG: "●", G.LARGE_DOG: "●", G.CAT: "●"},
    I.WINDOW_SAFETY:  {G.SMALL_DOG: "○", G.LARGE_DOG: "○", G.CAT: "●●"},
    I.VERTICAL_SPACE: {G.SMALL_DOG: "○", G.LARGE_DOG: "○", G.CAT: "●●"},
    I.STAIRS_RISK:    {G.SMALL_DOG: "●", G.LARGE_DOG: "●●", G.CAT: "○"},
    I.HIDING_SPACE:   {G.SMALL_DOG: "○", G.LARGE_DOG: "○", G.CAT: "●●"},
    I.FEEDING_ENV:    {G.SMALL_DOG: "●", G.LARGE_DOG: "●", G.CAT: "●"},
}

#: 분석 항목 × 반려동물 그룹 → 가중치
WEIGHT: dict[I, dict[G, int]] = {
    item: {group: _SYMBOL[sym] for group, sym in row.items()}
    for item, row in _MATRIX.items()
}


def weight_of(item: I, group: G) -> int:
    """분석 항목의 가중치를 반환한다.

    Args:
        item: 분석 항목.
        group: 반려동물 그룹.

    Returns:
        1 이상 3 이하의 가중치.

    Raises:
        KeyError: 분석 기준이 정의되지 않은 그룹인 경우.
    """
    return WEIGHT[item][group]
