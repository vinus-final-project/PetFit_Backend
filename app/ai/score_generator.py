"""Pet Fit Score 산출 (파이프라인 11단계).

**규칙 기반으로 산출한다.** 생성형 AI는 점수를 만들지 않는다.
동일한 입력에 대해 항상 동일한 점수를 반환해야 한다.

    score(C) = 100 × ( 1 − Σ(w_i × p_i) / Σ(w_i) )
    total    = round(safety×0.40 + activity×0.25 + rest×0.20 + environment×0.15)

가중치 합에서 제외하는 항목은 두 가지다.

    산출 불가   : 판정할 탐지 객체가 정의되지 않은 항목
    공간 미적용 : 해당 공간에서 평가하지 않는 항목

감점 0으로 처리하지 않는다. 만점인 것처럼 계산되어 점수가 부풀려진다.
"""

from collections.abc import Collection
from dataclasses import dataclass, field

from app.rules.importance import weight_of
from app.rules.penalty import penalty_of
from app.rules.space_rules import is_applicable
from app.schemas.enums import (
    AnalysisItem,
    AnimalGroup,
    ScoreCategory,
    SpaceType,
)
from app.utils.rounding import round_half_up

__all__ = ["PetFitScore", "ItemBreakdown", "generate"]


@dataclass(frozen=True)
class ItemBreakdown:
    """분석 항목별 감점 근거. 점수 산출 과정을 추적하기 위해 보관한다."""

    item: AnalysisItem
    weight: int
    penalty: float

    @property
    def weighted(self) -> float:
        return self.weight * self.penalty


@dataclass(frozen=True)
class PetFitScore:
    """Pet Fit Score 산출 결과."""

    total: int
    safety: int
    activity: int
    rest: int
    environment: int
    breakdown: dict[ScoreCategory, list[ItemBreakdown]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        """API 응답용 딕셔너리로 변환한다."""
        return {
            "total": self.total,
            "safety": self.safety,
            "activity": self.activity,
            "rest": self.rest,
            "environment": self.environment,
        }


def _category_score(
    category: ScoreCategory,
    group: AnimalGroup,
    space: SpaceType,
    objects: Collection[str],
    occupancy_ratio: float,
) -> tuple[int, list[ItemBreakdown]]:
    """평가 항목별 점수를 산출한다.

    산출 가능한 항목이 하나도 없으면 100으로 처리한다.
    """
    rows: list[ItemBreakdown] = []
    for item in AnalysisItem:
        if item.category is not category:
            continue
        if not is_applicable(item, space):
            continue
        penalty = penalty_of(item, objects, occupancy_ratio)
        if penalty is None:
            continue
        rows.append(ItemBreakdown(item, weight_of(item, group), penalty))

    if not rows:
        return 100, rows

    numerator = sum(r.weighted for r in rows)
    denominator = sum(r.weight for r in rows)
    return round_half_up(100 * (1 - numerator / denominator)), rows


def generate(
    group: AnimalGroup,
    space: SpaceType,
    objects: Collection[str],
    occupancy_ratio: float,
) -> PetFitScore:
    """Pet Fit Score를 산출한다.

    Args:
        group: 반려동물 그룹.
        space: 촬영한 공간 종류.
        objects: 탐지 신뢰 기준을 통과한 객체의 한글 이름 집합.
        occupancy_ratio: 프레임별 점유율의 중앙값.

    Returns:
        종합 점수와 항목별 점수, 감점 근거.

    Raises:
        KeyError: 분석 기준이 정의되지 않은 반려동물 그룹인 경우.
    """
    scores: dict[ScoreCategory, int] = {}
    breakdown: dict[ScoreCategory, list[ItemBreakdown]] = {}

    for category in ScoreCategory:
        value, rows = _category_score(category, group, space, objects, occupancy_ratio)
        scores[category] = value
        breakdown[category] = rows

    total = round_half_up(
        sum(scores[c] * c.weight for c in ScoreCategory)
    )

    return PetFitScore(
        total=total,
        safety=scores[ScoreCategory.SAFETY],
        activity=scores[ScoreCategory.ACTIVITY],
        rest=scores[ScoreCategory.REST],
        environment=scores[ScoreCategory.ENVIRONMENT],
        breakdown=breakdown,
    )
