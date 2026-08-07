"""객체 클래스 매핑 검증."""

import pytest

from app.rules.object_map import (
    CUSTOM_TRAINED,
    OBJECT_NAMES,
    PRIMARY_OBJECTS,
    is_known,
    to_korean,
)
from app.rules.penalty import penalty_of
from app.schemas.enums import AnalysisItem as I


class TestObjectNames:
    def test_target_object_count(self) -> None:
        """MVP 탐지 대상은 12종이다."""
        assert len(OBJECT_NAMES) == 12

    def test_korean_names_are_unique(self) -> None:
        assert len(set(OBJECT_NAMES.values())) == len(OBJECT_NAMES)

    def test_to_korean(self) -> None:
        assert to_korean("cable") == "전선"
        assert to_korean("cat_tower") == "캣타워"

    def test_unknown_class_returns_none(self) -> None:
        """매핑표에 없는 코드는 저장하지 않고 무시한다."""
        assert to_korean("person") is None
        assert not is_known("person")

    def test_custom_trained_subset_of_targets(self) -> None:
        assert CUSTOM_TRAINED <= set(OBJECT_NAMES)

    def test_custom_trained_count(self) -> None:
        """COCO 사전학습에 없어 커스텀 학습이 필요한 클래스는 8종이다."""
        assert len(CUSTOM_TRAINED) == 8

    def test_coco_available_classes_not_custom(self) -> None:
        for code in ("sofa", "bed", "chair", "table"):
            assert code not in CUSTOM_TRAINED


class TestPrimaryObjects:
    def test_primary_objects_are_mapped_names(self) -> None:
        assert PRIMARY_OBJECTS <= set(OBJECT_NAMES.values())

    @pytest.mark.parametrize("name", sorted(PRIMARY_OBJECTS))
    def test_primary_object_affects_some_penalty(self, name: str) -> None:
        """감점 근거 객체는 최소 한 항목의 감점률을 바꿔야 한다."""
        changed = any(
            penalty_of(item, {name}, 0.3) != penalty_of(item, set(), 0.3)
            for item in I
        )
        assert changed, f"{name} 은 어떤 감점 규칙에도 영향을 주지 않는다"
