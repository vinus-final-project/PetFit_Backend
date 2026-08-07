"""객체 클래스 코드 · 한글 명칭 매핑.

YOLO는 영문 클래스 코드를 출력하고 데이터베이스와 API는 한글 객체명을 사용한다.
매핑표에 없는 코드는 저장하지 않고 무시한다.
"""

__all__ = ["OBJECT_NAMES", "CUSTOM_TRAINED", "to_korean", "is_known"]

#: 클래스 코드 → 한글 객체명 (탐지 대상 12종)
OBJECT_NAMES: dict[str, str] = {
    "sofa": "소파",
    "bed": "침대",
    "chair": "의자",
    "table": "테이블",
    "window": "창문",
    "cable": "전선",
    "stairs": "계단",
    "carpet": "카펫",
    "feeder": "급식기",
    "water_dispenser": "급수기",
    "pet_bed": "반려동물 침대",
    "cat_tower": "캣타워",
}

#: 커스텀 학습이 필요한 클래스. COCO 사전학습에 존재하지 않는다.
CUSTOM_TRAINED: frozenset[str] = frozenset({
    "window", "cable", "stairs", "carpet",
    "feeder", "water_dispenser", "pet_bed", "cat_tower",
})

#: 감점 규칙의 판정 근거로 사용되는 객체. 분석 대표 프레임 선정 기준이 된다.
PRIMARY_OBJECTS: frozenset[str] = frozenset({
    "소파", "침대", "반려동물 침대", "창문", "전선",
    "계단", "카펫", "급식기", "급수기", "캣타워",
})


def to_korean(class_code: str) -> str | None:
    """클래스 코드를 한글 객체명으로 변환한다.

    Args:
        class_code: YOLO 클래스 코드.

    Returns:
        한글 객체명. 매핑표에 없으면 None.
    """
    return OBJECT_NAMES.get(class_code)


def is_known(class_code: str) -> bool:
    """매핑표에 정의된 클래스인지 여부를 반환한다."""
    return class_code in OBJECT_NAMES
