"""수치 반올림 유틸리티.

Pet Fit Score는 규칙 기반으로 산출되며 동일 입력에 항상 동일한 값을 반환해야 한다.
Python 내장 ``round()`` 는 은행가 반올림(Round Half to Even)을 사용하므로
``round(0.5) == 0``, ``round(49.5) == 50`` 처럼 결과가 직관과 다르다.

AI 분석 정의서의 반올림 규칙에 따라 Round Half Up 으로 통일한다.
"""

from decimal import Decimal, ROUND_HALF_UP

__all__ = ["round_half_up", "normalize"]

#: 임계값 비교 전 정규화 자릿수. Bounding Box 좌표의 NUMERIC(5,4) 와 자릿수를 맞춘다.
COMPARISON_PRECISION = 4


def round_half_up(value: float) -> int:
    """0.5를 올림하여 정수로 반올림한다.

    Args:
        value: 반올림할 실수.

    Returns:
        Round Half Up 규칙을 적용한 정수.

    Examples:
        >>> round_half_up(49.5)
        50
        >>> round_half_up(0.5)
        1
        >>> round_half_up(66.67)
        67
    """
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def normalize(value: float, precision: int = COMPARISON_PRECISION) -> float:
    """임계값 비교 전 부동소수점 오차를 제거한다.

    합집합 면적과 confidence는 부동소수점 연산 결과이므로 임계값과 정확히
    일치하지 않는다. ``0.05 + 0.35`` 는 ``0.39999999999999997`` 이 되어
    "0.40 이하" 판정이 계산 순서에 따라 달라진다.

    Args:
        value: 정규화할 실수.
        precision: 유지할 소수 자릿수. 기본값은 4.

    Returns:
        지정한 자릿수로 Round Half Up 한 실수.

    Examples:
        >>> normalize(0.05 + 0.35)
        0.4
    """
    exp = Decimal("1").scaleb(-precision)
    return float(Decimal(str(value)).quantize(exp, rounding=ROUND_HALF_UP))
