"""시각 처리.

**저장·표시 모두 KST(UTC+9)를 기준으로 한다.**

한국은 서머타임을 시행하지 않으므로 고정 오프셋으로 충분하다. 로컬 시간 저장의
고전적 문제인 "1년에 한 번 중복되는 시각"과 "존재하지 않는 시각"이 발생하지 않는다.

MySQL의 ``DATETIME`` 은 타임존을 보존하지 않는다. 따라서 저장된 naive 값이 어느
타임존인지는 **약속으로 정해야 하며, 그 약속이 이 모듈이다.**

``CURRENT_TIMESTAMP(6)`` 는 세션 타임존을 따른다. 연결 시점에 ``+09:00`` 으로
고정하지 않으면 서버 OS 설정에 따라 값이 달라진다. 고정 작업은
``app/db/session.py`` 가 담당한다.
"""

from datetime import datetime, timedelta, timezone

__all__ = ["KST", "DB_TIME_ZONE", "now", "now_naive", "as_aware", "to_iso"]

#: 한국 표준시. DST가 없어 고정 오프셋으로 표현할 수 있다.
#: ZoneInfo는 Windows에서 tzdata 패키지를 추가로 요구하므로 사용하지 않는다.
KST = timezone(timedelta(hours=9))

#: MySQL 세션에 설정할 타임존. 저장 기준과 반드시 일치해야 한다.
DB_TIME_ZONE = "+09:00"


def now() -> datetime:
    """현재 시각을 KST aware datetime으로 반환한다."""
    return datetime.now(KST)


def now_naive() -> datetime:
    """현재 시각을 KST 기준 naive datetime으로 반환한다.

    ``DATETIME`` 컬럼에 직접 대입할 때 사용한다. aware 값을 그대로 넣으면
    드라이버가 오프셋을 잘라내면서 의도치 않은 변환이 일어날 수 있다.
    """
    return datetime.now(KST).replace(tzinfo=None)


def as_aware(value: datetime) -> datetime:
    """DB에서 읽은 naive 값에 KST 오프셋을 붙인다.

    이미 타임존이 있으면 KST로 변환한다. 외부 입력이 UTC로 들어오는 경우를
    대비한 방어 로직이다.

    Args:
        value: 변환할 시각.

    Returns:
        KST 오프셋이 붙은 시각.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


def to_iso(value: datetime) -> str:
    """API 응답용 ISO 8601 문자열로 변환한다.

    명세서 예시가 초 단위까지만 표기하므로 마이크로초는 버린다.

    Returns:
        ``2026-08-06T14:30:00+09:00`` 형식의 문자열.
    """
    return as_aware(value).replace(microsecond=0).isoformat()
