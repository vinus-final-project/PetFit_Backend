"""시각 처리 검증.

저장·표시 모두 KST 기준이다. DB의 DATETIME은 타임존을 보존하지 않으므로
naive 값을 어느 타임존으로 해석할지는 약속이며, 그 약속을 검증한다.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.utils.timeutil import DB_TIME_ZONE, KST, as_aware, now, now_naive, to_iso


class TestKst:
    def test_offset_is_nine_hours(self) -> None:
        assert KST.utcoffset(None) == timedelta(hours=9)

    def test_db_time_zone_matches_offset(self) -> None:
        """DB 세션에 설정하는 값과 애플리케이션 오프셋이 같아야 한다.

        어긋나면 CURRENT_TIMESTAMP(6)로 채운 값과 애플리케이션이 쓴 값이
        서로 다른 기준을 갖게 된다.
        """
        hours = int(DB_TIME_ZONE.split(":")[0])
        assert timedelta(hours=hours) == KST.utcoffset(None)

    def test_no_dst(self) -> None:
        """한국은 서머타임이 없다. 연중 오프셋이 일정해야 한다."""
        offsets = {
            KST.utcoffset(datetime(2026, m, 15)) for m in range(1, 13)
        }
        assert offsets == {timedelta(hours=9)}


class TestAsAware:
    def test_naive_is_treated_as_kst(self) -> None:
        """DB에서 읽은 naive 값은 KST다. UTC로 해석하면 9시간 어긋난다."""
        result = as_aware(datetime(2026, 8, 6, 14, 30))
        assert result == datetime(2026, 8, 6, 14, 30, tzinfo=KST)

    def test_naive_hour_is_not_shifted(self) -> None:
        assert as_aware(datetime(2026, 8, 6, 14, 30)).hour == 14

    def test_utc_input_converted(self) -> None:
        """외부에서 UTC가 들어오면 KST로 변환한다."""
        utc = datetime(2026, 8, 6, 5, 30, tzinfo=timezone.utc)
        assert as_aware(utc) == datetime(2026, 8, 6, 14, 30, tzinfo=KST)

    def test_already_kst_unchanged(self) -> None:
        value = datetime(2026, 8, 6, 14, 30, tzinfo=KST)
        assert as_aware(value) == value


class TestToIso:
    def test_matches_spec_format(self) -> None:
        assert to_iso(datetime(2026, 8, 6, 14, 30)) == "2026-08-06T14:30:00+09:00"

    def test_microseconds_dropped(self) -> None:
        """명세 예시는 초 단위까지만 표기한다."""
        assert to_iso(datetime(2026, 8, 6, 14, 30, 0, 123456)) == "2026-08-06T14:30:00+09:00"

    def test_offset_always_present(self) -> None:
        assert to_iso(datetime(2026, 1, 1, 0, 0)).endswith("+09:00")
        assert to_iso(datetime(2026, 7, 1, 0, 0)).endswith("+09:00")

    def test_round_trip(self) -> None:
        original = datetime(2026, 8, 6, 14, 30, 42)
        assert datetime.fromisoformat(to_iso(original)).replace(tzinfo=None) == original


class TestNow:
    def test_now_is_aware(self) -> None:
        assert now().tzinfo is KST

    def test_now_naive_has_no_tzinfo(self) -> None:
        """DATETIME 컬럼에 직접 대입하므로 naive 여야 한다."""
        assert now_naive().tzinfo is None

    def test_now_and_now_naive_agree(self) -> None:
        a, b = now(), now_naive()
        assert abs((a.replace(tzinfo=None) - b).total_seconds()) < 1

    def test_now_naive_is_kst_wall_clock(self) -> None:
        """naive 값의 시:분이 KST 벽시계와 같아야 한다."""
        expected = datetime.now(timezone.utc).astimezone(KST)
        assert abs((now_naive() - expected.replace(tzinfo=None)).total_seconds()) < 1


class TestConsistency:
    @pytest.mark.parametrize(
        "moment",
        [
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 6, 15, 12, 0, 0),
            datetime(2026, 12, 31, 23, 59, 59),
        ],
    )
    def test_store_then_display_preserves_wall_clock(self, moment: datetime) -> None:
        """저장한 벽시계 시각이 응답에서 그대로 보여야 한다.

        KST 저장 기준의 핵심 이점이다. DB를 직접 조회했을 때
        사용자가 본 시각과 같은 값이 나온다.
        """
        assert to_iso(moment).startswith(moment.strftime("%Y-%m-%dT%H:%M:%S"))
