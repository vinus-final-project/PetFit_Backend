"""프레임 추출 검증.

계산식은 순수 함수로 따로 시험하고, 실제 동작은 **진짜 H.264 파일을 만들어**
확인한다. 가짜 객체로 대신하면 컨테이너 해석·타임스탬프·축소를 하나도
시험하지 못한다.
"""

import pytest
from PIL import Image

from app.ai.pipeline import PipelineError
from app.ai.vision.frames import (
    TOO_FEW_FRAMES_MESSAGE,
    UNREADABLE_MESSAGE,
    extract,
    sample_timestamps,
    target_frame_count,
)
from app.core.constants import FRAME_MAX, FRAME_MAX_EDGE, FRAME_MIN
from app.schemas.enums import AnalysisStage
from app.utils.video import rotation_degrees


class TestTargetFrameCount:
    """목표 장수 = clamp(길이 x 3, 15, 30)"""

    def test_short_video_is_raised_to_minimum(self) -> None:
        """3초는 9장이 되어 부족하다."""
        assert target_frame_count(3.0) == FRAME_MIN

    def test_long_video_is_capped(self) -> None:
        """30초는 90장이 되어 처리 시간이 세 배가 된다."""
        assert target_frame_count(30.0) == FRAME_MAX

    def test_middle_range_uses_three_per_second(self) -> None:
        assert target_frame_count(7.0) == 21
        assert target_frame_count(8.4) == 25

    @pytest.mark.parametrize("duration", [3.0, 4.9, 5.0, 5.1, 9.9, 10.0, 10.1, 30.0])
    def test_always_within_contract_range(self, duration: float) -> None:
        """PipelineResult 가 15~30장을 요구한다."""
        assert FRAME_MIN <= target_frame_count(duration) <= FRAME_MAX

    def test_boundary_between_clamp_and_rate(self) -> None:
        """5초 = 15장이 최솟값과 정확히 맞물리는 지점이다."""
        assert target_frame_count(5.0) == 15
        assert target_frame_count(10.0) == 30


class TestSampleTimestamps:
    def test_samples_the_middle_of_each_interval(self) -> None:
        """0초부터 시작하면 촬영 시작의 흔들린 장면을 쓰게 된다."""
        assert sample_timestamps(6.0, 3) == [1.0, 3.0, 5.0]

    def test_count_matches(self) -> None:
        assert len(sample_timestamps(8.4, 25)) == 25

    def test_is_ascending(self) -> None:
        stamps = sample_timestamps(8.4, 25)
        assert stamps == sorted(stamps)

    def test_stays_inside_the_video(self) -> None:
        """마지막 시각이 영상 길이를 넘으면 그 프레임은 얻을 수 없다."""
        stamps = sample_timestamps(8.4, 25)
        assert 0.0 < stamps[0]
        assert stamps[-1] < 8.4


class TestExtract:
    def test_returns_expected_count(self, make_video) -> None:
        path = make_video(seconds=5.0, fps=30)
        assert extract(path).count == target_frame_count(5.0)

    def test_count_is_within_contract_range(self, make_video) -> None:
        for seconds in (3.0, 5.0, 12.0):
            result = extract(make_video(name=f"c{seconds}.mp4", seconds=seconds))
            assert FRAME_MIN <= result.count <= FRAME_MAX

    def test_reports_duration(self, make_video) -> None:
        result = extract(make_video(seconds=6.0))
        assert result.duration == pytest.approx(6.0, abs=0.2)

    def test_frames_are_numbered_from_zero(self, make_video) -> None:
        frames = extract(make_video()).frames
        assert [f.number for f in frames] == list(range(len(frames)))

    def test_timestamps_ascend(self, make_video) -> None:
        stamps = [f.timestamp for f in extract(make_video()).frames]
        assert stamps == sorted(stamps)

    def test_frames_are_distinct(self, make_video) -> None:
        """같은 프레임을 두 번 쓰면 detection_frame_count 가 부풀려진다.

        오탐 필터가 이 값을 쓰므로, 중복이 섞이면 1프레임짜리 오탐이 채택된다.
        """
        frames = extract(make_video()).frames
        assert len({f.timestamp for f in frames}) == len(frames)

    def test_frames_span_the_video(self, make_video) -> None:
        """앞부분만 뽑으면 공간 전체를 보지 못한다."""
        result = extract(make_video(seconds=6.0))
        assert result.frames[0].timestamp < 1.0
        assert result.frames[-1].timestamp > 5.0

    def test_frames_differ_in_content(self, make_video) -> None:
        """색이 시간에 따라 변하도록 만든 영상이다. 같은 장면이 반복되면 안 된다."""
        frames = extract(make_video()).frames
        first = frames[0].image.getpixel((5, 5))
        last = frames[-1].image.getpixel((5, 5))
        assert first != last


class TestResize:
    def test_large_frame_is_shrunk(self, make_video) -> None:
        """4K 원본 30장을 그대로 들고 있으면 동시 처리에서 메모리가 터진다."""
        frames = extract(make_video(width=1920, height=1080)).frames
        assert max(frames[0].width, frames[0].height) == FRAME_MAX_EDGE

    def test_aspect_ratio_is_kept(self, make_video) -> None:
        """비율이 바뀌면 정규화 좌표가 실제 위치와 어긋난다."""
        frames = extract(make_video(width=1920, height=1080)).frames
        assert frames[0].width / frames[0].height == pytest.approx(16 / 9, abs=0.01)

    def test_small_frame_is_not_enlarged(self, make_video) -> None:
        """확대하면 화질은 그대로면서 메모리만 늘어난다."""
        frames = extract(make_video(width=640, height=360)).frames
        assert (frames[0].width, frames[0].height) == (640, 360)

    def test_portrait_video_is_shrunk_by_long_edge(self, make_video) -> None:
        frames = extract(make_video(width=1080, height=1920)).frames
        assert frames[0].height == FRAME_MAX_EDGE
        assert frames[0].width < frames[0].height


class TestFailure:
    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(PipelineError) as e:
            extract(tmp_path / "none.mp4")
        assert e.value.stage is AnalysisStage.FRAME_EXTRACTION
        assert e.value.message == UNREADABLE_MESSAGE

    def test_not_a_video(self, tmp_path) -> None:
        path = tmp_path / "fake.mp4"
        path.write_bytes(b"not a video at all")

        with pytest.raises(PipelineError) as e:
            extract(path)
        assert e.value.stage is AnalysisStage.FRAME_EXTRACTION

    def test_too_few_frames_is_rejected(self, make_video) -> None:
        """저속 촬영 영상은 15장을 채울 수 없다.

        부족분을 복제해 채우면 오탐 필터가 무력해지므로 실패로 처리한다.
        """
        path = make_video(seconds=4.0, fps=2)

        with pytest.raises(PipelineError) as e:
            extract(path)
        assert e.value.message == TOO_FEW_FRAMES_MESSAGE
        assert e.value.stage is AnalysisStage.FRAME_EXTRACTION

    def test_error_message_hides_internals(self, tmp_path) -> None:
        """파일 경로나 라이브러리 예외를 사용자에게 노출하지 않는다."""
        path = tmp_path / "secret-name.mp4"
        path.write_bytes(b"\x00\x01\x02")

        with pytest.raises(PipelineError) as e:
            extract(path)
        assert "secret-name" not in e.value.message
        assert "av" not in e.value.message.lower()


class TestRotation:
    """세로 촬영 영상 대응.

    회전을 적용하지 않으면 방을 옆으로 눕혀서 분석하게 된다.
    """

    class _Stream:
        def __init__(self, metadata):
            self.metadata = metadata

    def test_no_metadata(self) -> None:
        assert rotation_degrees(self._Stream({})) == 0

    @pytest.mark.parametrize("raw,expected", [("90", 90), ("180", 180), ("270", 270)])
    def test_reads_angle(self, raw: str, expected: int) -> None:
        assert rotation_degrees(self._Stream({"rotate": raw})) == expected

    def test_negative_angle_is_normalized(self) -> None:
        assert rotation_degrees(self._Stream({"rotate": "-90"})) == 270

    def test_non_right_angle_is_ignored(self) -> None:
        """45도 회전은 되돌릴 수 없다. 원본 그대로 둔다."""
        assert rotation_degrees(self._Stream({"rotate": "45"})) == 0

    def test_garbage_is_ignored(self) -> None:
        assert rotation_degrees(self._Stream({"rotate": "sideways"})) == 0

    def test_rotation_swaps_dimensions(self) -> None:
        """90도 회전은 가로세로가 바뀐다. 축소보다 먼저 적용해야 한다."""
        from app.ai.vision.frames import _prepare

        rotated = _prepare(Image.new("RGB", (1920, 1080)), 90, FRAME_MAX_EDGE)
        assert rotated.height > rotated.width
        assert max(rotated.width, rotated.height) == FRAME_MAX_EDGE
