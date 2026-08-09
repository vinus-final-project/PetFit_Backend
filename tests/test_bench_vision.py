"""측정 스크립트 검증.

측정은 Mac Studio 에서 하고 슬롯당 30분뿐이다. **스크립트가 거기서 깨지면
슬롯 하나가 통째로 날아간다.** 여기서 미리 돌려 본다.

숫자의 옳고 그름은 보지 않는다. 그건 측정으로 정할 일이다. 여기서는
**스크립트가 끝까지 도는지**와 인자 처리만 확인한다.
"""

import pytest

from scripts.bench_vision import (
    Measurement,
    Stopwatch,
    find_videos,
    main,
    parse_args,
    peak_memory_mb,
)


@pytest.fixture
def samples(tmp_path, make_video):
    """영상 두 개가 든 폴더."""
    folder = tmp_path / "samples"
    folder.mkdir()
    for name, seconds in (("v01.mp4", 5.0), ("v02.mp4", 6.0)):
        make_video(name=name, seconds=seconds).rename(folder / name)
    return folder


class TestArguments:
    def test_requires_a_model_or_stub(self, capsys) -> None:
        """둘 다 없으면 어떤 탐지기를 쓸지 알 수 없다."""
        with pytest.raises(SystemExit):
            parse_args(["--videos", "."])

    def test_stub_alone_is_enough(self) -> None:
        args = parse_args(["--videos", ".", "--stub"])
        assert args.stub is True

    def test_model_alone_is_enough(self) -> None:
        args = parse_args(["--videos", ".", "--model", "yolo.pt"])
        assert args.model == "yolo.pt"

    def test_defaults(self) -> None:
        args = parse_args(["--videos", ".", "--stub"])

        assert args.batch_size == 8
        assert args.max_edge == 1280
        assert args.tracking is False
        assert args.sweep_iou is None

    def test_tracking_flag(self) -> None:
        args = parse_args(["--videos", ".", "--stub", "--tracking"])
        assert args.tracking is True


class TestFindVideos:
    def test_picks_video_files(self, tmp_path) -> None:
        for name in ("a.mp4", "b.mov", "c.m4v"):
            (tmp_path / name).touch()
        assert len(find_videos(tmp_path)) == 3

    def test_ignores_other_files(self, tmp_path) -> None:
        (tmp_path / "a.mp4").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "frame.jpg").touch()

        assert [p.name for p in find_videos(tmp_path)] == ["a.mp4"]

    def test_is_sorted(self, tmp_path) -> None:
        for name in ("c.mp4", "a.mp4", "b.mp4"):
            (tmp_path / name).touch()
        assert [p.name for p in find_videos(tmp_path)] == ["a.mp4", "b.mp4", "c.mp4"]

    def test_case_insensitive_suffix(self, tmp_path) -> None:
        (tmp_path / "A.MP4").touch()
        assert len(find_videos(tmp_path)) == 1


class TestStopwatch:
    def test_records_each_label(self) -> None:
        watch = Stopwatch()
        watch.measure("a", lambda: None)
        watch.measure("b", lambda: None)

        assert set(watch.timings) == {"a", "b"}

    def test_returns_the_value(self) -> None:
        assert Stopwatch().measure("x", lambda: 42) == 42

    def test_times_are_not_negative(self) -> None:
        watch = Stopwatch()
        watch.measure("x", lambda: sum(range(1000)))
        assert watch.timings["x"] >= 0


class TestMeasurement:
    def test_total_sums_the_stages(self) -> None:
        m = Measurement(
            name="v", duration=5.0, frame_count=15, detections=10, objects=3,
            occupancy=0.2, occupancy_spread=0.1,
            timings={"extract": 1.0, "detect": 2.0},
        )
        assert m.total == 3.0


class TestRun:
    def test_runs_end_to_end(self, samples, capsys) -> None:
        assert main(["--videos", str(samples), "--stub"]) == 0

        out = capsys.readouterr().out
        assert "영상별" in out
        assert "단계별 소요" in out
        assert "활동 공간 점유율" in out

    def test_sweep_is_printed(self, samples, capsys) -> None:
        main(["--videos", str(samples), "--stub", "--sweep-iou", "0.2,0.5"])

        out = capsys.readouterr().out
        assert "IoU 임계값 스윕" in out

    def test_writes_csv(self, samples, tmp_path) -> None:
        target = tmp_path / "out.csv"
        main(["--videos", str(samples), "--stub", "--csv", str(target)])

        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("video,duration")
        assert len(lines) == 3          # 헤더 + 영상 2개

    def test_missing_folder(self, tmp_path, capsys) -> None:
        assert main(["--videos", str(tmp_path / "none"), "--stub"]) == 1

    def test_empty_folder(self, tmp_path, capsys) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["--videos", str(empty), "--stub"]) == 1

    def test_tracking_mode_uses_the_id_tracker(self, samples, capsys) -> None:
        """스텁 탐지기는 ID 를 붙이지 않으므로 전부 IoU 로 복구된다."""
        assert main(["--videos", str(samples), "--stub", "--tracking"]) == 0
        assert "TrackIdTracker" in capsys.readouterr().out


class TestPeakMemory:
    def test_returns_a_positive_number(self) -> None:
        """tracemalloc 은 모델 가중치를 세지 못한다. OS 값을 쓴다."""
        assert peak_memory_mb() > 0
