"""Vision 파이프라인 측정 도구.

성능평가 진행 가이드의 슬롯 3·4·5를 실제 영상으로 재기 위한 하네스다.
지금 코드에는 **실측 없이 정해진 값이 둘** 있다.

    TRACKING_IOU_THRESHOLD = 0.30      같은 물체로 볼 최소 겹침
    OCCUPANCY_THRESHOLDS   = 0.40/0.60/0.75

이 스크립트는 값을 정해 주지 않는다. **정할 근거를 만든다.**

사용 예::

    # 모델 없이 배관만 확인 (어디서든 동작)
    python -m scripts.bench_vision --videos ./samples --stub

    # 실제 모델 (Mac Studio)
    python -m scripts.bench_vision --videos ./samples \\
        --model ./models/yolo26m.pt --device mps

    # 추적기 비교
    python -m scripts.bench_vision --videos ./samples --model ./models/yolo26m.pt \\
        --device mps --tracking

    # IoU 임계값 스윕
    python -m scripts.bench_vision --videos ./samples --stub \\
        --sweep-iou 0.1,0.2,0.3,0.4,0.5,0.7

**탐지는 한 번만 한다.** 임계값 스윕은 같은 탐지 결과를 다시 묶을 뿐이므로
추론을 반복하지 않는다. 임계값마다 추론하면 스윕 한 번에 수십 분이 걸린다.
"""

import argparse
import csv
import resource
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.vision import frames as frame_module  # noqa: E402
from app.ai.vision.detector import StubDetector  # noqa: E402
from app.ai.vision.imaging import select_analysis_frame  # noqa: E402
from app.ai.vision.occupancy import frame_ratios, occupancy_ratio  # noqa: E402
from app.ai.vision.tracking import IouTracker, TrackIdTracker, adopt  # noqa: E402
from app.core.constants import (  # noqa: E402
    OCCUPANCY_THRESHOLDS,
    PROCESSING_TIMEOUT_SECONDS,
    TRACKING_IOU_THRESHOLD,
)
from app.rules.object_map import to_korean  # noqa: E402

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")


@dataclass
class Measurement:
    """영상 1개의 측정 결과."""

    name: str
    duration: float
    frame_count: int
    detections: int
    objects: int
    occupancy: float
    occupancy_spread: float
    timings: dict[str, float] = field(default_factory=dict)
    names: tuple[str, ...] = ()

    @property
    def total(self) -> float:
        return sum(self.timings.values())


class Stopwatch:
    """단계별 소요 시간을 재는 도구."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def measure(self, label: str, call):
        start = time.perf_counter()
        value = call()
        self.timings[label] = time.perf_counter() - start
        return value


def build_detector(args):
    """탐지기를 만든다. 모델이 없으면 스텁을 쓴다."""
    if args.stub:
        return StubDetector(), "stub"

    from app.ai.vision.yolo_detector import YoloDetector

    detector = YoloDetector(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        tracking=args.tracking,
    )
    mode = "track" if args.tracking else "predict"
    return detector, f"{Path(args.model).name} ({mode})"


def measure(path: Path, detector, tracker, args) -> Measurement:
    """영상 1개를 처리하며 단계별 시간을 잰다."""
    watch = Stopwatch()

    video = watch.measure(
        "extract", lambda: frame_module.extract(path, args.max_edge)
    )
    rows = watch.measure("detect", lambda: detector.detect(video.frames))
    ratios = watch.measure("occupancy", lambda: frame_ratios(rows))
    objects = watch.measure("track", lambda: adopt(tracker.track(rows)))
    watch.measure(
        "select",
        lambda: select_analysis_frame(
            video.frames, rows, {o.class_code for o in objects}
        ),
    )

    return Measurement(
        name=path.name,
        duration=video.duration,
        frame_count=video.count,
        detections=sum(len(r) for r in rows),
        objects=len(objects),
        occupancy=occupancy_ratio(rows),
        occupancy_spread=(max(ratios) - min(ratios)) if ratios else 0.0,
        timings=watch.timings,
        names=tuple(sorted(to_korean(o.class_code) or o.class_code for o in objects)),
    )


def sweep_iou(path: Path, detector, thresholds: Sequence[float], args) -> dict:
    """임계값별 객체 수를 센다. 탐지는 한 번만 한다."""
    video = frame_module.extract(path, args.max_edge)
    rows = detector.detect(video.frames)
    return {t: len(adopt(IouTracker(threshold=t).track(rows))) for t in thresholds}


def peak_memory_mb() -> float:
    """이 프로세스가 쓴 최대 상주 메모리.

    ``tracemalloc`` 은 파이썬 할당만 세므로 모델 가중치가 잡히지 않는다.
    운영체제가 보는 값을 쓴다. Linux 는 KB, macOS 는 바이트로 보고한다.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform == "linux" else peak / (1024 * 1024)


def find_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def print_measurements(results: Sequence[Measurement]) -> None:
    print("\n== 영상별 ==")
    header = f"{'영상':<24}{'길이':>7}{'프레임':>7}{'탐지':>7}{'객체':>6}{'점유율':>9}{'소요':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name[:23]:<24}{r.duration:>6.1f}s{r.frame_count:>7}"
            f"{r.detections:>7}{r.objects:>6}{r.occupancy:>9.4f}{r.total:>7.2f}s"
        )


def print_stages(results: Sequence[Measurement]) -> None:
    print("\n== 단계별 소요 (영상당 평균) ==")
    labels = ("extract", "detect", "occupancy", "track", "select")
    total = sum(r.total for r in results) / len(results)

    for label in labels:
        values = [r.timings.get(label, 0.0) for r in results]
        mean = statistics.mean(values)
        share = mean / total * 100 if total else 0
        bar = "#" * max(1, round(share / 2)) if mean else ""
        print(f"  {label:<12}{mean:>7.3f}s{share:>7.1f}%  {bar}")

    print(f"  {'합계':<12}{total:>7.3f}s")
    print(
        f"\n  처리 제한 {PROCESSING_TIMEOUT_SECONDS}초의 "
        f"{total / PROCESSING_TIMEOUT_SECONDS * 100:.1f}% 를 쓴다."
    )
    print("  나머지는 환경 분석(12단계) 몫이다. 여기가 실제 병목인지 함께 재야 한다.")


def print_occupancy(results: Sequence[Measurement]) -> None:
    print("\n== 활동 공간 점유율 ==")
    values = sorted(r.occupancy for r in results)
    print(f"  중앙값 범위 : {values[0]:.4f} ~ {values[-1]:.4f}")
    print(f"  현재 임계값 : {' / '.join(str(t) for t, _ in OCCUPANCY_THRESHOLDS)}")

    for threshold, penalty in OCCUPANCY_THRESHOLDS:
        hit = sum(1 for v in values if v >= threshold)
        print(f"    {threshold:.2f} 이상 {hit:>2}/{len(values)}개 영상 (감점률 {penalty})")

    if values[-1] < OCCUPANCY_THRESHOLDS[0][0]:
        print(
            "\n  경고: 모든 영상이 첫 임계값 미만이다. 지금 값으로는 활동 공간"
            " 감점이 절대 발생하지 않는다. 임계값을 낮추거나 근거를 다시 잡아야 한다."
        )


def print_sweep(sweeps: dict[str, dict]) -> None:
    print("\n== IoU 임계값 스윕 (채택 객체 수) ==")
    thresholds = sorted(next(iter(sweeps.values())).keys())

    header = f"{'영상':<24}" + "".join(f"{t:>7.2f}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for name, counts in sweeps.items():
        print(f"{name[:23]:<24}" + "".join(f"{counts[t]:>7}" for t in thresholds))

    totals = [sum(c[t] for c in sweeps.values()) for t in thresholds]
    print(f"{'합계':<24}" + "".join(f"{v:>7}" for v in totals))

    if len(set(totals)) == 1:
        print(
            "\n  전 구간에서 결과가 같다. 이 영상들로는 임계값을 정할 수 없다."
            "\n  카메라가 빠르게 움직이거나 객체가 많은 영상이 필요하다."
        )
    else:
        print(f"\n  현재 설정값은 {TRACKING_IOU_THRESHOLD} 이다.")
        print("  값이 커질수록 같은 물체가 여러 건으로 쪼개진다.")
        print("  정답은 육안으로 센 고유 객체 수와 대조해야 나온다.")


def write_csv(path: Path, results: Sequence[Measurement]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video", "duration", "frames", "detections", "objects",
             "occupancy", "extract", "detect", "occupancy_calc", "track",
             "select", "total", "names"]
        )
        for r in results:
            writer.writerow([
                r.name, f"{r.duration:.2f}", r.frame_count, r.detections, r.objects,
                f"{r.occupancy:.4f}",
                *(f"{r.timings.get(k, 0):.4f}" for k in
                  ("extract", "detect", "occupancy", "track", "select")),
                f"{r.total:.4f}", " ".join(r.names),
            ])
    print(f"\nCSV 저장: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Vision 파이프라인 측정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--videos", type=Path, required=True, help="영상 폴더")
    parser.add_argument("--model", help="가중치 경로")
    parser.add_argument("--stub", action="store_true", help="모델 없이 스텁으로 실행")
    parser.add_argument("--device", help="mps | cuda | cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-edge", type=int, default=1280)
    parser.add_argument(
        "--tracking", action="store_true",
        help="탐지기가 추적까지 수행한다. TrackIdTracker 를 쓴다",
    )
    parser.add_argument(
        "--sweep-iou", help="IoU 임계값 목록. 예: 0.1,0.2,0.3,0.5",
    )
    parser.add_argument("--csv", type=Path, help="결과를 CSV로 저장")

    args = parser.parse_args(argv)
    if not args.stub and not args.model:
        parser.error("--model 또는 --stub 중 하나가 필요하다")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.videos.is_dir():
        print(f"영상 폴더가 없다: {args.videos}")
        return 1

    videos = find_videos(args.videos)
    if not videos:
        print(f"영상이 없다: {args.videos}")
        return 1

    detector, label = build_detector(args)
    tracker = TrackIdTracker() if args.tracking else IouTracker()

    print(f"탐지기 : {label}")
    print(f"추적기 : {type(tracker).__name__}")
    print(f"영상   : {len(videos)}개")

    results = [measure(p, detector, tracker, args) for p in videos]

    print_measurements(results)
    print_stages(results)
    print_occupancy(results)

    if args.sweep_iou:
        thresholds = [float(v) for v in args.sweep_iou.split(",")]
        sweeps = {p.name: sweep_iou(p, detector, thresholds, args) for p in videos}
        print_sweep(sweeps)

    print(f"\nPeak 메모리 : {peak_memory_mb():.0f} MB")

    if args.csv:
        write_csv(args.csv, results)

    print("\n※ 탐지 정확도(mAP)는 여기서 재지 않는다. 정답 라벨이 필요하다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
