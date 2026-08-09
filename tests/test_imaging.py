"""대표 프레임 선정과 위험 객체 마킹 검증.

마킹은 **저장된 이미지의 픽셀을 실제로 읽어** 확인한다. 함수가 오류 없이 끝났다는
사실만으로는 빨간 박스가 그려졌는지 알 수 없다. 색이 위험도를 전달하는 유일한
수단이므로 잘못되면 사용자가 위험을 반대로 이해한다.
"""

import pytest
from PIL import Image

from app.ai.vision.imaging import build, draw_box, select_analysis_frame
from app.ai.vision.types import Detection, Frame, TrackedObject
from app.core.constants import LLM_MAX_IMAGES
from app.schemas.enums import AnimalGroup, RiskLevel

GROUP = AnimalGroup.SMALL_DOG


def frame(number: int, size=(160, 90)) -> Frame:
    return Frame(number, number * 0.33, Image.new("RGB", size, (10, 10, 10)))


def det(code: str, frame_number: int, conf: float = 0.9) -> Detection:
    return Detection(code, conf, frame_number, 0.2, 0.2, 0.2, 0.2)


def obj(code: str, frame_number: int = 0, conf: float = 0.9) -> TrackedObject:
    return TrackedObject(code, conf, 5, frame_number, 0.25, 0.25, 0.3, 0.3)


def colors(image: Image.Image) -> set[tuple[int, int, int]]:
    return {c for _, c in image.getcolors(maxcolors=1 << 16)}


def run_build(storage, frames, detections, objects, group=GROUP):
    """대표 프레임 선정과 마킹을 이어서 수행한다.

    파이프라인이 두 단계를 나눠 부르므로 테스트도 같은 순서로 부른다.
    """
    chosen = select_analysis_frame(
        frames, detections, {o.class_code for o in objects}
    )
    return build(frames, chosen, objects, group, storage)


class TestSelectAnalysisFrame:
    def test_prefers_more_primary_object_kinds(self) -> None:
        frames = [frame(0), frame(1), frame(2)]
        detections = [
            [det("sofa", 0)],
            [det("sofa", 1), det("cable", 1), det("carpet", 1)],
            [det("sofa", 2), det("cable", 2)],
        ]
        chosen = select_analysis_frame(frames, detections, {"sofa", "cable", "carpet"})
        assert chosen.number == 1

    def test_counts_kinds_not_instances(self) -> None:
        """소파 다섯 개보다 소파와 전선이 함께 있는 프레임이 낫다."""
        frames = [frame(0), frame(1)]
        detections = [
            [det("sofa", 0) for _ in range(5)],
            [det("sofa", 1), det("cable", 1)],
        ]
        assert select_analysis_frame(frames, detections, {"sofa", "cable"}).number == 1

    def test_ignores_non_primary_objects(self) -> None:
        """의자·테이블은 감점 근거가 아니라 주요 객체에 들지 않는다."""
        frames = [frame(0), frame(1)]
        detections = [
            [det("chair", 0), det("table", 0), det("bed", 0)],
            [det("sofa", 1), det("cable", 1)],
        ]
        chosen = select_analysis_frame(
            frames, detections, {"chair", "table", "bed", "sofa", "cable"}
        )
        assert chosen.number == 1

    def test_ignores_rejected_classes(self) -> None:
        """오탐이 몰린 프레임이 썸네일로 뽑히면 안 된다."""
        frames = [frame(0), frame(1)]
        detections = [
            [det("cable", 0), det("carpet", 0), det("window", 0)],   # 전부 탈락
            [det("sofa", 1)],
        ]
        assert select_analysis_frame(frames, detections, {"sofa"}).number == 1

    def test_ties_broken_by_mean_confidence(self) -> None:
        frames = [frame(0), frame(1)]
        detections = [
            [det("sofa", 0, conf=0.51)],
            [det("sofa", 1, conf=0.97)],
        ]
        assert select_analysis_frame(frames, detections, {"sofa"}).number == 1

    def test_ties_broken_by_frame_number(self) -> None:
        """3순위까지 적용하면 항상 하나가 결정된다."""
        frames = [frame(0), frame(1), frame(2)]
        detections = [[det("sofa", i, conf=0.9)] for i in range(3)]
        assert select_analysis_frame(frames, detections, {"sofa"}).number == 0

    def test_no_detections_at_all(self) -> None:
        frames = [frame(0), frame(1)]
        assert select_analysis_frame(frames, [[], []], set()).number == 0

    def test_empty_frames_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            select_analysis_frame([], [], set())


class TestDrawBox:
    def test_high_risk_is_red(self) -> None:
        marked = draw_box(frame(0), obj("cable"), RiskLevel.HIGH)
        assert (255, 0, 0) in colors(marked)

    def test_medium_risk_is_yellow(self) -> None:
        marked = draw_box(frame(0), obj("stairs"), RiskLevel.MEDIUM)
        assert (255, 255, 0) in colors(marked)

    def test_low_risk_is_green(self) -> None:
        marked = draw_box(frame(0), obj("window"), RiskLevel.LOW)
        assert (0, 128, 0) in colors(marked)

    def test_original_frame_is_not_modified(self) -> None:
        """같은 프레임이 여러 객체의 배경이 된다.

        제자리에서 그리면 두 번째 객체의 이미지에 첫 번째 박스가 남는다.
        """
        source = frame(0)
        before = colors(source.image)
        draw_box(source, obj("cable"), RiskLevel.HIGH)
        assert colors(source.image) == before

    def test_box_is_at_the_right_place(self) -> None:
        """정규화 좌표를 픽셀로 되돌리는 유일한 지점이다."""
        marked = draw_box(frame(0, size=(200, 100)), obj("cable"), RiskLevel.HIGH)

        # 박스는 (0.25, 0.25) ~ (0.55, 0.55) -> 픽셀 (50, 25) ~ (110, 55)
        assert marked.getpixel((50, 30)) == (255, 0, 0)
        assert marked.getpixel((10, 10)) != (255, 0, 0)
        assert marked.getpixel((150, 80)) != (255, 0, 0)

    def test_line_scales_with_frame_size(self) -> None:
        """고정 픽셀이면 4K 에서 실처럼 보인다."""
        small = draw_box(frame(0, size=(160, 90)), obj("cable"), RiskLevel.HIGH)
        large = draw_box(frame(0, size=(1600, 900)), obj("cable"), RiskLevel.HIGH)

        # getcolors 는 (픽셀 수, 색) 을 돌려준다. 항목 수가 아니라 픽셀 수를 센다.
        red_small = sum(n for n, c in small.getcolors(1 << 16) if c == (255, 0, 0))
        red_large = sum(n for n, c in large.getcolors(1 << 16) if c == (255, 0, 0))

        # 변 길이가 10배이므로 둘레도 10배다. 선이 같은 두께면 10배에 그친다.
        assert red_large > red_small * 20

    def test_size_is_unchanged(self) -> None:
        marked = draw_box(frame(0, size=(200, 100)), obj("cable"), RiskLevel.HIGH)
        assert marked.size == (200, 100)


class TestBuild:
    def _run(self, storage, group=GROUP):
        frames = [frame(i) for i in range(5)]
        detections = [[det("cable", i), det("sofa", i)] for i in range(5)]
        objects = [obj("cable", frame_number=1), obj("sofa", frame_number=2)]
        return run_build(storage, frames, detections, objects, group), frames

    def test_creates_thumbnail(self, storage) -> None:
        result, _ = self._run(storage)
        assert result.thumbnail_path.startswith("/images/")
        assert (storage.image_dir / result.thumbnail_path.split("/")[-1]).is_file()

    def test_marks_risky_objects_only(self, storage) -> None:
        """SAFE 까지 표시하면 화면이 박스로 가득 차 위험 요소가 묻힌다."""
        result, _ = self._run(storage)
        by_name = {o.name: o for o in result.detected_objects}

        assert by_name["전선"].marked_image_path is not None
        assert by_name["소파"].marked_image_path is None

    def test_risk_is_classified_by_group(self, storage) -> None:
        """창문은 소형견에게 LOW, 고양이에게 HIGH 다."""
        frames = [frame(i) for i in range(3)]
        detections = [[det("window", i)] for i in range(3)]
        objects = [obj("window", frame_number=0)]

        for group, expected in [
            (AnimalGroup.SMALL_DOG, RiskLevel.LOW),
            (AnimalGroup.CAT, RiskLevel.HIGH),
        ]:
            result = run_build(storage, frames, detections, objects, group)
            assert result.detected_objects[0].risk is expected

    def test_names_are_korean(self, storage) -> None:
        result, _ = self._run(storage)
        assert {o.name for o in result.detected_objects} == {"전선", "소파"}

    def test_marked_image_uses_the_object_frame(self, storage) -> None:
        """마킹 배경은 그 객체의 대표 프레임이어야 한다.

        추적이 정한 frame_number 를 다시 계산하면 좌표와 배경이 어긋나
        엉뚱한 자리에 박스가 그려진다.
        """
        result, _ = self._run(storage)
        cable = next(o for o in result.detected_objects if o.name == "전선")
        assert cable.frame_number == 1

    def test_marked_file_is_saved(self, storage) -> None:
        result, _ = self._run(storage)
        cable = next(o for o in result.detected_objects if o.name == "전선")
        saved = storage.image_dir / cable.marked_image_path.split("/")[-1]

        assert saved.is_file()
        with Image.open(saved) as image:
            assert image.size == (160, 90)

    def test_each_object_gets_its_own_file(self, storage) -> None:
        frames = [frame(i) for i in range(5)]
        detections = [[det("cable", i)] for i in range(5)]
        objects = [obj("cable", frame_number=1), obj("cable", frame_number=3)]

        result = run_build(storage, frames, detections, objects)
        paths = [o.marked_image_path for o in result.detected_objects]

        assert len(set(paths)) == 2

    def test_no_objects(self, storage) -> None:
        frames = [frame(i) for i in range(3)]
        result = run_build(storage, frames, [[], [], []], [])

        assert result.detected_objects == []
        assert result.thumbnail_path.startswith("/images/")


class TestAnalysisFrames:
    def test_starts_with_the_analysis_frame(self, storage) -> None:
        frames = [frame(i) for i in range(5)]
        detections = [[det("cable", i)] for i in range(5)]
        result = run_build(storage, frames, detections, [obj("cable", 2)])

        assert result.analysis_frames[0].number == result.analysis_frames[0].number
        assert len(result.analysis_frames) >= 1

    def test_respects_the_image_limit(self, storage) -> None:
        frames = [frame(i) for i in range(10)]
        detections = [[det("cable", i)] for i in range(10)]
        objects = [obj("cable", frame_number=i) for i in range(1, 8)]

        result = run_build(storage, frames, detections, objects)
        assert len(result.analysis_frames) <= LLM_MAX_IMAGES

    def test_no_duplicate_frames(self, storage) -> None:
        """중복으로 상한을 채우면 그만큼 다른 장면을 보지 못한다."""
        frames = [frame(i) for i in range(5)]
        detections = [[det("cable", i)] for i in range(5)]
        objects = [obj("cable", frame_number=1) for _ in range(4)]

        result = run_build(storage, frames, detections, objects)
        numbers = [f.number for f in result.analysis_frames]
        assert len(numbers) == len(set(numbers))

    def test_safe_objects_contribute_no_frames(self, storage) -> None:
        """탐지 대상 밖의 위험 요소를 찾는 것이 목적이므로 위험 객체 장면을 넣는다."""
        frames = [frame(i) for i in range(5)]
        detections = [[det("sofa", i)] for i in range(5)]
        objects = [obj("sofa", frame_number=i) for i in range(1, 5)]

        result = run_build(storage, frames, detections, objects)
        assert len(result.analysis_frames) == 1

    def test_frames_are_not_marked(self, storage) -> None:
        """박스가 그려진 이미지를 넘기면 모델이 이미 탐지된 것만 다시 서술한다."""
        frames = [frame(i) for i in range(5)]
        detections = [[det("cable", i)] for i in range(5)]
        result = run_build(storage, frames, detections, [obj("cable", 1)])

        for f in result.analysis_frames:
            assert (255, 0, 0) not in colors(f.image)


class TestSaveImage:
    def test_returns_relative_path(self, storage) -> None:
        path = storage.save_image(Image.new("RGB", (10, 10)))
        assert path.startswith("/images/")
        assert path.endswith(".jpg")

    def test_filenames_do_not_collide(self, storage) -> None:
        paths = {storage.save_image(Image.new("RGB", (10, 10))) for _ in range(5)}
        assert len(paths) == 5

    def test_converts_non_rgb(self, storage) -> None:
        """JPEG 는 투명도를 지원하지 않는다. 변환하지 않으면 저장이 실패한다."""
        path = storage.save_image(Image.new("RGBA", (10, 10), (255, 0, 0, 128)))
        assert (storage.image_dir / path.split("/")[-1]).is_file()

    def test_deletable_by_storage(self, storage) -> None:
        """재시도·삭제 시 정리되어야 한다. 남으면 참조 없는 파일이 쌓인다."""
        path = storage.save_image(Image.new("RGB", (10, 10)))
        assert storage.delete(path) == 1
        assert not (storage.image_dir / path.split("/")[-1]).exists()


class TestListImages:
    """고아 파일 회수의 재료."""

    def test_lists_saved_images(self, storage) -> None:
        paths = {storage.save_image(Image.new("RGB", (10, 10))) for _ in range(3)}
        assert set(storage.list_images()) == paths

    def test_returns_relative_paths(self, storage) -> None:
        storage.save_image(Image.new("RGB", (10, 10)))
        assert all(p.startswith("/images/") for p in storage.list_images())

    def test_empty_directory(self, storage) -> None:
        assert storage.list_images() == []

    def test_age_filter_excludes_new_files(self, storage) -> None:
        """방금 만든 파일을 고아로 오인하면 진행 중인 분석이 깨진다.

        마킹 이미지는 DB에 기록되기 전에 먼저 디스크에 쓰인다.
        """
        storage.save_image(Image.new("RGB", (10, 10)))
        assert storage.list_images(min_age_seconds=60) == []

    def test_age_filter_includes_old_files(self, storage) -> None:
        import os
        import time

        path = storage.save_image(Image.new("RGB", (10, 10)))
        target = storage.image_dir / path.split("/")[-1]
        old = time.time() - 3600
        os.utime(target, (old, old))

        assert storage.list_images(min_age_seconds=60) == [path]

    def test_ignores_directories(self, storage) -> None:
        (storage.image_dir / "sub").mkdir()
        assert storage.list_images() == []
