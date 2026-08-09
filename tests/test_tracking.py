"""객체 추적과 탐지 신뢰 기준 검증.

여기서 틀리면 증상이 두 방향으로 나타난다.

    미합침 : 소파 하나가 응답에 11건으로 나열된다
    과합침 : 나란히 놓인 의자 두 개가 하나가 된다

점수는 존재 여부로 판정하므로 둘 다 점수를 바꾸지 않는다. 대신 사용자가 보는
목록과 마킹 이미지 수가 달라진다.

``detection_frame_count`` 는 오탐 필터의 근거이므로 부풀려지면 안 된다.
"""

import pytest

from app.ai.vision.tracking import IouTracker, Tracker, adopt
from app.ai.vision.types import Detection, TrackedObject
from app.core.constants import ADOPTION_CONFIDENCE, MIN_DETECTION_FRAMES
from app.utils.geometry import BoundingBox, intersection_area, iou


def det(frame: int, x: float, y: float, w: float = 0.2, h: float = 0.2,
        conf: float = 0.9, code: str = "sofa") -> Detection:
    return Detection(code, conf, frame, x, y, w, h)


def obj(conf: float = 0.9, frames: int = 5, code: str = "sofa") -> TrackedObject:
    return TrackedObject(code, conf, frames, 0, 0.1, 0.1, 0.2, 0.2)


class TestIou:
    def test_identical_boxes(self) -> None:
        box = BoundingBox(0.1, 0.1, 0.4, 0.4)
        assert iou(box, box) == 1.0

    def test_disjoint_boxes(self) -> None:
        assert iou(BoundingBox(0, 0, 0.2, 0.2), BoundingBox(0.5, 0.5, 0.2, 0.2)) == 0.0

    def test_touching_edges_do_not_overlap(self) -> None:
        assert iou(BoundingBox(0, 0, 0.2, 0.2), BoundingBox(0.2, 0, 0.2, 0.2)) == 0.0

    def test_half_overlap(self) -> None:
        a = BoundingBox(0.0, 0.0, 0.2, 0.2)
        b = BoundingBox(0.1, 0.0, 0.2, 0.2)
        # 교집합 0.02, 합집합 0.06
        assert iou(a, b) == pytest.approx(1 / 3)

    def test_contained_box(self) -> None:
        outer = BoundingBox(0.0, 0.0, 0.4, 0.4)
        inner = BoundingBox(0.1, 0.1, 0.2, 0.2)
        assert iou(outer, inner) == pytest.approx(0.04 / 0.16)

    def test_zero_area_box(self) -> None:
        assert iou(BoundingBox(0, 0, 0, 0), BoundingBox(0, 0, 0.2, 0.2)) == 0.0

    def test_intersection_area(self) -> None:
        a = BoundingBox(0.0, 0.0, 0.4, 0.4)
        b = BoundingBox(0.2, 0.2, 0.4, 0.4)
        assert intersection_area(a, b) == pytest.approx(0.04)

    def test_is_symmetric(self) -> None:
        a = BoundingBox(0.1, 0.2, 0.3, 0.4)
        b = BoundingBox(0.2, 0.1, 0.4, 0.3)
        assert iou(a, b) == iou(b, a)


class TestMerging:
    def test_same_object_across_frames_becomes_one(self) -> None:
        rows = [[det(i, 0.3, 0.3)] for i in range(5)]
        result = IouTracker().track(rows)

        assert len(result) == 1
        assert result[0].detection_frame_count == 5

    def test_wobbling_box_still_merges(self) -> None:
        """실제 탐지 박스는 프레임마다 미세하게 떨린다."""
        rows = [[det(i, 0.30 + i * 0.004, 0.30 - i * 0.004)] for i in range(10)]
        result = IouTracker().track(rows)

        assert len(result) == 1
        assert result[0].detection_frame_count == 10

    def test_distant_objects_stay_separate(self) -> None:
        rows = [[det(i, 0.05, 0.05), det(i, 0.7, 0.7)] for i in range(5)]
        result = IouTracker().track(rows)

        assert len(result) == 2
        assert all(o.detection_frame_count == 5 for o in result)

    def test_different_classes_never_merge(self) -> None:
        """겹쳐 있어도 소파와 카펫은 같은 물체가 아니다."""
        rows = [[det(i, 0.3, 0.3, code="sofa"), det(i, 0.3, 0.3, code="carpet")]
                for i in range(5)]
        result = IouTracker().track(rows)

        assert len(result) == 2
        assert {o.class_code for o in result} == {"sofa", "carpet"}

    def test_one_track_takes_one_detection_per_frame(self) -> None:
        """이 제약이 없으면 나란히 놓인 의자 두 개가 하나로 합쳐진다."""
        rows = [[det(i, 0.30, 0.3), det(i, 0.38, 0.3)] for i in range(5)]
        result = IouTracker().track(rows)

        assert len(result) == 2

    def test_object_moving_across_the_frame(self) -> None:
        """카메라가 방을 훑으면 물체가 화면을 가로지른다.

        최근 박스가 아니라 첫 박스와 견주면 몇 프레임 만에 겹침이 사라져
        하나의 소파가 여러 건으로 쪼개진다.
        """
        rows = [[det(i, 0.05 + i * 0.05, 0.3)] for i in range(10)]
        result = IouTracker().track(rows)

        assert len(result) == 1
        assert result[0].detection_frame_count == 10

    def test_reappearing_object_is_rejoined(self) -> None:
        """카메라가 다른 곳을 비추다 돌아온 경우다."""
        rows = [[det(0, 0.3, 0.3)], [], [], [], [det(4, 0.3, 0.3)]]
        result = IouTracker().track(rows)

        assert len(result) == 1
        assert result[0].detection_frame_count == 2

    def test_empty_input(self) -> None:
        assert IouTracker().track([]) == []

    def test_all_frames_empty(self) -> None:
        assert IouTracker().track([[], [], []]) == []

    def test_satisfies_protocol(self) -> None:
        assert isinstance(IouTracker(), Tracker)


class TestRepresentativeValues:
    def test_confidence_is_the_maximum(self) -> None:
        rows = [[det(0, 0.3, 0.3, conf=0.61)],
                [det(1, 0.3, 0.3, conf=0.95)],
                [det(2, 0.3, 0.3, conf=0.72)]]
        assert IouTracker().track(rows)[0].confidence == 0.95

    def test_frame_number_is_where_confidence_peaked(self) -> None:
        rows = [[det(0, 0.3, 0.3, conf=0.61)],
                [det(1, 0.3, 0.3, conf=0.95)],
                [det(2, 0.3, 0.3, conf=0.72)]]
        assert IouTracker().track(rows)[0].frame_number == 1

    def test_box_comes_from_the_representative_frame(self) -> None:
        """좌표는 대표 프레임에서 관측된 값이어야 마킹이 맞는 자리에 그려진다."""
        rows = [[det(0, 0.30, 0.30, conf=0.5)],
                [det(1, 0.34, 0.32, w=0.25, conf=0.9)]]
        result = IouTracker().track(rows)[0]

        assert (result.x, result.y, result.width) == (0.34, 0.32, 0.25)

    def test_ties_broken_by_area_then_frame(self) -> None:
        """동점이면 면적이 큰 쪽, 그래도 같으면 앞 프레임이다."""
        rows = [[det(0, 0.30, 0.30, w=0.20, h=0.20, conf=0.9)],
                [det(1, 0.30, 0.30, w=0.30, h=0.30, conf=0.9)],
                [det(2, 0.30, 0.30, w=0.30, h=0.30, conf=0.9)]]
        assert IouTracker().track(rows)[0].frame_number == 1

    def test_frame_count_counts_frames_not_detections(self) -> None:
        """같은 프레임에서 두 번 잡혀도 1프레임이다.

        그대로 세면 탐지 프레임 수가 부풀려져 오탐 필터가 무력해진다.
        """
        rows = [[det(0, 0.30, 0.30), det(0, 0.31, 0.30)], [det(1, 0.30, 0.30)]]
        result = IouTracker().track(rows)

        assert sum(o.detection_frame_count for o in result) <= 3
        assert all(o.detection_frame_count <= 2 for o in result)


class TestThreshold:
    def test_high_threshold_splits_more(self) -> None:
        rows = [[det(i, 0.30 + i * 0.05, 0.3)] for i in range(6)]

        loose = IouTracker(threshold=0.10).track(rows)
        strict = IouTracker(threshold=0.90).track(rows)

        assert len(loose) < len(strict)

    def test_threshold_boundary_is_inclusive(self) -> None:
        """기준값과 정확히 같으면 같은 물체로 본다."""
        a = BoundingBox(0.0, 0.0, 0.2, 0.2)
        b = BoundingBox(0.1, 0.0, 0.2, 0.2)
        exact = iou(a, b)   # 1/3

        rows = [[det(0, a.x, a.y)], [det(1, b.x, b.y)]]
        assert len(IouTracker(threshold=exact).track(rows)) == 1


class TestAdopt:
    def test_low_confidence_is_dropped(self) -> None:
        assert adopt([obj(conf=0.39)]) == []

    def test_confidence_boundary_is_inclusive(self) -> None:
        assert len(adopt([obj(conf=ADOPTION_CONFIDENCE)])) == 1

    def test_single_frame_object_is_dropped(self) -> None:
        """30프레임 중 1장에서만 잡힌 물체는 오탐일 가능성이 높다."""
        assert adopt([obj(frames=1)]) == []

    def test_frame_count_boundary_is_inclusive(self) -> None:
        assert len(adopt([obj(frames=MIN_DETECTION_FRAMES)])) == 1

    def test_both_conditions_must_hold(self) -> None:
        assert adopt([obj(conf=0.95, frames=1)]) == []
        assert adopt([obj(conf=0.30, frames=20)]) == []

    def test_low_confidence_but_many_frames_is_still_dropped(self) -> None:
        """전선처럼 confidence 가 낮은 물체를 살리려면 임계값 자체를 낮춰야 한다.

        프레임 수가 많다고 임계값을 면제하지 않는다.
        """
        assert adopt([obj(conf=0.35, frames=30, code="cable")]) == []

    def test_floating_point_boundary(self) -> None:
        """0.05 + 0.35 는 0.39999999999999997 이다. 판정이 흔들리면 안 된다."""
        assert len(adopt([obj(conf=0.05 + 0.35)])) == 1

    def test_order_is_kept(self) -> None:
        objects = [obj(code="sofa"), obj(code="cable"), obj(code="carpet")]
        assert [o.class_code for o in adopt(objects)] == ["sofa", "cable", "carpet"]

    def test_empty_input(self) -> None:
        assert adopt([]) == []


class TestWithDetector:
    """기준 장면을 끝까지 통과시킨다."""

    def _rows(self):
        from PIL import Image

        from app.ai.vision.detector import StubDetector
        from app.ai.vision.types import Frame

        frames = [Frame(i, i * 0.33, Image.new("RGB", (16, 9))) for i in range(25)]
        return StubDetector().detect(frames)

    def test_five_objects_survive(self) -> None:
        adopted = adopt(IouTracker().track(self._rows()))
        assert {o.class_code for o in adopted} == {
            "cable", "window", "carpet", "sofa", "water_dispenser"
        }

    def test_no_duplicates(self) -> None:
        """흔들리는 박스 때문에 하나의 물체가 여러 건으로 쪼개지면 안 된다."""
        adopted = adopt(IouTracker().track(self._rows()))
        assert len(adopted) == 5

    def test_frame_counts_match_the_scene(self) -> None:
        adopted = adopt(IouTracker().track(self._rows()))
        counts = {o.class_code: o.detection_frame_count for o in adopted}

        assert counts == {
            "cable": 12, "carpet": 15, "sofa": 21, "window": 9, "water_dispenser": 4
        }

    def test_confidences_match_the_scene(self) -> None:
        adopted = adopt(IouTracker().track(self._rows()))
        confs = {o.class_code: o.confidence for o in adopted}

        assert confs == {
            "cable": 0.94, "window": 0.91, "carpet": 0.89,
            "sofa": 0.98, "water_dispenser": 0.86,
        }
