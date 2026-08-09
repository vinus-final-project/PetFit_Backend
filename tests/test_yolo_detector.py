"""YOLO 탐지기 변환 검증.

**ultralytics 없이 돌아간다.** 실제 추론은 Mac Studio 에서 확인하고, 여기서는
모델 출력을 ``Detection`` 으로 옮기는 부분만 본다.

버그가 실제로 생기는 곳이 여기다. 좌표계(xyxy vs xywh), 정규화 여부, 클래스
색인에서 이름 찾기, 별칭 처리. 추론 자체는 라이브러리가 하므로 우리가 틀릴 수
있는 부분이 아니다.
"""

import pytest
from PIL import Image

from app.ai.vision.types import Frame
from app.ai.vision.yolo_detector import (
    BATCH_SIZE,
    DEFAULT_TRACKER,
    MISSING_PACKAGE,
    YoloDetector,
    to_class_code,
    to_detections,
)
from app.core.constants import DETECTION_CONFIDENCE

#: COCO 클래스 색인 일부. 실제 이름을 그대로 쓴다.
COCO_NAMES = {
    0: "person",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    62: "tv",
    58: "potted plant",
}


class _Column(list):
    """텐서 흉내. ultralytics 는 torch 텐서를 돌려주며 tolist() 를 지원한다."""

    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, coords, confidences, classes):
        self.xyxyn = _Column(coords)
        self.conf = _Column(confidences)
        self.cls = _Column(classes)

    def __len__(self):
        return len(self.conf)


class _Result:
    """ultralytics Results 흉내."""

    def __init__(self, coords=(), confidences=(), classes=(), names=None):
        self.names = COCO_NAMES if names is None else names
        self.boxes = _Boxes(list(coords), list(confidences), list(classes))


class _Model:
    """predict 호출을 기록하는 모델 흉내."""

    def __init__(self, results_by_call=None):
        self.calls = []
        self._results = results_by_call

    def predict(self, images, conf=None, device=None, verbose=None):
        self.calls.append({"count": len(images), "conf": conf, "device": device})
        if self._results is not None:
            return self._results[len(self.calls) - 1]
        return [_Result() for _ in images]


def detector(model, **kwargs) -> YoloDetector:
    """모델 적재를 건너뛰고 가짜 모델을 끼운다."""
    instance = YoloDetector.__new__(YoloDetector)
    instance._model = model
    instance._device = kwargs.get("device")
    instance._threshold = kwargs.get("threshold", DETECTION_CONFIDENCE)
    instance._batch_size = kwargs.get("batch_size", BATCH_SIZE)
    instance._tracking = kwargs.get("tracking", False)
    instance._tracker = kwargs.get("tracker", DEFAULT_TRACKER)
    return instance


def frames(count: int) -> list[Frame]:
    image = Image.new("RGB", (16, 9))
    return [Frame(i, i * 0.33, image) for i in range(count)]


class TestClassCode:
    def test_coco_couch_becomes_sofa(self) -> None:
        assert to_class_code("couch") == "sofa"

    def test_coco_dining_table_becomes_table(self) -> None:
        assert to_class_code("dining table") == "table"

    def test_case_and_spacing_do_not_matter(self) -> None:
        """표기 차이가 남으면 테이블이 통째로 누락된다."""
        assert to_class_code("Dining Table") == "table"
        assert to_class_code("  COUCH  ") == "sofa"
        assert to_class_code("dining-table") == "table"

    def test_custom_names_pass_through(self) -> None:
        """커스텀 학습본은 우리 이름을 그대로 낸다."""
        for code in ("cable", "cat_tower", "water_dispenser", "carpet"):
            assert to_class_code(code) == code

    def test_unmapped_name_is_normalized_only(self) -> None:
        assert to_class_code("Potted Plant") == "potted_plant"


class TestToDetections:
    def test_empty_result(self) -> None:
        assert to_detections(_Result(), 0) == []

    def test_result_without_boxes(self) -> None:
        class Bare:
            names = {}
            boxes = None

        assert to_detections(Bare(), 0) == []

    def test_converts_xyxy_to_xywh(self) -> None:
        """모델은 좌상단·우하단을 주고 우리는 위치와 크기를 쓴다."""
        result = _Result([[0.2, 0.3, 0.6, 0.8]], [0.9], [57])
        box = to_detections(result, 0)[0]

        assert (box.x, box.y) == pytest.approx((0.2, 0.3))
        assert (box.width, box.height) == pytest.approx((0.4, 0.5))

    def test_maps_class_index_to_code(self) -> None:
        result = _Result([[0.1, 0.1, 0.2, 0.2]], [0.9], [57])
        assert to_detections(result, 0)[0].class_code == "sofa"

    def test_keeps_frame_number(self) -> None:
        result = _Result([[0.1, 0.1, 0.2, 0.2]], [0.9], [57])
        assert to_detections(result, 17)[0].frame_number == 17

    def test_drops_below_threshold(self) -> None:
        result = _Result(
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], [0.24, 0.26], [57, 59]
        )
        codes = [d.class_code for d in to_detections(result, 0)]
        assert codes == ["bed"]

    def test_drops_unmapped_classes(self) -> None:
        """사람·TV·화분이 통과하면 활동 공간 점유율에 섞인다."""
        result = _Result(
            [[0.0, 0.0, 0.1, 0.1]] * 4,
            [0.9] * 4,
            [0, 62, 58, 57],       # person, tv, potted plant, couch
        )
        codes = [d.class_code for d in to_detections(result, 0)]
        assert codes == ["sofa"]

    def test_unknown_class_index_is_skipped(self, caplog) -> None:
        result = _Result([[0.1, 0.1, 0.2, 0.2]], [0.9], [999])
        assert to_detections(result, 0) == []

    def test_clamps_to_the_frame(self) -> None:
        """모델 박스는 프레임 경계를 넘을 수 있다. 넘으면 점유율이 1.0을 넘는다."""
        result = _Result([[-0.1, -0.2, 1.3, 1.4]], [0.9], [57])
        box = to_detections(result, 0)[0]

        assert (box.x, box.y) == (0.0, 0.0)
        assert box.x + box.width <= 1.0
        assert box.y + box.height <= 1.0

    def test_zero_area_box_is_dropped(self) -> None:
        result = _Result([[0.5, 0.5, 0.5, 0.5]], [0.9], [57])
        assert to_detections(result, 0) == []

    def test_fully_outside_box_is_dropped(self) -> None:
        result = _Result([[1.2, 1.2, 1.5, 1.5]], [0.9], [57])
        assert to_detections(result, 0) == []

    def test_confidence_is_a_float(self) -> None:
        """텐서 스칼라가 그대로 들어가면 DB 저장과 비교가 어긋난다."""
        result = _Result([[0.1, 0.1, 0.2, 0.2]], [0.87], [57])
        assert isinstance(to_detections(result, 0)[0].confidence, float)

    def test_multiple_instances_of_one_class(self) -> None:
        """전선이 두 곳에 있으면 2건이다."""
        result = _Result(
            [[0.0, 0.0, 0.2, 0.2], [0.6, 0.6, 0.8, 0.8]], [0.9, 0.8], [56, 56]
        )
        assert len(to_detections(result, 0)) == 2

    def test_custom_weights_names(self) -> None:
        """커스텀 학습본 클래스도 그대로 통과해야 한다."""
        names = {0: "cable", 1: "cat_tower", 2: "water_dispenser"}
        result = _Result(
            [[0.1, 0.1, 0.2, 0.2]] * 3, [0.9] * 3, [0, 1, 2], names=names
        )
        codes = {d.class_code for d in to_detections(result, 0)}
        assert codes == {"cable", "cat_tower", "water_dispenser"}


class TestDetect:
    def test_returns_one_row_per_frame(self) -> None:
        given = frames(20)
        assert len(detector(_Model()).detect(given)) == len(given)

    def test_no_frames(self) -> None:
        assert detector(_Model()).detect([]) == []

    def test_splits_into_batches(self) -> None:
        """30장을 한꺼번에 넘기면 메모리 사용이 크게 튄다."""
        model = _Model()
        detector(model, batch_size=8).detect(frames(20))

        assert [c["count"] for c in model.calls] == [8, 8, 4]

    def test_single_batch_when_it_fits(self) -> None:
        model = _Model()
        detector(model, batch_size=32).detect(frames(20))
        assert len(model.calls) == 1

    def test_passes_threshold_to_the_model(self) -> None:
        """모델 쪽에서 걸러야 후처리할 박스 수가 줄어든다."""
        model = _Model()
        detector(model, threshold=0.25).detect(frames(3))
        assert model.calls[0]["conf"] == 0.25

    def test_passes_device(self) -> None:
        model = _Model()
        detector(model, device="mps").detect(frames(3))
        assert model.calls[0]["device"] == "mps"

    def test_frame_numbers_are_preserved_across_batches(self) -> None:
        """배치로 나눠도 프레임 번호가 밀리면 마킹 배경이 어긋난다."""
        box = [[0.1, 0.1, 0.2, 0.2]]
        results = [
            [_Result(box, [0.9], [57]) for _ in range(2)],
            [_Result(box, [0.9], [57]) for _ in range(2)],
            [_Result(box, [0.9], [57])],
        ]
        rows = detector(_Model(results), batch_size=2).detect(frames(5))

        assert [row[0].frame_number for row in rows] == [0, 1, 2, 3, 4]

    def test_satisfies_the_detector_protocol(self) -> None:
        from app.ai.vision.detector import Detector

        assert isinstance(detector(_Model()), Detector)


class TestTrackingMode:
    """추적을 함께 수행하는 모드.

    켜면 추론 한 번으로 탐지와 추적을 모두 얻는다. 따로 추적하면 같은 프레임을
    두 번 추론하게 되어 처리 시간이 두 배가 된다.
    """

    class _Tracking(_Model):
        def __init__(self, ids=None):
            super().__init__()
            self.track_calls = []
            self._ids = ids

        def track(self, image, conf=None, device=None, tracker=None,
                  persist=None, verbose=None):
            self.track_calls.append(
                {"tracker": tracker, "persist": persist, "device": device}
            )
            box = [[0.1, 0.1, 0.2, 0.2]]
            result = _Result(box, [0.9], [57])
            if self._ids is not None:
                result.boxes.id = _Column([self._ids])
            return [result]

    def test_uses_track_not_predict(self) -> None:
        model = self._Tracking()
        detector(model, tracking=True).detect(frames(5))

        assert len(model.track_calls) == 5
        assert model.calls == []

    def test_persists_state_between_frames(self) -> None:
        """추적기는 프레임을 순서대로 받아 상태를 이어가야 한다.

        persist 가 없으면 호출마다 상태가 초기화되어 ID 가 매 프레임 새로 발급된다.
        """
        model = self._Tracking()
        detector(model, tracking=True).detect(frames(3))

        assert all(c["persist"] is True for c in model.track_calls)

    def test_one_frame_per_call(self) -> None:
        """여러 장을 한 번에 넘기면 앞뒤 관계가 사라진다."""
        model = self._Tracking()
        detector(model, tracking=True, batch_size=8).detect(frames(6))

        assert len(model.track_calls) == 6

    def test_passes_the_tracker_config(self) -> None:
        model = self._Tracking()
        detector(model, tracking=True, tracker="bytetrack.yaml").detect(frames(2))

        assert model.track_calls[0]["tracker"] == "bytetrack.yaml"

    def test_default_tracker_is_botsort(self) -> None:
        model = self._Tracking()
        detector(model, tracking=True).detect(frames(1))

        assert model.track_calls[0]["tracker"] == DEFAULT_TRACKER

    def test_track_id_reaches_the_detection(self) -> None:
        model = self._Tracking(ids=42)
        rows = detector(model, tracking=True).detect(frames(3))

        assert all(row[0].track_id == 42 for row in rows)

    def test_missing_id_is_none(self) -> None:
        """추적기가 확신하지 못한 탐지에는 ID 가 붙지 않는다."""
        model = self._Tracking(ids=None)
        rows = detector(model, tracking=True).detect(frames(3))

        assert all(row[0].track_id is None for row in rows)

    def test_predict_mode_has_no_ids(self) -> None:
        rows = detector(_Model()).detect(frames(3))
        assert all(d.track_id is None for row in rows for d in row)


class TestMissingPackage:
    def test_message_tells_how_to_install(self) -> None:
        """설치되지 않은 환경에서 원인을 알 수 있어야 한다."""
        assert "requirements-ai.txt" in MISSING_PACKAGE

    def test_module_imports_without_ultralytics(self) -> None:
        """API 계층과 테스트는 torch 없이 돌아가야 한다."""
        import importlib

        module = importlib.import_module("app.ai.vision.yolo_detector")
        assert module.to_class_code("couch") == "sofa"
