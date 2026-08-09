"""탐지기 규약 검증.

여기서 시험하는 것은 가짜 탐지기가 아니라 **규약 자체**다. 실제 YOLO 구현도
같은 조건을 지켜야 하므로, 이 파일의 검증은 구현체가 바뀌어도 그대로 쓴다.

규약을 지키지 않은 탐지기를 통과시키면 증상이 엉뚱한 곳에서 나타난다.
매핑표 밖 클래스가 새어 나가면 활동성 점수가 깎이고, 좌표가 프레임을 넘으면
점유율이 1.0을 초과한다. 원인을 탐지기까지 되짚기 어렵다.
"""

import pytest
from PIL import Image

from app.ai.vision.detector import (
    DEFAULT_SCENE,
    Detector,
    PlantedObject,
    StubDetector,
    clamp_box,
)
from app.ai.vision.types import Frame
from app.core.constants import DETECTION_CONFIDENCE
from app.rules.object_map import is_known
from app.utils.geometry import BoundingBox

BOX = BoundingBox(0.2, 0.2, 0.3, 0.3)


def frames(count: int = 25) -> list[Frame]:
    """검증용 프레임. 이미지 내용은 쓰지 않으므로 작게 만든다.

    **번호는 1부터다.** 추출기가 그렇게 매긴다. 0부터 만들면 실제와 다른 조건에서
    시험하게 되어, 기준 장면의 탐지 프레임 수가 어긋난다.
    """
    image = Image.new("RGB", (16, 9))
    return [
        Frame(number=i, timestamp=i * 0.33, image=image)
        for i in range(1, count + 1)
    ]


class TestContract:
    """구현체가 바뀌어도 유지되어야 하는 조건."""

    def test_returns_one_list_per_frame(self) -> None:
        """인덱스로 프레임과 대응시키므로 길이가 같아야 한다."""
        given = frames(25)
        result = StubDetector().detect(given)
        assert len(result) == len(given)

    def test_detection_belongs_to_its_frame(self) -> None:
        """result[i] 안의 탐지는 전부 frames[i] 에서 나온 것이어야 한다."""
        given = frames(25)
        for frame, detections in zip(given, StubDetector().detect(given)):
            assert all(d.frame_number == frame.number for d in detections)

    def test_empty_frame_yields_empty_list(self) -> None:
        """아무것도 없는 프레임은 None 이 아니라 빈 목록이다."""
        result = StubDetector([]).detect(frames(3))
        assert result == [[], [], []]

    def test_drops_unknown_class_code(self) -> None:
        """매핑표에 없는 클래스는 탐지기가 버린다.

        통과시키면 활동 공간 점유율에 섞인다. 점유율은 4단계이고 오탐 필터는
        6단계라 뒤에서 막을 수 없다.
        """
        detector = StubDetector([
            PlantedObject("person", BOX, frames=range(5)),
            PlantedObject("tv", BOX, frames=range(5)),
            PlantedObject("sofa", BOX, frames=range(5)),
        ])

        codes = {d.class_code for row in detector.detect(frames(5)) for d in row}
        assert codes == {"sofa"}

    def test_all_codes_are_known(self) -> None:
        """기준 장면 전체가 매핑표 안에 있어야 한다."""
        detector = StubDetector()
        for row in detector.detect(frames(25)):
            assert all(is_known(d.class_code) for d in row)

    def test_drops_below_threshold(self) -> None:
        """프레임 단위 임계값 미만은 포함하지 않는다."""
        detector = StubDetector([
            PlantedObject("sofa", BOX, frames=range(5), confidence=0.24),
            PlantedObject("bed", BOX, frames=range(5), confidence=0.26),
        ])

        codes = {d.class_code for row in detector.detect(frames(5)) for d in row}
        assert codes == {"bed"}

    def test_threshold_is_the_documented_value(self) -> None:
        """기본 임계값이 상수와 같아야 한다. 실제 모델의 conf 인자와 동일한 값이다."""
        detector = StubDetector([
            PlantedObject("sofa", BOX, frames=[1], confidence=DETECTION_CONFIDENCE),
        ])
        assert len(detector.detect(frames(1))[0]) == 1

    def test_coordinates_stay_inside_frame(self) -> None:
        """좌표가 프레임을 벗어나면 점유율이 1.0을 넘는다."""
        detector = StubDetector([
            PlantedObject("sofa", BoundingBox(0.9, 0.9, 0.5, 0.5), frames=range(5)),
        ])

        for row in detector.detect(frames(5)):
            for d in row:
                assert 0.0 <= d.x and 0.0 <= d.y
                assert d.x + d.width <= 1.0
                assert d.y + d.height <= 1.0

    def test_satisfies_protocol(self) -> None:
        assert isinstance(StubDetector(), Detector)


class TestClampBox:
    def test_inside_box_is_unchanged(self) -> None:
        assert clamp_box(0.2, 0.3, 0.4, 0.1) == BoundingBox(0.2, 0.3, 0.4, 0.1)

    def test_overflowing_box_is_cut(self) -> None:
        assert clamp_box(0.8, 0.8, 0.5, 0.5) == pytest.approx(
            BoundingBox(0.8, 0.8, 0.2, 0.2)
        )

    def test_negative_origin_is_cut(self) -> None:
        assert clamp_box(-0.2, -0.1, 0.5, 0.5) == pytest.approx(
            BoundingBox(0.0, 0.0, 0.3, 0.4)
        )

    def test_fully_outside_box_has_no_area(self) -> None:
        assert clamp_box(1.4, 1.4, 0.2, 0.2).area == 0.0

    def test_full_frame_box_is_kept(self) -> None:
        assert clamp_box(0.0, 0.0, 1.0, 1.0).area == 1.0


class TestPlantedObject:
    def test_jitter_is_deterministic(self) -> None:
        """난수를 쓰지 않는다. 같은 프레임이면 항상 같은 박스다."""
        planted = PlantedObject("sofa", BOX, frames=range(9), jitter=0.01)
        assert planted.box_at(4) == planted.box_at(4)
        assert planted.box_at(4) == planted.box_at(7)

    def test_jitter_moves_the_box(self) -> None:
        """흔들리지 않으면 추적을 시험한 것이 아니다."""
        planted = PlantedObject("sofa", BOX, frames=range(9), jitter=0.01)
        assert planted.box_at(0) != planted.box_at(1)

    def test_no_jitter_keeps_the_box(self) -> None:
        planted = PlantedObject("sofa", BOX, frames=range(9))
        assert planted.box_at(5) == BOX

    def test_confidence_by_frame_overrides(self) -> None:
        """추적이 최댓값 프레임을 대표로 고르는지 시험하려면 프레임별 값이 필요하다."""
        planted = PlantedObject(
            "sofa", BOX, frames=range(5), confidence=0.5,
            confidence_by_frame={3: 0.97},
        )
        assert planted.confidence_at(0) == 0.5
        assert planted.confidence_at(3) == 0.97

    def test_frames_outside_range_are_ignored(self) -> None:
        """추출 범위 밖에 심어도 예외가 나지 않는다."""
        detector = StubDetector([PlantedObject("sofa", BOX, frames=[1, 99])])
        result = detector.detect(frames(3))
        assert len(result) == 3
        assert len(result[0]) == 1


class TestDefaultScene:
    """기준 장면. 전 구간 검증의 출발점이다."""

    def test_contains_five_objects(self) -> None:
        assert len(DEFAULT_SCENE) == 5

    def test_matches_stub_pipeline_objects(self) -> None:
        """StubPipeline 과 같은 구성이어야 56점 재현을 대조할 수 있다."""
        codes = {p.class_code for p in DEFAULT_SCENE}
        assert codes == {"cable", "window", "carpet", "sofa", "water_dispenser"}

    def test_every_object_appears_in_multiple_frames(self) -> None:
        """탐지 프레임 수가 1이면 오탐 필터에서 전부 탈락한다."""
        assert all(len(list(p.frames)) >= 2 for p in DEFAULT_SCENE)

    def test_detects_all_five_over_the_video(self) -> None:
        given = frames(25)
        codes = {d.class_code for row in StubDetector().detect(given) for d in row}
        assert len(codes) == 5

    def test_objects_are_spread_across_frames(self) -> None:
        """전 객체가 한 프레임에 몰려 있으면 대표 프레임 선정을 시험할 수 없다."""
        rows = StubDetector().detect(frames(25))
        kinds = [len({d.class_code for d in row}) for row in rows]
        assert max(kinds) < 5
