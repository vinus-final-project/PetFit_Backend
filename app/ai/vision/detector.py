"""객체 탐지 (3단계).

**모델이 필요한 유일한 단계다.** 여기에 선을 그어 나머지 단계를 모델 없이
구현하고 검증한다. 실제 YOLO 구현은 별도 파일로 추가하며, 이 파일과 이후
단계는 수정하지 않는다.

탐지기가 지켜야 할 규약은 네 가지다.

    1. 반환 목록의 길이는 입력 프레임 수와 같다. 인덱스로 대응시킨다.
    2. ``confidence`` 는 DETECTION_CONFIDENCE 이상만 포함한다.
    3. ``class_code`` 는 탐지 대상 12종에 속하는 것만 포함한다.
    4. 좌표는 정규화 값이며 프레임 경계를 벗어나지 않는다.

3번이 중요하다. COCO 사전학습 모델은 사람·TV·화분도 함께 내놓는데, 이것이
통과하면 **활동 공간 점유율에 섞인다.** 점유율은 4단계이고 오탐 필터는 6단계라
필터로는 막을 수 없다.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.ai.vision.types import Detection, Frame
from app.core.constants import DETECTION_CONFIDENCE
from app.rules.object_map import is_known
from app.utils.geometry import BoundingBox

__all__ = ["Detector", "PlantedObject", "StubDetector", "DEFAULT_SCENE", "clamp_box"]


@runtime_checkable
class Detector(Protocol):
    """프레임에서 객체를 탐지한다.

    **동기 메서드다.** 추론은 CPU·GPU를 점유하는 블로킹 작업이므로 ``async`` 로
    선언하면 구현체가 비동기인 척만 하고 이벤트 루프는 그대로 막힌다. 호출하는
    쪽이 ``asyncio.to_thread`` 로 감싼다.
    """

    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """프레임 목록 전체를 한 번에 탐지한다.

        프레임 1장씩이 아니라 목록으로 받는다. 배치 추론이 장당 추론보다 빠르고,
        호출 횟수만큼 발생하는 전처리·후처리 오버헤드가 사라진다.

        Args:
            frames: 추출된 프레임. 순서를 유지한다.

        Returns:
            프레임별 탐지 결과. ``result[i]`` 가 ``frames[i]`` 에 대응하며,
            아무것도 탐지되지 않은 프레임은 빈 목록이다.
        """
        ...


def clamp_box(x: float, y: float, width: float, height: float) -> BoundingBox:
    """박스를 프레임 안쪽으로 자른다.

    탐지 모델의 박스는 프레임 경계를 넘을 수 있다. 그대로 두면 점유율 계산에서
    프레임 밖 면적이 더해져 1.0을 넘는다.

    Args:
        x, y: 좌측 상단 좌표. 정규화 값.
        width, height: 크기. 정규화 값.

    Returns:
        경계 안으로 잘린 박스. 프레임과 겹치지 않으면 크기가 0이다.
    """
    # 이미 안쪽이면 값을 그대로 돌려준다. 뺄셈을 거치면 0.4 가 0.4000000000000001 이
    # 되어, 자를 필요가 없는 박스까지 부동소수점 오차를 얻는다.
    if 0.0 <= x and 0.0 <= y and x + width <= 1.0 and y + height <= 1.0:
        return BoundingBox(x, y, width, height)

    left = min(max(x, 0.0), 1.0)
    top = min(max(y, 0.0), 1.0)
    right = min(max(x + width, 0.0), 1.0)
    bottom = min(max(y + height, 0.0), 1.0)
    return BoundingBox(left, top, max(right - left, 0.0), max(bottom - top, 0.0))


@dataclass(frozen=True)
class PlantedObject:
    """가짜 탐지기에 심어놓는 객체 1개.

    프레임마다 결과를 일일이 나열하면 테스트가 읽히지 않는다. **어느 프레임에
    나타나는지**만 선언하고 나머지는 탐지기가 만든다.

    Attributes:
        class_code: 클래스 코드. 매핑표에 없는 값을 넣으면 탐지기가 버린다.
        box: 기준 Bounding Box. 정규화 좌표.
        frames: 등장하는 프레임 번호.
        confidence: 기본 confidence.
        confidence_by_frame: 특정 프레임의 confidence 를 덮어쓴다.
            추적이 최댓값 프레임을 대표로 고르는지 시험할 때 쓴다.
        jitter: 프레임마다 박스를 흔드는 폭.
    """

    class_code: str
    box: BoundingBox
    frames: Sequence[int]
    confidence: float = 0.90
    confidence_by_frame: Mapping[int, float] = field(default_factory=dict)
    jitter: float = 0.0

    def confidence_at(self, frame_number: int) -> float:
        """해당 프레임의 confidence."""
        return self.confidence_by_frame.get(frame_number, self.confidence)

    def box_at(self, frame_number: int) -> BoundingBox:
        """해당 프레임의 Bounding Box.

        난수를 쓰지 않는다. 프레임 번호로 흔들림이 결정되므로 테스트는 몇 번을
        돌려도 같은 결과를 얻는다.

        실제 탐지 박스는 프레임마다 미세하게 떨린다. 흔들리지 않는 박스로만
        시험하면 추적을 시험한 것이 아니다.

        경계 처리는 하지 않는다. 규약을 지키는 것은 탐지기의 책임이며, 여기서도
        자르면 두 곳이 같은 일을 한다.
        """
        if not self.jitter:
            return self.box

        # -1, 0, +1 을 순환시킨다. 대칭이라 평균 위치가 기준 박스와 같다.
        shift = ((frame_number % 3) - 1) * self.jitter
        return BoundingBox(
            self.box.x + shift, self.box.y - shift, self.box.width, self.box.height
        )


#: 전 구간 검증에 쓰는 기준 장면.
#:
#: ``app.ai.stub.StubPipeline`` 과 같은 5개 객체다. Vision 파이프라인이 이 장면을
#: 처리하면 소형견·거실 기준 종합 56점이 나와야 한다. AI 분석 정의서의 계산
#: 예시와 같은 값이므로, 파이프라인 전체가 규칙과 어긋나지 않았는지 확인할 수 있다.
DEFAULT_SCENE: tuple[PlantedObject, ...] = (
    PlantedObject("cable", BoundingBox(0.1250, 0.7400, 0.2000, 0.0800),
                  frames=range(0, 12), confidence=0.94, jitter=0.004),
    PlantedObject("window", BoundingBox(0.6000, 0.1000, 0.2500, 0.4000),
                  frames=range(14, 23), confidence=0.91, jitter=0.004),
    PlantedObject("carpet", BoundingBox(0.2000, 0.5500, 0.5000, 0.3000),
                  frames=range(0, 15), confidence=0.89, jitter=0.004),
    PlantedObject("sofa", BoundingBox(0.4000, 0.3500, 0.3500, 0.2500),
                  frames=range(0, 21), confidence=0.98, jitter=0.004),
    PlantedObject("water_dispenser", BoundingBox(0.8000, 0.6500, 0.1000, 0.1200),
                  frames=range(20, 24), confidence=0.86, jitter=0.004),
)


class StubDetector:
    """심어놓은 객체를 그대로 돌려주는 탐지기.

    실제 모델 없이 4~10단계 전 구간을 실행하고 검증하기 위한 구현이다.
    규약은 실제 탐지기와 동일하게 지킨다. 규약을 지키지 않으면 이 탐지기로
    통과한 코드가 실제 모델에서 깨진다.
    """

    def __init__(
        self,
        objects: Iterable[PlantedObject] = DEFAULT_SCENE,
        threshold: float = DETECTION_CONFIDENCE,
    ) -> None:
        """
        Args:
            objects: 심어놓을 객체.
            threshold: 프레임 단위 탐지 임계값. 실제 모델의 ``conf`` 인자와 같다.
        """
        self._objects = tuple(objects)
        self._threshold = threshold
        #: 호출 횟수. 파이프라인이 탐지를 한 번만 부르는지 확인하는 데 쓴다.
        self.call_count = 0

    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """규약을 지켜 탐지 결과를 만든다."""
        self.call_count += 1

        planted_by_frame: dict[int, list[Detection]] = {f.number: [] for f in frames}

        for planted in self._objects:
            # 매핑표에 없는 코드는 여기서 버린다. 통과시키면 점유율에 섞인다.
            if not is_known(planted.class_code):
                continue

            for number in planted.frames:
                bucket = planted_by_frame.get(number)
                if bucket is None:
                    # 심어놓은 프레임이 추출 범위 밖일 수 있다. 조용히 넘긴다.
                    continue

                confidence = planted.confidence_at(number)
                if confidence < self._threshold:
                    continue

                # 좌표 규약은 여기서 보장한다. 심어놓은 박스가 프레임을 넘더라도
                # 밖으로 나가지 않는다. 실제 모델의 박스도 경계를 넘는다.
                box = clamp_box(*planted.box_at(number))
                if box.area <= 0.0:
                    continue

                bucket.append(
                    Detection(
                        class_code=planted.class_code,
                        confidence=confidence,
                        frame_number=number,
                        x=box.x,
                        y=box.y,
                        width=box.width,
                        height=box.height,
                    )
                )

        return [planted_by_frame[f.number] for f in frames]
