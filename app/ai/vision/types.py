"""Vision 단계 사이를 오가는 자료형.

단계마다 딕셔너리를 주고받으면 오타가 실행 시점까지 드러나지 않는다.
값의 모양을 여기서 못박는다.

좌표는 **모두 정규화 값(0.0~1.0)** 이다. 프레임의 가로·세로를 각각 1.0으로 보며
원점은 좌측 상단이다. 탐지 직후에 정규화하고, 픽셀로 되돌리는 곳은 마킹뿐이다.
중간에 두 좌표계를 섞으면 점유율이 프레임 해상도에 따라 달라진다.

최종 객체 목록은 ``app.ai.pipeline.DetectedObject`` 를 그대로 사용한다.
대응하는 타입을 여기 또 만들면 필드가 열 개인 변환 함수가 생기고, 계약이 바뀔 때
두 곳을 고쳐야 한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL.Image import Image

from app.ai.pipeline import DetectedObject
from app.utils.geometry import BoundingBox

__all__ = [
    "Frame",
    "Detection",
    "TrackedObject",
    "VisionResult",
    "ImageSink",
    "ImageStore",
]


@runtime_checkable
class ImageSink(Protocol):
    """이미지를 저장할 수 있는 대상.

    AI 계층이 서비스 계층의 ``Storage`` 를 직접 임포트하지 않기 위한 계약이다.
    임포트하면 의존 방향이 뒤집혀, 저장소를 고칠 때마다 파이프라인이 영향을 받는다.

    Vision 이 저장소에서 필요로 하는 것은 이 메서드 하나뿐이다.
    """

    def save_image(self, image: Image) -> str:
        """이미지를 저장하고 DB 저장용 상대 경로를 돌려준다."""
        ...


@runtime_checkable
class ImageStore(ImageSink, Protocol):
    """저장과 삭제가 모두 필요한 대상.

    Vision 이 이미지를 만든 뒤 환경 분석에서 실패하면, 만든 파일이 DB 참조 없이
    디스크에 남는다. 실패 경로에서 지우려면 삭제까지 필요하다.
    """

    def delete(self, *relative_paths: "str | None") -> int:
        """상대 경로로 지정한 파일을 삭제하고 삭제된 수를 돌려준다."""
        ...


@dataclass(frozen=True, eq=False)
class Frame:
    """추출된 프레임 1장.

    ``eq=False`` 인 이유가 둘이다.

    PIL 이미지는 ``__eq__`` 를 정의하면서 ``__hash__`` 는 정의하지 않는다.
    ``frozen=True`` 가 자동 생성하는 ``__hash__`` 가 이미지를 해싱하려 들면
    실행 중에 실패한다.

    비교가 성립하더라도 프레임 두 장을 대조하려고 1280x720 픽셀을 전부 읽게 된다.
    프레임은 동일성으로만 구분한다.

    Attributes:
        number: 추출 순서. **1부터 시작한다.** 영상의 원본 프레임 번호가 아니다.
            DB의 ``ck_detected_object_frame_number`` 가 1 이상을 요구하므로
            0부터 매기면 저장 단계에서 제약에 걸린다. DB 명세서도 "0은 존재할
            수 없는 값" 으로 정의한다.
        timestamp: 영상 시작점 기준 시각(초).
        image: 축소된 프레임 이미지. RGB.
    """

    number: int
    timestamp: float
    image: Image

    @property
    def width(self) -> int:
        """저장된 이미지의 가로 픽셀. 원본이 아니라 축소 후 값이다."""
        return self.image.width

    @property
    def height(self) -> int:
        """저장된 이미지의 세로 픽셀. 원본이 아니라 축소 후 값이다."""
        return self.image.height


@dataclass(frozen=True)
class Detection:
    """프레임 1장에서 탐지된 객체 1건.

    같은 프레임에 같은 클래스가 여러 번 나올 수 있다. 전선이 두 곳에 있으면 2건이다.

    ``class_code`` 는 **항상 탐지 대상 12종 중 하나다.** 매핑표에 없는 코드는
    탐지기가 이미 버렸다. 점유율(4단계)이 필터(6단계)보다 먼저 계산되므로,
    사람이나 TV를 여기까지 들여보내면 점유율에 섞여 활동성 점수가 깎인다.

    한글 변환은 아직 하지 않는다. 추적·필터에서 탈락할 객체를 변환하는 것은
    낭비이므로 채택이 확정된 뒤에 수행한다.
    """

    class_code: str
    confidence: float
    frame_number: int
    x: float
    y: float
    width: float
    height: float

    #: 추적 ID. 탐지기가 추적까지 함께 수행한 경우에만 채워진다.
    #:
    #: ultralytics 의 BoT-SORT 는 ``model.track()`` 안에서 탐지와 추적을 함께
    #: 한다. 결과의 ID 를 여기 실어 두면 추론을 한 번만 하고도 추적 결과를 쓸 수
    #: 있다. 따로 추적하면 같은 프레임을 두 번 추론하게 된다.
    #:
    #: 추적기가 확신하지 못한 탐지에는 ID 가 붙지 않으므로 None 일 수 있다.
    track_id: int | None = None

    @property
    def box(self) -> BoundingBox:
        """점유율 계산에 넘길 형태. ``union_area()`` 가 그대로 받는다."""
        return BoundingBox(self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class TrackedObject:
    """여러 프레임의 탐지를 하나로 통합한 객체.

    ``Detection`` 이 "프레임 7번에서 소파를 봤다" 라면 이쪽은 "이 영상에 소파가
    하나 있다" 이다.

    산출 방식은 AI 설계서를 따른다.

        confidence            : 탐지된 프레임 중 최댓값
        frame_number          : confidence 가 가장 높은 프레임
        detection_frame_count : 탐지된 프레임 수

    ``frame_number`` 가 그대로 객체 대표 프레임이 된다. 9단계에서 다시 계산하지
    않는다. 좌표도 그 프레임에서 관측된 값이다.
    """

    class_code: str
    confidence: float
    detection_frame_count: int
    frame_number: int
    x: float
    y: float
    width: float
    height: float

    @property
    def box(self) -> BoundingBox:
        """대표 프레임에서의 Bounding Box."""
        return BoundingBox(self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class VisionResult:
    """Vision 파이프라인의 산출물 전체.

    부분 성공은 정의하지 않는다. 이 객체가 반환되면 1~10단계가 모두 성공한 것이다.

    Attributes:
        capture_duration: 영상 길이(초).
        frame_count: 추출한 프레임 수. 15 이상 30 이하.
        occupancy_ratio: 프레임별 활동 공간 점유율의 중앙값. 0.0 이상 1.0 이하.
        thumbnail_path: 분석 대표 프레임의 저장 경로.
        detected_objects: 필터를 통과한 객체. 위험도와 마킹 경로가 채워져 있다.
        analysis_frames: 환경 분석에 넘길 원본 프레임.
    """

    capture_duration: float
    frame_count: int
    occupancy_ratio: float
    thumbnail_path: str

    #: 분석 대표 프레임의 번호.
    #:
    #: ``analysis_frames[0].number`` 와 같지만 명시해 둔다. 12단계가 이 값을
    #: 입력으로 받는데, "목록의 첫 번째가 대표" 라는 규칙을 주석으로만 두면
    #: 목록 구성이 바뀌었을 때 조용히 다른 프레임을 가리키게 된다.
    thumbnail_frame: int = 0

    detected_objects: Sequence[DetectedObject] = field(default_factory=tuple)

    #: 환경 분석(12단계) 입력용 원본 프레임. **마킹하지 않은 것이어야 한다.**
    #: 분석 대표 1장 + 위험 객체 대표 최대 3장으로 ``LLM_MAX_IMAGES`` 를 넘지 않는다.
    #: 여기에 담지 않으면 담당자가 영상을 다시 열어 프레임을 추출해야 한다.
    analysis_frames: Sequence[Frame] = field(default_factory=tuple)

    @property
    def object_names(self) -> set[str]:
        """점수 산출에 넘길 객체 이름 집합.

        Score Generator 는 인스턴스 수가 아니라 존재 여부로 감점을 판정한다.
        """
        return {o.name for o in self.detected_objects}

    @property
    def marked_image_paths(self) -> list[str]:
        """생성된 이미지 경로 전체. 썸네일을 포함한다.

        Vision 이 끝난 뒤 환경 분석에서 실패하면 여기서 만든 이미지가 참조 없이
        남는다. 정리하려면 경로를 알아야 한다.
        """
        paths = [self.thumbnail_path] if self.thumbnail_path else []
        paths += [o.marked_image_path for o in self.detected_objects
                  if o.marked_image_path]
        return paths
