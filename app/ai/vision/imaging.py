"""대표 프레임 선정과 위험 객체 마킹 (9·10단계).

**대표 프레임은 두 종류다.**

    분석 대표 프레임 : 분석당 1장. 목록 화면의 썸네일
    객체 대표 프레임 : 위험 객체당 1장. 마킹 이미지의 배경

객체 대표 프레임은 추적이 이미 정했다. ``TrackedObject.frame_number`` 를 그대로
쓰며 다시 계산하지 않는다. 두 곳에서 따로 정하면 마킹 좌표와 배경 프레임이
어긋나 엉뚱한 자리에 박스가 그려진다.

마킹은 **객체 1개당 이미지 1장**이고, 그 이미지에는 그 객체의 박스 하나만
그린다. 객체마다 배경 프레임이 다르기 때문이다. ``SAFE`` 는 생성하지 않는다.
전부 표시하면 화면이 박스로 가득 차 위험 요소가 묻힌다.
"""

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from PIL import ImageDraw
from PIL.Image import Image

from app.ai.pipeline import DetectedObject
from app.ai.vision.types import Detection, Frame, ImageSink, TrackedObject
from app.core.constants import LLM_MAX_IMAGES
from app.rules.object_map import PRIMARY_OBJECTS, to_korean
from app.rules.risk_rules import classify
from app.schemas.enums import AnimalGroup, RiskLevel

__all__ = ["Visuals", "select_analysis_frame", "draw_box", "build"]

logger = logging.getLogger(__name__)

#: 마킹 선 두께를 프레임 짧은 변에 대한 비율로 정한다.
#: 고정 픽셀로 두면 720p 에서 적당한 두께가 4K 에서는 실처럼 보인다.
LINE_RATIO = 0.006
LINE_MIN = 2


@dataclass(frozen=True)
class Visuals:
    """시각화 산출물.

    Attributes:
        thumbnail_path: 분석 대표 프레임의 저장 경로.
        detected_objects: 위험도와 마킹 경로가 채워진 객체 목록.
        analysis_frames: 환경 분석에 넘길 원본 프레임. 마킹하지 않은 것이다.
    """

    thumbnail_path: str
    detected_objects: list[DetectedObject]
    analysis_frames: list[Frame]


def select_analysis_frame(
    frames: Sequence[Frame],
    detections: Sequence[Sequence[Detection]],
    adopted: Collection[str],
) -> Frame:
    """분석 대표 프레임을 고른다.

    AI 설계서의 우선순위를 그대로 적용한다.

        1. 포함된 주요 객체의 종류 수가 많을수록
        2. 그 프레임 탐지 객체의 평균 confidence 가 높을수록
        3. 프레임 번호가 작을수록

    3순위까지 적용하면 프레임 번호가 유일하므로 항상 하나가 결정된다.

    채택되지 않은 클래스는 세지 않는다. 그러지 않으면 오탐이 몰린 프레임이
    "객체가 많은 프레임" 으로 뽑혀 썸네일이 된다.

    Args:
        frames: 추출된 프레임. 비어 있으면 안 된다.
        detections: 프레임별 탐지 결과. ``frames`` 와 인덱스가 대응한다.
        adopted: 오탐 필터를 통과한 클래스 코드.

    Returns:
        분석 대표 프레임.

    Raises:
        ValueError: 프레임이 없는 경우. 추출 단계에서 이미 걸러진다.
    """
    if not frames:
        raise ValueError("대표 프레임을 고를 프레임이 없다")

    keep = set(adopted)
    best_index = max(
        range(len(frames)),
        key=lambda i: _rank_of(detections[i] if i < len(detections) else (), keep, i),
    )
    return frames[best_index]


def draw_box(frame: Frame, obj: TrackedObject, risk: RiskLevel) -> Image:
    """프레임 위에 객체의 Bounding Box 하나를 그린다.

    원본을 수정하지 않는다. 같은 프레임이 여러 객체의 배경으로 쓰이므로,
    제자리에서 그리면 두 번째 객체의 이미지에 첫 번째 박스가 남는다.

    Args:
        frame: 배경이 될 객체 대표 프레임.
        obj: 그릴 객체. 좌표는 정규화 값이다.
        risk: 위험 수준. 색상을 결정한다.

    Returns:
        박스가 그려진 새 이미지.
    """
    canvas = frame.image.copy()
    width, height = canvas.size

    # 정규화 좌표를 픽셀로 되돌리는 유일한 지점이다.
    left = obj.x * width
    top = obj.y * height
    right = (obj.x + obj.width) * width
    bottom = (obj.y + obj.height) * height

    line = max(LINE_MIN, round(min(width, height) * LINE_RATIO))
    ImageDraw.Draw(canvas).rectangle(
        (left, top, right, bottom), outline=risk.marking_color, width=line
    )
    return canvas


def build(
    frames: Sequence[Frame],
    analysis_frame: Frame,
    objects: Sequence[TrackedObject],
    group: AnimalGroup,
    storage: ImageSink,
) -> Visuals:
    """위험도를 판정하고 이미지를 만들어 저장한다 (8·10단계).

    대표 프레임 선정(9단계)은 ``select_analysis_frame()`` 이 이미 끝냈다.
    두 단계를 한 함수에 두면 사용자에게 표시하는 진행 단계를 나눌 수 없다.
    마킹은 이미지를 여러 장 인코딩하므로 선정보다 훨씬 오래 걸린다.

    디스크에 쓰는 블로킹 호출이다. 호출하는 쪽이 스레드에서 실행한다.

    Args:
        frames: 추출된 프레임.
        analysis_frame: 선정된 분석 대표 프레임.
        objects: 오탐 필터를 통과한 객체.
        group: 반려동물 그룹. 위험도 판정에 쓴다.
        storage: 이미지 저장소.

    Returns:
        썸네일 경로, 마킹까지 끝난 객체 목록, 환경 분석용 원본 프레임.
    """
    by_number = {f.number: f for f in frames}
    thumbnail_path = storage.save_image(analysis_frame.image)

    detected: list[DetectedObject] = []
    for obj in _in_display_order(objects, group):
        name = to_korean(obj.class_code)
        if name is None:
            # 탐지기가 규약을 지키면 도달하지 않는다.
            logger.warning("매핑표에 없는 클래스가 통과했다: %s", obj.class_code)
            continue

        risk = classify(name, group)
        frame = by_number.get(obj.frame_number)

        marked_path = None
        if risk is not RiskLevel.SAFE and frame is not None:
            marked_path = storage.save_image(draw_box(frame, obj, risk))

        detected.append(
            DetectedObject(
                name=name,
                risk=risk,
                confidence=obj.confidence,
                detection_frame_count=obj.detection_frame_count,
                frame_number=obj.frame_number,
                x=obj.x,
                y=obj.y,
                width=obj.width,
                height=obj.height,
                marked_image_path=marked_path,
            )
        )

    return Visuals(
        thumbnail_path=thumbnail_path,
        detected_objects=detected,
        analysis_frames=_frames_for_analysis(analysis_frame, detected, by_number),
    )


def _rank_of(
    row: Sequence[Detection], adopted: Collection[str], index: int
) -> tuple[int, float, int]:
    """분석 대표 프레임 선정 기준을 정렬 가능한 값으로 만든다.

    프레임 번호는 작을수록 우선이므로 부호를 뒤집는다.
    """
    kept = [d for d in row if d.class_code in adopted]

    primary = {
        code
        for code in (d.class_code for d in kept)
        if to_korean(code) in PRIMARY_OBJECTS
    }
    mean_confidence = sum(d.confidence for d in kept) / len(kept) if kept else 0.0

    return len(primary), mean_confidence, -index


def _in_display_order(
    objects: Sequence[TrackedObject], group: AnimalGroup
) -> list[TrackedObject]:
    """위험한 것부터 나열한다.

    응답의 최종 정렬은 스키마 계층이 맡는다. 여기서 순서를 정하는 이유는
    환경 분석에 넘길 프레임 3장을 **위험한 객체부터** 고르기 위해서다.
    """
    def key(obj: TrackedObject) -> tuple[int, float, int]:
        name = to_korean(obj.class_code) or ""
        return -classify(name, group).rank, -obj.confidence, obj.frame_number

    return sorted(objects, key=key)


def _frames_for_analysis(
    analysis_frame: Frame,
    detected: Sequence[DetectedObject],
    by_number: dict[int, Frame],
) -> list[Frame]:
    """환경 분석에 넘길 원본 프레임을 고른다.

    분석 대표 1장에 위험 객체 대표를 더해 최대 ``LLM_MAX_IMAGES`` 장이다.
    **마킹하지 않은 원본이어야 한다.** 박스가 그려진 이미지를 넣으면 모델이
    이미 탐지된 객체를 다시 서술하게 되어, 탐지 대상 밖의 위험 요소를 찾는다는
    목적이 사라진다.

    같은 프레임을 두 번 넣지 않는다. 장수 상한을 중복으로 채우면 그만큼 다른
    장면을 보지 못한다.
    """
    picked = [analysis_frame]
    seen = {analysis_frame.number}

    for obj in detected:
        if len(picked) >= LLM_MAX_IMAGES:
            break
        if obj.risk is RiskLevel.SAFE or obj.frame_number in seen:
            continue

        frame = by_number.get(obj.frame_number)
        if frame is None:
            continue

        picked.append(frame)
        seen.add(obj.frame_number)

    return picked
