"""YOLO 객체 탐지 (3단계 실제 구현).

``Detector`` 규약을 ultralytics 로 구현한다. **이 파일만 추가하면 실제 모델이
붙는다.** 나머지 Vision 파일은 수정하지 않는다.

ultralytics 를 **모듈 최상단에서 임포트하지 않는다.** torch 를 함께 끌어와
1~2GB 를 차지하므로, 설치하지 않은 환경에서도 다른 코드가 동작해야 한다.
API 계층·서비스 계층·테스트는 모두 이 파일 없이 돌아간다.

변환과 추론을 나눈 이유가 있다. 좌표계·클래스 색인·정규화 여부처럼 **버그가
실제로 생기는 곳은 변환**이고, 그쪽은 모델 없이 검증할 수 있다.
"""

import logging
from collections.abc import Iterator, Sequence

from app.ai.vision.detector import clamp_box
from app.ai.vision.types import Detection, Frame
from app.core.constants import DETECTION_CONFIDENCE
from app.rules.object_map import is_known

__all__ = [
    "YoloDetector",
    "to_detections",
    "to_class_code",
    "CLASS_ALIASES",
    "DEFAULT_TRACKER",
]

logger = logging.getLogger(__name__)

#: 한 번에 추론할 프레임 수.
#:
#: 30장을 한꺼번에 넘기면 1280x720 기준으로 메모리 사용이 크게 튄다. Mac Studio
#: 한 대에서 동시 2건에 VLM 까지 함께 돌리므로 여유를 둔다.
BATCH_SIZE = 8

#: 추적기 설정 파일. ultralytics 가 함께 배포한다.
#:
#: 1차 성능평가에서 BoT-SORT 를 잠정 선정했으나 중복 제거 정확도가 게이트에
#: 미달했다. ``bytetrack.yaml`` 로 바꿔 재측정할 수 있다.
DEFAULT_TRACKER = "botsort.yaml"

#: 모델이 쓰는 클래스명 -> 탐지 대상 코드.
#:
#: COCO 사전학습 가중치는 우리 코드와 다른 이름을 쓴다. 커스텀 학습본은 우리
#: 이름을 그대로 내므로 이 표를 거쳐도 값이 바뀌지 않는다.
CLASS_ALIASES: dict[str, str] = {
    "couch": "sofa",
    "dining_table": "table",
}

#: ultralytics 미설치 안내.
MISSING_PACKAGE = (
    "ultralytics 가 설치되어 있지 않다. "
    "pip install -r requirements-ai.txt 로 설치한다."
)


def to_class_code(raw: str) -> str:
    """모델이 내놓은 클래스명을 탐지 대상 코드로 바꾼다.

    표기 차이를 먼저 없앤 뒤 별칭을 적용한다. ``"Dining Table"`` 과
    ``"dining table"`` 이 다른 클래스로 취급되면 테이블이 통째로 누락된다.

    Args:
        raw: 모델의 클래스명.

    Returns:
        탐지 대상 코드. 대응하는 것이 없으면 정규화된 이름을 그대로 돌려준다.

    Examples:
        >>> to_class_code("dining table")
        'table'
        >>> to_class_code("Couch")
        'sofa'
        >>> to_class_code("cat_tower")
        'cat_tower'
    """
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(key, key)


def to_detections(
    result, frame_number: int, threshold: float = DETECTION_CONFIDENCE
) -> list[Detection]:
    """ultralytics 결과를 ``Detection`` 목록으로 바꾼다.

    규약 네 가지를 여기서 전부 지킨다.

        - 임계값 미만 제외
        - 매핑표에 없는 클래스 제외
        - 좌표를 정규화 값으로
        - 프레임 경계 안으로

    **정규화 좌표(``xyxyn``)를 쓴다.** 픽셀 좌표를 받아 직접 나누면 letterbox
    패딩이 섞여 어긋난다. 좌표계 변환은 라이브러리에 맡긴다.

    Args:
        result: ultralytics ``Results`` 객체.
        frame_number: 이 결과가 나온 프레임 번호.
        threshold: 프레임 단위 탐지 임계값.

    Returns:
        해당 프레임의 탐지 목록.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {}) or {}
    coords = boxes.xyxyn.tolist()
    confidences = boxes.conf.tolist()
    class_indexes = boxes.cls.tolist()
    track_ids = _track_ids(boxes, len(confidences))

    detections: list[Detection] = []
    for (x1, y1, x2, y2), confidence, index, track_id in zip(
        coords, confidences, class_indexes, track_ids
    ):
        if confidence < threshold:
            continue

        raw = names.get(int(index))
        if raw is None:
            logger.warning("모델이 알 수 없는 클래스 색인을 냈다: %s", index)
            continue

        code = to_class_code(raw)
        if not is_known(code):
            # 사람·TV·화분 등이 여기서 걸러진다. 통과하면 점유율에 섞인다.
            continue

        box = clamp_box(x1, y1, x2 - x1, y2 - y1)
        if box.area <= 0.0:
            continue

        detections.append(
            Detection(
                class_code=code,
                confidence=float(confidence),
                frame_number=frame_number,
                x=box.x,
                y=box.y,
                width=box.width,
                height=box.height,
                track_id=track_id,
            )
        )

    return detections


def _track_ids(boxes, count: int) -> list[int | None]:
    """추적 ID를 꺼낸다.

    ``predict()`` 결과에는 ``id`` 가 없고, ``track()`` 결과라도 추적기가 확신하지
    못한 탐지에는 붙지 않는다. 없으면 전부 None 이다.
    """
    raw = getattr(boxes, "id", None)
    if raw is None:
        return [None] * count
    return [int(value) for value in raw.tolist()]


class YoloDetector:
    """ultralytics 가중치로 프레임을 탐지한다.

    가중치 경로만 바꾸면 커스텀 학습본으로 교체된다. 파이프라인과 이후 단계는
    수정하지 않는다.
    """

    def __init__(
        self,
        weights,
        device: str | None = None,
        threshold: float = DETECTION_CONFIDENCE,
        batch_size: int = BATCH_SIZE,
        tracking: bool = False,
        tracker: str = DEFAULT_TRACKER,
    ) -> None:
        """모델을 즉시 적재한다.

        서버가 뜰 때 실패하는 편이 낫다. 첫 요청에서 실패하면 사용자에게는
        분석 실패로 보이고, 원인이 설정 문제라는 사실이 드러나지 않는다.

        Args:
            weights: 가중치 파일 경로.
            device: ``mps`` · ``cuda`` · ``cpu``. None 이면 자동 선택한다.
            threshold: 프레임 단위 탐지 임계값.
            batch_size: 한 번에 추론할 프레임 수. 추적 모드에서는 무시된다.
            tracking: 추적까지 함께 수행할지 여부. 켜면 결과에 ``track_id`` 가
                실리며 ``TrackIdTracker`` 로 통합할 수 있다.
            tracker: 추적기 설정 파일. ``botsort.yaml`` 또는 ``bytetrack.yaml``.

        Raises:
            RuntimeError: ultralytics 가 설치되어 있지 않은 경우.
        """
        self._model = _load(weights)
        self._device = device
        self._threshold = threshold
        self._batch_size = max(1, batch_size)
        self._tracking = tracking
        self._tracker = tracker

    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """프레임 목록을 추론한다.

        블로킹 호출이다. 호출하는 쪽이 스레드에서 실행한다.
        """
        if not frames:
            return []

        rows = self._track(frames) if self._tracking else self._predict(frames)

        # 규약상 프레임 수와 길이가 같아야 한다. 어긋나면 파이프라인이 막지만,
        # 원인이 여기라는 사실은 남지 않으므로 로그를 남긴다.
        if len(rows) != len(frames):
            logger.error(
                "추론 결과 수가 프레임 수와 다르다: %s != %s", len(rows), len(frames)
            )

        return rows

    def _predict(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """배치로 나눠 추론한다. 추적하지 않는다."""
        rows: list[list[Detection]] = []
        for chunk in _batched(frames, self._batch_size):
            results = self._model.predict(
                [f.image for f in chunk],
                conf=self._threshold,
                device=self._device,
                verbose=False,
            )
            rows.extend(
                to_detections(result, frame.number, self._threshold)
                for frame, result in zip(chunk, results)
            )
        return rows

    def _track(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """추적까지 함께 수행한다.

        **배치를 쓰지 않는다.** 추적기는 프레임을 순서대로 하나씩 받아 상태를
        이어가야 한다. 여러 장을 한 번에 넘기면 앞뒤 관계가 사라져 ID가 매 프레임
        새로 발급된다. ``persist=True`` 가 호출 사이의 상태를 유지한다.

        배치를 못 쓰므로 ``predict`` 보다 느리다. 추적 품질과 속도의 교환이며,
        어느 쪽이 나은지는 성능평가에서 정한다.
        """
        rows: list[list[Detection]] = []
        for frame in frames:
            results = self._model.track(
                frame.image,
                conf=self._threshold,
                device=self._device,
                tracker=self._tracker,
                persist=True,
                verbose=False,
            )
            rows.append(to_detections(results[0], frame.number, self._threshold))
        return rows


def _load(weights):
    """가중치를 적재한다. ultralytics 를 여기서만 임포트한다."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(MISSING_PACKAGE) from exc

    logger.info("탐지 모델 적재: %s", weights)
    return YOLO(str(weights))


def _batched(items: Sequence[Frame], size: int) -> Iterator[Sequence[Frame]]:
    """목록을 고정 크기로 자른다."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
