"""객체 추적과 탐지 신뢰 기준 적용 (5·6단계).

프레임마다 흩어진 탐지를 물체 단위로 접고, 오탐을 걸러낸다.

    프레임 3  소파 0.91
    프레임 4  소파 0.95   ->  소파 · confidence 0.95 · 21프레임 · 대표 프레임 5
    프레임 5  소파 0.98

통합 결과의 산출 방식은 AI 설계서를 따른다.

    confidence            : 탐지된 프레임 중 최댓값
    frame_number          : confidence 가 가장 높은 프레임
    detection_frame_count : 탐지된 프레임 수

**추적기를 교체할 수 있게 둔다.** 1차 성능평가에서 BoT-SORT 의 중복 제거
정확도가 0.287 로 게이트(0.85)에 크게 미달했다. 원인이 데이터 오염일 수도
있으나 구조적인 이유도 있다. 추적기는 연속 프레임을 가정하는데 우리는 초당
3프레임만 뽑으므로, 카메라가 움직이는 중에는 같은 물체의 박스가 프레임 간에
거의 겹치지 않는다.

여기 기본 구현으로 두는 IoU 병합은 모델이 필요 없고 지금 검증할 수 있다.
재측정 결과에 따라 다른 구현으로 바꾼다.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.ai.vision.types import Detection, TrackedObject
from app.core.constants import (
    ADOPTION_CONFIDENCE,
    MIN_DETECTION_FRAMES,
    TRACKING_IOU_THRESHOLD,
)
from app.utils.geometry import iou
from app.utils.rounding import normalize

__all__ = ["Tracker", "IouTracker", "adopt"]

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracker(Protocol):
    """프레임별 탐지를 물체 단위로 통합한다."""

    def track(self, detections: Sequence[Sequence[Detection]]) -> list[TrackedObject]:
        """통합 결과를 돌려준다.

        Args:
            detections: 프레임별 탐지 결과. 시각 순서대로여야 한다.

        Returns:
            통합된 객체 목록. 아직 오탐 필터를 거치지 않았다.
        """
        ...


@dataclass
class _Track:
    """통합 중인 물체 하나. 완성되면 ``TrackedObject`` 가 된다."""

    class_code: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def last_box(self):
        """가장 최근에 관측된 박스.

        처음 박스가 아니라 최근 박스와 견준다. 카메라가 방을 훑으면 물체가
        화면을 가로질러 이동하는데, 첫 박스를 기준으로 삼으면 몇 프레임 만에
        겹침이 사라져 같은 물체가 여러 건으로 쪼개진다.
        """
        return self.detections[-1].box

    def finalize(self) -> TrackedObject:
        """대표값을 정해 통합 객체를 만든다.

        대표 프레임은 우선순위 세 단계로 정한다. AI 설계서의 객체 대표 프레임
        선정 기준과 같은 순서다.

            1. confidence 가 높을수록
            2. Bounding Box 면적이 클수록
            3. 프레임 번호가 작을수록

        3순위까지 적용하면 프레임 번호가 유일하므로 항상 하나가 결정된다.
        """
        best = max(
            self.detections,
            key=lambda d: (d.confidence, d.box.area, -d.frame_number),
        )
        return TrackedObject(
            class_code=self.class_code,
            confidence=best.confidence,
            # 같은 프레임에서 두 번 잡힌 경우를 한 번으로 센다. 그대로 세면
            # 탐지 프레임 수가 부풀려져 오탐 필터가 무력해진다.
            detection_frame_count=len({d.frame_number for d in self.detections}),
            frame_number=best.frame_number,
            x=best.x,
            y=best.y,
            width=best.width,
            height=best.height,
        )


class IouTracker:
    """박스 겹침으로 같은 물체를 판정하는 추적기.

    프레임을 시각 순서로 훑으며, 각 탐지를 겹침이 가장 큰 기존 물체에 붙인다.
    붙일 곳이 없으면 새 물체로 시작한다.

    **한 물체는 한 프레임에서 탐지를 하나만 받는다.** 이 제약이 없으면 나란히
    놓인 의자 두 개가 한 물체로 합쳐진다.

    오래된 물체를 만료시키지 않는다. 카메라가 다른 곳을 비추다 돌아왔을 때
    같은 물체를 다시 이어붙이기 위해서다. 그동안 물체가 크게 움직였다면 어차피
    겹침이 사라져 새 물체가 된다.
    """

    def __init__(self, threshold: float = TRACKING_IOU_THRESHOLD) -> None:
        """
        Args:
            threshold: 같은 물체로 볼 최소 IoU.
        """
        # 비교하는 두 값을 같은 자릿수로 맞춘다. 한쪽만 정규화하면 기준값과
        # 정확히 같은 겹침이 미달로 판정된다.
        self._threshold = normalize(threshold)

    def track(self, detections: Sequence[Sequence[Detection]]) -> list[TrackedObject]:
        """프레임별 탐지를 물체 단위로 통합한다."""
        tracks_by_class: dict[str, list[_Track]] = {}

        for row in detections:
            for class_code, group in _group_by_class(row).items():
                tracks = tracks_by_class.setdefault(class_code, [])
                self._assign(tracks, group, class_code)

        return [t.finalize() for tracks in tracks_by_class.values() for t in tracks]

    def _assign(
        self, tracks: list[_Track], group: list[Detection], class_code: str
    ) -> None:
        """한 프레임의 탐지들을 기존 물체에 붙이거나 새 물체로 만든다.

        겹침이 큰 쌍부터 확정하는 그리디 방식이다. 먼저 나온 탐지가 좋은 자리를
        차지하는 것을 막는다.
        """
        pairs = [
            (iou(track.last_box, det.box), ti, di)
            for ti, track in enumerate(tracks)
            for di, det in enumerate(group)
        ]
        # 겹침 내림차순. 동점일 때 순서가 흔들리지 않도록 색인까지 정렬 기준에 넣는다.
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        taken_tracks: set[int] = set()
        taken_dets: set[int] = set()

        for overlap, ti, di in pairs:
            if normalize(overlap) < self._threshold:
                # 정렬되어 있으므로 이후 쌍은 전부 기준에 못 미친다.
                break
            if ti in taken_tracks or di in taken_dets:
                continue
            tracks[ti].detections.append(group[di])
            taken_tracks.add(ti)
            taken_dets.add(di)

        for di, det in enumerate(group):
            if di not in taken_dets:
                tracks.append(_Track(class_code=class_code, detections=[det]))


def adopt(
    objects: Sequence[TrackedObject],
    confidence: float = ADOPTION_CONFIDENCE,
    min_frames: int = MIN_DETECTION_FRAMES,
) -> list[TrackedObject]:
    """탐지 신뢰 기준을 적용해 오탐을 걸러낸다 (6단계).

    두 조건을 함께 쓴다. 전선처럼 가늘고 배경과 유사한 물체는 실재해도
    confidence 가 낮으므로, 임계값만 높이면 가장 위험한 것을 놓친다. 반면
    탐지 프레임 수는 오탐과 실재를 잘 구분한다. 30프레임 중 1장에서만 잡힌
    물체는 오탐일 가능성이 높다.

    채택되지 않은 물체는 저장하지 않으며 점수와 마킹 어디에도 쓰지 않는다.

    Args:
        objects: 통합된 객체 목록.
        confidence: 채택 임계값.
        min_frames: 채택에 필요한 최소 탐지 프레임 수.

    Returns:
        채택된 객체. 입력 순서를 유지한다.
    """
    # 양쪽을 같은 자릿수로 맞춘다. 한쪽만 정규화하면 기준값과 정확히 같은
    # confidence 가 미달로 판정된다.
    threshold = normalize(confidence)
    adopted = [
        o
        for o in objects
        if normalize(o.confidence) >= threshold
        and o.detection_frame_count >= min_frames
    ]

    dropped = len(objects) - len(adopted)
    if dropped:
        logger.debug("탐지 신뢰 기준으로 %s건을 제외했다", dropped)

    return adopted


def _group_by_class(row: Sequence[Detection]) -> dict[str, list[Detection]]:
    """한 프레임의 탐지를 클래스별로 나눈다.

    서로 다른 클래스는 절대 같은 물체가 아니므로 비교할 필요가 없다.
    """
    grouped: dict[str, list[Detection]] = {}
    for det in row:
        grouped.setdefault(det.class_code, []).append(det)
    return grouped
