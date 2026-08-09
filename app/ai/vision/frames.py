"""영상 처리와 프레임 추출 (1·2단계).

추출 기준은 AI 분석 정의서를 따른다.

    목표 장수 = clamp(영상 길이(초) x 3, 15, 30)
    추출 간격 = 영상 길이 / 목표 장수

**되감기(seek)를 쓰지 않고 순차 디코딩한다.** 3~30초 영상이라 전부 디코딩해도
수백 프레임이고, 되감기는 B프레임이 있는 파일에서 목표 시각과 다른 프레임을
돌려주는 경우가 있다. 정확도를 택한다.

이미지로 변환하는 것은 **선택된 프레임뿐이다.** 디코딩한 모든 프레임을 변환하면
30장을 얻으려고 900장을 만들게 된다.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import ImageOps
from PIL.Image import Image

from app.ai.pipeline import PipelineError
from app.ai.vision.types import Frame
from app.core.constants import FRAME_MAX, FRAME_MAX_EDGE, FRAME_MIN, FRAME_RATE
from app.schemas.enums import AnalysisStage
from app.utils.video import duration_seconds, rotation_degrees

__all__ = ["VideoFrames", "target_frame_count", "sample_timestamps", "extract"]

logger = logging.getLogger(__name__)

#: 사용자에게 표시하는 실패 사유. 내부 오류를 노출하지 않는다.
UNREADABLE_MESSAGE = "영상을 읽을 수 없습니다. 다시 촬영해주세요."
TOO_FEW_FRAMES_MESSAGE = "영상에서 분석할 장면을 충분히 찾지 못했습니다. 다시 촬영해주세요."


@dataclass(frozen=True)
class VideoFrames:
    """추출 결과.

    Attributes:
        duration: 영상 길이(초).
        frames: 추출된 프레임. 15장 이상 30장 이하이며 시각 순서대로다.
    """

    duration: float
    frames: Sequence[Frame]

    @property
    def count(self) -> int:
        return len(self.frames)


def target_frame_count(duration: float) -> int:
    """영상 길이에서 목표 프레임 수를 구한다.

    초당 3프레임이 기준이지만 결과는 15~30장 범위로 자른다. 3초 영상은 9장이
    되어 부족하고, 30초 영상은 90장이 되어 처리 시간이 세 배가 된다.

    Args:
        duration: 영상 길이(초).

    Returns:
        목표 장수. 15 이상 30 이하.

    Examples:
        >>> target_frame_count(3.0)
        15
        >>> target_frame_count(7.0)
        21
        >>> target_frame_count(30.0)
        30
    """
    return min(max(round(duration * FRAME_RATE), FRAME_MIN), FRAME_MAX)


def sample_timestamps(duration: float, count: int) -> list[float]:
    """추출할 시각을 구한다.

    구간의 **가운데**를 고른다. 0초부터 시작하면 첫 프레임이 촬영을 시작하며
    흔들린 장면이 되고, 마지막 구간은 통째로 버려진다.

    Args:
        duration: 영상 길이(초).
        count: 목표 장수. 1 이상.

    Returns:
        오름차순 시각 목록.

    Examples:
        >>> sample_timestamps(6.0, 3)
        [1.0, 3.0, 5.0]
    """
    interval = duration / count
    return [(i + 0.5) * interval for i in range(count)]


def extract(video_path: Path, max_edge: int = FRAME_MAX_EDGE) -> VideoFrames:
    """영상을 열어 프레임을 추출한다.

    블로킹 호출이다. 호출하는 쪽이 ``asyncio.to_thread`` 로 감싼다.

    Args:
        video_path: 영상 파일 경로.
        max_edge: 보관할 프레임의 긴 변 픽셀 상한.

    Returns:
        영상 길이와 프레임 목록.

    Raises:
        PipelineError: 영상을 열 수 없거나 프레임이 부족한 경우.
            단계는 ``FRAME_EXTRACTION`` 이다.
    """
    try:
        with av.open(str(video_path)) as container:
            streams = container.streams.video
            if not streams:
                raise PipelineError(UNREADABLE_MESSAGE, AnalysisStage.FRAME_EXTRACTION)

            stream = streams[0]
            duration = duration_seconds(container, stream)
            if not duration or duration <= 0:
                raise PipelineError(UNREADABLE_MESSAGE, AnalysisStage.FRAME_EXTRACTION)

            rotation = rotation_degrees(stream)
            targets = sample_timestamps(duration, target_frame_count(duration))
            frames = _collect(container, stream, targets, rotation, max_edge)
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001
        # av 는 손상된 파일에 다양한 예외를 낸다. 원인을 사용자에게 노출하지 않는다.
        logger.info("프레임 추출 실패 %s: %s: %s", video_path, type(exc).__name__, exc)
        raise PipelineError(
            UNREADABLE_MESSAGE, AnalysisStage.FRAME_EXTRACTION
        ) from exc

    if len(frames) < FRAME_MIN:
        # 저속 촬영이나 손상으로 실제 프레임이 모자란 경우다. 부족분을 같은 프레임으로
        # 채우면 detection_frame_count 가 부풀려져 오탐 필터가 무력해진다.
        logger.info("프레임 부족 %s: %s장", video_path, len(frames))
        raise PipelineError(TOO_FEW_FRAMES_MESSAGE, AnalysisStage.FRAME_EXTRACTION)

    return VideoFrames(duration=duration, frames=tuple(frames))


def _collect(
    container,
    stream,
    targets: Sequence[float],
    rotation: int,
    max_edge: int,
) -> list[Frame]:
    """목표 시각에 해당하는 프레임을 순차 디코딩으로 골라낸다.

    같은 프레임을 두 번 쓰지 않는다. 영상의 실제 프레임 수가 목표 장수보다 적으면
    그만큼 덜 나오며, 충분한지는 호출한 쪽이 판정한다.
    """
    picked: list[Frame] = []
    target_index = 0
    frame_index = 0
    last_used = -1

    for decoded in container.decode(stream):
        # 컨테이너에 따라 time 이 없다. 프레임 순번으로 대체한다.
        seconds = decoded.time
        if seconds is None:
            seconds = frame_index / float(stream.average_rate or 30)

        while target_index < len(targets) and seconds >= targets[target_index]:
            target_index += 1
            if frame_index == last_used:
                # 이 프레임은 이미 앞의 목표에 썼다. 중복 대신 건너뛴다.
                continue
            picked.append(
                Frame(
                    number=len(picked),
                    timestamp=float(seconds),
                    image=_prepare(decoded.to_image(), rotation, max_edge),
                )
            )
            last_used = frame_index

        frame_index += 1
        if target_index >= len(targets):
            break

    return picked


def _prepare(image: Image, rotation: int, max_edge: int) -> Image:
    """회전을 되돌리고 크기를 줄인다.

    회전을 **먼저** 적용한다. 축소를 먼저 하면 세로 영상의 긴 변이 뒤바뀐 상태로
    계산되어, 회전 후 크기가 상한을 넘거나 불필요하게 작아진다.
    """
    if rotation:
        # 저장된 각도만큼 되돌린다. PIL 의 rotate 는 반시계 방향이다.
        image = image.rotate(-rotation, expand=True)

    if max(image.width, image.height) > max_edge:
        image = ImageOps.contain(image, (max_edge, max_edge))

    return image
