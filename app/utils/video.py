"""영상 메타데이터 읽기.

업로드 검증(서비스 계층)과 프레임 추출(AI 계층)이 같은 값을 읽는다. 각자
구현하면 한쪽만 고쳐졌을 때 **검증은 통과했는데 추출은 실패하는** 상태가 된다.
공용 위치에 하나만 둔다.
"""

import av

__all__ = ["duration_seconds", "rotation_degrees"]


def duration_seconds(container, stream) -> float | None:
    """영상 길이를 초 단위로 구한다.

    컨테이너 길이를 우선 쓰고, 없으면 스트림 값으로 대체한다.
    스트림에만 길이가 있는 파일이 있다.

    Args:
        container: 열린 ``av`` 컨테이너.
        stream: 비디오 스트림.

    Returns:
        영상 길이(초). 어느 쪽에도 없으면 None.
    """
    if container.duration is not None:
        return container.duration / av.time_base
    if stream.duration is not None and stream.time_base:
        return float(stream.duration * stream.time_base)
    return None


def rotation_degrees(stream) -> int:
    """영상의 회전 각도를 구한다.

    휴대폰으로 세로 촬영한 영상은 가로 화면으로 저장되고 회전 정보만 따로
    붙는 경우가 있다. 회전을 적용하지 않으면 **방을 옆으로 눕혀서 분석하게 되어**
    탐지 정확도가 떨어지고, 마킹 이미지도 돌아간 채로 사용자에게 표시된다.

    Args:
        stream: 비디오 스트림.

    Returns:
        0 · 90 · 180 · 270 중 하나. 정보가 없거나 해석할 수 없으면 0.

    Note:
        ``rotate`` 태그만 읽는다. 회전을 **표시 행렬(display matrix)** 로만 기록한
        파일은 여기서 0이 나온다. PyAV 17 이 해당 값을 노출하지 않는다.
        실제 촬영 영상으로 확인이 필요하다.
    """
    try:
        raw = stream.metadata.get("rotate")
    except AttributeError:
        return 0

    if raw is None:
        return 0

    try:
        degrees = int(round(float(raw)))
    except (TypeError, ValueError):
        return 0

    return degrees % 360 if degrees % 90 == 0 else 0
