"""파일 저장소.

업로드 영상의 검증·저장·삭제를 담당한다. 생성된 이미지의 삭제도 여기서 처리한다.

**파일명은 UUID를 사용한다.** 사용자가 올린 파일명을 그대로 쓰면 경로 조작
(``../../etc/passwd``)과 파일명 충돌이 발생한다.

영상 판별에는 PyAV를 사용한다. ``ffprobe`` 는 정확하지만 외부 바이너리라
``requirements.txt`` 로 설치되지 않아, 팀원마다 설치 상태가 달라진다.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import av

from app.core.constants import (
    VIDEO_MAX_BYTES,
    VIDEO_MAX_SECONDS,
    VIDEO_MIN_SECONDS,
)
from app.core.exceptions import ErrorCode, PetFitError
from app.utils.video import duration_seconds

#: 저장 이미지 품질. 화면 표시용이라 원본 품질이 필요하지 않다.
#: 분석 1건이 최대 12장을 만들므로 용량이 그대로 저장소 사용량이 된다.
IMAGE_QUALITY = 85

__all__ = ["VideoInfo", "Storage", "ALLOWED_CODECS"]

logger = logging.getLogger(__name__)

#: 허용 비디오 코덱. MP4 컨테이너라도 코덱이 다르면 디코딩에 실패할 수 있다.
ALLOWED_CODECS: frozenset[str] = frozenset({"h264"})

#: 업로드를 읽어들일 청크 크기. 전체를 메모리에 올리지 않는다.
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class VideoInfo:
    """검증을 통과한 영상 정보."""

    path: Path
    duration: float
    width: int
    height: int
    codec: str

    @property
    def relative_path(self) -> str:
        """DB에 저장할 경로. 저장소 루트가 바뀌어도 참조가 깨지지 않도록 상대 경로를 쓴다."""
        return f"/videos/{self.path.name}"


class Storage:
    """영상·이미지 파일을 관리한다.

    Args:
        root: 저장소 루트. 하위에 ``videos/`` 와 ``images/`` 를 만든다.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self.video_dir = self._root / "videos"
        self.image_dir = self._root / "images"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    # --- 저장 -------------------------------------------------------------

    async def save_video(self, upload) -> VideoInfo:
        """업로드된 영상을 검증하고 저장한다.

        크기 검증을 **읽는 도중에** 수행한다. 전부 읽은 뒤 검사하면 100MB 제한이
        있어도 그보다 큰 파일이 디스크와 메모리를 먼저 소비한다.

        Args:
            upload: FastAPI ``UploadFile``. ``read`` 를 제공하는 객체면 된다.

        Returns:
            검증을 통과한 영상 정보.

        Raises:
            PetFitError: 영상이 없거나 크기·형식·길이 조건을 위반한 경우.
        """
        if upload is None or not getattr(upload, "filename", None):
            raise PetFitError(ErrorCode.VIDEO_REQUIRED, field="video")

        path = self.video_dir / f"{uuid4()}.mp4"
        written = 0

        try:
            with path.open("wb") as f:
                while chunk := await upload.read(CHUNK_SIZE):
                    written += len(chunk)
                    if written > VIDEO_MAX_BYTES:
                        raise PetFitError(ErrorCode.VIDEO_TOO_LARGE, field="video")
                    f.write(chunk)

            if written == 0:
                raise PetFitError(ErrorCode.VIDEO_REQUIRED, field="video")

            info = await asyncio.to_thread(self._probe, path)
        except Exception:
            # 검증에 실패한 파일을 남기지 않는다. 참조 없는 파일은 추적이 불가능하다.
            path.unlink(missing_ok=True)
            raise

        return info

    def _probe(self, path: Path) -> VideoInfo:
        """영상 메타데이터를 읽고 검증한다. 블로킹 호출이므로 스레드에서 실행한다."""
        try:
            with av.open(str(path)) as container:
                streams = container.streams.video
                if not streams:
                    raise PetFitError(ErrorCode.VIDEO_FORMAT_INVALID, field="video")

                stream = streams[0]
                codec = (stream.codec_context.name or "").lower()
                width = stream.codec_context.width
                height = stream.codec_context.height

                duration = duration_seconds(container, stream)
        except PetFitError:
            raise
        except Exception as exc:  # noqa: BLE001
            # av 는 손상된 파일에 다양한 예외를 낸다. 원인을 노출하지 않고 형식 오류로 통일한다.
            logger.info("영상 판별 실패: %s: %s", type(exc).__name__, exc)
            raise PetFitError(ErrorCode.VIDEO_FORMAT_INVALID, field="video") from exc

        if codec not in ALLOWED_CODECS:
            raise PetFitError(ErrorCode.VIDEO_FORMAT_INVALID, field="video")

        if duration is None or not VIDEO_MIN_SECONDS <= duration <= VIDEO_MAX_SECONDS:
            raise PetFitError(ErrorCode.VIDEO_DURATION_INVALID, field="video")

        return VideoInfo(
            path=path, duration=duration, width=width, height=height, codec=codec
        )

    # --- 삭제 -------------------------------------------------------------

    def delete(self, *relative_paths: str | None) -> int:
        """상대 경로로 지정한 파일을 삭제한다.

        삭제 실패는 예외를 내지 않고 로그만 남긴다. 파일이 이미 없다고 해서
        DB 행 삭제까지 막으면, 지울 수 없는 분석이 영구히 남는다.

        Args:
            relative_paths: ``/videos/...`` 또는 ``/images/...`` 형식. None은 무시한다.

        Returns:
            실제로 삭제된 파일 수.
        """
        removed = 0
        for rel in relative_paths:
            if not rel:
                continue
            path = self._resolve(rel)
            if path is None:
                logger.warning("저장소 밖 경로 삭제 시도를 차단했다: %s", rel)
                continue
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("파일 삭제 실패 %s: %s", path, exc)
        return removed

    def list_images(self, min_age_seconds: float = 0.0) -> list[str]:
        """저장된 이미지의 상대 경로를 돌려준다.

        Args:
            min_age_seconds: 이 시간보다 오래된 파일만 포함한다. 0이면 전부.
                **진행 중인 분석이 방금 만든 파일을 고아로 오인하지 않기 위해
                필요하다.** 마킹 이미지는 DB에 기록되기 전에 먼저 디스크에 쓰인다.

        Returns:
            ``/images/...`` 형식의 상대 경로.
        """
        cutoff = time.time() - min_age_seconds
        found: list[str] = []

        for path in self.image_dir.iterdir():
            try:
                if not path.is_file() or path.stat().st_mtime > cutoff:
                    continue
            except OSError:
                # 열거 도중 삭제될 수 있다. 다음 실행에서 다시 잡힌다.
                continue
            found.append(self.image_path(path.name))

        return found

    def _resolve(self, relative: str) -> Path | None:
        """상대 경로를 실제 경로로 바꾼다.

        저장소 밖을 가리키면 None을 반환한다. DB 값이 오염되어도
        임의 파일이 지워지지 않도록 막는다.
        """
        candidate = (self._root / relative.lstrip("/")).resolve()
        try:
            candidate.relative_to(self._root.resolve())
        except ValueError:
            return None
        return candidate

    def image_path(self, filename: str) -> str:
        """이미지 파일명을 DB 저장용 상대 경로로 바꾼다."""
        return f"/images/{filename}"

    def save_image(self, image) -> str:
        """이미지를 저장하고 DB 저장용 상대 경로를 돌려준다.

        마킹 이미지와 대표 프레임이 이 경로로 저장된다. 파일명은 UUID이며
        원본 파일명을 쓰지 않는다. 삭제·재시도 시 정리는 ``delete()`` 가 맡으므로
        저장도 여기에 두어야 경로 규칙이 한곳에 남는다.

        디스크에 쓰는 블로킹 호출이다. 호출하는 쪽이 스레드에서 실행한다.

        Args:
            image: PIL 이미지.

        Returns:
            ``/images/<uuid>.jpg`` 형식의 상대 경로.
        """
        filename = f"{uuid4()}.jpg"

        # JPEG 는 투명도를 지원하지 않는다. RGB 가 아닌 이미지는 저장이 실패한다.
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(self.image_dir / filename, "JPEG", quality=IMAGE_QUALITY, optimize=True)
        return self.image_path(filename)
