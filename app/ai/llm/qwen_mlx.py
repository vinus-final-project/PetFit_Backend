"""Qwen3-VL 환경 분석 (12단계 실제 구현).

`VisionLLM` 규약을 MLX 로 구현한다. **이 파일만 추가하면 실제 모델이 붙는다.**
프롬프트·검증·재생성은 수정하지 않는다.

    모델 : mlx-community/Qwen3-VL-32B-Instruct-4bit

`mlx_vlm` 을 **모듈 최상단에서 임포트하지 않는다.** Apple Silicon macOS 에서만
설치되므로, 개발 PC(Windows)에서도 API 계층·서비스 계층·테스트가 전부 돌아가야
한다. `YoloDetector` 가 ultralytics 를 다루는 방식과 같다.

변환과 추론을 나눈다. **버그가 실제로 생기는 곳은 변환**(대화 구성, 이미지 전달,
응답 추출)이고 그쪽은 모델 없이 검증할 수 있다. 추론 호출은 얇게 유지한다.

로컬 실행이라 생활공간 이미지가 외부로 나가지 않는다. 외부 API 를 쓰면 사용자의
실내 사진이 전송되므로, 모델 선정에서 `데이터 처리 위치` 가 평가 항목에 있다.
"""

import asyncio
import logging
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

from app.ai.llm.base import LLMError
from app.ai.prompts import Prompt

__all__ = ["QwenMLX", "build_messages", "extract_text", "DEFAULT_MODEL"]

logger = logging.getLogger(__name__)

#: 기본 가중치. 성능평가 진행 가이드가 받도록 지정한 4bit 양자화본이다.
DEFAULT_MODEL = "mlx-community/Qwen3-VL-32B-Instruct-4bit"

#: 생성 상한.
#:
#: 출력은 위험 요소 5건·서술 4건·추천 5건의 한국어 JSON 이다. 한국어는 토큰이
#: 많이 들어가므로 여유를 둔다. 모자라면 JSON 이 중간에 끊겨 파싱에 실패하고
#: 재생성으로 이어져 오히려 느려진다.
MAX_TOKENS = 1600

#: 표본 온도.
#:
#: 서술은 생성물이지만 **형식은 지켜져야 한다.** 온도가 높으면 코드펜스나 설명
#: 문장이 붙어 재생성이 늘어난다. 0 은 같은 문장을 반복하는 경우가 있어 조금 준다.
TEMPERATURE = 0.2

#: mlx-vlm 미설치 안내.
MISSING_PACKAGE = (
    "mlx-vlm 이 설치되어 있지 않다. "
    "Apple Silicon macOS 에서 pip install -r requirements-ai.txt 로 설치한다."
)


def build_messages(prompt: Prompt) -> list[dict[str, str]]:
    """프롬프트를 대화 형식으로 바꾼다.

    예시는 **주고받은 대화로 넣는다.** 지시문 안에 예시를 문자열로 이어 붙이면
    모델이 그것을 분석 대상으로 오해해, 예시에 있던 창문이나 전선을 실제 결과에
    섞어 낸다.

    Args:
        prompt: 제공자에 중립적인 프롬프트.

    Returns:
        role/content 메시지 목록. 마지막 항목이 이번 분석 대상이다.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt.system}]
    for shot in prompt.few_shots:
        messages.append({"role": "user", "content": shot.user})
        messages.append({"role": "assistant", "content": shot.assistant})
    messages.append({"role": "user", "content": prompt.user})
    return messages


def extract_text(result: object) -> str:
    """생성 결과에서 본문을 꺼낸다.

    `mlx_vlm.generate` 는 버전에 따라 문자열을 주기도 하고 `text` 를 가진 객체를
    주기도 한다. 어느 쪽이든 원문을 그대로 돌려준다. **손질하지 않는다.**
    어댑터가 응답을 고치면 형식 준수율을 측정할 수 없다.
    """
    if isinstance(result, str):
        return result
    for attribute in ("text", "output", "content"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    return str(result)


class QwenMLX:
    """MLX 로 실행하는 Qwen3-VL.

    Args:
        model_path: 가중치 식별자 또는 로컬 경로.
        max_tokens: 생성 상한.
        temperature: 표본 온도.

    Attributes:
        name: 모델 식별자. 성능평가 기록에 쓴다.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> None:
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._loaded: tuple | None = None
        self._load_lock = threading.Lock()
        # 한 모델 인스턴스에 동시에 생성을 걸지 않는다. 서버는 분석 2건을 동시에
        # 처리하므로 12단계가 겹칠 수 있는데, MLX 는 동시 호출을 보장하지 않는다.
        # GPU 가 하나뿐이라 직렬화해도 실제 처리량은 줄지 않는다.
        self._generate_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_path

    async def complete(self, prompt: Prompt, images: Sequence[bytes]) -> str:
        """모델을 호출하고 응답 원문을 받는다.

        **별도 스레드에서 실행한다.** MLX 생성은 블로킹이며 32B 모델은 수 초가
        걸린다. 이벤트 루프에서 직접 돌리면 그동안 진행 상태 조회가 멈춰,
        클라이언트의 2초 폴링이 응답을 받지 못한다.

        Raises:
            LLMError: 패키지 미설치, 가중치 로드 실패, 생성 실패.
        """
        return await asyncio.to_thread(self._run, prompt, images)

    # --- 내부 -------------------------------------------------------------

    def _run(self, prompt: Prompt, images: Sequence[bytes]) -> str:
        """로드·변환·생성을 수행한다. 블로킹이므로 스레드에서 호출한다."""
        model, processor, config = self._ensure_loaded()
        messages = build_messages(prompt)

        with _image_files(images) as paths:
            try:
                from mlx_vlm import generate
                from mlx_vlm.prompt_utils import apply_chat_template
            except ImportError as exc:
                raise LLMError(MISSING_PACKAGE) from exc

            # 한 모델에 동시에 생성을 걸지 않는다. 서버가 분석 2건을 동시에
            # 처리하므로 12단계가 겹칠 수 있는데, MLX 는 동시 호출을 보장하지 않는다.
            # GPU 가 하나뿐이라 직렬화해도 실제 처리량은 줄지 않는다.
            with self._generate_lock:
                try:
                    formatted = apply_chat_template(
                        processor, config, messages, num_images=len(paths)
                    )
                    result = generate(
                        model,
                        processor,
                        formatted,
                        paths,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        verbose=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Qwen 생성 실패")
                    raise LLMError(f"생성에 실패했다: {type(exc).__name__}") from exc

        return extract_text(result)

    def _ensure_loaded(self) -> tuple:
        """가중치를 한 번만 읽는다.

        4bit 32B 는 20GB 가까이 차지하고 로드에 수십 초가 걸린다. 분석마다 다시
        읽으면 처리 제한 시간(180초)을 혼자 다 쓴다. 잠금을 두는 이유는 동시
        2건이 각각 로드를 시작하면 메모리를 두 배로 쓰기 때문이다.
        """
        if self._loaded is not None:
            return self._loaded

        with self._load_lock:
            if self._loaded is not None:
                return self._loaded

            try:
                from mlx_vlm import load
                from mlx_vlm.utils import load_config
            except ImportError as exc:
                raise LLMError(MISSING_PACKAGE) from exc

            logger.info("Qwen 가중치를 읽는다: %s", self._model_path)
            try:
                model, processor = load(self._model_path)
                config = load_config(self._model_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Qwen 가중치 로드 실패")
                raise LLMError(f"가중치를 읽지 못했다: {type(exc).__name__}") from exc

            self._loaded = (model, processor, config)
            return self._loaded


class _image_files:
    """이미지 바이트를 임시 파일로 풀어 경로 목록을 준다.

    `VisionLLM` 규약이 바이트를 넘기는 이유는 제공자마다 요구가 다르기 때문이다.
    OpenAI 는 base64 문자열을, MLX 는 경로나 PIL 이미지를 받는다. 경로는 버전과
    무관하게 받아들여지므로 여기서는 임시 파일을 쓴다.

    최대 4장이라 입출력 비용은 32B 추론에 비하면 무시할 수 있다.
    """

    def __init__(self, images: Sequence[bytes]) -> None:
        self._images = [data for data in images if data]
        self._directory: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> list[str]:
        if not self._images:
            return []
        self._directory = tempfile.TemporaryDirectory(prefix="petfit-vlm-")
        root = Path(self._directory.name)

        paths: list[str] = []
        for index, data in enumerate(self._images):
            path = root / f"frame-{index}.jpg"
            path.write_bytes(data)
            paths.append(str(path))
        return paths

    def __exit__(self, *exc_info) -> None:
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
