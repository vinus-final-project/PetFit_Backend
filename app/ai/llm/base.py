"""환경 분석 모델 계약.

**12단계 본체와 모델 구현의 경계다.** 본체는 이 프로토콜만 알고, 모델은 이
프로토콜만 지킨다. `app/ai/pipeline.py` 가 서비스와 AI를 가르는 것과 같은 방식이다.

후보가 둘이라 경계가 필요하다.

    GPT-4o      : 외부 API. 생활공간 이미지가 외부로 전송된다
    Qwen2.5-VL  : MLX 로컬 실행. 전송이 발생하지 않는다

메시지 형식과 이미지 전달 방식이 서로 다르므로, 제공자별 요청 조립은 각
어댑터가 맡는다. 선정 결과가 바뀌어도 본체와 프롬프트는 수정하지 않는다.

**비전 입력은 필수 조건이다.** 이미지를 처리하지 못하면 탐지 대상 12종 밖의
위험 요소(`OBSERVED`)를 찾을 수 없어 이미지를 보내는 의미가 사라진다.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.ai.prompts import Prompt

__all__ = ["VisionLLM", "LLMError"]


class LLMError(Exception):
    """모델 호출 실패.

    12단계 본체가 잡아 재시도한다. 원인 메시지를 사용자에게 그대로 노출하지 않는다.
    """


@runtime_checkable
class VisionLLM(Protocol):
    """이미지를 함께 받을 수 있는 생성 모델."""

    @property
    def name(self) -> str:
        """모델 식별자. 로그와 성능평가 기록에 쓴다."""
        ...

    async def complete(self, prompt: Prompt, images: Sequence[bytes]) -> str:
        """프롬프트와 이미지를 전달하고 응답 원문을 받는다.

        응답을 해석하거나 검증하지 않는다. JSON 파싱과 스키마 확인은
        `app.ai.validation` 이 담당한다. 어댑터가 응답을 손질하면 형식 준수율을
        측정할 수 없다.

        Args:
            prompt: 제공자에 중립적인 프롬프트.
            images: 마킹하지 않은 원본 프레임. 최대 `LLM_MAX_IMAGES` 장이며
                비어 있을 수 있다.

        Returns:
            모델이 반환한 원문.

        Raises:
            LLMError: 호출에 실패한 경우.
        """
        ...
