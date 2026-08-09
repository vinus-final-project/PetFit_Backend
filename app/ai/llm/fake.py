"""고정 응답을 돌려주는 모델.

**실제 모델 없이 12단계 전 구간을 돌린다.** 프롬프트 조립·검증·재생성·실패 처리는
모델의 품질과 무관한 로직이므로, 응답을 고정하면 전부 결정적으로 검증할 수 있다.

가중치 다운로드나 API 키를 기다리지 않아도 되고, 재생성 흐름처럼 실제 모델로는
재현하기 어려운 경우도 응답을 순서대로 지정해 만들 수 있다.

`StubPipeline` 이 AI 없이 서비스 계층을 검증하는 것과 같은 역할이다.
"""

from collections.abc import Sequence

from app.ai.llm.base import LLMError
from app.ai.prompts import Prompt

__all__ = ["FakeLLM"]


class FakeLLM:
    """미리 정한 응답을 순서대로 돌려준다.

    Args:
        responses: 호출 순서대로 반환할 값. `Exception` 을 넣으면 그 시점에
            호출 실패를 일으킨다. 목록이 소진되면 마지막 값을 계속 반환한다.
            재생성 흐름을 시험할 때 "두 번 실패 후 성공" 같은 순서를 만든다.
        name: 모델 식별자.

    Attributes:
        prompts: 받은 프롬프트 기록. 재생성 시 지시문이 실렸는지 확인한다.
        image_counts: 호출마다 함께 받은 이미지 장수.
    """

    def __init__(self, *responses: str | Exception, name: str = "fake") -> None:
        if not responses:
            raise ValueError("응답을 하나 이상 지정해야 한다")
        self._responses = list(responses)
        self._name = name
        self.prompts: list[Prompt] = []
        self.image_counts: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    async def complete(self, prompt: Prompt, images: Sequence[bytes]) -> str:
        """기록을 남기고 다음 응답을 반환한다."""
        self.prompts.append(prompt)
        self.image_counts.append(len(images))

        # 마지막 응답은 소진하지 않는다. 호출 횟수를 미리 알지 못해도
        # "이후로는 계속 성공" 같은 시나리오를 쓸 수 있다.
        value = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]

        if isinstance(value, Exception):
            raise value
        return value


def always_fails(message: str = "모델 호출에 실패했다") -> FakeLLM:
    """호출할 때마다 실패하는 모델. 재생성 소진 경로를 시험한다."""
    return FakeLLM(LLMError(message), name="always-fails")
