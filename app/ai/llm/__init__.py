"""환경 분석 모델 어댑터.

`base.VisionLLM` 이 유일한 계약이다. 12단계 본체는 이 계약만 알고, 제공자별
차이는 어댑터 안에 가둔다. 성능평가로 모델이 확정되어도 본체는 수정하지 않는다.
"""

from app.ai.llm.base import LLMError, VisionLLM
from app.ai.llm.fake import FakeLLM

__all__ = ["VisionLLM", "LLMError", "FakeLLM"]

# `qwen_mlx` 는 여기서 임포트하지 않는다. 모듈 자체는 mlx-vlm 없이도 읽히지만,
# 기본 노출 대상에 두면 제공자 선택과 무관하게 항상 불러오게 된다.
# 사용하는 쪽에서 `from app.ai.llm.qwen_mlx import QwenMLX` 로 직접 가져온다.
