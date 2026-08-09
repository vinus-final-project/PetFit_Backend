"""환경 분석 모델 어댑터.

`base.VisionLLM` 이 유일한 계약이다. 12단계 본체는 이 계약만 알고, 제공자별
차이는 어댑터 안에 가둔다. 성능평가로 모델이 확정되어도 본체는 수정하지 않는다.
"""

from app.ai.llm.base import VisionLLM
from app.ai.llm.fake import FakeLLM

__all__ = ["VisionLLM", "FakeLLM"]
