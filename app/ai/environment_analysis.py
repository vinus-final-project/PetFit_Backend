"""환경 분석 (파이프라인 12단계).

객체 탐지 결과·Pet Fit Score·원본 프레임을 입력으로 받아 위험 요소·분석 서술·
개선 추천을 생성한다. **점수는 만들지 않는다.** 이미 산출된 값을 입력으로 받아
서술이 점수와 어긋나지 않게 한다.

흐름은 네 단계다.

    조립 → 호출 → 검증 → (거절이면) 재생성

재생성은 사유를 프롬프트에 실어 다시 부른다. 무엇이 잘못됐는지 알리지 않으면
같은 실패가 반복된다. 정해진 횟수를 소진하면 분석 실패로 전환한다.

**모델 없이 전 구간을 검증할 수 있다.** `FakeLLM` 에 응답을 순서대로 지정하면
재생성과 실패 경로까지 결정적으로 재현된다.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.ai.llm.base import VisionLLM
from app.ai.pipeline import PipelineError
from app.ai.prompts import AnalysisContext, build, select_image_frames
from app.ai.validation import AnalysisOutput, Rejection, Repair, validate
from app.schemas.enums import AnalysisStage

__all__ = [
    "EnvironmentAnalyzer",
    "AnalysisReport",
    "Attempt",
    "ImageLoader",
    "MAX_REGENERATIONS",
    "FAILURE_MESSAGE",
]

logger = logging.getLogger(__name__)

#: 거절 시 다시 시도하는 횟수. 최초 호출을 포함하면 최대 4회 호출한다.
#:
#: 프롬프트 설계서의 "재생성 3회 실패 → 분석 실패로 처리"를 따른다.
#: 호출 1회가 수 초이므로 서버 처리 제한 시간(180초) 안에서 조정한다.
MAX_REGENERATIONS = 3

#: 사용자에게 표시할 실패 사유. 모델의 오류 메시지를 그대로 노출하지 않는다.
FAILURE_MESSAGE = "환경 분석에 실패했습니다."

#: 프레임 번호를 원본 이미지 바이트로 바꾸는 함수.
#:
#: 프레임 추출은 2단계(Vision)의 산출물이므로 여기서 파일을 직접 읽지 않는다.
#: 읽을 수 없으면 None을 반환한다.
ImageLoader = Callable[[int], "bytes | None"]


@dataclass(frozen=True)
class Attempt:
    """호출 1회의 결과.

    성능평가의 JSON 형식 준수율·재생성 횟수 측정에 쓴다. 몇 번 만에 통과했는지
    남기지 않으면 모델 비교에서 "되긴 된다"는 것 외에 말할 수 있는 게 없다.
    """

    #: 거절 사유. 통과했으면 None.
    rejection: Rejection | None = None
    #: 재호출 없이 고친 항목.
    repairs: tuple[Repair, ...] = ()
    #: 호출 자체가 실패한 경우의 원인 유형.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.rejection is None and self.error is None


@dataclass(frozen=True)
class AnalysisReport:
    """12단계 산출물과 생성 과정 기록."""

    output: AnalysisOutput
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)
    #: 실제로 함께 보낸 이미지 장수. 0이면 관찰 근거 없이 생성한 결과다.
    images_sent: int = 0

    @property
    def regenerations(self) -> int:
        """재생성 횟수. 0이면 첫 호출에 통과했다."""
        return max(len(self.attempts) - 1, 0)


class EnvironmentAnalyzer:
    """생성형 모델로 환경 분석 결과를 만든다.

    Args:
        llm: 환경 분석 모델. 비전 입력을 지원해야 한다.
        max_regenerations: 거절 시 다시 시도할 횟수.
        include_few_shots: 프롬프트에 예시를 포함할지 여부.
    """

    def __init__(
        self,
        llm: VisionLLM,
        *,
        max_regenerations: int = MAX_REGENERATIONS,
        include_few_shots: bool = True,
    ) -> None:
        self._llm = llm
        self._max_regenerations = max_regenerations
        self._include_few_shots = include_few_shots

    async def analyze(
        self,
        context: AnalysisContext,
        load_image: ImageLoader | None = None,
    ) -> AnalysisReport:
        """환경 분석을 수행한다.

        Args:
            context: 12단계 입력. 점수는 이미 산출된 값이어야 한다.
            load_image: 프레임 번호를 원본 이미지로 바꾸는 함수. 생략하거나
                읽지 못하면 이미지 없이 진행한다.

        Returns:
            생성 결과와 시도 기록.

        Raises:
            PipelineError: 재생성 횟수를 소진한 경우. 실패 단계를 함께 전달한다.
        """
        images = self._load_images(context, load_image)
        detected = [o.name for o in context.objects]

        attempts: list[Attempt] = []
        rejection: Rejection | None = None

        for _ in range(self._max_regenerations + 1):
            prompt = build(
                context,
                retry=rejection,
                include_few_shots=self._include_few_shots,
                has_images=bool(images),
            )

            try:
                raw = await self._llm.complete(prompt, images)
            except Exception as exc:  # noqa: BLE001
                # 모델은 네트워크 오류·메모리 부족 등 다양한 예외를 낸다.
                # 어떤 것이든 한 번의 실패로 세고 다시 시도한다.
                logger.info("환경 분석 호출 실패: %s", type(exc).__name__)
                attempts.append(Attempt(error=type(exc).__name__))
                rejection = None
                continue

            result = validate(raw, detected)
            attempts.append(
                Attempt(rejection=result.rejection, repairs=result.repairs)
            )

            if result.ok and result.value is not None:
                if result.repairs:
                    logger.info("환경 분석 응답 복구: %s", [r.value for r in result.repairs])
                return AnalysisReport(
                    output=result.value,
                    attempts=tuple(attempts),
                    images_sent=len(images),
                )

            rejection = result.rejection
            logger.info("환경 분석 응답 거절: %s", rejection.value if rejection else "?")

        logger.warning(
            "환경 분석 %d회 시도 실패: %s",
            len(attempts),
            [a.rejection.value if a.rejection else a.error for a in attempts],
        )
        raise PipelineError(FAILURE_MESSAGE, AnalysisStage.ENVIRONMENT_ANALYSIS)

    def _load_images(
        self, context: AnalysisContext, load_image: ImageLoader | None
    ) -> list[bytes]:
        """보낼 원본 프레임을 읽는다.

        **읽지 못해도 분석을 중단하지 않는다.** 이미지가 없으면 탐지 결과만으로
        서술할 수 있다. 다만 관찰 근거가 없으므로 프롬프트가 `OBSERVED` 생성을
        금지한다. 이미지 하나 때문에 전체 분석을 실패로 만들 이유가 없다.
        """
        if load_image is None:
            return []

        images: list[bytes] = []
        for number in select_image_frames(context):
            try:
                data = load_image(number)
            except Exception as exc:  # noqa: BLE001
                logger.info("프레임 %s 읽기 실패: %s", number, type(exc).__name__)
                continue
            if data:
                images.append(data)

        if not images:
            logger.info("이미지 없이 환경 분석을 진행한다")
        return images
