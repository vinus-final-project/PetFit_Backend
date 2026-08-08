"""파이프라인 스텁 구현.

**실제 AI 모델 없이 전 구간을 동작시킨다.** 프론트엔드 연동과 상태 전이 검증이
AI 성능평가를 기다리지 않고 진행되도록 하는 것이 목적이다.

가짜인 것과 진짜인 것을 구분한다.

    가짜 : 탐지 결과, 위험 요소 서술, 분석 서술, 추천 문구, 이미지 경로
    진짜 : 위험도 판정, 점수 산출, 단계 전이, 소요 시간

점수는 실제 ``score_generator`` 를 거치므로, 프론트가 받는 점수는 문서의 계산
규칙과 정확히 일치한다. 화면 개발에 쓰기에 충분하다.

실제 파이프라인이 완성되면 이 파일을 교체한다. 서비스 계층은 수정하지 않는다.
"""

import asyncio
from pathlib import Path

from app.ai.pipeline import (
    DetectedObject,
    PipelineError,
    PipelineResult,
    Recommendation,
    RiskFactor,
    StageCallback,
)
from app.rules.risk_rules import classify
from app.schemas.enums import (
    AnalysisStage,
    AnimalGroup,
    RecommendationType,
    RiskSource,
    SpaceType,
)

__all__ = ["StubPipeline", "STAGE_DELAYS"]

#: 단계별 지연(초). 실제 파이프라인의 대략적인 비중을 흉내 낸다.
#: 프론트가 진행률 표시와 폴링 간격(2초)을 실제 조건에 가깝게 시험할 수 있어야 한다.
STAGE_DELAYS: dict[AnalysisStage, float] = {
    AnalysisStage.FRAME_EXTRACTION: 1.0,
    AnalysisStage.OBJECT_DETECTION: 3.5,
    AnalysisStage.OBJECT_TRACKING: 1.0,
    AnalysisStage.FRAME_SELECTION: 0.7,
    AnalysisStage.RISK_MARKING: 0.6,
    AnalysisStage.SCORE_CALCULATION: 0.4,
    AnalysisStage.ENVIRONMENT_ANALYSIS: 1.8,
}

#: 고정 탐지 결과. AI 분석 정의서의 계산 예시와 동일한 구성이다.
#: 소형견·거실 기준으로 종합 56점이 나온다.
_FIXTURE: tuple[tuple[str, float, int, int, float, float, float, float], ...] = (
    # (이름, confidence, 탐지 프레임 수, 프레임 번호, x, y, w, h)
    ("전선",   0.94, 12,  3, 0.1250, 0.7400, 0.2000, 0.0800),
    ("창문",   0.91,  9, 18, 0.6000, 0.1000, 0.2500, 0.4000),
    ("카펫",   0.89, 15,  7, 0.2000, 0.5500, 0.5000, 0.3000),
    ("소파",   0.98, 21,  5, 0.4000, 0.3500, 0.3500, 0.2500),
    ("급수기", 0.86,  4, 22, 0.8000, 0.6500, 0.1000, 0.1200),
)

_ANALYSIS_TEXT = (
    "활동 공간은 충분하지만 바닥에 노출된 전선이 위험 요소입니다.",
    "휴식 공간은 소파로 일부 확보되어 있으나 전용 잠자리가 없습니다.",
)

_RECOMMENDATIONS = (
    (RecommendationType.SAFETY, "전선을 벽면으로 정리하거나 몰딩으로 덮어주세요.", RiskSource.DETECTED),
    (RecommendationType.REST, "조용한 곳에 전용 잠자리를 마련해주세요.", RiskSource.DETECTED),
    (RecommendationType.SAFETY, "창가 화분을 반려동물이 닿지 않는 곳으로 옮겨주세요.", RiskSource.OBSERVED),
)


class StubPipeline:
    """고정 결과를 반환하는 파이프라인.

    Args:
        image_dir: 마킹 이미지 경로의 기준 디렉터리. 실제 파일은 만들지 않는다.
        speed: 지연 배율. 0으로 두면 대기 없이 즉시 끝나므로 테스트에서 사용한다.
        fail_at: 지정한 단계에서 PipelineError를 발생시킨다. 실패 흐름 시험용.
    """

    def __init__(
        self,
        image_dir: str = "/images",
        speed: float = 1.0,
        fail_at: AnalysisStage | None = None,
    ) -> None:
        self._image_dir = image_dir.rstrip("/")
        self._speed = speed
        self._fail_at = fail_at

    async def run(
        self,
        video_path: Path,
        group: AnimalGroup,
        space: SpaceType,
        on_stage: StageCallback,
    ) -> PipelineResult:
        """단계를 순서대로 진행하며 고정 결과를 만든다."""
        for stage in AnalysisStage:
            await on_stage(stage)

            if stage is self._fail_at:
                raise PipelineError(_failure_message(stage), stage)

            delay = STAGE_DELAYS[stage] * self._speed
            if delay > 0:
                await asyncio.sleep(delay)

        return PipelineResult(
            capture_duration=8.4,
            frame_count=25,
            thumbnail_path=f"{self._image_dir}/stub-thumbnail.jpg",
            occupancy_ratio=0.35,
            detected_objects=tuple(self._objects(group)),
            risk_factors=(
                RiskFactor("전선이 바닥에 엉킨 채 노출되어 있습니다.", RiskSource.DETECTED),
                RiskFactor(
                    "창가 선반에 화분이 놓여 있어 흙을 파헤치거나 삼킬 수 있습니다.",
                    RiskSource.OBSERVED,
                ),
            ),
            analysis=_ANALYSIS_TEXT,
            recommendations=tuple(
                Recommendation(type=t, text=x, priority=i, source=s)
                for i, (t, x, s) in enumerate(_RECOMMENDATIONS, start=1)
            ),
        )

    def _objects(self, group: AnimalGroup) -> list[DetectedObject]:
        """고정 탐지 결과에 **실제 위험도 판정**을 적용한다.

        위험도는 가짜로 두지 않는다. 그룹에 따라 창문이 고양이에게 HIGH,
        소형견에게 LOW가 되는 동작을 프론트가 실제로 확인할 수 있어야 한다.
        """
        objects = []
        for i, (name, conf, frames, frame_no, x, y, w, h) in enumerate(_FIXTURE, start=1):
            risk = classify(name, group)
            objects.append(
                DetectedObject(
                    name=name,
                    risk=risk,
                    confidence=conf,
                    detection_frame_count=frames,
                    frame_number=frame_no,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    # SAFE는 마킹 이미지를 생성하지 않는다.
                    marked_image_path=(
                        None if risk.marking_color is None
                        else f"{self._image_dir}/stub-mark-{i}.jpg"
                    ),
                )
            )
        return objects


def _failure_message(stage: AnalysisStage) -> str:
    """단계별 실패 사유. 사용자에게 표시하는 문구다."""
    return {
        AnalysisStage.FRAME_EXTRACTION: "영상에서 프레임을 추출하지 못했습니다.",
        AnalysisStage.OBJECT_DETECTION: "객체 탐지에 실패했습니다.",
        AnalysisStage.OBJECT_TRACKING: "객체 정리 중 오류가 발생했습니다.",
        AnalysisStage.FRAME_SELECTION: "대표 프레임을 선정하지 못했습니다.",
        AnalysisStage.RISK_MARKING: "위험 객체 마킹에 실패했습니다.",
        AnalysisStage.SCORE_CALCULATION: "점수 산출 중 오류가 발생했습니다.",
        AnalysisStage.ENVIRONMENT_ANALYSIS: "환경 분석에 실패했습니다.",
    }[stage]
