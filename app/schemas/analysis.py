"""분석 API 요청·응답 DTO.

API 명세서의 공통 데이터 타입 및 API-001~008과 1:1로 대응한다.

공통 규칙
    필드 명명 : camelCase (파이썬 쪽은 snake_case, alias로 변환)
    날짜·시간 : ISO 8601, KST 오프셋 포함 (`2026-08-06T14:30:00+09:00`)
    식별자    : Integer(int64)
    목록 응답 : 배열을 객체로 래핑

시각 변환은 `app.utils.timeutil` 이 담당한다. 저장·표시 모두 KST 기준이다.

요청 DTO는 필드를 모두 Optional로 선언하고 **직접 검증한다.**
필수로 선언하면 FastAPI가 먼저 422를 반환해 `ANIMAL_GROUP_REQUIRED` 같은
명세상의 오류 코드를 내보낼 수 없다.
"""

from datetime import datetime
from typing import Any, Iterable, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from pydantic.alias_generators import to_camel

from app.core.constants import (
    MAX_RETRY_COUNT,
    PAGE_SIZE_DEFAULT,
    PAGE_SIZE_MAX,
)
from app.core.exceptions import ErrorCode, PetFitError
from app.utils.timeutil import KST, to_iso
from app.schemas.enums import (
    AnalysisStage,
    AnalysisStatus,
    AnimalGroup,
    RecommendationType,
    RiskLevel,
    RiskSource,
    SpaceType,
)

__all__ = [
    "CamelModel",
    "ErrorResponse",
    "MessageResponse",
    "CodeNameItem",
    "AnimalListResponse",
    "SpaceListResponse",
    "AnalysisCreateForm",
    "AnalysisListQuery",
    "PetFitScoreOut",
    "DetectedObjectOut",
    "RiskFactorOut",
    "RecommendationOut",
    "AnalysisAccepted",
    "RetryAccepted",
    "AnalysisStatusOut",
    "AnalysisSummaryOut",
    "AnalysisDetailOut",
    "Pagination",
    "AnalysisListOut",
    "sort_detected_objects",
    "sort_risk_factors",
    "KST",
    "to_iso",
]



class CamelModel(BaseModel):
    """camelCase 직렬화를 적용하는 기반 모델.

    `populate_by_name` 을 켜서 파이썬 쪽에서는 snake_case로 생성할 수 있게 한다.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


# =============================================================================
# 공통 응답
# =============================================================================


class ErrorResponse(CamelModel):
    """공통 오류 응답.

    `code` 는 클라이언트 분기 처리에, `message` 는 화면 표시에 사용한다.
    서버 내부 오류 메시지를 `message` 에 노출하지 않는다.
    """

    code: str = Field(description="오류 코드")
    message: str = Field(description="사용자 표시용 메시지 (한국어)")
    field: str | None = Field(default=None, description="검증 오류가 발생한 필드")
    status: AnalysisStatus | None = Field(
        default=None, description="상태 충돌인 경우 현재 분석 상태"
    )


class MessageResponse(CamelModel):
    """단순 처리 결과 메시지. API-005 삭제 응답에 사용한다."""

    message: str


class CodeNameItem(CamelModel):
    """코드·이름 쌍. 목록형 메타 API의 항목이다."""

    code: str
    name: str


class AnimalListResponse(CamelModel):
    """API-001 반려동물 그룹 목록."""

    animals: list[CodeNameItem]

    @classmethod
    def build(cls) -> "AnimalListResponse":
        """분석 기준이 정의된 그룹만 반환한다.

        확장 그룹(소동물·조류·파충류)은 코드만 예약되어 있으므로 노출하지 않는다.
        """
        return cls(
            animals=[
                CodeNameItem(code=g.value, name=g.label) for g in AnimalGroup.analyzable()
            ]
        )


class SpaceListResponse(CamelModel):
    """API-008 공간 종류 목록."""

    spaces: list[CodeNameItem]

    @classmethod
    def build(cls) -> "SpaceListResponse":
        return cls(spaces=[CodeNameItem(code=s.value, name=s.label) for s in SpaceType])


# =============================================================================
# 요청
# =============================================================================


def validate_device_id(raw: str | None) -> str:
    """X-Device-Id 헤더를 검증한다.

    Args:
        raw: 헤더 값.

    Returns:
        검증을 통과한 기기 식별자.

    Raises:
        PetFitError: 헤더가 없거나 UUID v4 형식이 아닌 경우.
    """
    if not raw or not raw.strip():
        raise PetFitError(ErrorCode.DEVICE_ID_REQUIRED, field="X-Device-Id")

    value = raw.strip()
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise PetFitError(ErrorCode.DEVICE_ID_INVALID, field="X-Device-Id") from exc

    if parsed.version != 4:
        raise PetFitError(ErrorCode.DEVICE_ID_INVALID, field="X-Device-Id")

    return str(parsed)


class AnalysisCreateForm(CamelModel):
    """API-002 분석 요청 폼.

    multipart/form-data 로 전송되며 영상 파일은 별도 파라미터로 받는다.
    필드를 Optional로 두고 `validated()` 에서 명세상의 오류 코드를 직접 낸다.
    """

    animal_group: str | None = None
    space_type: str | None = None

    def validated(self) -> tuple[AnimalGroup, SpaceType]:
        """폼 값을 Enum으로 변환한다.

        Returns:
            (반려동물 그룹, 공간 종류).

        Raises:
            PetFitError: 값이 없거나 허용되지 않는 경우.
        """
        if not self.animal_group:
            raise PetFitError(ErrorCode.ANIMAL_GROUP_REQUIRED, field="animalGroup")
        try:
            group = AnimalGroup(self.animal_group)
        except ValueError as exc:
            raise PetFitError(
                ErrorCode.ANIMAL_GROUP_UNSUPPORTED, field="animalGroup"
            ) from exc
        if group not in AnimalGroup.analyzable():
            raise PetFitError(ErrorCode.ANIMAL_GROUP_UNSUPPORTED, field="animalGroup")

        if not self.space_type:
            raise PetFitError(ErrorCode.SPACE_TYPE_REQUIRED, field="spaceType")
        try:
            space = SpaceType(self.space_type)
        except ValueError as exc:
            raise PetFitError(ErrorCode.SPACE_TYPE_INVALID, field="spaceType") from exc

        return group, space


class AnalysisListQuery(CamelModel):
    """API-003 분석 이력 조회 쿼리."""

    page: int = 1
    size: int = PAGE_SIZE_DEFAULT
    status: str | None = None

    def validated(self) -> tuple[int, int, AnalysisStatus | None]:
        """쿼리 파라미터를 검증한다.

        Returns:
            (페이지 번호, 페이지 크기, 상태 필터).

        Raises:
            PetFitError: 범위를 벗어나거나 정의되지 않은 값인 경우.
        """
        if self.page < 1:
            raise PetFitError(ErrorCode.PAGE_INVALID, field="page")
        if not 1 <= self.size <= PAGE_SIZE_MAX:
            raise PetFitError(ErrorCode.SIZE_INVALID, field="size")

        status: AnalysisStatus | None = None
        if self.status:
            try:
                status = AnalysisStatus(self.status)
            except ValueError as exc:
                raise PetFitError(ErrorCode.STATUS_INVALID, field="status") from exc

        return self.page, self.size, status


# =============================================================================
# 응답 구성 객체
# =============================================================================


class PetFitScoreOut(CamelModel):
    """Object : PetFitScore."""

    total: int = Field(ge=0, le=100)
    safety: int = Field(ge=0, le=100)
    activity: int = Field(ge=0, le=100)
    rest: int = Field(ge=0, le=100)
    environment: int = Field(ge=0, le=100)

    @classmethod
    def from_model(cls, row: Any) -> "PetFitScoreOut":
        """analysis 행에서 점수를 추출한다."""
        return cls(
            total=row.total_score,
            safety=row.safety_score,
            activity=row.activity_score,
            rest=row.rest_score,
            environment=row.environment_score,
        )


class DetectedObjectOut(CamelModel):
    """Object : DetectedObject.

    탐지된 **인스턴스 단위** 로 반환한다. 동일한 `name` 이 여러 번 나타날 수 있다.
    """

    name: str
    risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    marked_image: str | None = Field(
        default=None, description="마킹 이미지 경로. SAFE는 null"
    )

    @classmethod
    def from_model(cls, row: Any) -> "DetectedObjectOut":
        return cls(
            name=row.object_name,
            risk=RiskLevel(row.risk_level),
            confidence=float(row.confidence),
            marked_image=row.marked_image_path,
        )


class RiskFactorOut(CamelModel):
    """Object : RiskFactor.

    `analysis.risk_factors` JSON 컬럼에 `{text, source}` 형태로 저장된다.
    """

    text: str
    source: RiskSource

    @classmethod
    def from_json(cls, item: dict) -> "RiskFactorOut":
        return cls(text=item["text"], source=RiskSource(item["source"]))


class RecommendationOut(CamelModel):
    """Object : Recommendation."""

    type: RecommendationType
    text: str
    priority: int = Field(ge=1, description="1부터 시작, 낮을수록 우선")
    source: RiskSource

    @classmethod
    def from_model(cls, row: Any) -> "RecommendationOut":
        return cls(
            type=RecommendationType(row.recommendation_type),
            text=row.recommendation_text,
            priority=row.priority,
            source=RiskSource(row.source),
        )


def sort_detected_objects(
    items: Iterable[tuple[int, DetectedObjectOut]],
) -> list[DetectedObjectOut]:
    """탐지 객체를 명세의 정렬 규칙으로 정렬한다.

    동일 입력에 항상 동일한 순서를 보장한다.

        1. risk       HIGH → MEDIUM → LOW → SAFE
        2. confidence 내림차순
        3. name       가나다순
        4. 내부 식별자 오름차순

    한글 완성형(가~힣)은 유니코드 코드포인트 순서가 가나다순과 일치하므로
    별도 로케일 정렬이 필요 없다.

    Args:
        items: (내부 식별자, DTO) 쌍. 식별자는 4순위 정렬에만 사용한다.

    Returns:
        정렬된 DTO 목록.
    """
    return [
        dto
        for _, dto in sorted(
            items,
            key=lambda pair: (
                -pair[1].risk.rank,
                -pair[1].confidence,
                pair[1].name,
                pair[0],
            ),
        )
    ]


def sort_risk_factors(items: Sequence[RiskFactorOut]) -> list[RiskFactorOut]:
    """위험 요소를 정렬한다.

    `DETECTED` 가 먼저 오고, 같은 `source` 안에서는 생성 순서를 유지한다.
    파이썬의 정렬은 안정 정렬이므로 원래 순서가 보존된다.
    """
    return sorted(items, key=lambda f: 0 if f.source is RiskSource.DETECTED else 1)


# =============================================================================
# 응답
# =============================================================================


class AnalysisAccepted(CamelModel):
    """API-002 분석 요청 접수 (202 Accepted)."""

    analysis_id: int
    status: AnalysisStatus = AnalysisStatus.PENDING


class RetryAccepted(CamelModel):
    """API-007 재시도 접수 (202 Accepted)."""

    analysis_id: int
    status: AnalysisStatus = AnalysisStatus.PENDING
    retry_count: int = Field(ge=0, le=MAX_RETRY_COUNT)


class AnalysisStatusOut(CamelModel):
    """API-006 분석 진행 상태.

    실패 시에도 `stage` 와 `progress` 를 유지한다. 실패 지점을 구조화된 값으로
    남겨야 클라이언트가 재촬영과 재시도 중 무엇을 안내할지 분기할 수 있다.
    """

    analysis_id: int
    status: AnalysisStatus
    stage: AnalysisStage | None = None
    progress: int = Field(ge=0, le=100)
    retry_count: int = Field(ge=0, le=MAX_RETRY_COUNT)
    can_retry: bool
    message: str | None = Field(default=None, description="실패 사유 (FAILED일 때만)")

    @classmethod
    def from_model(cls, row: Any, *, can_retry: bool) -> "AnalysisStatusOut":
        """analysis 행에서 상태 응답을 만든다.

        Args:
            row: analysis 모델 인스턴스.
            can_retry: 서버가 종합 판정한 재시도 가능 여부.
        """
        status = AnalysisStatus(row.status)
        return cls(
            analysis_id=row.analysis_id,
            status=status,
            stage=AnalysisStage(row.stage) if row.stage else None,
            progress=100 if status is AnalysisStatus.COMPLETED else row.progress,
            retry_count=row.retry_count,
            can_retry=can_retry,
            message=row.error_message,
        )


class AnalysisSummaryOut(CamelModel):
    """Object : AnalysisSummary. 이력 목록에서 사용한다.

    `COMPLETED` 가 아닌 항목은 `petFitScore` 와 `thumbnailImage` 가 null이다.
    """

    analysis_id: int
    animal_group: AnimalGroup
    space_type: SpaceType
    status: AnalysisStatus
    pet_fit_score: PetFitScoreOut | None = None
    thumbnail_image: str | None = None
    can_retry: bool
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        return to_iso(value)

    @classmethod
    def from_model(cls, row: Any, *, can_retry: bool) -> "AnalysisSummaryOut":
        completed = AnalysisStatus(row.status) is AnalysisStatus.COMPLETED
        return cls(
            analysis_id=row.analysis_id,
            animal_group=AnimalGroup(row.animal_group),
            space_type=SpaceType(row.space_type),
            status=AnalysisStatus(row.status),
            pet_fit_score=PetFitScoreOut.from_model(row) if completed else None,
            thumbnail_image=row.thumbnail_path if completed else None,
            can_retry=can_retry,
            created_at=row.created_at,
        )


class AnalysisDetailOut(CamelModel):
    """Object : AnalysisDetail. `status` 가 COMPLETED인 경우에만 반환한다."""

    analysis_id: int
    animal_group: AnimalGroup
    space_type: SpaceType
    status: AnalysisStatus
    pet_fit_score: PetFitScoreOut
    thumbnail_image: str
    detected_objects: list[DetectedObjectOut]
    risk_factors: list[RiskFactorOut]
    analysis: list[str]
    recommendations: list[RecommendationOut]
    created_at: datetime
    completed_at: datetime

    @field_serializer("created_at", "completed_at")
    def _serialize_datetime(self, value: datetime) -> str:
        return to_iso(value)

    @classmethod
    def from_model(
        cls,
        row: Any,
        objects: Sequence[Any],
        recommendations: Sequence[Any],
    ) -> "AnalysisDetailOut":
        """analysis 행과 자식 행들로 상세 응답을 만든다.

        Args:
            row: analysis 모델 인스턴스.
            objects: detected_object 행 목록.
            recommendations: recommendation 행 목록.
        """
        return cls(
            analysis_id=row.analysis_id,
            animal_group=AnimalGroup(row.animal_group),
            space_type=SpaceType(row.space_type),
            status=AnalysisStatus(row.status),
            pet_fit_score=PetFitScoreOut.from_model(row),
            thumbnail_image=row.thumbnail_path,
            detected_objects=sort_detected_objects(
                (o.object_id, DetectedObjectOut.from_model(o)) for o in objects
            ),
            risk_factors=sort_risk_factors(
                [RiskFactorOut.from_json(f) for f in (row.risk_factors or [])]
            ),
            analysis=list(row.analysis_result or []),
            recommendations=sorted(
                (RecommendationOut.from_model(r) for r in recommendations),
                key=lambda r: r.priority,
            ),
            created_at=row.created_at,
            completed_at=row.completed_at,
        )


class Pagination(CamelModel):
    """Object : Pagination."""

    page: int = Field(ge=1)
    size: int = Field(ge=1, le=PAGE_SIZE_MAX)
    total_count: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool

    @classmethod
    def build(cls, page: int, size: int, total_count: int) -> "Pagination":
        """페이지 정보를 계산한다.

        전체 항목이 0건이면 `totalPages` 는 0이다.
        """
        total_pages = -(-total_count // size) if total_count else 0
        return cls(
            page=page,
            size=size,
            total_count=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
        )


class AnalysisListOut(CamelModel):
    """API-003 분석 이력 조회 응답.

    항목이 없어도 `analyses` 는 빈 배열로 반환한다. null을 반환하지 않는다.
    """

    analyses: list[AnalysisSummaryOut]
    pagination: Pagination
