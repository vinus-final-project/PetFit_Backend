"""메타 정보 API.

API-001 반려동물 그룹 목록, API-008 공간 종류 목록.

DB를 조회하지 않는다. 두 목록 모두 코드에 정의된 고정 도메인이므로
마스터 테이블을 두지 않는다.
"""

from fastapi import APIRouter

from app.schemas.analysis import AnimalListResponse, SpaceListResponse

router = APIRouter(tags=["meta"])


@router.get(
    "/animals",
    response_model=AnimalListResponse,
    summary="반려동물 그룹 목록 조회",
)
async def list_animals() -> AnimalListResponse:
    """분석이 가능한 반려동물 그룹만 반환한다.

    확장 그룹(소동물·조류·파충류)은 Enum에 코드만 예약되어 있다. 분석 기준이
    정의되기 전에는 노출하지 않는다. 목록에 없는 코드로 분석을 요청하면 400이다.
    """
    return AnimalListResponse.build()


@router.get(
    "/spaces",
    response_model=SpaceListResponse,
    summary="공간 종류 목록 조회",
)
async def list_spaces() -> SpaceListResponse:
    """공간 종류 4종을 반환한다.

    공간에 따라 평가하는 분석 항목이 달라지므로, 서로 다른 공간의 점수는
    직접 비교할 수 없다. 결과 화면에 공간 종류를 함께 표시해야 한다.
    """
    return SpaceListResponse.build()
