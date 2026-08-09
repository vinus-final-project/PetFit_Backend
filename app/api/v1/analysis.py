"""분석 API.

API-002 분석 요청 · API-003 이력 조회 · API-004 상세 조회 · API-005 삭제 ·
API-006 진행 상태 · API-007 재시도.

**라우터는 조율만 한다.** 상태 전이·소유권 검사·파일 정리는 모두 서비스 계층에
있다. 여기서 status를 직접 다루면 `stage` · `progress` · `completed_at` 의
정합성이 깨져 DB CHECK 제약에 걸린다.

`analysisId` 로 접근하는 모든 API는 소유권을 검사한다. 불일치 시 403이 아니라
404를 반환한다. 403은 해당 ID의 분석이 존재한다는 사실을 노출한다. 검사는
서비스 계층의 `get_owned()` 가 담당하므로 라우터에서 다시 확인하지 않는다.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, Path, Query, UploadFile

from app.api.deps import (
    DbSession,
    DeviceId,
    QueueDep,
    ServiceDep,
    StorageDep,
    video_file,
)
from app.core.constants import PAGE_SIZE_DEFAULT, PAGE_SIZE_MAX
from app.core.exceptions import ErrorCode, PetFitError
from app.schemas.analysis import (
    AnalysisAccepted,
    AnalysisCreateForm,
    AnalysisDetailOut,
    AnalysisListOut,
    AnalysisListQuery,
    AnalysisStatusOut,
    AnalysisSummaryOut,
    MessageResponse,
    Pagination,
    RetryAccepted,
)
from app.schemas.enums import AnimalGroup, SpaceType

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])

#: 대기열 초과로 접수하지 못한 분석에 남기는 사유.
QUEUE_FULL_MESSAGE = ErrorCode.QUEUE_FULL.message

AnalysisId = Annotated[int, Path(description="분석 ID")]


def _reject_if_queue_full(queue: QueueDep) -> None:
    """대기열이 가득 찼으면 접수 전에 거절한다.

    작업을 등록하는 시점에도 `submit()` 이 같은 검사를 한다. 그런데 그때는
    이미 영상을 저장하고 행을 만든 뒤라, 되돌리는 비용이 크다. 먼저 확인해서
    대부분의 경우 아무것도 만들지 않고 끝낸다.

    판정은 큐에게 묻는다. 여기서 상수와 직접 비교하면 큐가 다른 용량으로
    만들어졌을 때 사전 검사와 `submit()` 의 결과가 달라진다.
    """
    if queue.is_full:
        raise PetFitError(ErrorCode.QUEUE_FULL)


# =============================================================================
# API-002 영상 분석 요청
# =============================================================================


@router.post(
    "",
    status_code=202,
    response_model=AnalysisAccepted,
    summary="영상 기반 생활환경 분석 요청",
)
async def create_analysis(
    device_id: DeviceId,
    session: DbSession,
    service: ServiceDep,
    storage: StorageDep,
    queue: QueueDep,
    video: Annotated[UploadFile | None, File(description="촬영 영상 (MP4/H.264)")] = None,
    animal_group: Annotated[str | None, Form(alias="animalGroup")] = None,
    space_type: Annotated[str | None, Form(alias="spaceType")] = None,
) -> AnalysisAccepted:
    """분석을 접수하고 즉시 분석 ID를 반환한다.

    영상 분석은 수십 초가 걸리므로 동기로 처리하면 요청 타임아웃이 발생한다.
    접수만 하고 202를 반환한 뒤 백그라운드에서 처리한다. 클라이언트는 API-006으로
    진행 상태를 폴링한다.

    검증 순서는 **비용이 낮은 것부터**다. 폼 값과 대기열을 먼저 보고, 마지막에
    영상을 읽는다. 어차피 거절할 요청 때문에 100MB를 디스크에 쓰지 않는다.
    """
    group, space = AnalysisCreateForm(
        animal_group=animal_group, space_type=space_type
    ).validated()

    _reject_if_queue_full(queue)

    info = await storage.save_video(video)

    try:
        # 기기당 동시 1건 제한은 서비스 계층이 확인한다. 위반 시 409가 나므로
        # 이미 저장한 영상을 지운다. 참조 없는 파일은 이후 추적할 수 없다.
        row = await service.create(device_id, group, space, info.relative_path)
        # 워커가 별도 세션으로 이 행을 조회한다. 커밋 전에 등록하면 워커가
        # 행을 찾지 못해 첫 단계 진행률이 유실된다.
        await session.commit()
    except Exception:
        storage.delete(info.relative_path)
        raise

    try:
        queue.submit(row.analysis_id, info.path, group, space)
    except PetFitError:
        # 사전 검사와 등록 사이에 대기열이 찬 경우다. 행을 지우는 대신 실패로
        # 남긴다. 사용자는 여유가 생겼을 때 재시도할 수 있고, 영상을 다시
        # 올리지 않아도 된다.
        await service.mark_failed(row, QUEUE_FULL_MESSAGE)
        await session.commit()
        raise

    return AnalysisAccepted(analysis_id=row.analysis_id)


# =============================================================================
# API-003 분석 이력 조회
# =============================================================================


@router.get(
    "",
    response_model=AnalysisListOut,
    summary="분석 이력 조회",
)
async def list_analyses(
    device_id: DeviceId,
    service: ServiceDep,
    page: Annotated[int, Query(description="페이지 번호 (1부터 시작)")] = 1,
    size: Annotated[int, Query(description=f"페이지당 항목 수 (최대 {PAGE_SIZE_MAX})")] = (
        PAGE_SIZE_DEFAULT
    ),
    status: Annotated[str | None, Query(description="분석 상태 필터")] = None,
) -> AnalysisListOut:
    """요청한 기기의 분석 이력만 페이지 단위로 반환한다.

    정렬은 `createdAt` 내림차순으로 고정한다.

    목록에 `FAILED` 항목이 함께 표시되므로 항목마다 `canRetry` 를 실어 보낸다.
    이 값이 없으면 클라이언트는 재시도를 눌러 409를 받아야 불가능함을 알게 된다.
    """
    page, size, status_filter = AnalysisListQuery(
        page=page, size=size, status=status
    ).validated()

    rows, total, retryable = await service.list_history(
        device_id, page, size, status_filter
    )

    return AnalysisListOut(
        analyses=[
            AnalysisSummaryOut.from_model(
                row, can_retry=row.analysis_id in retryable
            )
            for row in rows
        ],
        pagination=Pagination.build(page, size, total),
    )


# =============================================================================
# API-006 분석 진행 상태 조회
# =============================================================================


@router.get(
    "/{analysis_id}/status",
    response_model=AnalysisStatusOut,
    summary="분석 진행 상태 조회",
)
async def get_analysis_status(
    analysis_id: AnalysisId,
    device_id: DeviceId,
    service: ServiceDep,
) -> AnalysisStatusOut:
    """분석 상태·단계·진행률을 반환한다. 클라이언트는 2초 주기로 폴링한다.

    실패해도 `stage` 와 `progress` 를 0으로 되돌리지 않는다. 실패 지점이
    남아야 클라이언트가 재촬영과 재시도 중 무엇을 안내할지 분기할 수 있다.
    프레임 추출·객체 탐지에서 실패했다면 영상 품질 문제일 가능성이 높아
    재촬영을, 점수 산출·환경 분석에서 실패했다면 재시도를 권한다.
    """
    row = await service.get_owned(analysis_id, device_id)
    return AnalysisStatusOut.from_model(row, can_retry=await service.can_retry(row))


# =============================================================================
# API-007 실패한 분석 재시도
# =============================================================================


@router.post(
    "/{analysis_id}/retry",
    status_code=202,
    response_model=RetryAccepted,
    summary="실패한 분석 재시도",
)
async def retry_analysis(
    analysis_id: AnalysisId,
    device_id: DeviceId,
    session: DbSession,
    service: ServiceDep,
    storage: StorageDep,
    queue: QueueDep,
) -> RetryAccepted:
    """이미 업로드된 영상으로 분석을 다시 수행한다. 영상을 재전송하지 않는다.

    기존 `analysisId` 를 그대로 쓴다. 새 ID를 만들면 동일한 영상에 대한 실패
    기록이 이력에 중복으로 쌓인다.

    이전 결과와 생성된 이미지 파일 정리는 서비스 계층이 처리한다. 파일명이
    UUID라 재생성 시 새 이름이 붙으므로, 지우지 않으면 DB 참조가 끊긴 채
    디스크에 잔류한다.
    """
    _reject_if_queue_full(queue)

    row = await service.prepare_retry(analysis_id, device_id)
    await session.commit()

    try:
        queue.submit(
            row.analysis_id,
            video_file(storage, row.video_path),
            AnimalGroup(row.animal_group),
            SpaceType(row.space_type),
        )
    except PetFitError:
        await service.mark_failed(row, QUEUE_FULL_MESSAGE)
        await session.commit()
        raise

    return RetryAccepted(analysis_id=row.analysis_id, retry_count=row.retry_count)


# =============================================================================
# API-004 분석 결과 상세 조회
# =============================================================================


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetailOut,
    summary="분석 결과 상세 조회",
)
async def get_analysis(
    analysis_id: AnalysisId,
    device_id: DeviceId,
    service: ServiceDep,
) -> AnalysisDetailOut:
    """완료된 분석의 상세 결과를 반환한다.

    `status` 가 `COMPLETED` 가 아니면 409를 반환한다. 진행 중인 분석은 점수와
    이미지 경로가 아직 없으므로 상세 응답을 구성할 수 없다.
    """
    detail = await service.get_detail(analysis_id, device_id)
    return AnalysisDetailOut.from_model(
        detail.analysis, detail.objects, detail.recommendations
    )


# =============================================================================
# API-005 분석 결과 삭제
# =============================================================================


@router.delete(
    "/{analysis_id}",
    response_model=MessageResponse,
    summary="분석 결과 삭제",
)
async def delete_analysis(
    analysis_id: AnalysisId,
    device_id: DeviceId,
    session: DbSession,
    service: ServiceDep,
) -> MessageResponse:
    """분석 결과와 함께 업로드 영상·생성 이미지를 삭제한다.

    진행 중인 분석은 삭제할 수 없다. 백그라운드 작업이 삭제된 행을 갱신하려
    시도하고, 쓰는 중인 파일이 남는다. `PROCESSING` 은 처리 제한 시간을
    초과하면 `FAILED` 로 전환되므로 그 뒤에 삭제할 수 있다.
    """
    await service.delete(analysis_id, device_id)
    await session.commit()
    return MessageResponse(message="분석 결과가 삭제되었습니다.")
