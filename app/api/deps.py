"""API 공통 의존성.

라우터가 반복해서 필요로 하는 것을 모은다.

**검증 로직을 여기서 다시 작성하지 않는다.** `app.schemas.analysis` 의 검증 함수를
그대로 호출한다. 엔드포인트마다 직접 검증하면 같은 상황에 서로 다른 오류 코드가
나가고, 클라이언트는 분기 조건을 엔드포인트별로 따로 관리해야 한다.

저장소와 큐는 애플리케이션 수명 동안 하나만 존재한다. 모듈 전역이 아니라
``app.state`` 에 두어 테스트가 다른 구현으로 교체할 수 있게 한다.
"""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import session_scope
from app.schemas.analysis import validate_device_id
from app.services.analysis_service import AnalysisService
from app.services.queue import AnalysisQueue
from app.services.storage import Storage

__all__ = [
    "get_db",
    "get_device_id",
    "get_storage",
    "get_queue",
    "get_service",
    "video_file",
    "DbSession",
    "DeviceId",
    "StorageDep",
    "QueueDep",
    "ServiceDep",
]


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """요청 단위 세션을 제공한다. 정상 종료 시 커밋하고 예외 시 롤백한다.

    FastAPI는 라우터에서 발생한 예외를 이 제너레이터 안으로 다시 던지므로,
    ``session_scope`` 의 롤백 경로가 그대로 동작한다. 라우터는 커밋을 신경 쓰지
    않아도 되지만, 백그라운드 워커가 조회해야 하는 행은 **라우터에서 먼저 커밋한다.**
    """
    async with session_scope() as session:
        yield session


def get_device_id(x_device_id: Annotated[str | None, Header()] = None) -> str:
    """``X-Device-Id`` 헤더를 검증한다.

    헤더를 필수로 선언하지 않는다. 필수로 두면 FastAPI가 먼저 422를 반환해
    명세상의 ``DEVICE_ID_REQUIRED`` 를 내보낼 수 없다.
    """
    return validate_device_id(x_device_id)


def _from_state(request: Request, name: str):
    """앱 상태에서 구성 요소를 꺼낸다.

    수명주기 시작 전에 접근하면 ``None`` 이 조용히 흘러들어가 엉뚱한 곳에서
    깨지므로, 여기서 즉시 실패시킨다.
    """
    value = getattr(request.app.state, name, None)
    if value is None:
        raise RuntimeError(f"app.state.{name} 가 초기화되지 않았다")
    return value


def get_storage(request: Request) -> Storage:
    """파일 저장소."""
    return _from_state(request, "storage")


def get_queue(request: Request) -> AnalysisQueue:
    """분석 작업 큐."""
    return _from_state(request, "queue")


def get_service(
    session: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[Storage, Depends(get_storage)],
) -> AnalysisService:
    """분석 서비스.

    ``get_db`` 는 요청 안에서 캐시되므로, 라우터가 따로 주입받는 세션과
    같은 인스턴스를 공유한다. 커밋 시점이 어긋나지 않는다.
    """
    return AnalysisService(session, storage)


def video_file(storage: Storage, relative_path: str) -> Path:
    """DB에 저장된 상대 경로를 실제 영상 파일 경로로 바꾼다.

    재시도는 업로드된 영상을 다시 쓰지만 DB에는 ``/videos/<uuid>.mp4`` 형태의
    상대 경로만 있다. 큐에는 실제 경로를 넘겨야 한다.

    파일명만 취해 저장소 디렉터리에 붙인다. DB 값이 오염되어도 저장소 밖을
    가리킬 수 없다.
    """
    return storage.video_dir / Path(relative_path).name


DbSession = Annotated[AsyncSession, Depends(get_db)]
DeviceId = Annotated[str, Depends(get_device_id)]
StorageDep = Annotated[Storage, Depends(get_storage)]
QueueDep = Annotated[AnalysisQueue, Depends(get_queue)]
ServiceDep = Annotated[AnalysisService, Depends(get_service)]
