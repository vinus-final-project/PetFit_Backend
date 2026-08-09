"""애플리케이션 진입점.

앱 조립·전역 예외 처리·CORS·정적 파일 서빙·수명주기를 담당한다.

    uvicorn app.main:app --reload

**모든 오류는 하나의 형식으로 나간다.** `code` / `message` / `field` / `status`.
라우터가 개별적으로 오류 응답을 만들면 클라이언트는 엔드포인트마다 다른 본문을
처리해야 한다. 예외를 던지는 것은 서비스 계층이고, 응답으로 바꾸는 것은 여기다.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ai.stub import StubPipeline
from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.exceptions import ErrorCode, PetFitError
from app.db.session import session_scope
from app.services.analysis_service import AnalysisService
from app.services.queue import AnalysisQueue
from app.services.storage import Storage

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)

#: 요청 검증 실패를 명세상의 오류 코드로 옮기는 표.
#:
#: 요청 DTO는 필드를 Optional로 선언하고 직접 검증하므로, FastAPI 단계에서
#: 걸리는 것은 타입이 어긋난 쿼리 파라미터와 경로 변수뿐이다.
_QUERY_ERROR_CODES: dict[str, ErrorCode] = {
    "page": ErrorCode.PAGE_INVALID,
    "size": ErrorCode.SIZE_INVALID,
    "status": ErrorCode.STATUS_INVALID,
}


def _error_response(error: PetFitError) -> JSONResponse:
    """표준 오류 본문을 응답으로 만든다."""
    return JSONResponse(
        status_code=error.code.http_status, content=error.to_response()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """시작 시 큐를 준비하고 중단된 분석을 정리한다.

    큐는 인프로세스라 **재시작하면 사라진다.** DB에 남은 PENDING·PROCESSING 행을
    정리하지 않으면 해당 기기는 새 분석도, 삭제도, 재시도도 할 수 없는 상태로
    영구히 잠긴다. 기기당 동시 1건 제한이 사라지지 않는 유령 작업에 묶이기 때문이다.
    """
    storage: Storage = app.state.storage

    # 파이프라인 교체 지점. 실제 구현이 완성되면 이 한 줄만 바꾼다.
    # API·서비스·DB·프론트는 수정하지 않는다.
    app.state.queue = AnalysisQueue(StubPipeline(), storage)

    try:
        async with session_scope() as session:
            service = AnalysisService(session, storage)
            await service.cleanup_interrupted()
            # 마킹 이미지는 DB에 기록되기 전에 디스크에 먼저 쓰인다. 그 사이에
            # 분석이 취소되면 경로가 어디에도 남지 않아 재시도·삭제로 정리되지
            # 않는다. 시작할 때 참조 없는 파일을 회수한다.
            await service.cleanup_orphan_images()
    except Exception:  # noqa: BLE001
        # DB가 준비되지 않아도 앱은 뜨게 둔다. /animals·/spaces 는 DB 없이 동작하므로
        # 프론트가 화면 개발을 계속할 수 있다. 다만 정리되지 않은 행이 남았음을 남긴다.
        logger.exception("중단된 분석 정리에 실패했다. DB 연결을 확인해야 한다")

    yield

    await app.state.queue.drain()


def create_app() -> FastAPI:
    """앱을 조립한다. 테스트가 독립된 인스턴스를 만들 수 있도록 함수로 둔다."""
    settings = get_settings()

    app = FastAPI(
        title="Pet Fit API",
        description="영상 기반 AI 반려동물 생활환경 분석 서비스",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 저장소는 생성자에서 디렉터리를 만든다. 정적 파일 마운트보다 먼저 있어야 한다.
    app.state.storage = Storage(settings.storage_root)

    # 회원 기능이 없어 쿠키·인증 헤더를 쓰지 않는다. 자격 증명을 허용하지 않으므로
    # 와일드카드 출처가 안전하다. Capacitor 앱은 capacitor://localhost 등
    # 고정되지 않은 출처로 요청한다.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_exception_handlers(app)

    app.include_router(api_router)

    # 마킹 이미지와 대표 프레임만 공개한다. **업로드 영상은 서빙하지 않는다.**
    # 파일명이 UUID라 추측은 어렵지만, 생활공간 원본 영상을 공개 경로에 두지 않는다.
    app.mount(
        "/images",
        StaticFiles(directory=app.state.storage.image_dir),
        name="images",
    )

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """전역 예외 핸들러를 등록한다."""

    @app.exception_handler(PetFitError)
    async def _handle_petfit_error(
        request: Request, exc: PetFitError
    ) -> JSONResponse:
        """서비스 계층이 던진 오류를 명세 형식으로 변환한다."""
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """FastAPI의 422를 명세상의 오류 코드로 바꾼다.

        기본 422 본문은 `code` 가 없어 클라이언트가 분기할 수 없다.
        """
        for error in exc.errors():
            location = error.get("loc") or ()
            if not location:
                continue

            # 경로 변수가 정수가 아니면 존재하지 않는 분석이다.
            # 404로 통일해야 ID 존재 여부가 새어나가지 않는다.
            if location[0] == "path":
                return _error_response(PetFitError(ErrorCode.ANALYSIS_NOT_FOUND))

            field = str(location[-1])
            code = _QUERY_ERROR_CODES.get(field)
            if code is not None:
                return _error_response(PetFitError(code, field=field))

            # 본문 검증 실패는 multipart 파싱 자체가 실패한 경우다.
            # 폼 값은 DTO가 직접 검증하므로 여기까지 오지 않는다.
            if location[0] == "body":
                return _error_response(
                    PetFitError(ErrorCode.VIDEO_REQUIRED, field="video")
                )

        logger.warning("예상하지 못한 요청 검증 실패: %s", exc.errors())
        return _error_response(PetFitError(ErrorCode.INTERNAL_ERROR))

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """정의되지 않은 경로·메서드도 같은 형식으로 응답한다.

        문서화된 엔드포인트에서는 발생하지 않으므로, 여기서 쓰는 코드는
        API 명세서의 오류 코드 목록에 포함되지 않는다.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": _http_error_code(exc.status_code),
                "message": "요청을 처리할 수 없습니다.",
                "field": None,
                "status": None,
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """처리되지 않은 예외. 내부 메시지를 응답에 노출하지 않는다."""
        logger.exception(
            "처리되지 않은 오류: %s %s", request.method, request.url.path
        )
        return _error_response(PetFitError(ErrorCode.INTERNAL_ERROR))


def _http_error_code(status_code: int) -> str:
    """HTTP 상태 코드를 문자열 코드로 바꾼다."""
    return {
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
    }.get(status_code, "REQUEST_FAILED")


app = create_app()
