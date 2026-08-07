"""API 오류 코드와 예외.

모든 오류 응답은 code / message / field / status 형식을 따른다.
``code`` 는 클라이언트 분기 처리에, ``message`` 는 화면 표시에 사용한다.
서버 내부 오류 메시지를 ``message`` 에 그대로 노출하지 않는다.
"""

from enum import Enum
from http import HTTPStatus

__all__ = ["ErrorCode", "PetFitError"]


class ErrorCode(str, Enum):
    """오류 코드. (코드, HTTP 상태, 사용자 메시지)"""

    DEVICE_ID_REQUIRED = "DEVICE_ID_REQUIRED"
    DEVICE_ID_INVALID = "DEVICE_ID_INVALID"
    VIDEO_REQUIRED = "VIDEO_REQUIRED"
    VIDEO_FORMAT_INVALID = "VIDEO_FORMAT_INVALID"
    VIDEO_DURATION_INVALID = "VIDEO_DURATION_INVALID"
    ANIMAL_GROUP_REQUIRED = "ANIMAL_GROUP_REQUIRED"
    ANIMAL_GROUP_UNSUPPORTED = "ANIMAL_GROUP_UNSUPPORTED"
    SPACE_TYPE_REQUIRED = "SPACE_TYPE_REQUIRED"
    SPACE_TYPE_INVALID = "SPACE_TYPE_INVALID"
    PAGE_INVALID = "PAGE_INVALID"
    SIZE_INVALID = "SIZE_INVALID"
    STATUS_INVALID = "STATUS_INVALID"

    ANALYSIS_NOT_FOUND = "ANALYSIS_NOT_FOUND"

    ANALYSIS_IN_PROGRESS = "ANALYSIS_IN_PROGRESS"
    ANALYSIS_NOT_COMPLETED = "ANALYSIS_NOT_COMPLETED"
    ANALYSIS_NOT_DELETABLE = "ANALYSIS_NOT_DELETABLE"
    ANALYSIS_NOT_RETRYABLE = "ANALYSIS_NOT_RETRYABLE"
    RETRY_LIMIT_EXCEEDED = "RETRY_LIMIT_EXCEEDED"

    VIDEO_TOO_LARGE = "VIDEO_TOO_LARGE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    QUEUE_FULL = "QUEUE_FULL"

    @property
    def http_status(self) -> int:
        return _HTTP_STATUS[self]

    @property
    def message(self) -> str:
        return _MESSAGES[self]


_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.DEVICE_ID_REQUIRED: HTTPStatus.BAD_REQUEST,
    ErrorCode.DEVICE_ID_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.VIDEO_REQUIRED: HTTPStatus.BAD_REQUEST,
    ErrorCode.VIDEO_FORMAT_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.VIDEO_DURATION_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.ANIMAL_GROUP_REQUIRED: HTTPStatus.BAD_REQUEST,
    ErrorCode.ANIMAL_GROUP_UNSUPPORTED: HTTPStatus.BAD_REQUEST,
    ErrorCode.SPACE_TYPE_REQUIRED: HTTPStatus.BAD_REQUEST,
    ErrorCode.SPACE_TYPE_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.PAGE_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.SIZE_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.STATUS_INVALID: HTTPStatus.BAD_REQUEST,
    ErrorCode.ANALYSIS_NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorCode.ANALYSIS_IN_PROGRESS: HTTPStatus.CONFLICT,
    ErrorCode.ANALYSIS_NOT_COMPLETED: HTTPStatus.CONFLICT,
    ErrorCode.ANALYSIS_NOT_DELETABLE: HTTPStatus.CONFLICT,
    ErrorCode.ANALYSIS_NOT_RETRYABLE: HTTPStatus.CONFLICT,
    ErrorCode.RETRY_LIMIT_EXCEEDED: HTTPStatus.CONFLICT,
    ErrorCode.VIDEO_TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    ErrorCode.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
    ErrorCode.QUEUE_FULL: HTTPStatus.SERVICE_UNAVAILABLE,
}

_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.DEVICE_ID_REQUIRED: "기기 식별자가 필요합니다.",
    ErrorCode.DEVICE_ID_INVALID: "기기 식별자 형식이 올바르지 않습니다.",
    ErrorCode.VIDEO_REQUIRED: "영상이 첨부되지 않았습니다.",
    ErrorCode.VIDEO_FORMAT_INVALID: "MP4 형식의 영상만 업로드할 수 있습니다.",
    ErrorCode.VIDEO_DURATION_INVALID: "영상 길이는 3초 이상 30초 이하여야 합니다.",
    ErrorCode.ANIMAL_GROUP_REQUIRED: "반려동물 그룹을 선택해주세요.",
    ErrorCode.ANIMAL_GROUP_UNSUPPORTED: "현재 지원하지 않는 반려동물 그룹입니다.",
    ErrorCode.SPACE_TYPE_REQUIRED: "공간 종류를 선택해주세요.",
    ErrorCode.SPACE_TYPE_INVALID: "지원하지 않는 공간 종류입니다.",
    ErrorCode.PAGE_INVALID: "페이지 번호는 1 이상이어야 합니다.",
    ErrorCode.SIZE_INVALID: "페이지당 항목 수는 1 이상 50 이하여야 합니다.",
    ErrorCode.STATUS_INVALID: "지원하지 않는 상태 값입니다.",
    ErrorCode.ANALYSIS_NOT_FOUND: "분석 결과를 찾을 수 없습니다.",
    ErrorCode.ANALYSIS_IN_PROGRESS: "이미 진행 중인 분석이 있습니다.",
    ErrorCode.ANALYSIS_NOT_COMPLETED: "분석이 진행 중입니다.",
    ErrorCode.ANALYSIS_NOT_DELETABLE: "분석이 진행 중이어서 삭제할 수 없습니다.",
    ErrorCode.ANALYSIS_NOT_RETRYABLE: "실패한 분석만 다시 시도할 수 있습니다.",
    ErrorCode.RETRY_LIMIT_EXCEEDED: "재시도 횟수를 초과했습니다. 다시 촬영해주세요.",
    ErrorCode.VIDEO_TOO_LARGE: "영상 크기가 100MB를 초과했습니다.",
    ErrorCode.INTERNAL_ERROR: "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
    ErrorCode.QUEUE_FULL: "요청이 많아 처리할 수 없습니다. 잠시 후 다시 시도해주세요.",
}


class PetFitError(Exception):
    """API 오류. 전역 예외 핸들러가 표준 응답으로 변환한다."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        field: str | None = None,
        status: str | None = None,
        message: str | None = None,
    ) -> None:
        self.code = code
        self.field = field
        self.status = status
        self.message = message or code.message
        super().__init__(self.message)

    def to_response(self) -> dict[str, str | None]:
        """표준 오류 응답 본문을 반환한다."""
        return {
            "code": self.code.value,
            "message": self.message,
            "field": self.field,
            "status": self.status,
        }
