"""파이프라인 임계값 상수.

성능평가 결과로 조정되는 값을 한곳에 모은다.
DB CHECK 제약에는 포함하지 않는다. 값이 바뀔 때마다 스키마 마이그레이션이 발생한다.
"""

__all__ = [
    "FRAME_RATE", "FRAME_MIN", "FRAME_MAX",
    "DETECTION_CONFIDENCE", "ADOPTION_CONFIDENCE", "MIN_DETECTION_FRAMES",
    "OCCUPANCY_THRESHOLDS",
    "VIDEO_MAX_BYTES", "VIDEO_MIN_SECONDS", "VIDEO_MAX_SECONDS",
    "PROCESSING_TIMEOUT_SECONDS", "MAX_RETRY_COUNT",
    "MAX_CONCURRENT_ANALYSIS", "MAX_QUEUE_SIZE", "MAX_ANALYSIS_PER_DEVICE",
    "LLM_MAX_IMAGES", "PAGE_SIZE_DEFAULT", "PAGE_SIZE_MAX",
]

# --- 프레임 추출 ---------------------------------------------------------
#: 기준 추출률(초당 프레임). 결과가 범위를 벗어나면 간격을 조정한다.
FRAME_RATE = 3
FRAME_MIN = 15
FRAME_MAX = 30

# --- 탐지 신뢰 기준 ------------------------------------------------------
#: 프레임 단위 탐지 임계값
DETECTION_CONFIDENCE = 0.25
#: 추적 통합 후 객체 채택 임계값
ADOPTION_CONFIDENCE = 0.40
#: 채택에 필요한 최소 탐지 프레임 수
MIN_DETECTION_FRAMES = 2

# --- 활동 공간 점유율 ----------------------------------------------------
#: (임계값, 감점률). 오름차순으로 평가하며 초과 시 1.0을 적용한다.
OCCUPANCY_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (0.40, 0.0),
    (0.60, 0.4),
    (0.75, 0.7),
)

# --- 업로드 제한 ---------------------------------------------------------
VIDEO_MAX_BYTES = 100 * 1024 * 1024
VIDEO_MIN_SECONDS = 3.0
VIDEO_MAX_SECONDS = 30.0

# --- 처리 제한 -----------------------------------------------------------
PROCESSING_TIMEOUT_SECONDS = 180
MAX_RETRY_COUNT = 3
MAX_CONCURRENT_ANALYSIS = 2
MAX_QUEUE_SIZE = 10
MAX_ANALYSIS_PER_DEVICE = 1

# --- 환경 분석 -----------------------------------------------------------
#: LLM에 전송하는 원본 프레임 최대 장수
LLM_MAX_IMAGES = 4

# --- 페이지네이션 --------------------------------------------------------
PAGE_SIZE_DEFAULT = 20
PAGE_SIZE_MAX = 50
