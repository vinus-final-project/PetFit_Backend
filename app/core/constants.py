"""파이프라인 임계값 상수.

성능평가 결과로 조정되는 값을 한곳에 모은다.
DB CHECK 제약에는 포함하지 않는다. 값이 바뀔 때마다 스키마 마이그레이션이 발생한다.
"""

__all__ = [
    "FRAME_RATE", "FRAME_MIN", "FRAME_MAX", "FRAME_MAX_EDGE",
    "DETECTION_CONFIDENCE", "ADOPTION_CONFIDENCE", "MIN_DETECTION_FRAMES",
    "DETECT_CHUNK_FRAMES",
    "TRACKING_IOU_THRESHOLD",
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
#: 보관하는 프레임의 긴 변 픽셀 상한.
#: 추론은 640 letterbox 이고 마킹 이미지는 화면 표시용이라 이 정도면 충분하다.
#: 4K 원본을 그대로 들고 있으면 30장에 700MB가 넘어 동시 처리에서 메모리가 터진다.
FRAME_MAX_EDGE = 1280

# --- 탐지 신뢰 기준 ------------------------------------------------------
#: 프레임 단위 탐지 임계값
DETECTION_CONFIDENCE = 0.25
#: 추적 통합 후 객체 채택 임계값
ADOPTION_CONFIDENCE = 0.40
#: 채택에 필요한 최소 탐지 프레임 수
MIN_DETECTION_FRAMES = 2

#: 취소를 확인하는 간격(프레임).
#:
#: 추론은 스레드에서 실행되는데, **파이썬은 스레드를 강제 종료할 수 없다.**
#: 처리 제한 시간을 넘겨 작업이 취소돼도 스레드는 끝까지 돌며 CPU와 GPU를 계속
#: 점유한다. 이미 느린 상태에서 좀비가 쌓이면 동시 처리 제한이 무의미해지고,
#: 기본 스레드 풀(코어 수 + 4)이 소진되면 정상 요청까지 멈춘다.
#:
#: 프레임을 나눠 추론하면 사이사이가 취소 확인 지점이 되어, 낭비되는 작업이
#: 최대 이 값만큼으로 줄어든다. 없앨 수는 없고 줄일 수만 있다.
DETECT_CHUNK_FRAMES = 8

# --- 객체 추적 -----------------------------------------------------------
#: 같은 물체로 볼 최소 IoU.
#:
#: **실측되지 않은 값이다.** 초당 3프레임이라 프레임 간격이 0.33초이고, 카메라가
#: 방을 훑는 중이면 같은 물체의 박스가 크게 이동한다. 높이면 하나의 소파가 여러
#: 건으로 나열되고, 낮추면 나란히 놓인 의자 두 개가 하나로 합쳐진다.
#: 점수는 존재 여부로 판정하므로 과합침보다 미합침이 눈에 띈다.
TRACKING_IOU_THRESHOLD = 0.30

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
