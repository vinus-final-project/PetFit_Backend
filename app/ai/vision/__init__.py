"""Vision 파이프라인.

AI 설계서의 1~10단계 중 생성형 AI를 제외한 구간을 담당한다.

    1  Video Processor              frames.py
    2  Frame Extractor              frames.py
    3  Object Detection             detector.py
    4  Occupancy Calculator         occupancy.py
    5  Object Tracking              tracking.py
    6  Detection Filter             tracking.py
    7  Object Name Mapper           rules/object_map.py (완성)
    8  Risk Classifier              rules/risk_rules.py (완성)
    9  Representative Frame         imaging.py
    10 Risk Object Marking          imaging.py

**12단계(환경 분석)는 여기 없다.** 생성형 AI는 별도 담당이며, 두 산출물을
합쳐 ``Pipeline`` 계약을 만족시키는 조립은 상위에서 한다.
"""

from app.ai.vision.detector import (
    DEFAULT_SCENE,
    Detector,
    PlantedObject,
    StubDetector,
    clamp_box,
)
from app.ai.vision.pipeline import VisionPipeline
from app.ai.vision.tracking import IouTracker, Tracker, adopt

# ultralytics 를 최상단에서 임포트하지 않으므로 여기서 꺼내도 torch 가 딸려오지 않는다.
from app.ai.vision.yolo_detector import YoloDetector
from app.ai.vision.types import (
    Detection,
    Frame,
    ImageSink,
    TrackedObject,
    VisionResult,
)

__all__ = [
    "Frame",
    "Detection",
    "TrackedObject",
    "VisionResult",
    "ImageSink",
    "Detector",
    "StubDetector",
    "PlantedObject",
    "DEFAULT_SCENE",
    "clamp_box",
    "Tracker",
    "IouTracker",
    "adopt",
    "YoloDetector",
    "VisionPipeline",
]
