"""SQLAlchemy 모델.

``Base.metadata`` 가 모든 테이블을 인식하도록 이곳에서 모두 import 한다.
Alembic 자동 생성이 테이블을 놓치지 않게 하는 목적도 있다.
"""

from app.models.analysis import Analysis
from app.models.base import Base
from app.models.detected_object import DetectedObject
from app.models.recommendation import Recommendation

__all__ = ["Base", "Analysis", "DetectedObject", "Recommendation"]
