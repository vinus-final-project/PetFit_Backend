"""Detected_Object 테이블.

탐지 신뢰 기준을 통과한 객체를 **인스턴스 단위**로 저장한다.
동일 클래스가 여러 개 탐지되면 각각 별도 행이 된다.

좌표는 해당 객체의 ``frame_number`` 프레임을 기준으로 하는 정규화 좌표이다.
객체마다 기준 프레임이 다르므로 서로 다른 객체의 좌표를 한 이미지에 함께 그리지 않는다.
"""

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

__all__ = ["DetectedObject"]


class DetectedObject(Base):
    __tablename__ = "Detected_Object"

    object_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("Analysis.analysis_id", ondelete="CASCADE"), nullable=False
    )

    object_name: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, server_default="0")
    detection_frame_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False, server_default="SAFE")

    #: 해당 객체가 가장 잘 보이는 프레임. 객체마다 다를 수 있다.
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    #: 위험 객체만 생성한다. SAFE는 NULL.
    marked_image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)

    x: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, server_default="0")
    y: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, server_default="0")
    width: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, server_default="0")
    height: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, server_default="0")

    analysis = relationship("Analysis", back_populates="detected_objects")

    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('HIGH','MEDIUM','LOW','SAFE')", name="ck_detected_object_risk_level"
        ),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_detected_object_confidence"),
        CheckConstraint("detection_frame_count >= 1", name="ck_detected_object_frame_count"),
        CheckConstraint("frame_number >= 1", name="ck_detected_object_frame_number"),
        CheckConstraint("x BETWEEN 0 AND 1 AND y BETWEEN 0 AND 1", name="ck_detected_object_xy"),
        CheckConstraint(
            "width > 0 AND width <= 1 AND height > 0 AND height <= 1", name="ck_detected_object_wh"
        ),
        CheckConstraint("x + width <= 1 AND y + height <= 1", name="ck_detected_object_bounds"),
        CheckConstraint(
            "risk_level <> 'SAFE' OR marked_image_path IS NULL", name="ck_detected_object_marking"
        ),
        Index("ix_detected_object_analysis", "analysis_id"),
        Index("ix_detected_object_name", "object_name"),
        Index("ix_detected_object_risk", "risk_level"),
    )
