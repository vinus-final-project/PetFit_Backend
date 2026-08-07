"""Analysis 테이블.

영상 기반 AI 분석 결과를 저장하는 메인 테이블이다.
컬럼 상세와 제약은 DB 명세서를 따른다.

소수 컬럼은 ``NUMERIC`` 을 사용한다. ``FLOAT`` 는 이진 부동소수점이라
저장·조회 과정에서 오차가 발생하고, 정렬과 임계값 판정을 바꾼다.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, Index, Integer, Numeric, String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

__all__ = ["Analysis"]


class Analysis(Base):
    __tablename__ = "Analysis"

    analysis_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # --- 식별 ---
    device_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # --- 분석 조건 ---
    animal_group: Mapped[str] = mapped_column(String(30), nullable=False)
    space_type: Mapped[str] = mapped_column(String(30), nullable=False)

    # --- 상태 ---
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="PENDING")
    stage: Mapped[str | None] = mapped_column(String(30), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # --- 입력 ---
    video_path: Mapped[str] = mapped_column(String(255), nullable=False)
    capture_duration: Mapped[float] = mapped_column(Numeric(4, 1), nullable=False, server_default="0")
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # --- 산출 ---
    thumbnail_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    occupancy_ratio: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, server_default="0")
    total_score: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    safety_score: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    activity_score: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    rest_score: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    environment_score: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # --- 서술 ---
    risk_factors: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="'[]'")
    analysis_result: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="'[]'")

    # --- 시각 ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    detected_objects = relationship(
        "DetectedObject", back_populates="analysis", cascade="all, delete-orphan", passive_deletes=True
    )
    recommendations = relationship(
        "Recommendation", back_populates="analysis", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING','PROCESSING','COMPLETED','FAILED')", name="ck_analysis_status"
        ),
        CheckConstraint(
            "stage IS NULL OR stage IN ('FRAME_EXTRACTION','OBJECT_DETECTION','OBJECT_TRACKING',"
            "'FRAME_SELECTION','RISK_MARKING','SCORE_CALCULATION','ENVIRONMENT_ANALYSIS')",
            name="ck_analysis_stage",
        ),
        CheckConstraint(
            "animal_group IN ('small_dog','large_dog','cat')", name="ck_analysis_animal_group"
        ),
        CheckConstraint(
            "space_type IN ('living_room','bedroom','kitchen','balcony')", name="ck_analysis_space_type"
        ),
        CheckConstraint("progress BETWEEN 0 AND 100", name="ck_analysis_progress"),
        CheckConstraint("retry_count BETWEEN 0 AND 3", name="ck_analysis_retry_count"),
        CheckConstraint("total_score BETWEEN 0 AND 100", name="ck_analysis_total_score"),
        CheckConstraint("safety_score BETWEEN 0 AND 100", name="ck_analysis_safety_score"),
        CheckConstraint("activity_score BETWEEN 0 AND 100", name="ck_analysis_activity_score"),
        CheckConstraint("rest_score BETWEEN 0 AND 100", name="ck_analysis_rest_score"),
        CheckConstraint("environment_score BETWEEN 0 AND 100", name="ck_analysis_environment_score"),
        CheckConstraint("occupancy_ratio BETWEEN 0 AND 1", name="ck_analysis_occupancy_ratio"),
        CheckConstraint("capture_duration BETWEEN 3 AND 30", name="ck_analysis_capture_duration"),
        CheckConstraint("frame_count = 0 OR frame_count BETWEEN 15 AND 30", name="ck_analysis_frame_count"),
        CheckConstraint(
            "(status = 'PROCESSING') = (stage IS NOT NULL)", name="ck_analysis_stage_consistency"
        ),
        CheckConstraint(
            "(status = 'COMPLETED') = (completed_at IS NOT NULL)", name="ck_analysis_completed_at"
        ),
        CheckConstraint(
            "(status = 'FAILED') = (error_message IS NOT NULL)", name="ck_analysis_error_message"
        ),
        Index("ix_analysis_device_created", "device_id", created_at.desc()),
        Index("ix_analysis_device_status_created", "device_id", "status", created_at.desc()),
        Index("ix_analysis_animal_group", "animal_group"),
    )
