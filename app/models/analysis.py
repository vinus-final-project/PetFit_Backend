"""Analysis 테이블.

영상 기반 AI 분석 결과를 저장하는 메인 테이블이다.
컬럼 상세와 제약은 DB 명세서를 따른다.

소수 컬럼은 ``NUMERIC`` 을 사용한다. ``FLOAT`` 는 이진 부동소수점이라
저장·조회 과정에서 오차가 발생하고, 정렬과 임계값 판정을 바꾼다.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger, CheckConstraint, Index, Integer, Numeric, String, Text, text,
)
from sqlalchemy.dialects.mysql import DATETIME, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TABLE_ARGS

__all__ = ["Analysis"]


class Analysis(Base):
    __tablename__ = "analysis"

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
    risk_factors: Mapped[list] = mapped_column(JSON, nullable=False, server_default=text("(JSON_ARRAY())"))
    analysis_result: Mapped[list] = mapped_column(JSON, nullable=False, server_default=text("(JSON_ARRAY())"))

    # --- 시각 ---
    #: **KST(UTC+9) 기준으로 저장한다.** 한국은 서머타임이 없어 고정 오프셋으로 충분하다.
    #: MySQL에는 타임존을 보존하는 타입이 없다. TIMESTAMP는 2038년 상한이 있어 사용하지 않는다.
    #: CURRENT_TIMESTAMP(6)은 세션 타임존을 따르므로 연결 시점에 +09:00으로 고정한다.
    #: 고정 작업은 app/db/session.py 의 CONNECT_ARGS 가 담당한다.
    created_at: Mapped[datetime] = mapped_column(
        DATETIME(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)")
    )
    completed_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6), nullable=True)

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
        # PENDING 상태에서는 아직 영상을 분석하지 않아 0이다. frame_count와 동일하게 0을 허용한다.
        CheckConstraint(
            "capture_duration = 0 OR capture_duration BETWEEN 3 AND 30",
            name="ck_analysis_capture_duration",
        ),
        CheckConstraint("frame_count = 0 OR frame_count BETWEEN 15 AND 30", name="ck_analysis_frame_count"),
        # FAILED는 실패한 단계를 보존한다. 실패 지점에 따라 클라이언트가
        # 재촬영과 재시도 중 무엇을 안내할지 분기하기 때문이다.
        # 접수 직후(파이프라인 진입 전) 실패는 stage가 NULL일 수 있다.
        CheckConstraint(
            "(status = 'PROCESSING' AND stage IS NOT NULL)"
            " OR (status = 'FAILED')"
            " OR (status IN ('PENDING','COMPLETED') AND stage IS NULL)",
            name="ck_analysis_stage_consistency",
        ),
        # progress는 stage에서 파생되는 값이라 어긋날 수 있다.
        # 단계별 수치는 조정될 수 있으므로 CHECK에 넣지 않고,
        # 변하지 않는 경계값(PENDING=0, COMPLETED=100)만 강제한다.
        # 단계별 매핑은 AnalysisStage.progress 가 유일한 정본이다.
        CheckConstraint(
            "(status = 'PENDING' AND progress = 0)"
            " OR (status = 'COMPLETED' AND progress = 100)"
            " OR (status IN ('PROCESSING','FAILED'))",
            name="ck_analysis_progress_consistency",
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
        TABLE_ARGS,
    )
