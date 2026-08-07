"""Recommendation 테이블.

LLM이 생성한 환경 개선 추천을 유형·우선순위·판단 근거와 함께 저장한다.
``UNIQUE (analysis_id, priority)`` 로 우선순위 중복을 차단한다.
LLM이 중복된 우선순위를 생성할 수 있으므로 데이터베이스에서 막는다.
"""

from sqlalchemy import (
    BigInteger, CheckConstraint, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

__all__ = ["Recommendation"]


class Recommendation(Base):
    __tablename__ = "Recommendation"

    recommendation_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("Analysis.analysis_id", ondelete="CASCADE"), nullable=False
    )

    recommendation_type: Mapped[str] = mapped_column(String(30), nullable=False)
    recommendation_text: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    source: Mapped[str] = mapped_column(String(20), nullable=False, server_default="DETECTED")

    analysis = relationship("Analysis", back_populates="recommendations")

    __table_args__ = (
        CheckConstraint(
            "recommendation_type IN ('SAFETY','ACTIVITY','REST','ENVIRONMENT')",
            name="ck_recommendation_type",
        ),
        CheckConstraint("source IN ('DETECTED','OBSERVED')", name="ck_recommendation_source"),
        CheckConstraint("priority >= 1", name="ck_recommendation_priority"),
        UniqueConstraint("analysis_id", "priority", name="uq_recommendation_priority"),
        Index("ix_recommendation_analysis", "analysis_id"),
    )
