"""SQLAlchemy 선언적 베이스."""

from sqlalchemy.orm import DeclarativeBase

__all__ = ["Base"]


class Base(DeclarativeBase):
    """모든 모델의 베이스."""
