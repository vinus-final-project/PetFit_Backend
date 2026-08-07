"""데이터베이스 세션 관리.

비동기 SQLAlchemy 세션을 FastAPI 의존성으로 제공한다.
분석은 백그라운드에서 장시간 수행되므로, 요청 세션과 작업 세션을 분리한다.

**연결 시점에 세션 타임존을 KST로 고정한다.** ``CURRENT_TIMESTAMP(6)`` 는 세션
타임존을 따르므로, 고정하지 않으면 서버 OS 설정에 따라 ``created_at`` 이
달라진다. 개발 장비와 배포 서버의 타임존이 다르면 같은 코드가 다른 값을 남긴다.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine,
)

from app.core.config import get_settings
from app.utils.timeutil import DB_TIME_ZONE

__all__ = ["engine", "SessionLocal", "get_session", "session_scope", "CONNECT_ARGS"]

_settings = get_settings()

#: 모든 연결에 적용할 초기화 명령. 풀에서 새 연결을 만들 때마다 실행된다.
CONNECT_ARGS = {"init_command": f"SET time_zone = '{DB_TIME_ZONE}'"}

engine: AsyncEngine = create_async_engine(
    _settings.database_url,
    echo=_settings.debug,
    pool_pre_ping=True,
    connect_args=CONNECT_ARGS,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 의존성. 요청 단위 세션을 제공한다."""
    async with SessionLocal() as session:
        yield session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """백그라운드 작업용 세션 컨텍스트.

    요청 생명주기와 무관하게 동작하므로 커밋·롤백을 직접 처리한다.
    """
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
