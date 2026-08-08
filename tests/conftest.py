"""pytest 공통 설정."""

import sys
import warnings
from pathlib import Path

import pytest
from sqlalchemy import Integer, MetaData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import Base  # noqa: E402

#: 팀 표준 파이썬 버전.
REQUIRED_PYTHON = (3, 10)


def pytest_configure(config) -> None:
    """표준과 다른 파이썬으로 실행하면 경고한다.

    상위 버전에서는 3.11+ 전용 기능(asyncio.timeout, TaskGroup, StrEnum 등)을
    써도 테스트가 통과한다. 그 코드는 3.10 사용자 환경에서만 깨지므로,
    개발 단계에서 알아채지 못하면 발견이 늦어진다.
    """
    actual = sys.version_info[:2]
    if actual != REQUIRED_PYTHON:
        warnings.warn(
            f"팀 표준은 Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} 이다. "
            f"현재 {actual[0]}.{actual[1]} 로 실행 중이다. "
            "상위 버전 전용 문법을 써도 여기서는 통과하지만 다른 팀원 환경에서 깨진다.",
            RuntimeWarning,
            stacklevel=2,
        )


def sqlite_metadata() -> MetaData:
    """MySQL 모델을 SQLite에서 실행하기 위한 스키마 사본.

    **CHECK 제약은 그대로 유지한다.** 상태 정합성 검증이 테스트의 핵심이므로
    제약을 빼면 검증 의미가 사라진다.

    MySQL 서버가 없어도 서비스 계층을 검증할 수 있어야 CI에서 돌릴 수 있다.
    """
    md = MetaData()
    for table in Base.metadata.tables.values():
        copy = table.to_metadata(md)
        for col in copy.columns:
            # SQLite는 INTEGER PRIMARY KEY만 자동 증가한다. BIGINT는 증가하지 않는다.
            if col.primary_key:
                col.type = Integer()
            # MySQL 전용 기본값: JSON_ARRAY(), CURRENT_TIMESTAMP(6)
            default = getattr(col.server_default, "arg", "")
            if col.server_default is not None and "(" in str(default):
                col.server_default = None
    return md


@pytest.fixture
def storage(tmp_path):
    """임시 디렉터리를 쓰는 저장소."""
    from app.services.storage import Storage

    return Storage(tmp_path / "storage")
