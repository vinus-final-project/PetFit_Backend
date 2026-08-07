"""데이터베이스 초기화.

스키마 생성 → 테이블 생성 → 결과 검증까지 수행한다.

    python -m scripts.init_db          # 없는 테이블만 생성
    python -m scripts.init_db --drop   # 전체 삭제 후 재생성 (개발 전용)
    python -m scripts.init_db --check  # 생성하지 않고 현재 상태만 확인

`--drop` 은 데이터를 모두 지운다. 운영 환경에서는 마이그레이션 도구를 사용한다.
"""

import argparse
import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import DEFAULT_CHARSET, get_settings
from app.db.session import CONNECT_ARGS
from app.utils.timeutil import DB_TIME_ZONE

# 메타데이터 등록을 위해 모든 모델을 임포트한다.
# 임포트하지 않은 모델은 create_all 대상에서 누락된다.
from app.models import Analysis, Base, DetectedObject, Recommendation  # noqa: F401

COLLATION = "utf8mb4_unicode_ci"

#: 생성되어야 할 테이블. 실제 결과와 대조한다.
EXPECTED_TABLES = ("analysis", "detected_object", "recommendation")


def _print(mark: str, message: str) -> None:
    print(f"  {mark} {message}")


async def ensure_schema() -> None:
    """스키마가 없으면 생성한다.

    스키마 자체가 없으면 접속 단계에서 실패하므로, 서버에만 붙어서 먼저 만든다.
    """
    settings = get_settings()
    server_url = settings.build_url(database="")
    engine = create_async_engine(
        server_url, isolation_level="AUTOCOMMIT", connect_args=CONNECT_ARGS
    )

    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                    "WHERE SCHEMA_NAME = :name"
                ),
                {"name": settings.db_name},
            )
            if result.first():
                _print("·", f"스키마 `{settings.db_name}` 이미 존재")
                return

            await conn.execute(
                text(
                    f"CREATE DATABASE `{settings.db_name}` "
                    f"CHARACTER SET {DEFAULT_CHARSET} COLLATE {COLLATION}"
                )
            )
            _print("+", f"스키마 `{settings.db_name}` 생성 ({DEFAULT_CHARSET})")
    finally:
        await engine.dispose()


async def create_tables(drop: bool) -> None:
    """테이블을 생성한다."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, connect_args=CONNECT_ARGS)

    try:
        async with engine.begin() as conn:
            if drop:
                await conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
                await conn.run_sync(Base.metadata.drop_all)
                await conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
                _print("-", "기존 테이블 삭제")

            await conn.run_sync(Base.metadata.create_all)
        _print("+", "테이블 생성 완료")
    finally:
        await engine.dispose()


async def verify() -> bool:
    """생성 결과를 검증한다.

    Returns:
        기대한 테이블이 모두 존재하고 엔진·문자셋이 올바르면 True.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, connect_args=CONNECT_ARGS)
    ok = True

    try:
        async with engine.connect() as conn:
            version = (await conn.execute(text("SELECT VERSION()"))).scalar_one()
            print(f"\nMySQL {version}")
            if not version.startswith(("8.", "9.")):
                _print("!", "CHECK 제약은 MySQL 8.0.16 이상에서만 동작한다")
                ok = False

            # created_at 은 CURRENT_TIMESTAMP(6) 로 채워지며 세션 타임존을 따른다.
            # 고정되지 않으면 서버 설정에 따라 시각이 어긋난다.
            tz, db_now = (
                await conn.execute(text("SELECT @@session.time_zone, NOW(6)"))
            ).one()
            print("\n타임존")
            if tz == DB_TIME_ZONE:
                _print("+", f"세션 타임존 {tz} (KST)  현재 시각 {db_now}")
            else:
                _print("!", f"세션 타임존이 {tz} 다. {DB_TIME_ZONE} 이어야 한다")
                ok = False

            rows = (
                await conn.execute(
                    text(
                        "SELECT TABLE_NAME, ENGINE, TABLE_COLLATION, TABLE_ROWS "
                        "FROM information_schema.TABLES "
                        "WHERE TABLE_SCHEMA = :db ORDER BY TABLE_NAME"
                    ),
                    {"db": settings.db_name},
                )
            ).all()

            print("\n테이블")
            found = {r[0] for r in rows}
            for name, engine_name, collation, count in rows:
                mark = "+" if engine_name == "InnoDB" else "!"
                _print(mark, f"{name:<18} {engine_name:<8} {collation:<20} {count or 0}행")
                if engine_name != "InnoDB":
                    ok = False

            missing = set(EXPECTED_TABLES) - found
            for name in sorted(missing):
                _print("!", f"{name} 누락")
                ok = False

            idx = (
                await conn.execute(
                    text(
                        "SELECT TABLE_NAME, INDEX_NAME, "
                        "GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) "
                        "FROM information_schema.STATISTICS "
                        "WHERE TABLE_SCHEMA = :db "
                        "GROUP BY TABLE_NAME, INDEX_NAME ORDER BY TABLE_NAME, INDEX_NAME"
                    ),
                    {"db": settings.db_name},
                )
            ).all()

            print("\n인덱스")
            for table, index, columns in idx:
                _print("·", f"{table:<18} {index:<28} ({columns})")

            fks = (
                await conn.execute(
                    text(
                        "SELECT TABLE_NAME, CONSTRAINT_NAME, DELETE_RULE "
                        "FROM information_schema.REFERENTIAL_CONSTRAINTS "
                        "WHERE CONSTRAINT_SCHEMA = :db ORDER BY TABLE_NAME"
                    ),
                    {"db": settings.db_name},
                )
            ).all()

            print("\n외래키")
            for table, name, rule in fks:
                mark = "+" if rule == "CASCADE" else "!"
                _print(mark, f"{table:<18} {name:<32} ON DELETE {rule}")
                if rule != "CASCADE":
                    ok = False
    finally:
        await engine.dispose()

    return ok


async def main(drop: bool, check_only: bool) -> int:
    settings = get_settings()
    print(f"대상: {settings.safe_database_url}\n")

    try:
        if not check_only:
            await ensure_schema()
            await create_tables(drop)
        ok = await verify()
    except Exception as exc:  # noqa: BLE001
        print(f"\n실패: {type(exc).__name__}: {exc}")
        print("\n확인 사항")
        print("  1. MySQL 서버가 실행 중인지")
        print("  2. .env 의 DB_USER / DB_PASSWORD 가 올바른지")
        print(f"  3. {settings.db_host}:{settings.db_port} 로 접속 가능한지")
        return 1

    print("\n검증 통과" if ok else "\n검증 실패 — 위 ! 항목 확인")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PetFit 데이터베이스 초기화")
    parser.add_argument("--drop", action="store_true", help="기존 테이블 삭제 후 재생성")
    parser.add_argument("--check", action="store_true", help="생성하지 않고 상태만 확인")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(drop=args.drop, check_only=args.check)))
