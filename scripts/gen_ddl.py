"""SQLAlchemy 모델에서 MySQL DDL을 생성한다.

스키마 정본은 ``app/models/`` 이다. DDL 파일은 파생물이므로 직접 수정하지 않는다.

    python -m scripts.gen_ddl > migrations/001_initial.sql
"""

from sqlalchemy import create_mock_engine

from app.models import Base

HEADER = """-- PetFit 초기 스키마 (MySQL 8.0)
--
-- DB 명세서 · DB 설계서를 기준으로 생성한 DDL이다.
-- SQLAlchemy 모델(app/models/)에서 자동 생성했으므로 직접 수정하지 않는다.

BEGIN;
"""

FOOTER = "\nCOMMIT;"


def main() -> None:
    statements: list[str] = []
    engine = create_mock_engine(
        "mysql+pymysql://",
        lambda sql, *args, **kwargs: statements.append(
            str(sql.compile(dialect=engine.dialect)).strip()
        ),
    )
    Base.metadata.create_all(engine, checkfirst=False)
    print(HEADER)
    print(";\n\n".join(statements) + ";")
    print(FOOTER)


if __name__ == "__main__":
    main()
