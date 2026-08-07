"""SQLAlchemy 선언적 베이스.

DBMS는 MySQL 8.0을 사용한다.

테이블명은 **소문자 snake_case** 로 통일한다. MySQL의 테이블명 대소문자 구분은
``lower_case_table_names`` 설정과 파일시스템에 따라 달라진다. Windows에서 개발하고
Linux에 배포하면 대문자가 섞인 테이블을 찾지 못한다.
"""

from sqlalchemy.orm import DeclarativeBase

__all__ = ["Base", "TABLE_ARGS"]

#: 모든 테이블에 적용하는 MySQL 옵션.
#: utf8mb4 는 한글과 이모지를 모두 저장할 수 있는 완전한 UTF-8 이다.
TABLE_ARGS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_unicode_ci",
}


class Base(DeclarativeBase):
    """모든 모델의 베이스."""
