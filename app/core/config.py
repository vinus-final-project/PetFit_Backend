"""애플리케이션 설정.

환경변수로 주입하며 `.env` 파일을 지원한다.

데이터베이스 접속 정보는 두 가지 방식을 지원한다.

    분리형 : DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME
    통합형 : DATABASE_URL

통합형이 명시되면 그쪽이 우선한다. 컨테이너 환경은 URL 하나만 주입하는 경우가 많고,
로컬 개발은 항목별로 관리하는 편이 편하기 때문이다.

비밀번호는 URL 인코딩하여 조립한다. `@` `#` `:` `/` 가 포함되면 인코딩 없이는
접속 문자열 파싱이 깨진다.
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings", "DEFAULT_DRIVER"]

#: 비동기 MySQL 드라이버. asyncmy 는 C 확장이라 빌드 도구가 필요하다.
DEFAULT_DRIVER = "mysql+aiomysql"

#: 한글 객체명과 이모지를 저장하므로 utf8mb4 가 필수다.
DEFAULT_CHARSET = "utf8mb4"


class Settings(BaseSettings):
    """환경 설정."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 애플리케이션 -----------------------------------------------------
    app_name: str = "PetFit"
    debug: bool = False

    # --- 데이터베이스 (분리형) --------------------------------------------
    db_user: str = Field(default="petfit", description="MySQL 계정")
    db_password: str = Field(default="petfit", description="MySQL 비밀번호")
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=3306, ge=1, le=65535)
    db_name: str = Field(default="petfit", description="스키마 이름")

    # --- 데이터베이스 (통합형) --------------------------------------------
    database_url: str | None = Field(
        default=None,
        description="명시하면 분리형 설정보다 우선한다. 미지정 시 DB_* 로 조립한다.",
    )

    # --- 파일 저장소 ------------------------------------------------------
    storage_root: Path = Field(
        default=Path("./storage"),
        description="영상·이미지 저장 루트. 하위에 videos/ images/ 를 둔다.",
    )

    # --- AI ---------------------------------------------------------------
    yolo_model_path: Path = Field(default=Path("./models/yolo.pt"))
    llm_provider: str = Field(default="qwen", description="qwen | openai")
    llm_api_key: str | None = None

    @model_validator(mode="after")
    def _assemble_database_url(self) -> "Settings":
        """DATABASE_URL 이 없으면 DB_* 항목으로 조립한다."""
        if not self.database_url:
            object.__setattr__(self, "database_url", self.build_url())
        return self

    def build_url(self, *, driver: str = DEFAULT_DRIVER, database: str | None = None) -> str:
        """접속 문자열을 조립한다.

        Args:
            driver: SQLAlchemy 드라이버. 동기 접속이 필요하면 `mysql+pymysql` 을 넘긴다.
            database: 접속할 스키마. None 이면 `db_name` 을 사용한다.
                      스키마 생성 전 서버에만 접속할 때는 빈 문자열을 넘긴다.

        Returns:
            SQLAlchemy 접속 문자열.
        """
        schema = self.db_name if database is None else database
        return (
            f"{driver}://{quote_plus(self.db_user)}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{schema}?charset={DEFAULT_CHARSET}"
        )

    @property
    def safe_database_url(self) -> str:
        """비밀번호를 가린 접속 문자열. 로그 출력용이다."""
        url = self.database_url or ""
        if not self.db_password:
            return url
        return url.replace(quote_plus(self.db_password), "****", 1)

    @property
    def video_dir(self) -> Path:
        return self.storage_root / "videos"

    @property
    def image_dir(self) -> Path:
        return self.storage_root / "images"


@lru_cache
def get_settings() -> Settings:
    """설정 싱글턴을 반환한다."""
    return Settings()
