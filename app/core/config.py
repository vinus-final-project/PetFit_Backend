"""애플리케이션 설정.

환경변수로 주입하며 `.env` 파일을 지원한다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """환경 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "PetFit"
    debug: bool = False

    database_url: str = Field(
        default="postgresql+asyncpg://petfit:petfit@localhost:5432/petfit",
        description="PostgreSQL 연결 문자열",
    )

    storage_root: Path = Field(
        default=Path("./storage"),
        description="영상·이미지 저장 루트. 하위에 videos/ images/ 를 둔다.",
    )

    yolo_model_path: Path = Field(default=Path("./models/yolo.pt"))
    llm_provider: str = Field(default="qwen", description="qwen | openai")
    llm_api_key: str | None = None

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
