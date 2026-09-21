from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["fake", "openai"] = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 30.0
    openai_api_key: SecretStr | None = None
    load_sample_data: bool = True
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    min_retrieval_score: float = Field(default=0.2, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
