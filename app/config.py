from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.api_keys import parse_api_keys


class Settings(BaseSettings):
    """App settings, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

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
    store_backend: Literal["memory", "sqlite"] = "sqlite"
    store_path: str = "data/store.db"
    auth_required: bool = True
    api_keys: SecretStr | None = None
    audit_path: str = "data/audit.db"

    @field_validator("api_keys")
    @classmethod
    def check_api_keys(cls, value: SecretStr | None) -> SecretStr | None:
        """Refuse to start if the keys are badly formatted."""
        if value is not None:
            parse_api_keys(value.get_secret_value())
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
