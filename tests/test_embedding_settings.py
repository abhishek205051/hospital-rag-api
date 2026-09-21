import pytest
from pydantic import ValidationError

from app.config import Settings


def test_embedding_defaults(monkeypatch):
    for name in (
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_BASE_URL",
        "MIN_RETRIEVAL_SCORE",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "hashing"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_base_url is None
    assert settings.min_retrieval_score == 0.2


def test_embedding_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.45")
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.embedding_base_url == "http://localhost:11434/v1"
    assert settings.min_retrieval_score == 0.45


def test_invalid_embedding_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "banana")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_min_retrieval_score_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, min_retrieval_score=1.5)
