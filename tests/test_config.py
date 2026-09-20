import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_defaults_use_fake_provider(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "fake"
    assert settings.openai_api_key is None
    assert settings.llm_base_url is None


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "my-model"
    assert settings.openai_api_key.get_secret_value() == "sk-test-123"


def test_api_key_is_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    settings = Settings(_env_file=None)
    assert "sk-test-123" not in repr(settings)
    assert "sk-test-123" not in str(settings)


def test_invalid_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
