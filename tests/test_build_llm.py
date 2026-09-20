import pytest

from app import dependencies
from app.config import Settings
from app.dependencies import build_llm
from app.rag.llm import FakeLLM, OpenAICompatibleLLM


class RecordingOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_fake_provider_builds_fake_llm():
    settings = Settings(_env_file=None, llm_provider="fake")
    assert isinstance(build_llm(settings), FakeLLM)


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError):
        build_llm(settings)


def test_openai_provider_builds_client_with_settings(monkeypatch):
    monkeypatch.setattr(dependencies, "OpenAI", RecordingOpenAI)
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="sk-test-123",
        llm_base_url="http://localhost:11434/v1",
        llm_timeout_seconds=12.0,
    )
    llm = build_llm(settings)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm._client.kwargs == {
        "api_key": "sk-test-123",
        "base_url": "http://localhost:11434/v1",
        "timeout": 12.0,
    }
