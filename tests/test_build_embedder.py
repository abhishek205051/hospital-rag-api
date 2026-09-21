import pytest

from app import dependencies
from app.config import Settings, get_settings
from app.dependencies import build_embedder, get_pipeline
from app.rag.embeddings import HashingEmbedder, OpenAICompatibleEmbedder


class RecordingOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_hashing_provider_builds_hashing_embedder():
    settings = Settings(_env_file=None, embedding_provider="hashing")
    assert isinstance(build_embedder(settings), HashingEmbedder)


def test_openai_embedding_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, embedding_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError):
        build_embedder(settings)


def test_openai_embedding_provider_builds_client_with_settings(monkeypatch):
    monkeypatch.setattr(dependencies, "OpenAI", RecordingOpenAI)
    settings = Settings(
        _env_file=None,
        embedding_provider="openai",
        embedding_model="nomic-embed-text",
        embedding_base_url="http://localhost:11434/v1",
        openai_api_key="ollama",
        llm_timeout_seconds=12.0,
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder._model == "nomic-embed-text"
    assert embedder._client.kwargs == {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "timeout": 12.0,
    }


def test_pipeline_uses_min_score_from_settings(monkeypatch):
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")
    get_settings.cache_clear()
    get_pipeline.cache_clear()
    assert get_pipeline()._min_score == 0.5
