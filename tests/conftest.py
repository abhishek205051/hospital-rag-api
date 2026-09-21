import pytest

from app.config import get_settings
from app.dependencies import get_pipeline, get_store


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.2")
    monkeypatch.setenv("LOAD_SAMPLE_DATA", "true")
    monkeypatch.setenv("MAX_UPLOAD_MB", "10")
    get_settings.cache_clear()
    get_store.cache_clear()
    get_pipeline.cache_clear()
    yield
    get_settings.cache_clear()
    get_store.cache_clear()
    get_pipeline.cache_clear()
