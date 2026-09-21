import pytest

from app.config import get_settings
from app.dependencies import get_audit_log, get_pipeline, get_store


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.2")
    monkeypatch.setenv("LOAD_SAMPLE_DATA", "true")
    monkeypatch.setenv("MAX_UPLOAD_MB", "10")
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store.db"))
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("AUDIT_PATH", str(tmp_path / "audit.db"))
    get_settings.cache_clear()
    get_store.cache_clear()
    get_pipeline.cache_clear()
    get_audit_log.cache_clear()
    yield
    get_settings.cache_clear()
    get_store.cache_clear()
    get_pipeline.cache_clear()
    get_audit_log.cache_clear()
