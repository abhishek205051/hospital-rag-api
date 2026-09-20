from app.config import Settings, get_settings
from app.dependencies import get_store
from app.rag.sample_data import SAMPLE_CHUNKS


def test_document_settings_defaults(monkeypatch):
    monkeypatch.delenv("LOAD_SAMPLE_DATA", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    settings = Settings(_env_file=None)
    assert settings.load_sample_data is True
    assert settings.max_upload_mb == 10


def test_document_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("LOAD_SAMPLE_DATA", "false")
    monkeypatch.setenv("MAX_UPLOAD_MB", "3")
    settings = Settings(_env_file=None)
    assert settings.load_sample_data is False
    assert settings.max_upload_mb == 3


def test_store_starts_with_sample_data_by_default():
    assert len(get_store()) == len(SAMPLE_CHUNKS)


def test_store_can_start_empty(monkeypatch):
    monkeypatch.setenv("LOAD_SAMPLE_DATA", "false")
    get_settings.cache_clear()
    get_store.cache_clear()
    assert len(get_store()) == 0
