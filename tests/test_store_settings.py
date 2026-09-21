import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.dependencies import embedder_id, get_store
from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.sample_data import SAMPLE_CHUNKS
from app.rag.sqlite_store import SqliteVectorStore
from app.rag.vector_store import InMemoryVectorStore


def use_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE_BACKEND", "sqlite")
    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store.db"))
    get_settings.cache_clear()
    get_store.cache_clear()


def test_store_defaults(monkeypatch):
    monkeypatch.delenv("STORE_BACKEND", raising=False)
    monkeypatch.delenv("STORE_PATH", raising=False)
    settings = Settings(_env_file=None)
    assert settings.store_backend == "sqlite"
    assert settings.store_path == "data/store.db"


def test_store_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("STORE_PATH", "somewhere/else.db")
    settings = Settings(_env_file=None)
    assert settings.store_backend == "memory"
    assert settings.store_path == "somewhere/else.db"


def test_invalid_store_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("STORE_BACKEND", "banana")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_embedder_id_for_hashing():
    assert embedder_id(Settings(_env_file=None, embedding_provider="hashing")) == "hashing"


def test_embedder_id_for_openai():
    settings = Settings(
        _env_file=None, embedding_provider="openai", embedding_model="nomic-embed-text"
    )
    assert embedder_id(settings) == "openai:nomic-embed-text"


def test_memory_backend_builds_in_memory_store():
    assert isinstance(get_store(), InMemoryVectorStore)


def test_sqlite_backend_builds_sqlite_store_and_loads_samples_once(monkeypatch, tmp_path):
    use_sqlite(monkeypatch, tmp_path)
    store = get_store()
    assert isinstance(store, SqliteVectorStore)
    assert len(store) == len(SAMPLE_CHUNKS)
    get_store.cache_clear()
    assert len(get_store()) == len(SAMPLE_CHUNKS)


def test_sample_data_is_skipped_when_store_already_has_documents(monkeypatch, tmp_path):
    existing = SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "hashing")
    existing.add_chunks([Chunk("Existing document text", "mine.pdf", 1, 0)])
    use_sqlite(monkeypatch, tmp_path)
    assert len(get_store()) == 1
