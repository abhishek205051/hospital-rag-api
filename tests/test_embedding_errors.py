import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_pipeline, get_store
from app.main import app
from app.rag import loader
from app.rag.chunker import Chunk
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.llm import FakeLLM
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore


class FlakyEmbedder:
    def __init__(self):
        self.inner = HashingEmbedder()
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("The embedding request failed.")
        return self.inner.embed(texts)


class FakePage:
    def extract_text(self):
        return "Brand new page text"


class FakeReader:
    def __init__(self, stream):
        self.pages = [FakePage()]


@pytest.fixture
def flaky_store():
    embedder = FlakyEmbedder()
    store = InMemoryVectorStore(embedder)
    store.add_chunks([Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0)])
    embedder.fail = True
    return store


def test_query_returns_503_when_embeddings_fail(flaky_store):
    pipeline = RagPipeline(store=flaky_store, llm=FakeLLM())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        response = TestClient(app).post(
            "/query", json={"question": "What are the ICU visiting hours?"}
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "embedding" in response.json()["detail"].lower()


def test_upload_returns_503_and_keeps_old_document(flaky_store, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    app.dependency_overrides[get_store] = lambda: flaky_store
    try:
        response = TestClient(app).post(
            "/documents",
            files={"file": ("visiting_policy.pdf", b"%PDF-1.4", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert flaky_store.sources() == {"visiting_policy.pdf": 1}
