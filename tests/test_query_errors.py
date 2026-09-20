from fastapi.testclient import TestClient

from app.dependencies import get_pipeline
from app.main import app
from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.llm import LLMError
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore


class BrokenLLM:
    def generate(self, prompt: str) -> str:
        raise LLMError("The language model request failed.")


def test_query_returns_503_when_llm_fails():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks([Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0)])
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(store=store, llm=BrokenLLM())
    try:
        response = TestClient(app).post("/query", json={"question": "ICU visiting hours?"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
