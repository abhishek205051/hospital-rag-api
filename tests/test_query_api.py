import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_pipeline
from app.main import app
from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.llm import FakeLLM
from app.rag.pipeline import NOT_FOUND_MESSAGE, RagPipeline
from app.rag.vector_store import InMemoryVectorStore


def build_pipeline() -> RagPipeline:
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks(
        [
            Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
            Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 1),
        ]
    )
    return RagPipeline(store=store, llm=FakeLLM(response="Visiting hours are 4 to 6 PM."))


@pytest.fixture
def client():
    def override() -> RagPipeline:
        return build_pipeline()

    app.dependency_overrides[get_pipeline] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_returns_answer_and_sources(client):
    response = client.post("/query", json={"question": "What are the ICU visiting hours?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Visiting hours are 4 to 6 PM."
    assert body["refused"] is False
    assert body["sources"][0]["source"] == "visiting_policy.pdf"
    assert body["sources"][0]["page"] == 1


def test_query_refuses_clinical_question(client):
    response = client.post("/query", json={"question": "What dose should I give?"})
    body = response.json()
    assert response.status_code == 200
    assert body["refused"] is True
    assert body["reason"] == "clinical_advice"
    assert body["sources"] == []


def test_query_returns_not_found_for_unrelated_question(client):
    response = client.post("/query", json={"question": "Tell me about parking fees"})
    body = response.json()
    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["reason"] == "no_context"


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_rejects_missing_question(client):
    assert client.post("/query", json={}).status_code == 422


def test_query_refuses_blank_question(client):
    body = client.post("/query", json={"question": "   "}).json()
    assert body["refused"] is True
    assert body["reason"] == "empty"


def test_query_rejects_too_long_question(client):
    response = client.post("/query", json={"question": "a" * 1001})
    assert response.status_code == 422
