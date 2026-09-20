import pytest
from fastapi.testclient import TestClient

from app.api.documents import clean_filename
from app.config import Settings, get_settings
from app.dependencies import get_pipeline, get_store
from app.main import app
from app.rag import loader
from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.llm import FakeLLM
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore

TWO_PAGES = ["ICU visiting hours are 4 to 6 PM", "Wash hands with soap before patient contact"]


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def fake_reader(page_texts):
    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage(text) for text in page_texts]

    return FakeReader


class BrokenReader:
    def __init__(self, stream):
        raise ValueError("not a pdf")


@pytest.fixture
def store():
    return InMemoryVectorStore(HashingEmbedder())


@pytest.fixture
def client(store):
    pipeline = RagPipeline(store=store, llm=FakeLLM(response="Answer from fake."))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, name="policy.pdf", content=b"%PDF-1.4 test"):
    return client.post("/documents", files={"file": (name, content, "application/pdf")})


def test_upload_pdf_adds_chunks(client, store, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", fake_reader(TWO_PAGES))
    response = upload(client)
    assert response.status_code == 200
    assert response.json() == {
        "filename": "policy.pdf",
        "pages": 2,
        "chunks": 2,
        "replaced": False,
        "total_chunks": 2,
    }
    assert len(store) == 2


def test_uploaded_document_can_be_queried(client, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", fake_reader(TWO_PAGES))
    upload(client)
    response = client.post("/query", json={"question": "What are the ICU visiting hours?"})
    body = response.json()
    assert body["answer"] == "Answer from fake."
    assert body["sources"][0]["source"] == "policy.pdf"
    assert body["sources"][0]["page"] == 1


def test_reupload_replaces_old_document(client, store, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", fake_reader(["First page", "Second page"]))
    upload(client)
    monkeypatch.setattr(loader, "PdfReader", fake_reader(["Only page now"]))
    body = upload(client).json()
    assert body["replaced"] is True
    assert body["chunks"] == 1
    assert body["total_chunks"] == 1
    assert len(store) == 1


def test_non_pdf_file_returns_415(client):
    response = upload(client, name="notes.txt")
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


def test_clean_filename_removes_folders():
    assert clean_filename("../../evil.pdf") == "evil.pdf"
    assert clean_filename("C:\\docs\\policy.pdf") == "policy.pdf"
    assert clean_filename("policy.pdf") == "policy.pdf"
    assert clean_filename(None) == ""


def test_unreadable_pdf_returns_422(client, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", BrokenReader)
    response = upload(client)
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]


def test_pdf_without_text_returns_422(client, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", fake_reader(["", "   "]))
    response = upload(client)
    assert response.status_code == 422
    assert "No readable text" in response.json()["detail"]


def test_too_large_file_returns_413(client):
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, max_upload_mb=1)
    response = upload(client, content=b"%PDF" + b"0" * (1024 * 1024))
    assert response.status_code == 413


def test_missing_file_returns_422(client):
    assert client.post("/documents").status_code == 422


def test_list_documents_when_empty(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": [], "total_chunks": 0}


def test_list_documents_shows_chunk_counts(client, store):
    store.add_chunks(
        [
            Chunk("some text", "b.pdf", 1, 0),
            Chunk("more text", "a.pdf", 1, 0),
            Chunk("again", "a.pdf", 2, 1),
        ]
    )
    assert client.get("/documents").json() == {
        "documents": [{"source": "a.pdf", "chunks": 2}, {"source": "b.pdf", "chunks": 1}],
        "total_chunks": 3,
    }
