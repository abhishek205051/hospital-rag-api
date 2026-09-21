import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_audit_log, get_pipeline, get_store
from app.main import app
from app.rag import loader
from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.llm import LLMError
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore

ADMIN_KEY = "admin-secret-key-123456"
READER_KEY = "reader-secret-key-654321"
QUESTION = {"question": "What are the ICU visiting hours?"}


class FakePage:
    def extract_text(self):
        return "ICU visiting hours are 4 to 6 PM"


class FakeReader:
    def __init__(self, stream):
        self.pages = [FakePage()]


class BrokenLLM:
    def generate(self, prompt):
        raise LLMError("The language model request failed.")


def key(value):
    return {"X-API-Key": value}


def events():
    return get_audit_log().recent(limit=100)


def upload(client, api_key):
    return client.post(
        "/documents",
        files={"file": ("policy.pdf", b"%PDF-1.4", "application/pdf")},
        headers=key(api_key),
    )


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("API_KEYS", f"alice:admin:{ADMIN_KEY},bob:reader:{READER_KEY}")
    get_settings.cache_clear()
    get_store.cache_clear()
    get_pipeline.cache_clear()
    return TestClient(app)


def test_query_is_recorded_with_actor_and_sources(secured):
    secured.post("/query", json=QUESTION, headers=key(READER_KEY))
    event = events()[0]
    assert event.actor == "bob"
    assert event.role == "reader"
    assert event.action == "query"
    assert event.outcome == "answered"
    assert event.detail == "What are the ICU visiting hours?"
    assert "visiting_policy.pdf p.1" in event.sources


def test_personal_details_are_redacted_in_the_log(secured):
    body = {"question": "ICU visiting hours? Call 9876543210"}
    secured.post("/query", json=body, headers=key(ADMIN_KEY))
    detail = events()[0].detail
    assert "9876543210" not in detail
    assert "[PHONE]" in detail


def test_refused_query_is_recorded(secured):
    body = {"question": "What dose of paracetamol should I give?"}
    secured.post("/query", json=body, headers=key(ADMIN_KEY))
    event = events()[0]
    assert event.outcome == "refused:clinical_advice"
    assert event.sources == ""


def test_failed_login_is_recorded_without_the_key(secured):
    bad_key = "wrong-key-0000000000"
    secured.post("/query", json=QUESTION, headers=key(bad_key))
    event = events()[0]
    assert event.actor == "unknown"
    assert event.action == "auth_failed"
    assert event.outcome == "denied"
    assert event.detail == "/query"
    everything = " ".join(
        [event.actor, event.role, event.action, event.outcome, event.detail, event.sources]
    )
    assert bad_key not in everything


def test_forbidden_action_is_recorded(secured):
    upload(secured, READER_KEY)
    event = events()[0]
    assert event.actor == "bob"
    assert event.action == "forbidden"
    assert event.detail == "POST /documents"


def test_upload_and_replacement_are_recorded(secured, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    upload(secured, ADMIN_KEY)
    upload(secured, ADMIN_KEY)
    newest, older = events()[0], events()[1]
    assert (older.action, older.outcome, older.detail) == ("upload", "uploaded", "policy.pdf")
    assert (newest.action, newest.outcome) == ("upload", "replaced")


def test_rejected_upload_is_recorded(secured):
    secured.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        headers=key(ADMIN_KEY),
    )
    event = events()[0]
    assert event.outcome == "rejected:415"
    assert event.detail == "notes.txt"


def test_language_model_failure_is_recorded(secured):
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks([Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0)])
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(store=store, llm=BrokenLLM())
    try:
        response = secured.post("/query", json=QUESTION, headers=key(ADMIN_KEY))
    finally:
        app.dependency_overrides.pop(get_pipeline, None)
    assert response.status_code == 503
    assert events()[0].outcome == "error"


def test_admin_can_read_the_audit_log_newest_first(secured):
    secured.post("/query", json=QUESTION, headers=key(READER_KEY))
    other = {"question": "Tell me about parking fees"}
    secured.post("/query", json=other, headers=key(ADMIN_KEY))
    response = secured.get("/audit", headers=key(ADMIN_KEY))
    assert response.status_code == 200
    items = response.json()["events"]
    assert [item["actor"] for item in items] == ["alice", "bob"]
    assert items[0]["outcome"] == "no_context"


def test_reader_cannot_read_the_audit_log(secured):
    assert secured.get("/audit", headers=key(READER_KEY)).status_code == 403


def test_audit_log_needs_a_key(secured):
    assert secured.get("/audit").status_code == 401


def test_audit_limit_is_validated(secured):
    assert secured.get("/audit?limit=0", headers=key(ADMIN_KEY)).status_code == 422
