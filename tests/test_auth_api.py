import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_pipeline, get_store
from app.main import app
from app.rag import loader

ADMIN_KEY = "admin-secret-key-123456"
READER_KEY = "reader-secret-key-654321"
QUESTION = {"question": "What are the ICU visiting hours?"}


class FakePage:
    def extract_text(self):
        return "ICU visiting hours are 4 to 6 PM"


class FakeReader:
    def __init__(self, stream):
        self.pages = [FakePage()]


def key(value):
    return {"X-API-Key": value}


def reset_caches():
    get_settings.cache_clear()
    get_store.cache_clear()
    get_pipeline.cache_clear()


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
    reset_caches()
    return TestClient(app)


def test_health_needs_no_key(secured):
    assert secured.get("/health").status_code == 200


def test_missing_key_returns_401(secured):
    response = secured.post("/query", json=QUESTION)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_wrong_key_returns_401_without_echoing_it(secured):
    response = secured.post("/query", json=QUESTION, headers=key("wrong-key-0000000000"))
    assert response.status_code == 401
    assert "wrong-key-0000000000" not in response.text


def test_reader_key_can_query(secured):
    response = secured.post("/query", json=QUESTION, headers=key(READER_KEY))
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "visiting_policy.pdf"


def test_admin_key_can_query(secured):
    response = secured.post("/query", json=QUESTION, headers=key(ADMIN_KEY))
    assert response.status_code == 200


def test_reader_key_can_list_documents(secured):
    assert secured.get("/documents", headers=key(READER_KEY)).status_code == 200


def test_reader_key_cannot_upload(secured):
    response = upload(secured, READER_KEY)
    assert response.status_code == 403
    assert "admin" in response.json()["detail"]


def test_admin_key_can_upload(secured, monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    response = upload(secured, ADMIN_KEY)
    assert response.status_code == 200
    assert response.json()["filename"] == "policy.pdf"


def test_no_keys_configured_returns_503(secured, monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    reset_caches()
    response = secured.post("/query", json=QUESTION, headers=key(ADMIN_KEY))
    assert response.status_code == 503
    assert "No API keys" in response.json()["detail"]


def test_auth_can_be_switched_off(secured, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    reset_caches()
    assert secured.post("/query", json=QUESTION).status_code == 200


def test_docs_page_offers_an_authorize_button():
    schemes = app.openapi()["components"]["securitySchemes"]
    assert any(s["type"] == "apiKey" and s["name"] == "X-API-Key" for s in schemes.values())
