from fastapi.testclient import TestClient

from app.dependencies import get_store
from app.main import app
from app.rag.sqlite_store import StoreMismatchError


def test_store_mismatch_returns_503_with_helpful_message():
    def broken_store():
        raise StoreMismatchError("built with 'hashing', but the current setting is 'openai:m'")

    app.dependency_overrides[get_store] = broken_store
    try:
        response = TestClient(app).get("/documents")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "built with" in response.json()["detail"]
