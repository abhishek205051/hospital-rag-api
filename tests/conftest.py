import pytest

from app.config import get_settings
from app.dependencies import get_pipeline


@pytest.fixture(autouse=True)
def use_fake_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    get_pipeline.cache_clear()
    yield
    get_settings.cache_clear()
    get_pipeline.cache_clear()
