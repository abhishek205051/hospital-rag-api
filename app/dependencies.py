from functools import lru_cache

from openai import OpenAI

from app.config import Settings, get_settings
from app.core.audit import AuditLog
from app.rag.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from app.rag.llm import FakeLLM, LLMClient, OpenAICompatibleLLM
from app.rag.pipeline import RagPipeline
from app.rag.sample_data import SAMPLE_CHUNKS
from app.rag.sqlite_store import SqliteVectorStore
from app.rag.store_types import VectorStore
from app.rag.vector_store import InMemoryVectorStore

PLACEHOLDER_ANSWER = (
    "Placeholder answer: no real language model is connected yet. "
    "See the sources for the relevant policy text."
)


def _make_client(settings: Settings, base_url: str | None) -> OpenAI:
    """Create an OpenAI-style client (works for OpenAI and for Ollama)."""
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required when using the openai provider")
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=base_url or None,
        timeout=settings.llm_timeout_seconds,
    )


def build_llm(settings: Settings) -> LLMClient:
    """Create the AI client chosen in the settings."""
    if settings.llm_provider == "fake":
        return FakeLLM(response=PLACEHOLDER_ANSWER)
    client = _make_client(settings, settings.llm_base_url)
    return OpenAICompatibleLLM(client=client, model=settings.llm_model)


def build_embedder(settings: Settings) -> Embedder:
    """Create the fingerprint maker chosen in the settings."""
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    client = _make_client(settings, settings.embedding_base_url)
    return OpenAICompatibleEmbedder(client=client, model=settings.embedding_model)


def embedder_id(settings: Settings) -> str:
    """A short name for the embedding setting, saved inside the database file."""
    if settings.embedding_provider == "hashing":
        return "hashing"
    return f"openai:{settings.embedding_model}"


@lru_cache
def get_store() -> VectorStore:
    """Create the shared document store once. Uploaded documents are added to it."""
    settings = get_settings()
    embedder = build_embedder(settings)
    store: VectorStore
    if settings.store_backend == "sqlite":
        store = SqliteVectorStore(embedder, settings.store_path, embedder_id(settings))
    else:
        store = InMemoryVectorStore(embedder)
    if settings.load_sample_data and len(store) == 0:
        store.add_chunks(SAMPLE_CHUNKS)
    return store


@lru_cache
def get_pipeline() -> RagPipeline:
    """Build the pipeline once and reuse it for every request."""
    settings = get_settings()
    return RagPipeline(
        store=get_store(),
        llm=build_llm(settings),
        min_score=settings.min_retrieval_score,
    )


@lru_cache
def get_audit_log() -> AuditLog:
    """Open the audit log once and reuse it."""
    return AuditLog(get_settings().audit_path)
