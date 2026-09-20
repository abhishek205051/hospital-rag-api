from functools import lru_cache

from openai import OpenAI

from app.config import Settings, get_settings
from app.rag.embeddings import HashingEmbedder
from app.rag.llm import FakeLLM, LLMClient, OpenAICompatibleLLM
from app.rag.pipeline import RagPipeline
from app.rag.sample_data import SAMPLE_CHUNKS
from app.rag.vector_store import InMemoryVectorStore

PLACEHOLDER_ANSWER = (
    "Placeholder answer: no real language model is connected yet. "
    "See the sources for the relevant policy text."
)


def build_llm(settings: Settings) -> LLMClient:
    """Create the AI client chosen in the settings."""
    if settings.llm_provider == "fake":
        return FakeLLM(response=PLACEHOLDER_ANSWER)
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_base_url or None,
        timeout=settings.llm_timeout_seconds,
    )
    return OpenAICompatibleLLM(client=client, model=settings.llm_model)


@lru_cache
def get_store() -> InMemoryVectorStore:
    """Create the shared document store once. Uploaded documents are added to it."""
    store = InMemoryVectorStore(HashingEmbedder())
    if get_settings().load_sample_data:
        store.add_chunks(SAMPLE_CHUNKS)
    return store


@lru_cache
def get_pipeline() -> RagPipeline:
    """Build the pipeline once and reuse it for every request."""
    return RagPipeline(store=get_store(), llm=build_llm(get_settings()))
