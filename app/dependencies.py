from functools import lru_cache

from app.rag.embeddings import HashingEmbedder
from app.rag.llm import FakeLLM
from app.rag.pipeline import RagPipeline
from app.rag.sample_data import SAMPLE_CHUNKS
from app.rag.vector_store import InMemoryVectorStore

PLACEHOLDER_ANSWER = (
    "Placeholder answer: no real language model is connected yet. "
    "See the sources for the relevant policy text."
)


@lru_cache
def get_pipeline() -> RagPipeline:
    """Build the pipeline once and reuse it for every request."""
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks(SAMPLE_CHUNKS)
    return RagPipeline(store=store, llm=FakeLLM(response=PLACEHOLDER_ANSWER))
