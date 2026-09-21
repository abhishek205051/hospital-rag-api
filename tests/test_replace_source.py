import pytest

from app.rag.chunker import Chunk
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore


class FlakyEmbedder:
    def __init__(self):
        self.inner = HashingEmbedder()
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("The embedding request failed.")
        return self.inner.embed(texts)


def make_store(embedder=None):
    store = InMemoryVectorStore(embedder or HashingEmbedder())
    store.add_chunks(
        [
            Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
            Chunk("Visitors must sign in at the front desk", "visiting_policy.pdf", 2, 1),
            Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 2),
        ]
    )
    return store


def test_replace_source_swaps_old_chunks_for_new_ones():
    store = make_store()
    new_chunks = [Chunk("Visiting is closed at night", "visiting_policy.pdf", 1, 0)]
    assert store.replace_source("visiting_policy.pdf", new_chunks) is True
    assert store.sources() == {"visiting_policy.pdf": 1, "hand_hygiene.pdf": 1}
    assert len(store) == 2
    top = store.search("Visiting is closed at night", top_k=1)[0]
    assert top.chunk.text == "Visiting is closed at night"


def test_replace_source_adds_a_new_document():
    store = make_store()
    new_chunks = [Chunk("New policy text", "new_policy.pdf", 1, 0)]
    assert store.replace_source("new_policy.pdf", new_chunks) is False
    assert len(store) == 4


def test_failed_embedding_keeps_the_old_document():
    embedder = FlakyEmbedder()
    store = make_store(embedder)
    embedder.fail = True
    with pytest.raises(EmbeddingError):
        store.replace_source(
            "visiting_policy.pdf", [Chunk("Replacement", "visiting_policy.pdf", 1, 0)]
        )
    assert store.sources() == {"visiting_policy.pdf": 2, "hand_hygiene.pdf": 1}
