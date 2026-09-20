from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore


def make_store():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks(
        [
            Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
            Chunk("Visitors must sign in at the front desk", "visiting_policy.pdf", 2, 1),
            Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 2),
        ]
    )
    return store


def test_sources_counts_chunks_per_document():
    assert make_store().sources() == {"visiting_policy.pdf": 2, "hand_hygiene.pdf": 1}


def test_remove_source_deletes_only_that_document():
    store = make_store()
    assert store.remove_source("visiting_policy.pdf") == 2
    assert len(store) == 1
    assert store.sources() == {"hand_hygiene.pdf": 1}
    results = store.search("ICU visiting hours")
    assert all(result.chunk.source == "hand_hygiene.pdf" for result in results)


def test_remove_unknown_source_returns_zero():
    store = make_store()
    assert store.remove_source("nothing.pdf") == 0
    assert len(store) == 3
