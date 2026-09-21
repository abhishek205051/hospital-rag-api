import pytest

from app.rag.chunker import Chunk
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.sqlite_store import SqliteVectorStore, StoreMismatchError

CHUNKS = [
    Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
    Chunk("Visitors must sign in at the front desk", "visiting_policy.pdf", 2, 1),
    Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 2),
]


class FlakyEmbedder:
    def __init__(self):
        self.inner = HashingEmbedder()
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("The embedding request failed.")
        return self.inner.embed(texts)


def make_store(tmp_path, embedder=None):
    store = SqliteVectorStore(embedder or HashingEmbedder(), tmp_path / "store.db", "hashing")
    store.add_chunks(CHUNKS)
    return store


def test_add_chunks_increases_size(tmp_path):
    assert len(make_store(tmp_path)) == 3


def test_add_empty_list_changes_nothing(tmp_path):
    store = SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "hashing")
    store.add_chunks([])
    assert len(store) == 0


def test_search_returns_most_relevant_chunk_first(tmp_path):
    results = make_store(tmp_path).search("What are the ICU visiting hours?")
    assert results[0].chunk.source == "visiting_policy.pdf"
    assert results[0].chunk.page == 1


def test_search_finds_hand_hygiene_chunk(tmp_path):
    results = make_store(tmp_path).search("How long should I wash hands with soap?")
    assert results[0].chunk.source == "hand_hygiene.pdf"


def test_results_are_sorted_by_score(tmp_path):
    scores = [r.score for r in make_store(tmp_path).search("ICU visiting hours", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(tmp_path):
    assert len(make_store(tmp_path).search("hospital", top_k=2)) == 2


def test_search_empty_store_returns_empty_list(tmp_path):
    store = SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "hashing")
    assert store.search("anything") == []


def test_top_k_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        make_store(tmp_path).search("hours", top_k=0)


def test_sources_counts_chunks_per_document(tmp_path):
    assert make_store(tmp_path).sources() == {"visiting_policy.pdf": 2, "hand_hygiene.pdf": 1}


def test_remove_source_deletes_only_that_document(tmp_path):
    store = make_store(tmp_path)
    assert store.remove_source("visiting_policy.pdf") == 2
    assert store.sources() == {"hand_hygiene.pdf": 1}
    assert len(store) == 1


def test_remove_unknown_source_returns_zero(tmp_path):
    store = make_store(tmp_path)
    assert store.remove_source("nothing.pdf") == 0
    assert len(store) == 3


def test_replace_source_swaps_old_chunks_for_new_ones(tmp_path):
    store = make_store(tmp_path)
    new_chunks = [Chunk("Visiting is closed at night", "visiting_policy.pdf", 1, 0)]
    assert store.replace_source("visiting_policy.pdf", new_chunks) is True
    assert store.sources() == {"visiting_policy.pdf": 1, "hand_hygiene.pdf": 1}
    assert len(store) == 2
    assert store.search("Visiting is closed at night", top_k=1)[0].chunk.text == (
        "Visiting is closed at night"
    )


def test_replace_source_adds_a_new_document(tmp_path):
    store = make_store(tmp_path)
    new_chunks = [Chunk("New policy text", "new_policy.pdf", 1, 0)]
    assert store.replace_source("new_policy.pdf", new_chunks) is False
    assert len(store) == 4


def test_failed_embedding_keeps_the_old_document(tmp_path):
    embedder = FlakyEmbedder()
    store = make_store(tmp_path, embedder)
    embedder.fail = True
    with pytest.raises(EmbeddingError):
        store.replace_source(
            "visiting_policy.pdf", [Chunk("Replacement", "visiting_policy.pdf", 1, 0)]
        )
    assert store.sources() == {"visiting_policy.pdf": 2, "hand_hygiene.pdf": 1}


def test_documents_survive_reopening(tmp_path):
    make_store(tmp_path)
    reopened = SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "hashing")
    assert len(reopened) == 3
    assert reopened.sources() == {"visiting_policy.pdf": 2, "hand_hygiene.pdf": 1}
    top = reopened.search("What are the ICU visiting hours?")[0]
    assert top.chunk.source == "visiting_policy.pdf"
    assert top.chunk.page == 1


def test_saved_vectors_keep_their_precision(tmp_path):
    top = make_store(tmp_path).search("ICU visiting hours are 4 to 6 PM", top_k=1)[0]
    assert top.score == pytest.approx(1.0)


def test_different_embedder_is_rejected_when_documents_exist(tmp_path):
    make_store(tmp_path)
    with pytest.raises(StoreMismatchError) as info:
        SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "openai:nomic-embed-text")
    assert "hashing" in str(info.value)
    assert "openai:nomic-embed-text" in str(info.value)


def test_different_embedder_is_accepted_when_store_is_empty(tmp_path):
    SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "hashing")
    reopened = SqliteVectorStore(HashingEmbedder(), tmp_path / "store.db", "openai:other")
    assert len(reopened) == 0


def test_missing_folder_is_created(tmp_path):
    path = tmp_path / "new_folder" / "store.db"
    SqliteVectorStore(HashingEmbedder(), path, "hashing")
    assert path.exists()
