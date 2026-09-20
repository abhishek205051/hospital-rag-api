import pytest

from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore


def make_chunks():
    return [
        Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
        Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 1),
        Chunk("Discharge planning starts on the day of admission", "discharge_sop.pdf", 2, 2),
    ]


def make_store():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks(make_chunks())
    return store


def test_add_chunks_increases_size():
    assert len(make_store()) == 3


def test_add_empty_list_changes_nothing():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks([])
    assert len(store) == 0


def test_search_returns_most_relevant_chunk_first():
    results = make_store().search("What are the ICU visiting hours?")
    assert results[0].chunk.source == "visiting_policy.pdf"
    assert results[0].chunk.page == 1


def test_search_finds_hand_hygiene_chunk():
    results = make_store().search("How long should I wash hands with soap?")
    assert results[0].chunk.source == "hand_hygiene.pdf"


def test_results_are_sorted_by_score():
    scores = [r.score for r in make_store().search("ICU visiting hours", top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results():
    assert len(make_store().search("hospital", top_k=2)) == 2


def test_search_empty_store_returns_empty_list():
    assert InMemoryVectorStore(HashingEmbedder()).search("anything") == []


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        make_store().search("hours", top_k=0)
