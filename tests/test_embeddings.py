import pytest

from app.rag.embeddings import HashingEmbedder


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_same_text_gives_same_vector():
    embedder = HashingEmbedder()
    assert embedder.embed(["ICU visiting hours"]) == embedder.embed(["ICU visiting hours"])


def test_vector_has_requested_dimension():
    vector = HashingEmbedder(dimension=64).embed(["hello world"])[0]
    assert len(vector) == 64


def test_vector_is_normalized():
    vector = HashingEmbedder().embed(["hello world"])[0]
    assert sum(v * v for v in vector) == pytest.approx(1.0)


def test_empty_text_gives_zero_vector():
    vector = HashingEmbedder().embed([""])[0]
    assert all(v == 0.0 for v in vector)


def test_similar_texts_are_closer_than_unrelated_texts():
    a, b, c = HashingEmbedder().embed(
        ["icu visiting hours", "visiting hours in the icu", "hand hygiene procedure"]
    )
    assert cosine(a, b) > cosine(a, c)


def test_embed_many_texts_returns_one_vector_each():
    assert len(HashingEmbedder().embed(["a", "b", "c"])) == 3


def test_dimension_must_be_positive():
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=0)
