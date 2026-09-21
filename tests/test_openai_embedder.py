from types import SimpleNamespace

import pytest

from app.rag.embeddings import EmbeddingError, OpenAICompatibleEmbedder


class FakeEmbeddings:
    def __init__(self, vector=(3.0, 4.0), error=None, drop_last=False):
        self.vector = list(vector)
        self.error = error
        self.drop_last = drop_last
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        texts = kwargs["input"]
        if self.drop_last:
            texts = texts[:-1]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=list(self.vector)) for _ in texts]
        )


def make_client(embeddings):
    return SimpleNamespace(embeddings=embeddings)


def test_returns_one_normalized_vector_per_text():
    embedder = OpenAICompatibleEmbedder(client=make_client(FakeEmbeddings()), model="m")
    vectors = embedder.embed(["a", "b"])
    assert len(vectors) == 2
    assert vectors[0] == pytest.approx([0.6, 0.8])


def test_sends_model_and_texts():
    embeddings = FakeEmbeddings()
    OpenAICompatibleEmbedder(client=make_client(embeddings), model="my-model").embed(["x", "y"])
    assert embeddings.calls == [{"model": "my-model", "input": ["x", "y"]}]


def test_large_inputs_are_sent_in_batches():
    embeddings = FakeEmbeddings()
    embedder = OpenAICompatibleEmbedder(
        client=make_client(embeddings), model="m", batch_size=2
    )
    vectors = embedder.embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert [len(call["input"]) for call in embeddings.calls] == [2, 2, 1]


def test_empty_input_makes_no_requests():
    embeddings = FakeEmbeddings()
    assert OpenAICompatibleEmbedder(client=make_client(embeddings), model="m").embed([]) == []
    assert embeddings.calls == []


def test_zero_vector_stays_zero():
    embeddings = FakeEmbeddings(vector=(0.0, 0.0))
    embedder = OpenAICompatibleEmbedder(client=make_client(embeddings), model="m")
    assert embedder.embed(["a"]) == [[0.0, 0.0]]


def test_api_error_becomes_embedding_error():
    embeddings = FakeEmbeddings(error=RuntimeError("boom sk-secret"))
    embedder = OpenAICompatibleEmbedder(client=make_client(embeddings), model="m")
    with pytest.raises(EmbeddingError) as info:
        embedder.embed(["a"])
    assert "sk-secret" not in str(info.value)


def test_wrong_vector_count_becomes_embedding_error():
    embedder = OpenAICompatibleEmbedder(
        client=make_client(FakeEmbeddings(drop_last=True)), model="m"
    )
    with pytest.raises(EmbeddingError):
        embedder.embed(["a", "b"])


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        OpenAICompatibleEmbedder(client=make_client(FakeEmbeddings()), model="m", batch_size=0)
