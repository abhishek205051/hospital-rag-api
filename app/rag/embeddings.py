import hashlib
import math
import re
from typing import Protocol

from openai import OpenAI

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EmbeddingError(Exception):
    """Raised when the embedding service cannot produce vectors."""


class Embedder(Protocol):
    """Anything that can turn a list of texts into a list of number-vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...  # pragma: no cover


def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to length 1, so a dot product equals cosine similarity."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder:
    """A simple, free, offline embedder based on the words in the text.

    Good for building and testing the pipeline. It matches shared words,
    not meaning.
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dimension] += 1.0
        return _normalize(vector)


class OpenAICompatibleEmbedder:
    """Gets real embeddings from OpenAI, or from any server with the same API (Ollama)."""

    def __init__(self, client: OpenAI, model: str, batch_size: int = 64) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
                batch_vectors = [list(item.embedding) for item in response.data]
            except Exception as exc:
                raise EmbeddingError("The embedding request failed.") from exc
            if len(batch_vectors) != len(batch):
                raise EmbeddingError("The embedding service returned the wrong number of vectors.")
            vectors.extend(_normalize(vector) for vector in batch_vectors)
        return vectors
