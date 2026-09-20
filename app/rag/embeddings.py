import hashlib
import math
import re
from typing import Protocol

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that can turn a list of texts into a list of number-vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...  # pragma: no cover


class HashingEmbedder:
    """A simple, free, offline embedder based on the words in the text.

    Good for building and testing the pipeline. It matches shared words,
    not meaning. Swap in a real embedding model later.
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
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]
