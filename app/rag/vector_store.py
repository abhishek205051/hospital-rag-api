from dataclasses import dataclass

from app.rag.chunker import Chunk
from app.rag.embeddings import Embedder


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    """Keeps chunks and their vectors in memory and finds the best matches."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []
        query_vector = self._embedder.embed([query])[0]
        results = [
            SearchResult(chunk=chunk, score=_dot(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
