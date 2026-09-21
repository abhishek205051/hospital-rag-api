import threading
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
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        with self._lock:
            self._chunks.extend(chunks)
            self._vectors.extend(vectors)

    def replace_source(self, source: str, chunks: list[Chunk]) -> bool:
        """Swap one document's chunks for new ones.

        The new vectors are computed first, so if embedding fails the old
        document stays untouched. Returns True if an old version existed.
        """
        vectors = self._embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        with self._lock:
            keep = [i for i, chunk in enumerate(self._chunks) if chunk.source != source]
            replaced = len(keep) != len(self._chunks)
            self._chunks = [self._chunks[i] for i in keep] + list(chunks)
            self._vectors = [self._vectors[i] for i in keep] + vectors
        return replaced

    def remove_source(self, source: str) -> int:
        """Delete every chunk that came from one document. Returns how many were removed."""
        with self._lock:
            keep = [i for i, chunk in enumerate(self._chunks) if chunk.source != source]
            removed = len(self._chunks) - len(keep)
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = [self._vectors[i] for i in keep]
        return removed

    def sources(self) -> dict[str, int]:
        """Return how many chunks each document has."""
        counts: dict[str, int] = {}
        with self._lock:
            for chunk in self._chunks:
                counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return counts

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._lock:
            entries = list(zip(self._chunks, self._vectors))
        if not entries:
            return []
        query_vector = self._embedder.embed([query])[0]
        results = [
            SearchResult(chunk=chunk, score=_dot(query_vector, vector))
            for chunk, vector in entries
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
