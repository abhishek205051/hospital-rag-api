from typing import Protocol

from app.rag.chunker import Chunk
from app.rag.vector_store import SearchResult


class VectorStore(Protocol):
    """What the rest of the app needs from a document store."""

    def __len__(self) -> int: ...  # pragma: no cover

    def add_chunks(self, chunks: list[Chunk]) -> None: ...  # pragma: no cover

    def replace_source(self, source: str, chunks: list[Chunk]) -> bool: ...  # pragma: no cover

    def remove_source(self, source: str) -> int: ...  # pragma: no cover

    def sources(self) -> dict[str, int]: ...  # pragma: no cover

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...  # pragma: no cover
