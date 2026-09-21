import sqlite3
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.rag.chunker import Chunk
from app.rag.embeddings import Embedder
from app.rag.vector_store import SearchResult, _dot

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

INSERT_SQL = (
    "INSERT INTO chunks (source, page, chunk_index, text, vector) VALUES (?, ?, ?, ?, ?)"
)


class StoreMismatchError(Exception):
    """Raised when saved documents were built with a different embedding setting."""


def _pack(vector: list[float]) -> bytes:
    return array("d", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("d")
    values.frombytes(blob)
    return list(values)


def _to_rows(
    chunks: list[Chunk], vectors: list[list[float]]
) -> list[tuple[str, int, int, str, bytes]]:
    return [
        (chunk.source, chunk.page, chunk.chunk_index, chunk.text, _pack(vector))
        for chunk, vector in zip(chunks, vectors)
    ]


class SqliteVectorStore:
    """Saves chunks and their vectors in a SQLite file, so they survive restarts."""

    def __init__(self, embedder: Embedder, path: str | Path, embedder_id: str) -> None:
        self._embedder = embedder
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._check_embedder(conn, embedder_id)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _check_embedder(conn: sqlite3.Connection, embedder_id: str) -> None:
        """Refuse to mix vectors from two different embedding models."""
        row = conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if row is not None and row[0] != embedder_id:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count > 0:
                raise StoreMismatchError(
                    f"The saved documents were built with the embedding setting '{row[0]}', "
                    f"but the current setting is '{embedder_id}'. Switch the setting back, "
                    "or delete the database file (STORE_PATH) and upload your documents again."
                )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder', ?)", (embedder_id,)
        )

    def __len__(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        with self._connect() as conn:
            conn.executemany(INSERT_SQL, _to_rows(chunks, vectors))

    def replace_source(self, source: str, chunks: list[Chunk]) -> bool:
        """Swap one document's chunks for new ones in a single all-or-nothing step.

        The new vectors are computed first, so if embedding fails the old
        document stays untouched. Returns True if an old version existed.
        """
        vectors = self._embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM chunks WHERE source = ?", (source,)).rowcount
            conn.executemany(INSERT_SQL, _to_rows(chunks, vectors))
        return deleted > 0

    def remove_source(self, source: str) -> int:
        """Delete every chunk that came from one document. Returns how many were removed."""
        with self._connect() as conn:
            return int(conn.execute("DELETE FROM chunks WHERE source = ?", (source,)).rowcount)

    def sources(self) -> dict[str, int]:
        """Return how many chunks each document has."""
        with self._connect() as conn:
            rows = conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source").fetchall()
        return {source: count for source, count in rows}

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source, page, chunk_index, text, vector FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            return []
        query_vector = self._embedder.embed([query])[0]
        results = [
            SearchResult(
                chunk=Chunk(text=text, source=source, page=page, chunk_index=chunk_index),
                score=_dot(query_vector, _unpack(vector)),
            )
            for source, page, chunk_index, text, vector in rows
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
