import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("audit")

MAX_TEXT_LENGTH = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    role TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    detail TEXT NOT NULL,
    sources TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'The audit log is append-only');
END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'The audit log is append-only');
END;
"""

INSERT_SQL = (
    "INSERT INTO audit_events (timestamp, actor, role, action, outcome, detail, sources) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


@dataclass(frozen=True)
class AuditEvent:
    id: int
    timestamp: str
    actor: str
    role: str
    action: str
    outcome: str
    detail: str
    sources: str


class AuditLog:
    """An append-only record of who did what, saved in a SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def record(
        self,
        *,
        actor: str,
        role: str,
        action: str,
        outcome: str,
        detail: str = "",
        sources: str = "",
    ) -> None:
        """Save one event. A logging problem is reported but never breaks the request."""
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        values = (
            timestamp,
            actor,
            role,
            action,
            outcome,
            detail[:MAX_TEXT_LENGTH],
            sources[:MAX_TEXT_LENGTH],
        )
        try:
            with self._connect() as conn:
                conn.execute(INSERT_SQL, values)
        except sqlite3.Error:
            logger.exception("Could not write an audit event")

    def recent(self, limit: int = 50) -> list[AuditEvent]:
        """Return the newest events first."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, actor, role, action, outcome, detail, sources "
                "FROM audit_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [AuditEvent(*row) for row in rows]
