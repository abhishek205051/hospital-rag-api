import sqlite3
from contextlib import closing

import pytest

from app.core.audit import MAX_TEXT_LENGTH, AuditLog


def make_log(tmp_path):
    return AuditLog(tmp_path / "audit.db")


def add(log, **overrides):
    values = {
        "actor": "alice",
        "role": "admin",
        "action": "query",
        "outcome": "answered",
        "detail": "What are the visiting hours?",
        "sources": "visiting_policy.pdf p.1",
    }
    values.update(overrides)
    log.record(**values)


def test_record_and_read_back(tmp_path):
    log = make_log(tmp_path)
    add(log)
    [event] = log.recent()
    assert event.id == 1
    assert event.actor == "alice"
    assert event.role == "admin"
    assert event.action == "query"
    assert event.outcome == "answered"
    assert event.detail == "What are the visiting hours?"
    assert event.sources == "visiting_policy.pdf p.1"


def test_timestamp_is_utc(tmp_path):
    log = make_log(tmp_path)
    add(log)
    assert log.recent()[0].timestamp.endswith("+00:00")


def test_newest_events_come_first(tmp_path):
    log = make_log(tmp_path)
    add(log, detail="first")
    add(log, detail="second")
    assert [event.detail for event in log.recent()] == ["second", "first"]


def test_limit_restricts_number_of_events(tmp_path):
    log = make_log(tmp_path)
    for number in range(5):
        add(log, detail=f"question {number}")
    assert len(log.recent(limit=2)) == 2


def test_limit_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        make_log(tmp_path).recent(limit=0)


def test_long_text_is_truncated(tmp_path):
    log = make_log(tmp_path)
    add(log, detail="a" * 1000, sources="b" * 1000)
    event = log.recent()[0]
    assert len(event.detail) == MAX_TEXT_LENGTH
    assert len(event.sources) == MAX_TEXT_LENGTH


def test_events_cannot_be_changed(tmp_path):
    log = make_log(tmp_path)
    add(log)
    with closing(sqlite3.connect(tmp_path / "audit.db")) as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE audit_events SET actor = 'mallory'")
    assert log.recent()[0].actor == "alice"


def test_events_cannot_be_deleted(tmp_path):
    log = make_log(tmp_path)
    add(log)
    with closing(sqlite3.connect(tmp_path / "audit.db")) as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM audit_events")
    assert len(log.recent()) == 1


def test_events_survive_reopening(tmp_path):
    add(make_log(tmp_path))
    assert len(make_log(tmp_path).recent()) == 1


def test_missing_folder_is_created(tmp_path):
    path = tmp_path / "new_folder" / "audit.db"
    AuditLog(path)
    assert path.exists()


def test_write_failure_does_not_raise(tmp_path, monkeypatch):
    log = make_log(tmp_path)

    def broken():
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(log, "_connect", broken)
    add(log)
