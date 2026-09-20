import pytest

from app.rag.chunker import chunk_pages, chunk_text


def test_short_text_returns_single_chunk():
    assert chunk_text("Short text") == ["Short text"]


def test_long_text_is_split_within_size_limit():
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_chunks_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert chunks[0][-10:] == chunks[1][:10]


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_whitespace_is_normalized():
    assert chunk_text("hello   \n  world") == ["hello world"]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)


def test_chunk_pages_keeps_source_and_page():
    chunks = chunk_pages(["Page one text", "Page two text"], source="policy.pdf")
    assert [c.page for c in chunks] == [1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert all(c.source == "policy.pdf" for c in chunks)


def test_chunk_pages_skips_empty_pages():
    chunks = chunk_pages(["", "Only page two"], source="a.pdf")
    assert len(chunks) == 1
    assert chunks[0].page == 2
    assert chunks[0].chunk_index == 0
