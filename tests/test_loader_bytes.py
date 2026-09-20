import pytest

from app.rag import loader
from app.rag.loader import DocumentError, ingest_pdf_bytes


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def fake_reader(page_texts):
    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage(text) for text in page_texts]

    return FakeReader


class BrokenReader:
    def __init__(self, stream):
        raise ValueError("not a pdf")


def test_ingest_pdf_bytes_builds_labeled_chunks(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", fake_reader(["Page one text", "Page two text"]))
    document = ingest_pdf_bytes(b"%PDF-1.4", "policy.pdf")
    assert document.filename == "policy.pdf"
    assert document.pages == 2
    assert [chunk.page for chunk in document.chunks] == [1, 2]
    assert all(chunk.source == "policy.pdf" for chunk in document.chunks)


def test_ingest_pdf_bytes_redacts_personal_details(monkeypatch):
    text = "Duty desk 9876543210 or ward@example.com"
    monkeypatch.setattr(loader, "PdfReader", fake_reader([text]))
    stored = ingest_pdf_bytes(b"%PDF-1.4", "policy.pdf").chunks[0].text
    assert "9876543210" not in stored
    assert "ward@example.com" not in stored
    assert "[PHONE]" in stored
    assert "[EMAIL]" in stored


def test_unreadable_file_raises_document_error(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", BrokenReader)
    with pytest.raises(DocumentError):
        ingest_pdf_bytes(b"junk", "bad.pdf")


def test_pdf_without_text_raises_document_error(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", fake_reader(["", "   "]))
    with pytest.raises(DocumentError):
        ingest_pdf_bytes(b"%PDF-1.4", "scan.pdf")
