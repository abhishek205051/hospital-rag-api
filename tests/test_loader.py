from app.rag import loader


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, path):
        self.pages = [FakePage("First page text"), FakePage("Second page text")]


def test_load_pdf_pages_returns_text_per_page(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    assert loader.load_pdf_pages("policy.pdf") == ["First page text", "Second page text"]


def test_ingest_pdf_builds_chunks_with_source_and_page(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    chunks = loader.ingest_pdf("data/hand_hygiene.pdf")
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.source == "hand_hygiene.pdf" for c in chunks)
