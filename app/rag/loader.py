from pathlib import Path

from pypdf import PdfReader

from app.rag.chunker import Chunk, chunk_pages


def load_pdf_pages(path: str | Path) -> list[str]:
    """Return the text of each page of a PDF."""
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def ingest_pdf(path: str | Path, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Read a PDF and return labeled chunks ready for embedding."""
    pages = load_pdf_pages(path)
    return chunk_pages(pages, source=Path(path).name, chunk_size=chunk_size, overlap=overlap)
