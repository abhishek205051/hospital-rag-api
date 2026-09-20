import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.redaction import redact
from app.rag.chunker import Chunk, chunk_pages


class DocumentError(Exception):
    """Raised when an uploaded document cannot be read or has no usable text."""


@dataclass(frozen=True)
class IngestedDocument:
    filename: str
    pages: int
    chunks: list[Chunk]


def load_pdf_pages(path: str | Path) -> list[str]:
    """Return the text of each page of a PDF."""
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def ingest_pdf(path: str | Path, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Read a PDF file from disk and return labeled chunks ready for embedding."""
    pages = load_pdf_pages(path)
    return chunk_pages(pages, source=Path(path).name, chunk_size=chunk_size, overlap=overlap)


def ingest_pdf_bytes(
    data: bytes, filename: str, chunk_size: int = 500, overlap: int = 50
) -> IngestedDocument:
    """Read an uploaded PDF, hide personal details, and return labeled chunks."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise DocumentError("The file could not be read as a PDF.") from exc

    pages = [redact(page) for page in pages]
    chunks = chunk_pages(pages, source=filename, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise DocumentError("No readable text was found in this PDF. Scanned PDFs need OCR.")
    return IngestedDocument(filename=filename, pages=len(pages), chunks=chunks)
