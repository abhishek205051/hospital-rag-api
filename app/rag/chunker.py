from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping pieces of at most chunk_size characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be 0 or more and smaller than chunk_size")

    text = " ".join(text.split())
    if not text:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def chunk_pages(
    pages: list[str], source: str, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Chunk every page and label each chunk with its source file and page number."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text, chunk_size, overlap):
            chunks.append(
                Chunk(text=piece, source=source, page=page_number, chunk_index=len(chunks))
            )
    return chunks
