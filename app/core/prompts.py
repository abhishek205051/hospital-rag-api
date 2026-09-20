from app.rag.chunker import Chunk

PROMPT_VERSION = "v1"

SYSTEM_INSTRUCTIONS = (
    "You are a hospital policy assistant. "
    "Answer using ONLY the context below. "
    "If the context does not contain the answer, say you could not find it "
    "in the provided documents. "
    "Never give medical advice, diagnoses, or dosing. "
    "Cite sources as [file, page N]."
)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Combine the instructions, the retrieved chunks, and the question."""
    if not chunks:
        raise ValueError("at least one chunk is required")
    context = "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for c in chunks)
    return f"{SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
