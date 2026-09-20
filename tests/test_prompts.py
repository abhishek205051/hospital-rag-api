import pytest

from app.core.prompts import build_prompt
from app.rag.chunker import Chunk

CHUNKS = [
    Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
    Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 1),
]


def test_prompt_contains_question():
    prompt = build_prompt("What are the visiting hours?", CHUNKS)
    assert "What are the visiting hours?" in prompt


def test_prompt_contains_chunk_text_and_source_label():
    prompt = build_prompt("question", CHUNKS)
    assert "ICU visiting hours are 4 to 6 PM" in prompt
    assert "[visiting_policy.pdf, page 1]" in prompt
    assert "[hand_hygiene.pdf, page 3]" in prompt


def test_prompt_keeps_chunk_order():
    prompt = build_prompt("question", CHUNKS)
    assert prompt.index("visiting_policy.pdf") < prompt.index("hand_hygiene.pdf")


def test_prompt_tells_model_to_use_only_context():
    assert "ONLY the context" in build_prompt("question", CHUNKS)


def test_prompt_requires_at_least_one_chunk():
    with pytest.raises(ValueError):
        build_prompt("question", [])
