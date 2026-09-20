from app.rag.chunker import Chunk

# Synthetic sample text for demos. Not real hospital policy.
SAMPLE_CHUNKS: list[Chunk] = [
    Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
    Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 1),
    Chunk("Discharge planning starts on the day of admission", "discharge_sop.pdf", 2, 2),
]
