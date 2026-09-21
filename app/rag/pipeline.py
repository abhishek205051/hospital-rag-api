from dataclasses import dataclass

from app.core.guardrails import REFUSAL_MESSAGE, check_question
from app.core.prompts import build_prompt
from app.core.redaction import redact
from app.rag.llm import LLMClient
from app.rag.store_types import VectorStore

NOT_FOUND_MESSAGE = "I could not find this in the provided documents."
SNIPPET_LENGTH = 200


@dataclass(frozen=True)
class Source:
    source: str
    page: int
    snippet: str
    score: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[Source]
    refused: bool = False
    reason: str | None = None


class RagPipeline:
    """Guardrail, redact, retrieve, build prompt, ask the LLM, redact, return with sources."""

    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = 3,
        min_score: float = 0.2,
    ) -> None:
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, question: str) -> RagAnswer:
        check = check_question(question)
        if not check.allowed:
            return RagAnswer(
                answer=check.message or REFUSAL_MESSAGE,
                sources=[],
                refused=True,
                reason=check.reason,
            )

        clean_question = redact(question)
        results = [
            result
            for result in self._store.search(clean_question, top_k=self._top_k)
            if result.score >= self._min_score
        ]
        if not results:
            return RagAnswer(answer=NOT_FOUND_MESSAGE, sources=[], reason="no_context")

        prompt = build_prompt(clean_question, [result.chunk for result in results])
        answer_text = redact(self._llm.generate(prompt))
        sources = [
            Source(
                source=result.chunk.source,
                page=result.chunk.page,
                snippet=result.chunk.text[:SNIPPET_LENGTH],
                score=round(result.score, 3),
            )
            for result in results
        ]
        return RagAnswer(answer=answer_text, sources=sources)
