import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import get_settings
from app.dependencies import build_embedder
from app.rag.chunker import Chunk
from app.rag.embeddings import EmbeddingError
from app.rag.vector_store import InMemoryVectorStore, SearchResult

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
DEFAULT_CORPUS = EVAL_DIR / "corpus.json"
DEFAULT_CASES = EVAL_DIR / "golden_set.json"


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...  # pragma: no cover


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_source: str | None


@dataclass(frozen=True)
class CaseResult:
    question: str
    expected_source: str | None
    found_rank: int | None
    top_score: float


@dataclass(frozen=True)
class EvalReport:
    top_k: int
    answerable: int
    unanswerable: int
    hit_rate: float
    mrr: float
    min_answerable_top_score: float | None
    max_unanswerable_top_score: float | None
    results: list[CaseResult]


def load_corpus(path: str | Path) -> list[Chunk]:
    """Read the practice documents (source, page, text) as chunks."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Chunk(text=item["text"], source=item["source"], page=item["page"], chunk_index=index)
        for index, item in enumerate(items)
    ]


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read the test questions. expected_source is null for unanswerable questions."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(question=item["question"], expected_source=item.get("expected_source"))
        for item in items
    ]


def evaluate_retrieval(store: Retriever, cases: list[EvalCase], top_k: int = 3) -> EvalReport:
    """Measure how often the right document is found in the top results."""
    results: list[CaseResult] = []
    for case in cases:
        hits = store.search(case.question, top_k=top_k)
        top_score = hits[0].score if hits else 0.0
        found_rank: int | None = None
        if case.expected_source is not None:
            for position, found in enumerate(hits, start=1):
                if found.chunk.source == case.expected_source:
                    found_rank = position
                    break
        results.append(CaseResult(case.question, case.expected_source, found_rank, top_score))

    answerable = [r for r in results if r.expected_source is not None]
    unanswerable = [r for r in results if r.expected_source is None]

    hit_count = 0
    reciprocal_sum = 0.0
    for r in answerable:
        if r.found_rank is not None:
            hit_count += 1
            reciprocal_sum += 1 / r.found_rank

    return EvalReport(
        top_k=top_k,
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        hit_rate=hit_count / len(answerable) if answerable else 0.0,
        mrr=reciprocal_sum / len(answerable) if answerable else 0.0,
        min_answerable_top_score=min((r.top_score for r in answerable), default=None),
        max_unanswerable_top_score=max((r.top_score for r in unanswerable), default=None),
        results=results,
    )


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_report(report: EvalReport, label: str) -> str:
    """Turn a report into readable text."""
    low = report.min_answerable_top_score
    high = report.max_unanswerable_top_score
    lines = [
        f"Embedder: {label}",
        f"Questions: {report.answerable} answerable, {report.unanswerable} unanswerable "
        f"(top_k={report.top_k})",
        f"Hit rate: {report.hit_rate:.2f}",
        f"MRR: {report.mrr:.2f}",
        f"Lowest top score, answerable questions: {_score(low)}",
        f"Highest top score, unanswerable questions: {_score(high)}",
    ]
    if low is not None and high is not None:
        if low > high:
            lines.append(f"A MIN_RETRIEVAL_SCORE between {high:.2f} and {low:.2f} separates them.")
        else:
            lines.append("The score ranges overlap, so no single MIN_RETRIEVAL_SCORE separates them.")
    misses = [r for r in report.results if r.expected_source is not None and r.found_rank is None]
    if misses:
        lines.append("Missed questions:")
        lines.extend(f"- {r.question} (expected {r.expected_source})" for r in misses)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Run the evaluation. Optional argument: hashing or openai."""
    args = sys.argv[1:] if argv is None else argv
    settings = get_settings()
    if args:
        if args[0] not in ("hashing", "openai"):
            print("Usage: python -m app.evaluation [hashing|openai]")
            return 2
        settings = settings.model_copy(update={"embedding_provider": args[0]})
    label: str = settings.embedding_provider
    if label == "openai":
        label = f"openai ({settings.embedding_model})"
    try:
        store = InMemoryVectorStore(build_embedder(settings))
        store.add_chunks(load_corpus(DEFAULT_CORPUS))
        report = evaluate_retrieval(store, load_cases(DEFAULT_CASES))
    except EmbeddingError:
        print("Could not get embeddings. Is Ollama running and the model downloaded?")
        return 1
    print(format_report(report, label))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
