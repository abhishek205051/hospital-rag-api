from app.evaluation import (
    DEFAULT_CASES,
    DEFAULT_CORPUS,
    EvalCase,
    evaluate_retrieval,
    format_report,
    load_cases,
    load_corpus,
)
from app.rag.chunker import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.vector_store import InMemoryVectorStore, SearchResult

CHUNKS = [
    Chunk("ICU visiting hours are 4 to 6 PM", "visiting_policy.pdf", 1, 0),
    Chunk("Wash hands with soap before patient contact", "hand_hygiene.pdf", 3, 1),
    Chunk("Discharge planning starts on the day of admission", "discharge_sop.pdf", 2, 2),
]


def hit(source, score):
    return SearchResult(chunk=Chunk("text", source, 1, 0), score=score)


class StubStore:
    """Returns fixed search results, so ranks and scores are known."""

    def __init__(self, results_by_question):
        self.results_by_question = results_by_question

    def search(self, query, top_k=3):
        return self.results_by_question.get(query, [])[:top_k]


def real_store():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks(CHUNKS)
    return store


def test_perfect_hit_rate_and_mrr():
    cases = [
        EvalCase("What are the ICU visiting hours?", "visiting_policy.pdf"),
        EvalCase("How long should I wash hands with soap?", "hand_hygiene.pdf"),
    ]
    report = evaluate_retrieval(real_store(), cases, top_k=3)
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0
    assert report.answerable == 2
    assert report.unanswerable == 0


def test_miss_lowers_hit_rate_and_mrr():
    cases = [
        EvalCase("What are the ICU visiting hours?", "visiting_policy.pdf"),
        EvalCase("What are the ICU visiting hours?", "discharge_sop.pdf"),
    ]
    report = evaluate_retrieval(real_store(), cases, top_k=1)
    assert report.hit_rate == 0.5
    assert report.mrr == 0.5
    assert report.results[1].found_rank is None


def test_second_place_hit_counts_half():
    store = StubStore({"q": [hit("a.pdf", 0.9), hit("b.pdf", 0.5), hit("c.pdf", 0.1)]})
    report = evaluate_retrieval(store, [EvalCase("q", "b.pdf")], top_k=3)
    assert report.results[0].found_rank == 2
    assert report.hit_rate == 1.0
    assert report.mrr == 0.5


def test_unanswerable_questions_report_scores():
    store = StubStore({"q1": [hit("a.pdf", 0.9)], "q2": [hit("b.pdf", 0.3)]})
    cases = [EvalCase("q1", "a.pdf"), EvalCase("q2", None)]
    report = evaluate_retrieval(store, cases)
    assert report.answerable == 1
    assert report.unanswerable == 1
    assert report.hit_rate == 1.0
    assert report.min_answerable_top_score == 0.9
    assert report.max_unanswerable_top_score == 0.3


def test_no_hits_gives_zero_top_score():
    report = evaluate_retrieval(StubStore({}), [EvalCase("q", "a.pdf")])
    assert report.results[0].top_score == 0.0
    assert report.results[0].found_rank is None
    assert report.hit_rate == 0.0


def test_only_unanswerable_questions():
    store = StubStore({"q": [hit("a.pdf", 0.4)]})
    report = evaluate_retrieval(store, [EvalCase("q", None)])
    assert report.hit_rate == 0.0
    assert report.mrr == 0.0
    assert report.min_answerable_top_score is None
    assert report.max_unanswerable_top_score == 0.4


def test_format_report_shows_scores_and_threshold_hint():
    store = StubStore({"q1": [hit("a.pdf", 0.9)], "q2": [hit("b.pdf", 0.3)]})
    cases = [EvalCase("q1", "a.pdf"), EvalCase("q2", None)]
    text = format_report(evaluate_retrieval(store, cases), label="hashing")
    assert "Embedder: hashing" in text
    assert "Hit rate: 1.00" in text
    assert "MRR: 1.00" in text
    assert "between 0.30 and 0.90" in text


def test_format_report_warns_when_scores_overlap():
    store = StubStore({"q1": [hit("a.pdf", 0.3)], "q2": [hit("b.pdf", 0.6)]})
    cases = [EvalCase("q1", "a.pdf"), EvalCase("q2", None)]
    text = format_report(evaluate_retrieval(store, cases), label="hashing")
    assert "overlap" in text.lower()


def test_format_report_lists_missed_questions():
    report = evaluate_retrieval(StubStore({}), [EvalCase("Where is the cafeteria?", "a.pdf")])
    text = format_report(report, label="hashing")
    assert "Missed questions:" in text
    assert "Where is the cafeteria?" in text


def test_format_report_without_both_kinds_has_no_hint():
    store = StubStore({"q": [hit("a.pdf", 0.9)]})
    text = format_report(evaluate_retrieval(store, [EvalCase("q", "a.pdf")]), label="hashing")
    assert "n/a" in text
    assert "between" not in text


def test_load_corpus_and_cases(tmp_path):
    corpus_file = tmp_path / "corpus.json"
    corpus_file.write_text(
        '[{"source": "a.pdf", "page": 2, "text": "Some text"},'
        ' {"source": "b.pdf", "page": 1, "text": "Other text"}]',
        encoding="utf-8",
    )
    assert load_corpus(corpus_file) == [
        Chunk("Some text", "a.pdf", 2, 0),
        Chunk("Other text", "b.pdf", 1, 1),
    ]
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        '[{"question": "Q1?", "expected_source": "a.pdf"},'
        ' {"question": "Q2?", "expected_source": null},'
        ' {"question": "Q3?"}]',
        encoding="utf-8",
    )
    assert load_cases(cases_file) == [
        EvalCase("Q1?", "a.pdf"),
        EvalCase("Q2?", None),
        EvalCase("Q3?", None),
    ]


def test_shipped_eval_files_are_consistent():
    chunks = load_corpus(DEFAULT_CORPUS)
    cases = load_cases(DEFAULT_CASES)
    sources = {chunk.source for chunk in chunks}
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(case.expected_source in sources for case in cases if case.expected_source)
    assert sum(1 for case in cases if case.expected_source) >= 15
    assert sum(1 for case in cases if not case.expected_source) >= 3


def test_hashing_baseline_runs_on_shipped_data():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add_chunks(load_corpus(DEFAULT_CORPUS))
    report = evaluate_retrieval(store, load_cases(DEFAULT_CASES))
    assert report.answerable >= 15
    assert 0.0 <= report.hit_rate <= 1.0
    assert 0.0 <= report.mrr <= 1.0
