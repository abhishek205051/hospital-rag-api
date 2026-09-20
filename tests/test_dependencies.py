from app.dependencies import get_pipeline
from app.rag.pipeline import RagPipeline


def test_default_pipeline_answers_sample_question():
    answer = get_pipeline().answer("What are the ICU visiting hours?")
    assert answer.sources[0].source == "visiting_policy.pdf"


def test_default_pipeline_is_created_once():
    assert isinstance(get_pipeline(), RagPipeline)
    assert get_pipeline() is get_pipeline()
