from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.audit import AuditLog
from app.core.redaction import redact
from app.dependencies import get_pipeline
from app.rag.embeddings import EmbeddingError
from app.rag.llm import LLMError
from app.rag.pipeline import RagAnswer, RagPipeline
from app.schemas.query import QueryRequest, QueryResponse, SourceModel
from app.security import AuditDep, Principal, PrincipalDep

router = APIRouter()

LLM_UNAVAILABLE_MESSAGE = "The language model is unavailable. Please try again later."

PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]


def _outcome(result: RagAnswer) -> str:
    if result.refused:
        return f"refused:{result.reason}"
    if result.reason == "no_context":
        return "no_context"
    return "answered"


def _record_query(
    audit: AuditLog, principal: Principal, question: str, outcome: str, sources: str = ""
) -> None:
    audit.record(
        actor=principal.name,
        role=principal.role,
        action="query",
        outcome=outcome,
        detail=question,
        sources=sources,
    )


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: PipelineDep,
    principal: PrincipalDep,
    audit: AuditDep,
) -> QueryResponse:
    question = redact(request.question)
    try:
        result = pipeline.answer(request.question)
    except LLMError as exc:
        _record_query(audit, principal, question, "error")
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE_MESSAGE) from exc
    except EmbeddingError:
        _record_query(audit, principal, question, "error")
        raise

    sources_text = "; ".join(f"{s.source} p.{s.page}" for s in result.sources)
    _record_query(audit, principal, question, _outcome(result), sources_text)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceModel(source=s.source, page=s.page, snippet=s.snippet, score=s.score)
            for s in result.sources
        ],
        refused=result.refused,
        reason=result.reason,
    )
