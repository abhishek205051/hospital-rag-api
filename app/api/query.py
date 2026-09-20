from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_pipeline
from app.rag.llm import LLMError
from app.rag.pipeline import RagPipeline
from app.schemas.query import QueryRequest, QueryResponse, SourceModel

router = APIRouter()

LLM_UNAVAILABLE_MESSAGE = "The language model is unavailable. Please try again later."

PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: PipelineDep) -> QueryResponse:
    try:
        result = pipeline.answer(request.question)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE_MESSAGE) from exc
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceModel(source=s.source, page=s.page, snippet=s.snippet, score=s.score)
            for s in result.sources
        ],
        refused=result.refused,
        reason=result.reason,
    )
