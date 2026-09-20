from fastapi import APIRouter, Depends

from app.dependencies import get_pipeline
from app.rag.pipeline import RagPipeline
from app.schemas.query import QueryRequest, QueryResponse, SourceModel

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, pipeline: RagPipeline = Depends(get_pipeline)
) -> QueryResponse:
    result = pipeline.answer(request.question)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceModel(source=s.source, page=s.page, snippet=s.snippet, score=s.score)
            for s in result.sources
        ],
        refused=result.refused,
        reason=result.reason,
    )
