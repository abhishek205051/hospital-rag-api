from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceModel(BaseModel):
    source: str
    page: int
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    refused: bool
    reason: str | None = None
