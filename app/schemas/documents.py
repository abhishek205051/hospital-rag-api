from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    replaced: bool
    total_chunks: int


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int
