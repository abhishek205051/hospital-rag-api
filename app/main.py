from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.rag.embeddings import EmbeddingError

EMBEDDING_UNAVAILABLE_MESSAGE = "The embedding service is unavailable. Please try again later."

app = FastAPI(title="Hospital RAG API")
app.include_router(query_router)
app.include_router(documents_router)


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": EMBEDDING_UNAVAILABLE_MESSAGE})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
