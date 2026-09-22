from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.audit import router as audit_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.rag.embeddings import EmbeddingError
from app.rag.sqlite_store import StoreMismatchError

EMBEDDING_UNAVAILABLE_MESSAGE = "The embedding service is unavailable. Please try again later."
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Hospital RAG API")
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(audit_router)


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": EMBEDDING_UNAVAILABLE_MESSAGE})


@app.exception_handler(StoreMismatchError)
async def store_mismatch_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
