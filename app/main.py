from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.query import router as query_router

app = FastAPI(title="Hospital RAG API")
app.include_router(query_router)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
