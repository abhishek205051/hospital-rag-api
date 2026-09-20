from fastapi import FastAPI

from app.api.query import router as query_router

app = FastAPI(title="Hospital RAG API")
app.include_router(query_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
