from fastapi import FastAPI

app = FastAPI(title="Hospital RAG API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
