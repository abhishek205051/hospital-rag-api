from pathlib import PureWindowsPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.dependencies import get_store
from app.rag.loader import DocumentError, ingest_pdf_bytes
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse

router = APIRouter()

StoreDep = Annotated[InMemoryVectorStore, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def clean_filename(raw: str | None) -> str:
    """Keep only the file name, dropping any folder path the client sent."""
    return PureWindowsPath(raw or "").name


@router.post("/documents", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile, store: StoreDep, settings: SettingsDep
) -> DocumentUploadResponse:
    filename = clean_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"File is larger than {settings.max_upload_mb} MB."
        )

    try:
        document = ingest_pdf_bytes(data, filename)
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    replaced = store.remove_source(filename) > 0
    store.add_chunks(document.chunks)
    return DocumentUploadResponse(
        filename=filename,
        pages=document.pages,
        chunks=len(document.chunks),
        replaced=replaced,
        total_chunks=len(store),
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(store: StoreDep) -> DocumentListResponse:
    counts = store.sources()
    return DocumentListResponse(
        documents=[DocumentInfo(source=name, chunks=count) for name, count in sorted(counts.items())],
        total_chunks=len(store),
    )
