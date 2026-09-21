from pathlib import PureWindowsPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.core.redaction import redact
from app.dependencies import get_store
from app.rag.embeddings import EmbeddingError
from app.rag.loader import DocumentError, ingest_pdf_bytes
from app.rag.store_types import VectorStore
from app.schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse
from app.security import AdminDep, AuditDep, authenticate

router = APIRouter()

StoreDep = Annotated[VectorStore, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def clean_filename(raw: str | None) -> str:
    """Keep only the file name, dropping any folder path the client sent."""
    return PureWindowsPath(raw or "").name


def _ingest(
    file: UploadFile, filename: str, store: VectorStore, settings: Settings
) -> DocumentUploadResponse:
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

    replaced = store.replace_source(filename, document.chunks)
    return DocumentUploadResponse(
        filename=filename,
        pages=document.pages,
        chunks=len(document.chunks),
        replaced=replaced,
        total_chunks=len(store),
    )


@router.post("/documents", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile,
    store: StoreDep,
    settings: SettingsDep,
    principal: AdminDep,
    audit: AuditDep,
) -> DocumentUploadResponse:
    filename = clean_filename(file.filename)
    label = redact(filename)
    try:
        response = _ingest(file, filename, store, settings)
    except HTTPException as exc:
        audit.record(
            actor=principal.name,
            role=principal.role,
            action="upload",
            outcome=f"rejected:{exc.status_code}",
            detail=label,
        )
        raise
    except EmbeddingError:
        audit.record(
            actor=principal.name,
            role=principal.role,
            action="upload",
            outcome="error",
            detail=label,
        )
        raise

    audit.record(
        actor=principal.name,
        role=principal.role,
        action="upload",
        outcome="replaced" if response.replaced else "uploaded",
        detail=label,
    )
    return response


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(authenticate)],
)
def list_documents(store: StoreDep) -> DocumentListResponse:
    counts = store.sources()
    documents = [
        DocumentInfo(source=name, chunks=count) for name, count in sorted(counts.items())
    ]
    return DocumentListResponse(documents=documents, total_chunks=len(store))
