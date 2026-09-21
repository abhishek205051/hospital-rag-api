from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.schemas.audit import AuditEventModel, AuditListResponse
from app.security import AuditDep, require_admin

router = APIRouter()


@router.get(
    "/audit",
    response_model=AuditListResponse,
    dependencies=[Depends(require_admin)],
)
def read_audit_log(
    audit: AuditDep, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> AuditListResponse:
    events = audit.recent(limit)
    return AuditListResponse(events=[AuditEventModel(**asdict(event)) for event in events])
