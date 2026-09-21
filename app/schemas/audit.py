from pydantic import BaseModel


class AuditEventModel(BaseModel):
    id: int
    timestamp: str
    actor: str
    role: str
    action: str
    outcome: str
    detail: str
    sources: str


class AuditListResponse(BaseModel):
    events: list[AuditEventModel]
