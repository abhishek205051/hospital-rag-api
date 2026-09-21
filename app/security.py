from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings
from app.core.api_keys import find_key, parse_api_keys
from app.core.audit import AuditLog
from app.dependencies import get_audit_log

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

SettingsDep = Annotated[Settings, Depends(get_settings)]
AuditDep = Annotated[AuditLog, Depends(get_audit_log)]
ApiKeyDep = Annotated[str | None, Security(api_key_header)]

INVALID_KEY_MESSAGE = "Invalid or missing API key."
NO_KEYS_MESSAGE = "No API keys are configured on the server."
ADMIN_ONLY_MESSAGE = "This action requires an admin API key."


@dataclass(frozen=True)
class Principal:
    """Who is making the request."""

    name: str
    role: str


def authenticate(
    request: Request, api_key: ApiKeyDep, settings: SettingsDep, audit: AuditDep
) -> Principal:
    """Check the X-API-Key header and work out who is calling."""
    if not settings.auth_required:
        return Principal(name="anonymous", role="admin")

    raw = settings.api_keys.get_secret_value() if settings.api_keys is not None else ""
    entries = parse_api_keys(raw)
    if not entries:
        raise HTTPException(status_code=503, detail=NO_KEYS_MESSAGE)

    match = find_key(entries, api_key)
    if match is None:
        audit.record(
            actor="unknown",
            role="none",
            action="auth_failed",
            outcome="denied",
            detail=request.url.path,
        )
        raise HTTPException(
            status_code=401,
            detail=INVALID_KEY_MESSAGE,
            headers={"WWW-Authenticate": "APIKey"},
        )
    return Principal(name=match.name, role=match.role)


PrincipalDep = Annotated[Principal, Depends(authenticate)]


def require_admin(request: Request, principal: PrincipalDep, audit: AuditDep) -> Principal:
    """Allow only admin keys through."""
    if principal.role != "admin":
        audit.record(
            actor=principal.name,
            role=principal.role,
            action="forbidden",
            outcome="denied",
            detail=f"{request.method} {request.url.path}",
        )
        raise HTTPException(status_code=403, detail=ADMIN_ONLY_MESSAGE)
    return principal


AdminDep = Annotated[Principal, Depends(require_admin)]
