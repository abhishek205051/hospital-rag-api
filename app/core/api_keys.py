import hmac
from dataclasses import dataclass

VALID_ROLES = ("reader", "admin")
MIN_KEY_LENGTH = 16


@dataclass(frozen=True)
class ApiKeyEntry:
    name: str
    role: str
    key: str


def parse_api_keys(raw: str) -> list[ApiKeyEntry]:
    """Read entries written as name:role:key and separated by commas."""
    entries: list[ApiKeyEntry] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(":", 2)]
        if len(parts) != 3 or not parts[0] or not parts[2]:
            raise ValueError("Each API key entry must look like name:role:key")
        name, role, key = parts
        if role not in VALID_ROLES:
            raise ValueError("The role of an API key must be 'reader' or 'admin'")
        if len(key) < MIN_KEY_LENGTH:
            raise ValueError(f"API keys must be at least {MIN_KEY_LENGTH} characters long")
        entries.append(ApiKeyEntry(name=name, role=role, key=key))
    if len({entry.key for entry in entries}) != len(entries):
        raise ValueError("Two API key entries use the same key")
    return entries


def find_key(entries: list[ApiKeyEntry], candidate: str | None) -> ApiKeyEntry | None:
    """Return the entry whose key matches, comparing in constant time."""
    if not candidate:
        return None
    candidate_bytes = candidate.encode("utf-8")
    match: ApiKeyEntry | None = None
    for entry in entries:
        if hmac.compare_digest(entry.key.encode("utf-8"), candidate_bytes):
            match = entry
    return match
