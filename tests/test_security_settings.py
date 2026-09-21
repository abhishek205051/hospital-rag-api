import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.audit import AuditLog
from app.dependencies import get_audit_log

GOOD_KEYS = "alice:admin:admin-secret-key-123456"
BAD_KEYS = "alice:boss:admin-secret-key-123456"


def test_security_defaults(monkeypatch):
    for name in ("AUTH_REQUIRED", "API_KEYS", "AUDIT_PATH"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.auth_required is True
    assert settings.api_keys is None
    assert settings.audit_path == "data/audit.db"


def test_security_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setenv("API_KEYS", GOOD_KEYS)
    monkeypatch.setenv("AUDIT_PATH", "somewhere/audit.db")
    settings = Settings(_env_file=None)
    assert settings.auth_required is False
    assert settings.api_keys.get_secret_value() == GOOD_KEYS
    assert settings.audit_path == "somewhere/audit.db"


def test_malformed_api_keys_are_rejected(monkeypatch):
    monkeypatch.setenv("API_KEYS", BAD_KEYS)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_malformed_api_keys_do_not_leak_in_error(monkeypatch):
    monkeypatch.setenv("API_KEYS", BAD_KEYS)
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None)
    assert "admin-secret-key-123456" not in str(info.value)


def test_api_keys_are_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("API_KEYS", GOOD_KEYS)
    settings = Settings(_env_file=None)
    assert "admin-secret-key-123456" not in repr(settings)


def test_get_audit_log_uses_configured_path(tmp_path):
    log = get_audit_log()
    assert isinstance(log, AuditLog)
    assert (tmp_path / "audit.db").exists()
    assert get_audit_log() is log
