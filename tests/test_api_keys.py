import pytest

from app.core.api_keys import ApiKeyEntry, find_key, parse_api_keys

ADMIN_KEY = "admin-secret-key-123456"
READER_KEY = "reader-secret-key-654321"
RAW = f"alice:admin:{ADMIN_KEY},bob:reader:{READER_KEY}"


def test_parses_names_roles_and_keys():
    assert parse_api_keys(RAW) == [
        ApiKeyEntry("alice", "admin", ADMIN_KEY),
        ApiKeyEntry("bob", "reader", READER_KEY),
    ]


def test_ignores_spaces_and_blank_entries():
    raw = f" alice : admin : {ADMIN_KEY} , , bob:reader:{READER_KEY},"
    assert [entry.name for entry in parse_api_keys(raw)] == ["alice", "bob"]


def test_empty_string_gives_no_keys():
    assert parse_api_keys("") == []
    assert parse_api_keys("  ,  ") == []


def test_key_may_contain_a_colon():
    entry = parse_api_keys("svc:reader:abc:def-0123456789")[0]
    assert entry.key == "abc:def-0123456789"


def test_malformed_entries_are_rejected():
    for raw in ("alice", "alice:admin", ":admin:" + ADMIN_KEY, "alice:admin:"):
        with pytest.raises(ValueError):
            parse_api_keys(raw)


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        parse_api_keys(f"alice:boss:{ADMIN_KEY}")


def test_short_key_is_rejected():
    with pytest.raises(ValueError):
        parse_api_keys("alice:admin:short")


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError):
        parse_api_keys(f"alice:admin:{ADMIN_KEY},bob:reader:{ADMIN_KEY}")


def test_error_messages_never_contain_the_key():
    with pytest.raises(ValueError) as info:
        parse_api_keys(f"alice:boss:{ADMIN_KEY}")
    assert ADMIN_KEY not in str(info.value)


def test_find_key_returns_matching_entry():
    entries = parse_api_keys(RAW)
    assert find_key(entries, READER_KEY) == entries[1]


def test_find_key_rejects_wrong_key():
    assert find_key(parse_api_keys(RAW), "wrong-key-0000000000") is None


def test_find_key_rejects_missing_or_empty_key():
    entries = parse_api_keys(RAW)
    assert find_key(entries, None) is None
    assert find_key(entries, "") is None
