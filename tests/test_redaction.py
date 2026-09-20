from app.core.redaction import redact


def test_redacts_phone_number():
    assert redact("Call me at 9876543210") == "Call me at [PHONE]"


def test_redacts_phone_with_country_code():
    assert redact("Call +91 9876543210 now") == "Call [PHONE] now"


def test_redacts_dashed_phone_number():
    assert redact("Call 555-123-4567") == "Call [PHONE]"


def test_redacts_email():
    assert redact("Mail john@example.com") == "Mail [EMAIL]"


def test_redacts_multiple_items():
    text = "Email a@b.com or call 9876543210"
    assert redact(text) == "Email [EMAIL] or call [PHONE]"


def test_leaves_normal_text_unchanged():
    text = "Visiting hours are 4 to 6 PM in room 4021"
    assert redact(text) == text


def test_empty_string():
    assert redact("") == ""
