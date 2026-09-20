import re

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\d{10}|\d{3}[\s.-]\d{3}[\s.-]\d{4})(?!\d)"
)


def redact(text: str) -> str:
    """Replace emails and phone numbers with safe placeholders."""
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    return text
