import re
from dataclasses import dataclass

REFUSAL_MESSAGE = (
    "I can only answer questions about hospital policies and procedures "
    "from the provided documents. I cannot give medical advice, diagnoses, or dosing."
)

CLINICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat (dose|dosage)\b", re.IGNORECASE),
    re.compile(r"\bhow (much|many)\b.*\b(should|can) i (give|take|administer)\b", re.IGNORECASE),
    re.compile(r"\bdiagnose\b", re.IGNORECASE),
    re.compile(r"\bdo i have\b", re.IGNORECASE),
    re.compile(r"\bshould i (take|give)\b", re.IGNORECASE),
    re.compile(r"\bwhich (medicine|medication|drug)\b.*\b(prescribe|give|take)\b", re.IGNORECASE),
]

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore (all |any )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\b(reveal|show|print)\b.*\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\byou are now\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    message: str | None = None


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def check_question(question: str) -> GuardrailResult:
    """Decide whether a question may be sent on to the AI model."""
    if not question.strip():
        return GuardrailResult(allowed=False, reason="empty", message="Please enter a question.")
    if _matches_any(INJECTION_PATTERNS, question):
        return GuardrailResult(allowed=False, reason="prompt_injection", message=REFUSAL_MESSAGE)
    if _matches_any(CLINICAL_PATTERNS, question):
        return GuardrailResult(allowed=False, reason="clinical_advice", message=REFUSAL_MESSAGE)
    return GuardrailResult(allowed=True)
