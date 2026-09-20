from typing import Protocol


class LLMClient(Protocol):
    """Anything that can turn a prompt into an answer."""

    def generate(self, prompt: str) -> str: ...  # pragma: no cover


class FakeLLM:
    """A stand-in AI for tests and demos. Returns a fixed answer and records what it saw."""

    def __init__(self, response: str = "This is a placeholder answer.") -> None:
        self.response = response
        self.calls = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response
