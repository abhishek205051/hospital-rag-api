from typing import Protocol

from openai import OpenAI


class LLMError(Exception):
    """Raised when the language model cannot produce an answer."""


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


class OpenAICompatibleLLM:
    """Talks to OpenAI, or to any server with the same API (such as Ollama)."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise LLMError("The language model request failed.") from exc
        return content or ""
