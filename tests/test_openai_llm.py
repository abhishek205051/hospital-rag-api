from types import SimpleNamespace

import pytest

from app.rag.llm import LLMError, OpenAICompatibleLLM


class FakeCompletions:
    def __init__(self, content="Answer", error=None, empty=False):
        self.content = content
        self.error = error
        self.empty = empty
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        if self.empty:
            return SimpleNamespace(choices=[])
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_returns_text_from_model():
    llm = OpenAICompatibleLLM(client=make_client(FakeCompletions("Hello")), model="m")
    assert llm.generate("prompt") == "Hello"


def test_sends_model_prompt_and_zero_temperature():
    completions = FakeCompletions()
    OpenAICompatibleLLM(client=make_client(completions), model="my-model").generate("Q?")
    assert completions.kwargs["model"] == "my-model"
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Q?"}]
    assert completions.kwargs["temperature"] == 0.0


def test_none_content_becomes_empty_string():
    llm = OpenAICompatibleLLM(client=make_client(FakeCompletions(content=None)), model="m")
    assert llm.generate("prompt") == ""


def test_api_error_becomes_llm_error():
    completions = FakeCompletions(error=RuntimeError("boom sk-secret"))
    llm = OpenAICompatibleLLM(client=make_client(completions), model="m")
    with pytest.raises(LLMError) as info:
        llm.generate("prompt")
    assert "sk-secret" not in str(info.value)


def test_empty_choices_becomes_llm_error():
    llm = OpenAICompatibleLLM(client=make_client(FakeCompletions(empty=True)), model="m")
    with pytest.raises(LLMError):
        llm.generate("prompt")
