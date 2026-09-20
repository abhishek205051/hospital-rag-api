from app.rag.llm import FakeLLM


def test_fake_llm_returns_configured_response():
    assert FakeLLM(response="Hello").generate("any prompt") == "Hello"


def test_fake_llm_records_calls_and_last_prompt():
    llm = FakeLLM()
    llm.generate("first")
    llm.generate("second")
    assert llm.calls == 2
    assert llm.last_prompt == "second"


def test_fake_llm_has_default_response():
    assert FakeLLM().generate("prompt") != ""
