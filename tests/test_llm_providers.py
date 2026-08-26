import pytest

from rag.llm.providers import (
    OllamaProvider,
    _client_config_kwargs,
    _response_text,
    _tool_call_from_openai,
)


def test_response_text_accepts_plain_string():
    assert _response_text(" answer ") == "answer"


def test_response_text_uses_text_parts_and_omits_reasoning_parts():
    content = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "internal"}]},
        {"type": "text", "text": "grounded answer"},
    ]
    assert _response_text(content) == "grounded answer"


def test_response_text_rejects_unknown_shape():
    with pytest.raises(RuntimeError, match="unsupported"):
        _response_text({"text": "not a supported response"})


def test_databricks_client_config_bounds_the_request_and_retry_timeouts():
    # An unbounded serving-endpoint call can hang a Databricks App request for
    # as long as the endpoint takes; the agent loop cannot interrupt it.
    assert _client_config_kwargs(None, 90) == {"http_timeout_seconds": 90, "retry_timeout_seconds": 90}


def test_databricks_client_config_keeps_the_configured_profile():
    assert _client_config_kwargs("free-workspace", 45)["profile"] == "free-workspace"


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_ollama_provider_returns_a_native_tool_call(monkeypatch):
    captured = {}
    assistant = {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "final", "arguments": {"selected": ["S1", "S4"]}}}]}

    def fake_post(url, json=None, timeout=None):
        captured["url"], captured["body"] = url, json
        return _FakeResponse({"message": assistant})

    monkeypatch.setattr("rag.llm.providers.requests.post", fake_post)
    tools = [{"type": "function", "function": {"name": "final", "parameters": {}}}]
    messages = [{"role": "system", "content": "investigate"}, {"role": "user", "content": "Question: x"}]
    call = OllamaProvider("http://localhost:11434", "qwen").call_tool(messages, tools)

    assert call.name == "final"
    assert call.arguments == {"selected": ["S1", "S4"]}
    # The assistant turn comes back so the caller can append it verbatim and
    # keep the next request a pure extension of this one.
    assert call.message == assistant
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["tools"] == tools
    assert captured["body"]["messages"] == messages


def test_ollama_provider_raises_when_the_model_answers_without_calling_a_tool(monkeypatch):
    monkeypatch.setattr("rag.llm.providers.requests.post",
                        lambda url, json=None, timeout=None: _FakeResponse({"message": {"content": "I think S1."}}))
    with pytest.raises(RuntimeError, match="tool call"):
        OllamaProvider("http://localhost:11434", "qwen").call_tool([], [])


class _Function:
    name, arguments = "read_chunks", '{"labels": ["S1", "S2"]}'


class _Call:
    id, function = "call_abc", _Function()


class _Message:
    def __init__(self):
        self.tool_calls = [_Call()]

    def model_dump(self, exclude_none=False):
        return {"role": "assistant", "tool_calls": [{"id": "call_abc"}]}


def test_openai_tool_call_arguments_are_decoded_by_the_provider():
    # The OpenAI wire format delivers arguments as a JSON string; Ollama
    # delivers a dict. Providers normalize so the agent never decodes anything.
    call = _tool_call_from_openai(_Message())
    assert call.name == "read_chunks"
    assert call.arguments == {"labels": ["S1", "S2"]}
    assert call.call_id == "call_abc"
    assert call.message == {"role": "assistant", "tool_calls": [{"id": "call_abc"}]}


def test_openai_tool_call_raises_when_no_tool_was_called():
    class Empty:
        tool_calls = None

    with pytest.raises(RuntimeError, match="tool call"):
        _tool_call_from_openai(Empty())
