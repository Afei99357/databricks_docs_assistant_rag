import pytest

from rag.llm.providers import (
    DatabricksEndpointProvider,
    OllamaProvider,
    _client_config_kwargs,
    _tool_call_from_openai,
)


def test_databricks_client_config_bounds_the_request_and_retry_timeouts():
    # An unbounded serving-endpoint call can hang a Databricks App request for
    # as long as the endpoint takes; the agent loop cannot interrupt it.
    assert _client_config_kwargs(None, 90) == {"http_timeout_seconds": 90, "retry_timeout_seconds": 90}


def test_databricks_client_config_keeps_the_configured_profile():
    assert _client_config_kwargs("free-workspace", 45)["profile"] == "free-workspace"


class _Function:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Call:
    def __init__(self, name, arguments, call_id):
        self.id, self.function = call_id, _Function(name, arguments)


class _Message:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls, self.content = tool_calls, content

    def model_dump(self, exclude_none=False):
        return {"role": "assistant", "tool_calls": [{"id": c.id} for c in self.tool_calls or []]}


class _Completions:
    """Stands in for ``client.chat.completions``, recording the request."""

    def __init__(self, message):
        self.message, self.captured = message, {}

    def create(self, **kwargs):
        self.captured = kwargs
        return type("Completion", (), {"choices": [type("Choice", (), {"message": self.message})]})


def _provider(cls, message, *args, **kwargs):
    """Build a provider whose client is a recording stub."""
    provider = cls(*args, **kwargs)
    completions = _Completions(message)
    provider._client = lambda: type("Client", (), {"chat": type("Chat", (), {"completions": completions})})
    return provider, completions


def test_ollama_talks_to_the_openai_compatible_endpoint():
    # Ollama's own /api/chat and /api/generate routes work, but speaking the
    # same protocol as Databricks is what keeps the two runtimes at parity.
    assert OllamaProvider("http://localhost:11434/", "qwen").base_url == "http://localhost:11434/v1"


def test_ollama_disables_reasoning_through_the_vendor_extension():
    assert OllamaProvider("http://localhost:11434", "qwen").extra_body == {"think": False}


def test_the_provider_returns_a_native_tool_call():
    assistant = _Message([_Call("final", '{"selected": ["S1", "S4"]}', "call_abc")])
    provider, completions = _provider(OllamaProvider, assistant, "http://localhost:11434", "qwen")
    tools = [{"type": "function", "function": {"name": "final", "parameters": {}}}]
    messages = [{"role": "system", "content": "investigate"}, {"role": "user", "content": "Question: x"}]

    call = provider.call_tool(messages, tools)

    assert call.name == "final"
    assert call.arguments == {"selected": ["S1", "S4"]}
    assert call.call_id == "call_abc"
    # The assistant turn comes back so the caller can append it verbatim and
    # keep the next request a pure extension of this one.
    assert call.message == {"role": "assistant", "tool_calls": [{"id": "call_abc"}]}
    assert completions.captured["messages"] == messages
    assert completions.captured["tools"] == tools
    assert completions.captured["tool_choice"] == "required"
    assert completions.captured["model"] == "qwen"


def test_the_provider_raises_when_the_model_answers_without_calling_a_tool():
    provider, _ = _provider(OllamaProvider, _Message(content="I think S1."), "http://localhost:11434", "qwen")
    with pytest.raises(RuntimeError, match="prose instead of a tool call"):
        provider.call_tool([], [])


def test_complete_sends_the_prompt_as_a_single_user_turn():
    provider, completions = _provider(OllamaProvider, _Message(content=" grounded answer "),
                                      "http://localhost:11434", "qwen")

    assert provider.complete("Answer from the sources.") == "grounded answer"
    assert completions.captured["messages"] == [{"role": "user", "content": "Answer from the sources."}]
    assert "tools" not in completions.captured


def test_the_databricks_provider_shares_the_same_request_path():
    assistant = _Message([_Call("read_chunks", '{"labels": ["S1", "S2"]}', "call_xyz")])
    provider, completions = _provider(DatabricksEndpointProvider, assistant, "chat-endpoint")

    call = provider.call_tool([{"role": "user", "content": "Question: x"}], [])

    assert (call.name, call.arguments) == ("read_chunks", {"labels": ["S1", "S2"]})
    assert completions.captured["model"] == "chat-endpoint"
    assert completions.captured["tool_choice"] == "required"


def test_the_openai_normalizer_raises_when_no_tool_was_called():
    with pytest.raises(RuntimeError, match="tool call"):
        _tool_call_from_openai(_Message())
