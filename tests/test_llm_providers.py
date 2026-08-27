import pytest

from rag.llm.providers import (
    DatabricksEndpointProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    _client_config_kwargs,
    _tool_call_from_openai,
    _tool_calls_from_openai,
    capture_llm_usage,
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

    def __init__(self, message, usage=None):
        self.message, self.usage, self.captured = message, usage, {}

    def create(self, **kwargs):
        self.captured = kwargs
        return type("Completion", (), {
            "choices": [type("Choice", (), {"message": self.message})], "usage": self.usage,
        })


def _provider(cls, message, *args, usage=None, **kwargs):
    """Build a provider whose client is a recording stub."""
    provider = cls(*args, **kwargs)
    completions = _Completions(message, usage)
    provider._client = lambda: type("Client", (), {"chat": type("Chat", (), {"completions": completions})})
    return provider, completions


def test_ollama_talks_to_the_openai_compatible_endpoint():
    # Ollama's own /api/chat and /api/generate routes work, but speaking the
    # same protocol as Databricks is what keeps the two runtimes at parity.
    assert OllamaProvider("http://localhost:11434/", "qwen").base_url == "http://localhost:11434/v1"


def test_ollama_disables_reasoning_through_the_vendor_extension():
    assert OllamaProvider("http://localhost:11434", "qwen").extra_body == {"think": False}


def test_openai_compatible_provider_keeps_or_adds_the_v1_path():
    assert OpenAICompatibleProvider("http://intuition.local:1234", "muse").base_url == "http://intuition.local:1234/v1"
    assert OpenAICompatibleProvider("http://intuition.local:1234/v1", "muse").base_url == "http://intuition.local:1234/v1"


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
    assert completions.captured["parallel_tool_calls"] is True
    assert completions.captured["model"] == "qwen"


def test_the_provider_preserves_every_native_tool_call_in_one_assistant_turn():
    assistant = _Message([
        _Call("search_docs", '{"query": "definition"}', "call_one"),
        _Call("search_docs", '{"query": "limitations"}', "call_two"),
    ])
    calls = _tool_calls_from_openai(assistant)
    assert [(call.name, call.arguments, call.call_id) for call in calls] == [
        ("search_docs", {"query": "definition"}, "call_one"),
        ("search_docs", {"query": "limitations"}, "call_two"),
    ]
    assert calls[0].message == calls[1].message


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


def test_provider_captures_reported_usage_per_call():
    usage = type("Usage", (), {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29})()
    provider, _ = _provider(OllamaProvider, _Message(content="answer"), "http://localhost:11434", "qwen", usage=usage)

    with capture_llm_usage() as calls:
        provider.complete("Answer from sources.")

    assert len(calls) == 1
    assert calls[0].operation == "completion"
    assert (calls[0].input_tokens, calls[0].output_tokens, calls[0].total_tokens) == (21, 8, 29)


def test_provider_records_unknown_usage_without_estimating_tokens():
    provider, _ = _provider(OllamaProvider, _Message(content="answer"), "http://localhost:11434", "qwen")

    with capture_llm_usage() as calls:
        provider.complete("Answer from sources.")

    assert (calls[0].input_tokens, calls[0].output_tokens, calls[0].total_tokens) == (None, None, None)


def test_the_databricks_provider_shares_the_same_request_path():
    assistant = _Message([_Call("read_chunks", '{"labels": ["S1", "S2"]}', "call_xyz")])
    provider, completions = _provider(DatabricksEndpointProvider, assistant, "chat-endpoint")

    call = provider.call_tool([{"role": "user", "content": "Question: x"}], [])

    assert (call.name, call.arguments) == ("read_chunks", {"labels": ["S1", "S2"]})
    assert completions.captured["model"] == "chat-endpoint"
    assert completions.captured["tool_choice"] == "required"
    assert "parallel_tool_calls" not in completions.captured


def test_the_openai_normalizer_raises_when_no_tool_was_called():
    with pytest.raises(RuntimeError, match="tool call"):
        _tool_call_from_openai(_Message())
