import pytest

from rag.llm.providers import _client_config_kwargs, _response_text


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
