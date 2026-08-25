import pytest

from rag.llm.providers import _response_text


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
