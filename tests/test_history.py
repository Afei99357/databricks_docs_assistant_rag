from rag.history import _history_title


def test_history_title_normalizes_and_truncates_long_questions():
    assert _history_title("  A\nquestion  ") == "A question"
    title = _history_title("word " * 30)
    assert title.endswith("…")
    assert len(title) <= 88
