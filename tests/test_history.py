from rag.conversation import history_title


def test_history_title_normalizes_and_truncates_long_questions():
    assert history_title("  A\nquestion  ") == "A question"
    title = history_title("word " * 30)
    assert title.endswith("…")
    assert len(title) <= 88
