from rag.conversation import history_title
from rag.storage.databricks import DatabricksStore


class Recorder:
    """Captures statements and bound parameters instead of asserting on SQL text."""

    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or [[1]]

    def execute(self, statement, timeout_seconds=300, *, parameters=None):
        self.calls.append((statement, parameters or {}))
        return type("R", (), {"rows": self.rows})()


def test_conversation_title_is_trimmed_not_truncated_mid_word():
    assert history_title("  what   is  genie ") == "what is genie"
    assert history_title("x " * 100).endswith("…")


def test_create_conversation_binds_values_rather_than_inlining_them():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder().execute
    store.calls = []
    conversation_id = store.create_conversation("eric@example.com", "it's a question")
    statement, parameters = store.execute.__self__.calls[0]
    assert "it's" not in statement
    assert parameters["title"] == "it's a question"
    assert parameters["conversation_id"] == conversation_id


def test_append_turn_rejects_a_conversation_the_owner_does_not_own():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder(rows=[[0]]).execute
    try:
        store.append_turn("someone@else.com", "abc", question="q", resolved_query="q",
                          answer=type("A", (), {"text": "t", "supported": True,
                                                "provider": "p", "snapshot_id": "s"})(),
                          citation_ids=[], latency_ms=1)
    except PermissionError:
        return
    raise AssertionError("expected PermissionError")
