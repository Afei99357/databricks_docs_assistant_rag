from rag.agent.retrieval import RetrievalTrace, ToolStep
from rag.llm.grounding import GroundingTrace
from rag.llm.providers import LLMCallUsage
from rag.models import Answer, Chunk, RetrievalResult
from rag.store import DatabricksStore


class Recorder:
    def __init__(self):
        self.calls = []

    def execute(self, statement, timeout_seconds=300, *, parameters=None):
        self.calls.append((statement, parameters or {}))
        return type("R", (), {"rows": []})()


def _chunk():
    return Chunk("chunk-1", "doc", "version", 0, "evidence", ("Heading",), "https://docs.databricks.com/x", "Docs")


def test_request_trace_binds_answer_text_rather_than_inlining_it():
    store = Recorder()
    store.namespace = "cat.sch"
    DatabricksStore.record_request_trace(
        store, turn_id="t", conversation_id="c", owner="e@x.com",
        question="what's a governed tag?", resolved_query="governed tag",
        retrieval_trace=None, results=[], grounding_trace=None,
        answer=type("A", (), {"text": "It's this.", "supported": True,
                              "provider": "p", "snapshot_id": "s",
                              "citations": ()})(),
        latency_ms=5,
    )
    statement, parameters = store.calls[0]
    assert "It's this." not in statement
    assert parameters["final_answer_text"] == "It's this."


def test_request_trace_binds_retrieval_queries_and_citation_labels_as_json():
    store = Recorder()
    store.namespace = "cat.sch"
    chunk = _chunk()
    DatabricksStore.record_request_trace(
        store, turn_id=None, conversation_id=None, owner=None,
        question="What's it?", resolved_query="What's it?",
        retrieval_trace=RetrievalTrace(
            ("What's it?",), ("chunk-1",), "answered",
            (ToolStep("search_docs", "ok", query="What's it?", candidate_ids=("chunk-1",)),),
            "agent_satisfied",
        ),
        results=[RetrievalResult(chunk, 0.9, "snapshot")],
        grounding_trace=GroundingTrace("Answer [S1].", ("S1",), None),
        answer=Answer("Answer [S1].", (), True, "ollama", "snapshot"), latency_ms=12,
    )
    statement, parameters = store.calls[0]
    assert "array(" not in statement
    assert "What's it?" not in statement
    assert "from_json(:retrieval_queries" in statement
    assert "from_json(:parsed_citation_labels" in statement
    assert parameters["retrieval_queries"] == '["What\'s it?"]'
    assert parameters["parsed_citation_labels"] == '["S1"]'


def test_request_trace_writes_retrieval_rows_with_json_bound_id_arrays_when_turn_id_present():
    store = Recorder()
    store.namespace = "cat.sch"
    chunk = _chunk()
    DatabricksStore.record_request_trace(
        store, turn_id="turn", conversation_id="conversation", owner="user",
        question="What is it?", resolved_query="What is it?",
        retrieval_trace=RetrievalTrace(
            ("What is it?",), ("chunk-1",), "answered",
            (ToolStep("search_docs", "ok", query="What is it?",
                      candidate_ids=("chunk-1",), selected_chunk_ids=("chunk-1",)),),
            "agent_satisfied",
        ),
        results=[RetrievalResult(chunk, 0.9, "snapshot")],
        grounding_trace=GroundingTrace("Answer [S1].", ("S1",), None),
        answer=Answer("Answer [S1].", (), True, "ollama", "snapshot"), latency_ms=12,
        llm_usage=[LLMCallUsage("openai-compatible", "muse", "tool_call", 100, 20, 120, 50)],
    )
    assert len(store.calls) == 2
    trace_statement, trace_parameters = store.calls[0]
    assert "INSERT INTO cat.sch.rag_request_traces" in trace_statement
    assert '"input_tokens": 100' in trace_parameters["selected_evidence_json"]
    retrieval_statement, retrieval_parameters = store.calls[1]
    assert "INSERT INTO cat.sch.rag_retrieval_traces" in retrieval_statement
    assert "array(" not in retrieval_statement
    assert "chunk-1" not in retrieval_statement
    assert "from_json(:retrieved_chunk_ids" in retrieval_statement
    assert "from_json(:selected_chunk_ids" in retrieval_statement
    assert retrieval_parameters["retrieved_chunk_ids"] == '["chunk-1"]'
    assert retrieval_parameters["selected_chunk_ids"] == '["chunk-1"]'
    assert retrieval_parameters["trace_id"] == trace_parameters["trace_id"]


def test_request_trace_skips_retrieval_rows_when_turn_id_absent():
    store = Recorder()
    store.namespace = "cat.sch"
    DatabricksStore.record_request_trace(
        store, turn_id=None, conversation_id=None, owner=None,
        question="What is it?", resolved_query="What is it?",
        retrieval_trace=RetrievalTrace(
            ("What is it?",), ("chunk-1",), "answered",
            (ToolStep("search_docs", "ok", query="What is it?", candidate_ids=("chunk-1",)),),
            "agent_satisfied",
        ),
        results=[], grounding_trace=GroundingTrace(None, (), None),
        answer=Answer("Answer.", (), True, "ollama", "snapshot"), latency_ms=12,
    )
    assert len(store.calls) == 1


def test_feedback_binds_comment_and_chunk_ids_as_parameters_not_inlined_sql():
    store = Recorder()
    store.namespace = "cat.sch"
    DatabricksStore.record_feedback(
        store, turn_id="turn-1", owner="e@x.com", rating="up",
        comment="it's great", retrieved_chunk_ids=["chunk-1", "chunk-2"], latency_ms=42,
    )
    statement, parameters = store.calls[0]
    assert "it's great" not in statement
    assert "chunk-1" not in statement
    assert "array(" not in statement
    assert "from_json(:retrieved_chunk_ids" in statement
    assert parameters["comment"] == "it's great"
    assert parameters["retrieved_chunk_ids"] == '["chunk-1", "chunk-2"]'
    assert parameters["turn_id"] == "turn-1"
    assert parameters["owner_user_id"] == "e@x.com"
    assert parameters["rating"] == "up"
    assert parameters["latency_ms"] == 42
