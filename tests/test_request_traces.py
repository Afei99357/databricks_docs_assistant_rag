from rag.agent.retrieval import RetrievalTrace, ToolStep
from rag.llm.grounding import GroundingTrace
from rag.models import Answer, Chunk, RetrievalResult
from rag.store import DatabricksRequestTraceSink


class Store:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)


def test_request_trace_records_evidence_and_grounding_diagnostics():
    store = Store()
    chunk = Chunk("chunk-1", "doc", "version", 0, "evidence", ("Heading",), "https://docs.databricks.com/x", "Docs")
    DatabricksRequestTraceSink(
        store, "catalog.schema.rag_request_traces", provider="ollama", model="qwen",
        retrieval_table="catalog.schema.rag_retrieval_traces",
        agent_provider="openai-compatible", agent_model="muse",
    ).record(
        turn_id="turn", conversation_id="conversation", owner="user", question="What is it?",
        resolved_query="What is it?", retrieval_trace=RetrievalTrace(
            ("What is it?",), ("chunk-1",), "answered",
            (ToolStep("search_docs", "ok", query="What is it?", candidate_ids=("chunk-1",)),),
            "agent_satisfied",
        ),
        results=[RetrievalResult(chunk, 0.9, "snapshot")],
        grounding_trace=GroundingTrace("Answer [S1].", ("S1",), None),
        answer=Answer("Answer [S1].", (), True, "ollama", "snapshot"), latency_ms=12,
    )
    statement = store.statements[0]
    assert "INSERT INTO catalog.schema.rag_request_traces" in statement
    assert "selected_evidence_json" in statement
    assert "tool_steps" in statement
    assert "agent_satisfied" in statement
    assert "openai-compatible" in statement
    assert "muse" in statement
    assert "INSERT INTO catalog.schema.rag_retrieval_traces" in store.statements[1]
    assert "Answer [S1]." in statement
