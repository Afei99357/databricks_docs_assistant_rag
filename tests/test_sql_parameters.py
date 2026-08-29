from rag.agent.retrieval import RetrievalTrace, ToolStep
from rag.models import Chunk, Document, EmbeddingSpec, StoredEmbedding
from rag.storage.databricks import CHUNK_INSERT_BATCH, DatabricksStore, to_statement_parameters


def test_strings_bind_verbatim_without_escaping():
    params = {p.name: (p.type, p.value) for p in to_statement_parameters({"v": "it's \\ ok"})}
    assert params["v"] == ("STRING", "it's \\ ok")


def test_none_binds_as_a_null_typed_parameter():
    params = {p.name: (p.type, p.value) for p in to_statement_parameters({"v": None})}
    assert params["v"] == ("STRING", None)


def test_bool_and_numbers_get_their_own_types():
    params = {p.name: (p.type, p.value) for p in to_statement_parameters(
        {"flag": True, "count": 7, "score": 1.5}
    )}
    assert params["flag"] == ("BOOLEAN", "true")
    assert params["count"] == ("BIGINT", "7")
    assert params["score"] == ("DOUBLE", "1.5")


# `to_statement_parameters` binds an absent/None value as an untyped-looking
# STRING NULL (see test_none_binds_as_a_null_typed_parameter above). Databricks
# ANSI store assignment rejects assigning a STRING NULL into a non-STRING
# column ("Cannot write string to int"), so every nullable non-STRING column
# bound below must wrap its placeholder in an explicit CAST in the SQL text
# itself -- these tests guard the generated statement, not the parameter list.


class Recorder:
    """Captures statements and bound parameters instead of asserting on SQL text."""

    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows if rows is not None else [[1]]

    def execute(self, statement, timeout_seconds=300, *, parameters=None):
        self.calls.append((statement, parameters or {}))
        return type("R", (), {"rows": self.rows})()


def _chunk():
    return Chunk("chunk-1", "doc-1", "v1", 0, "text", (), "https://docs.databricks.com/x", "X")


def _document():
    return Document(
        "doc-1", "https://x", "https://x", "Title", "docs", None, "hash", "v1", "ok",
    )


def test_replace_document_chunks_casts_the_nullable_embedding_dimension():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder().execute

    # embedding_dimension defaults to None -- rag/workflow.py calls this
    # without embedding arguments on every changed document.
    store.replace_document_chunks(_document(), [_chunk()])

    insert_statement, insert_parameters = store.execute.__self__.calls[-1]
    assert "CAST(:r0_dim AS INT)" in insert_statement
    assert insert_parameters["r0_dim"] is None


def test_record_feedback_casts_the_nullable_latency_ms():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.provider, store.model = "p", "m"
    store.execute = Recorder().execute

    # payload.get("latency_ms") returns None when the key is absent.
    store.record_feedback({"rating": "up"})

    statement, parameters = store.execute.__self__.calls[-1]
    assert "CAST(:latency_ms AS BIGINT)" in statement
    assert parameters["latency_ms"] is None


def test_append_turn_casts_the_nullable_latency_ms():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder(rows=[[1], [0]]).execute
    answer = type("A", (), {"text": "t", "supported": True, "provider": "p", "snapshot_id": "s"})()

    store.append_turn("owner@x.com", "conv-1", question="q", resolved_query="q",
                       answer=answer, citation_ids=[], latency_ms=5)

    insert_statement, _ = store.execute.__self__.calls[2]
    assert "INSERT INTO" in insert_statement
    assert "CAST(:latency_ms AS BIGINT)" in insert_statement


def test_record_request_trace_casts_the_nullable_latency_ms_in_both_tables():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.provider, store.model = "p", "m"
    store.agent_provider, store.agent_model = "p", "m"
    store.execute = Recorder().execute
    answer = type("A", (), {"text": "t", "supported": True, "provider": "p", "snapshot_id": "s"})()
    retrieval_trace = RetrievalTrace(
        (), (), "answered",
        steps=(ToolStep("search_docs", "ok", query="q", candidate_ids=("chunk-1",)),),
    )

    store.record_request_trace(
        turn_id="turn-1", conversation_id="conv-1", owner="owner@x.com",
        question="q", resolved_query="q", retrieval_trace=retrieval_trace,
        results=[], grounding_trace=None, answer=answer, latency_ms=7,
    )

    request_trace_statement, _ = store.execute.__self__.calls[0]
    retrieval_trace_statement, _ = store.execute.__self__.calls[1]
    assert "CAST(:latency_ms AS BIGINT)" in request_trace_statement
    assert "CAST(:latency_ms AS BIGINT)" in retrieval_trace_statement


def test_save_embeddings_batches_instead_of_one_round_trip_per_row():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder().execute
    spec = EmbeddingSpec("model-a", "v1")
    # One row past a full batch, so this must split into two DELETE+INSERT pairs.
    embeddings = [
        StoredEmbedding(f"chunk-{i}", spec, (0.1, 0.2)) for i in range(CHUNK_INSERT_BATCH + 1)
    ]

    store.save_embeddings(embeddings)

    calls = store.execute.__self__.calls
    assert len(calls) == 4  # DELETE+INSERT per batch, not one pair per row
    first_delete, first_insert = calls[0], calls[1]
    second_insert = calls[3]
    assert "array_contains(from_json(:chunk_ids,'array<string>'),chunk_id)" in first_delete[0]
    assert first_delete[1]["model"] == "model-a"
    assert first_insert[0].count("VALUES") == 1
    assert first_insert[0].count("from_json(:r") == CHUNK_INSERT_BATCH
    assert second_insert[0].count("from_json(:r") == 1
    assert first_insert[1]["r0_id"] == "chunk-0"
    assert first_insert[1]["r0_dim"] == 2


def test_save_embeddings_groups_by_model_and_revision():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder().execute
    embeddings = [
        StoredEmbedding("chunk-1", EmbeddingSpec("model-a", "v1"), (0.1,)),
        StoredEmbedding("chunk-2", EmbeddingSpec("model-b", "v1"), (0.2,)),
    ]

    store.save_embeddings(embeddings)

    calls = store.execute.__self__.calls
    assert len(calls) == 4  # a separate DELETE+INSERT pair per (model, revision)
    models_deleted = {calls[0][1]["model"], calls[2][1]["model"]}
    assert models_deleted == {"model-a", "model-b"}


def test_save_embeddings_does_nothing_for_an_empty_list():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    store.execute = Recorder().execute

    store.save_embeddings([])

    assert store.execute.__self__.calls == []
