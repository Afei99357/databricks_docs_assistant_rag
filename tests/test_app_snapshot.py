import pytest

from rag.store import DatabricksStore, SqlResult


class Store:
    statement: str | None = None
    parameters: dict | None = None

    def execute(self, statement, timeout_seconds=300, *, parameters=None):
        self.statement = statement
        self.parameters = parameters
        return SqlResult(
            [],
            [["c1", "doc", "v1", "0", "body", '["Parent", "Section"]', "https://docs", "Title"]],
        )


def test_current_chunks_preserves_heading_path_from_statement_json():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "catalog.schema"
    store.execute = Store().execute
    chunks = DatabricksStore.current_chunks(store)
    assert chunks[0].heading_path == ("Parent", "Section")


def test_current_chunks_joins_documents_and_filters_status():
    recorder = Store()
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "catalog.schema"
    store.execute = recorder.execute
    DatabricksStore.current_chunks(store)
    assert "SELECT c.chunk_id,c.doc_id,c.document_version" in recorder.statement
    assert "JOIN catalog.schema.rag_documents d" in recorder.statement
    assert "WHERE d.status IN ('ok','pending_snapshot')" in recorder.statement
    assert "ORDER BY c.doc_id,c.document_version,c.position" in recorder.statement


def test_current_chunks_rejects_empty_source():
    class EmptyStore:
        def execute(self, statement, timeout_seconds=300, *, parameters=None):
            return SqlResult([], [])

    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "catalog.schema"
    store.execute = EmptyStore().execute
    with pytest.raises(RuntimeError, match="no chunks"):
        DatabricksStore.current_chunks(store)
