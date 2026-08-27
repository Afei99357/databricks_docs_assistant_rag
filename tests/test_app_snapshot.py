import pytest

from rag.store import SqlResult
from rag.workflow import load_current_chunks


class Store:
    statement: str | None = None

    def execute(self, statement):
        self.statement = statement
        return SqlResult([], [["c1", "doc", "v1", "0", "body", '["Parent", "Section"]', "https://docs", "Title"]])


def test_load_current_chunks_preserves_heading_path_from_statement_json():
    store = Store()
    chunks = load_current_chunks(store, "catalog.schema.rag_chunks")
    assert chunks[0].heading_path == ("Parent", "Section")


def test_load_current_chunks_qualifies_chunk_columns_when_joining_documents():
    store = Store()
    load_current_chunks(store, "catalog.schema.rag_chunks", "catalog.schema.rag_documents")
    assert "SELECT c.chunk_id,c.doc_id,c.document_version" in store.statement
    assert "ORDER BY c.doc_id,c.document_version,c.position" in store.statement


def test_load_current_chunks_rejects_empty_source():
    class EmptyStore:
        def execute(self, statement): return SqlResult([], [])
    with pytest.raises(RuntimeError, match="no chunks"):
        load_current_chunks(EmptyStore(), "catalog.schema.rag_chunks")
