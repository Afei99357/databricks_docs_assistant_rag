import pytest

from rag.store import SqlResult
from rag.workflow import load_current_chunks


class Store:
    def execute(self, statement):
        return SqlResult([], [["c1", "doc", "v1", "0", "body", '["Parent", "Section"]', "https://docs", "Title"]])


def test_load_current_chunks_preserves_heading_path_from_statement_json():
    chunks = load_current_chunks(Store(), "catalog.schema.rag_chunks")
    assert chunks[0].heading_path == ("Parent", "Section")


def test_load_current_chunks_rejects_empty_source():
    class EmptyStore:
        def execute(self, statement): return SqlResult([], [])
    with pytest.raises(RuntimeError, match="no chunks"):
        load_current_chunks(EmptyStore(), "catalog.schema.rag_chunks")
