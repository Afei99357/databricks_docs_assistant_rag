from rag.index.embeddings import HashEmbeddingProvider
from rag.index.runtime import ActiveSnapshotRetriever
from rag.index.service import build_and_activate
from rag.models import Chunk


def test_runtime_loads_only_active_completed_snapshot(tmp_path):
    chunks = [Chunk("id", "d", "v", 0, "Genie concepts evidence", (), "https://docs.databricks.com/x", "Docs")]
    build_and_activate(chunks, HashEmbeddingProvider(), tmp_path)
    retrieved = ActiveSnapshotRetriever(tmp_path, HashEmbeddingProvider(), 1).retrieve("Genie")
    assert retrieved[0].chunk.chunk_id == "id"


def test_runtime_exposes_chunk_and_document_tools(tmp_path):
    chunks = [
        Chunk("first", "doc", "v", 0, "Genie agent requirements", ("Requirements",), "https://docs.databricks.com/x", "Docs"),
        Chunk("second", "doc", "v", 1, "SELECT privilege is required", ("Requirements",), "https://docs.databricks.com/x", "Docs"),
        Chunk("other", "other-doc", "v", 0, "Unrelated content", (), "https://docs.databricks.com/y", "Other"),
    ]
    build_and_activate(chunks, HashEmbeddingProvider(), tmp_path)
    runtime = ActiveSnapshotRetriever(tmp_path, HashEmbeddingProvider(), 3)
    assert [item.chunk.chunk_id for item in runtime.read_chunks(["second"])] == ["second"]
    assert [item.chunk.chunk_id for item in runtime.related_chunks("first")] == ["first", "second"]
    assert {item.chunk.chunk_id for item in runtime.search_within_document("https://docs.databricks.com/x", "privilege", 3)} == {"first", "second"}
