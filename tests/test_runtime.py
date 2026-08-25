import pytest

from rag.index.embeddings import HashEmbeddingProvider
from rag.index.runtime import ActiveSnapshotRetriever
from rag.index.service import build_and_activate
from rag.models import Chunk


def test_runtime_loads_only_active_completed_snapshot(tmp_path):
    chunks = [Chunk("id", "d", "v", 0, "Genie concepts evidence", (), "https://docs.databricks.com/x", "Docs")]
    build_and_activate(chunks, HashEmbeddingProvider(), tmp_path)
    retrieved = ActiveSnapshotRetriever(tmp_path, HashEmbeddingProvider(), 1).retrieve("Genie")
    assert retrieved[0].chunk.chunk_id == "id"

