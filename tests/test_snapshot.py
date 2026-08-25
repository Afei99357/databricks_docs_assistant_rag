import importlib.util

import pytest

from rag.index.embeddings import HashEmbeddingProvider
from rag.index.faiss_store import FaissSnapshot, read_active_manifest
from rag.index.service import build_and_activate
from rag.models import Chunk


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="FAISS is an optional local dependency in this test environment")
def test_snapshot_validates_and_activates_atomically(tmp_path):
    chunks = [Chunk("a", "d", "v", 0, "Genie concepts", (), "https://docs.databricks.com/x", "X"), Chunk("b", "d", "v", 1, "Volume search", (), "https://docs.databricks.com/y", "Y")]
    published = build_and_activate(chunks, HashEmbeddingProvider(), tmp_path)
    assert read_active_manifest(tmp_path) == published.metadata.snapshot_id
    loaded = FaissSnapshot.load(published.local_directory, published.metadata.snapshot_id)
    assert loaded.index.ntotal == 2
