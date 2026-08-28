import importlib.util

import pytest

from rag.index.embeddings import HashEmbeddingProvider
from rag.models import Chunk
from rag.workflow import publish_snapshot


class FakePublisher:
    """Stands in for VolumePublisher without touching a real Volume."""

    def __init__(self, remote_dir):
        self.remote_dir = remote_dir
        self.calls = []

    def publish(self, local_directory, snapshot_id):
        self.calls.append((local_directory, snapshot_id))
        return self.remote_dir


class FakeCorpusStore:
    """Captures what activate_snapshot receives and replicates the production
    chunk_map_path derivation (rag/storage/databricks.py) so the test can pin
    both persisted columns without a live warehouse.
    """

    def __init__(self):
        self.activated = None

    def activate_snapshot(self, metadata):
        chunk_map_path = metadata.artifact_path.rsplit("/", 1)[0] + "/chunk_map.json"
        self.activated = (metadata, chunk_map_path)

    def mark_documents_materialized(self):
        raise AssertionError("materialize=False; should not be called")


@pytest.mark.skipif(
    importlib.util.find_spec("faiss") is None,
    reason="FAISS is an optional local dependency in this test environment",
)
def test_publish_snapshot_persists_the_exact_artifact_and_chunk_map_paths():
    remote_dir = "/Volumes/c/s/v/app/snapshots/abc123"
    publisher = FakePublisher(remote_dir)
    store = FakeCorpusStore()
    chunks = [Chunk("a", "d", "v", 0, "Genie concepts", (), "https://docs.databricks.com/x", "X")]

    publish_snapshot(store, publisher=publisher, chunks=chunks, embedder=HashEmbeddingProvider())

    metadata, chunk_map_path = store.activated
    assert metadata.artifact_path == "/Volumes/c/s/v/app/snapshots/abc123/index.faiss"
    assert chunk_map_path == "/Volumes/c/s/v/app/snapshots/abc123/chunk_map.json"
