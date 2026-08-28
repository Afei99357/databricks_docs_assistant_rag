from datetime import datetime, timezone

from rag.models import IndexSnapshot
from rag.store import DatabricksStore


class Recorder:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or []

    def execute(self, statement, timeout_seconds=300, *, parameters=None):
        self.calls.append((statement, parameters or {}))
        return type("R", (), {"rows": self.rows})()


def test_activate_snapshot_deactivates_the_previous_one_first():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    recorder = Recorder()
    store.execute = recorder.execute
    store.activate_snapshot(IndexSnapshot(
        snapshot_id="abc", embedding_model="qwen", embedding_dimension=1024,
        chunk_count=3, artifact_path="/Volumes/x/index.faiss", status="active",
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc), corpus_fingerprint="fp",
    ))
    assert "active=FALSE" in recorder.calls[0][0]
    assert recorder.calls[1][1]["snapshot_id"] == "abc"


def test_activate_snapshot_derives_the_chunk_map_path_alongside_the_artifact():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    recorder = Recorder()
    store.execute = recorder.execute
    store.activate_snapshot(IndexSnapshot(
        snapshot_id="abc", embedding_model="qwen", embedding_dimension=1024,
        chunk_count=3, artifact_path="/Volumes/x/snap/index.faiss", status="active",
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc), corpus_fingerprint="fp",
    ))
    assert recorder.calls[1][1]["chunk_map_path"] == "/Volumes/x/snap/chunk_map.json"
