from datetime import datetime, timezone

from rag.models import Document, IndexSnapshot
from rag.storage.databricks import DatabricksStore


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


def test_chunk_inserts_batch_so_large_documents_stay_under_the_parameter_ceiling():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    recorder = Recorder()
    store.execute = recorder.execute
    document = type("D", (), {"doc_id": "d", "document_version": "v"})()
    chunks = [
        type("C", (), {
            "chunk_id": f"c{i}", "doc_id": "d", "document_version": "v",
            "position": i, "text": "t", "heading_path": ("H",),
            "source_url": "u", "source_title": "T",
        })()
        for i in range(109)
    ]
    store.replace_document_chunks(document, chunks)
    inserts = [c for c in recorder.calls if c[0].lstrip().startswith("INSERT")]
    assert len(inserts) == 3  # 40 + 40 + 29
    assert all(len(c[1]) <= 512 for c in inserts)


def test_apostrophes_reach_the_parameters_not_the_statement():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    recorder = Recorder()
    store.execute = recorder.execute
    document = type("D", (), {"doc_id": "d", "document_version": "v"})()
    chunk = type("C", (), {
        "chunk_id": "c", "doc_id": "d", "document_version": "v",
        "position": 0, "text": "VALUES ('low', 'high')",
        "heading_path": (), "source_url": "u", "source_title": "T",
    })()
    store.replace_document_chunks(document, [chunk])
    insert = next(c for c in recorder.calls if c[0].lstrip().startswith("INSERT"))
    assert "'low'" not in insert[0]
    assert "VALUES ('low', 'high')" in insert[1].values()


def test_replace_document_chunks_deletes_the_prior_version_first():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    recorder = Recorder()
    store.execute = recorder.execute
    document = type("D", (), {"doc_id": "d", "document_version": "v2"})()
    store.replace_document_chunks(document, [])
    assert len(recorder.calls) == 1
    statement, parameters = recorder.calls[0]
    assert statement.lstrip().startswith("DELETE")
    assert parameters == {"doc_id": "d", "version": "v2"}


def test_upsert_document_binds_apostrophes_and_arrays_as_parameters():
    store = DatabricksStore.__new__(DatabricksStore)
    store.namespace = "cat.sch"
    recorder = Recorder()
    store.execute = recorder.execute
    document = Document(
        doc_id="d", requested_url="u", canonical_url="u",
        title="Databricks's docs", category="c", source_last_updated=None,
        content_hash="h", document_version="v", status="ok",
        source_origins=("crawl", "sitemap"),
    )
    store.upsert_document(document, action="published")
    statement, parameters = recorder.calls[0]
    assert "Databricks's docs" not in statement
    assert parameters["title"] == "Databricks's docs"
    assert "from_json(:source_origins" in statement
    assert parameters["source_origins"] == '["crawl", "sitemap"]'
    assert "current_timestamp()" in statement
    assert "retrieved_at" not in parameters


def test_corpus_methods_take_no_table_argument():
    import inspect

    for name in (
        "documents",
        "clear_indexed_content_hashes",
        "mark_documents_materialized",
        "active_snapshot_fingerprint",
    ):
        assert "table" not in inspect.signature(getattr(DatabricksStore, name)).parameters
