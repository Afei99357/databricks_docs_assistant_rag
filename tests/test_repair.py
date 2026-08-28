from rag.store import DatabricksStore


class Store:
    """Records statements and answers the pre-count query."""

    def __init__(self, count=275):
        self.statements = []
        self.count = count

    def execute(self, statement):
        self.statements.append(statement)
        return type("R", (), {"rows": [[self.count]]})()


def test_clear_indexed_content_hashes_reports_and_clears():
    store = Store(count=275)
    store.namespace = "cat.sch"
    affected = DatabricksStore.clear_indexed_content_hashes(store)
    assert affected == 275
    assert "UPDATE cat.sch.rag_documents SET indexed_content_hash = NULL" in store.statements[-1]


def test_clear_counts_only_documents_that_were_materialized():
    store = Store()
    store.namespace = "cat.sch"
    DatabricksStore.clear_indexed_content_hashes(store)
    assert "indexed_content_hash IS NOT NULL" in store.statements[0]


def test_clear_tolerates_an_empty_table():
    store = Store(count=None)
    store.namespace = "cat.sch"
    assert DatabricksStore.clear_indexed_content_hashes(store) == 0


def test_force_skips_the_fingerprint_gate(monkeypatch):
    """A repair changes chunk text but not document versions, so the
    fingerprint still matches and would otherwise skip the rebuild."""
    from rag import cli

    calls = []
    monkeypatch.setattr(cli, "read_active_fingerprint", lambda root: "same-fingerprint")
    monkeypatch.setattr(cli, "build_snapshot", lambda *a, **k: calls.append("built") or _Published())
    _install_stubs(monkeypatch, cli, fingerprint="same-fingerprint")

    cli.build_local_snapshot(force=False)
    assert calls == []          # unchanged corpus: correctly skipped

    cli.build_local_snapshot(force=True)
    assert calls == ["built"]   # repair: rebuilt anyway


class _Published:
    class metadata:
        snapshot_id = "snap"
        chunk_count = 1


def _install_stubs(monkeypatch, cli, *, fingerprint):
    monkeypatch.setenv("RAG_LOCAL_INDEX_DIR", "/tmp/does-not-matter")
    monkeypatch.setenv("RAG_CATALOG", "cat")
    monkeypatch.setenv("RAG_SCHEMA", "sch")
    monkeypatch.setenv("RAG_WAREHOUSE_ID", "wh")
    monkeypatch.setattr(cli, "DatabricksStore", lambda *a, **k: _Store())
    monkeypatch.setattr(
        cli, "refresh_sources", lambda *a, **k: type("R", (), {"corpus_fingerprint": fingerprint})()
    )
    monkeypatch.setattr(cli, "OllamaEmbeddingProvider", lambda *a, **k: object())


class _Store:
    def current_chunks(self): return ["chunk"]
    def mark_documents_materialized(self, *a, **k): pass
