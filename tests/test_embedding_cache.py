from rag.models import Chunk, EmbeddingSpec, StoredEmbedding
from rag.storage.sqlite import SQLiteStore


def _chunk(chunk_id="chunk-1"):
    return Chunk(chunk_id, "doc", "version", 0, "text", (), "https://example.invalid", "Example")


def test_sqlite_embedding_cache_reuses_only_compatible_vectors(tmp_path):
    store = SQLiteStore(str(tmp_path / "cache.sqlite"))
    chunk = _chunk()
    spec = EmbeddingSpec("model-a", "v1", 3)

    assert store.missing_embeddings([chunk], spec) == [chunk]
    store.save_embeddings([StoredEmbedding(chunk.chunk_id, spec, (0.1, 0.2, 0.3))])

    assert store.missing_embeddings([chunk], spec) == []
    assert store.embeddings_for([chunk], spec)[0].vector == (0.1, 0.2, 0.3)
    assert store.missing_embeddings([chunk], EmbeddingSpec("model-a", "v2", 3)) == [chunk]
    assert store.missing_embeddings([chunk], EmbeddingSpec("model-b", "v1", 3)) == [chunk]


def test_sqlite_schema_upgrade_adds_snapshot_revision_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE rag_index_snapshots (snapshot_id TEXT NOT NULL)")
    conn.close()

    SQLiteStore(str(path))

    check = sqlite3.connect(path)
    columns = {row[1] for row in check.execute("PRAGMA table_info(rag_index_snapshots)")}
    assert {"embedding_revision", "chunking_revision"} <= columns
