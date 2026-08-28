"""Thread-local SQLite implementation of the local storage protocols."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from uuid import uuid4

from rag.models import EmbeddingSpec, StoredEmbedding

_SCHEMA_FILE = Path(__file__).parent / "schema" / "sqlite.sql"


class SQLiteStore:
    def __init__(
        self, path: str, *, provider=None, model=None, agent_provider=None, agent_model=None
    ):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.provider, self.model = provider, model
        self.agent_provider, self.agent_model = agent_provider or provider, agent_model or model
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        sql = "\n".join(
            line
            for line in _SCHEMA_FILE.read_text().splitlines()
            if not line.lstrip().startswith("--")
        )
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(statement)
        self._add_missing_columns(
            conn,
            "rag_documents",
            {
                "chunked_content_hash": "TEXT",
                "chunked_source_last_updated": "TEXT",
                "chunked_document_version": "TEXT",
            },
        )
        self._add_missing_columns(
            conn,
            "rag_index_snapshots",
            {
                "embedding_revision": "TEXT NOT NULL DEFAULT 'v1'",
                "chunking_revision": "TEXT NOT NULL DEFAULT 'v1'",
            },
        )

    @staticmethod
    def _add_missing_columns(conn, table, columns):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def apply_schema(
        self, schema_file=None, *, catalog=None, schema=None, artifact_volume=None
    ) -> None:
        """No-op: local schema initialization occurs during construction."""

    def documents(self):
        rows = (
            self._connect()
            .execute(
                "SELECT doc_id,requested_url,canonical_url,title,category,source_last_updated,source_content_hash,document_version,status,source_origins,indexed_content_hash,indexed_source_last_updated,consecutive_404_count,error_message,chunked_content_hash,chunked_source_last_updated,chunked_document_version FROM rag_documents"
            )
            .fetchall()
        )
        from rag.models import Document

        return {
            r[0]: Document(
                *r[:9],
                tuple(json.loads(r[9]) if r[9] else ()),
                r[10],
                r[11],
                int(r[12] or 0),
                r[13],
            )
            for r in rows
        }

    def upsert_document(self, document, *, action=None):
        v = {
            "doc_id": document.doc_id,
            "requested_url": document.requested_url,
            "canonical_url": document.canonical_url,
            "title": document.title,
            "category": document.category,
            "source_last_updated": document.source_last_updated,
            "source_content_hash": document.content_hash,
            "document_version": document.document_version,
            "status": document.status,
            "source_origins": json.dumps(list(document.source_origins)),
            "indexed_content_hash": document.indexed_content_hash,
            "indexed_source_last_updated": document.indexed_source_last_updated,
            "consecutive_404_count": document.consecutive_404_count,
            "error_message": document.error_message,
            "chunked_content_hash": document.chunked_content_hash,
            "chunked_source_last_updated": document.chunked_source_last_updated,
            "chunked_document_version": document.chunked_document_version,
            "last_run_action": action,
        }
        cols = list(v)
        updates = ", ".join(f"{x}=excluded.{x}" for x in cols if x != "doc_id")
        self._connect().execute(
            f"INSERT INTO rag_documents ({','.join(cols)},retrieved_at) VALUES ({','.join(':' + x for x in cols)},CURRENT_TIMESTAMP) ON CONFLICT(doc_id) DO UPDATE SET {updates},retrieved_at=CURRENT_TIMESTAMP",
            v,
        )

    def replace_document_chunks(
        self, document, chunks, embedding_model=None, embedding_dimension=None
    ):
        c = self._connect()
        c.execute(
            "DELETE FROM rag_chunks WHERE doc_id=:d AND document_version=:v",
            {"d": document.doc_id, "v": document.document_version},
        )
        c.executemany(
            "INSERT INTO rag_chunks (chunk_id,doc_id,document_version,position,chunk_text,heading_path,source_url,source_title,embedding_model,embedding_dimension,embedding_created_at) VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            [
                (
                    x.chunk_id,
                    x.doc_id,
                    x.document_version,
                    x.position,
                    x.text,
                    json.dumps(list(x.heading_path)),
                    x.source_url,
                    x.source_title,
                    embedding_model,
                    embedding_dimension,
                )
                for x in chunks
            ],
        )

    def prune_document_chunks(self, document):
        self._connect().execute(
            "DELETE FROM rag_chunks WHERE doc_id=? AND document_version<>?",
            (document.doc_id, document.document_version),
        )

    def clear_indexed_content_hashes(self):
        c = self._connect()
        n = c.execute(
            "SELECT count(*) FROM rag_documents WHERE indexed_content_hash IS NOT NULL"
        ).fetchone()[0]
        c.execute("UPDATE rag_documents SET indexed_content_hash=NULL, chunked_content_hash=NULL, chunked_source_last_updated=NULL, chunked_document_version=NULL")
        return n

    def current_chunks(self):
        from rag.models import Chunk

        rows = (
            self._connect()
            .execute(
                "SELECT c.chunk_id,c.doc_id,c.document_version,c.position,c.chunk_text,c.heading_path,c.source_url,c.source_title FROM rag_chunks c JOIN rag_documents d ON c.doc_id=d.doc_id AND c.document_version=d.document_version WHERE d.status IN ('ok','pending_snapshot') ORDER BY c.doc_id,c.document_version,c.position"
            )
            .fetchall()
        )
        if not rows:
            raise RuntimeError(
                "no chunks found in rag_chunks; run the source refresh before building an index"
            )
        return [Chunk(*r[:5], tuple(json.loads(r[5]) if r[5] else ()), *r[6:]) for r in rows]

    def mark_documents_materialized(self):
        c = self._connect()
        c.execute(
            "UPDATE rag_documents SET indexed_content_hash=source_content_hash,indexed_source_last_updated=source_last_updated,status='ok',last_run_action='published' WHERE status='pending_snapshot'"
        )
        c.execute(
            "DELETE FROM rag_chunks WHERE doc_id IN (SELECT doc_id FROM rag_documents WHERE status='removed')"
        )

    def active_snapshot_fingerprint(self):
        r = (
            self._connect()
            .execute(
                "SELECT corpus_fingerprint FROM rag_index_snapshots WHERE active=1 ORDER BY created_at DESC LIMIT 1"
            )
            .fetchone()
        )
        return r[0] if r and r[0] else None

    def activate_snapshot(self, metadata):
        c = self._connect()
        c.execute("UPDATE rag_index_snapshots SET active=0 WHERE active=1")
        c.execute(
            "INSERT INTO rag_index_snapshots (snapshot_id,embedding_model,embedding_dimension,chunk_count,artifact_path,chunk_map_path,status,active,corpus_fingerprint,embedding_revision,chunking_revision,created_at) VALUES (?,?,?,?,?,?,?,1,?,?,?,CURRENT_TIMESTAMP)",
            (
                metadata.snapshot_id,
                metadata.embedding_model,
                metadata.embedding_dimension,
                metadata.chunk_count,
                metadata.artifact_path,
                metadata.artifact_path.rsplit("/", 1)[0] + "/chunk_map.json",
                metadata.status,
                metadata.corpus_fingerprint,
                metadata.embedding_revision,
                metadata.chunking_revision,
            ),
        )

    def missing_embeddings(self, chunks, spec: EmbeddingSpec):
        if not chunks:
            return []
        placeholders = ",".join("?" for _ in chunks)
        rows = (
            self._connect()
            .execute(
                f"SELECT chunk_id FROM rag_chunk_embeddings WHERE embedding_model=? AND embedding_revision=? AND chunk_id IN ({placeholders})",
                (spec.model, spec.revision, *(chunk.chunk_id for chunk in chunks)),
            )
            .fetchall()
        )
        present = {row[0] for row in rows}
        return [chunk for chunk in chunks if chunk.chunk_id not in present]

    def save_embeddings(self, embeddings: list[StoredEmbedding]):
        if embeddings:
            print(f"saving {len(embeddings)} embedding vectors to SQLite...", flush=True)
        self._connect().executemany(
            "INSERT OR REPLACE INTO rag_chunk_embeddings (chunk_id,embedding_model,embedding_revision,embedding_dimension,vector,created_at) VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
            [
                (
                    item.chunk_id,
                    item.spec.model,
                    item.spec.revision,
                    len(item.vector),
                    json.dumps(item.vector),
                )
                for item in embeddings
            ],
        )
        if embeddings:
            print(f"saved {len(embeddings)}/{len(embeddings)} embedding vectors to SQLite...", flush=True)

    def embeddings_for(self, chunks, spec: EmbeddingSpec):
        if not chunks:
            return []
        placeholders = ",".join("?" for _ in chunks)
        rows = (
            self._connect()
            .execute(
                f"SELECT chunk_id,embedding_dimension,vector FROM rag_chunk_embeddings WHERE embedding_model=? AND embedding_revision=? AND chunk_id IN ({placeholders})",
                (spec.model, spec.revision, *(chunk.chunk_id for chunk in chunks)),
            )
            .fetchall()
        )
        values = {row[0]: tuple(json.loads(row[2])) for row in rows}
        missing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in values]
        if missing:
            raise RuntimeError(f"missing compatible embeddings for {len(missing)} chunks")
        if spec.dimension and any(len(vector) != spec.dimension for vector in values.values()):
            raise ValueError(
                "stored embedding dimension does not match the selected embedding spec"
            )
        return [StoredEmbedding(chunk.chunk_id, spec, values[chunk.chunk_id]) for chunk in chunks]

    def create_conversation(self, owner, title):
        from rag.conversation import history_title

        i = uuid4().hex
        self._connect().execute(
            "INSERT INTO rag_conversations (conversation_id,owner_user_id,title,status,created_at,updated_at) VALUES (?,?,?,'active',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            (i, owner, history_title(title)),
        )
        return i

    def list_conversations(self, owner):
        return (
            self._connect()
            .execute(
                "SELECT conversation_id,title,updated_at FROM rag_conversations WHERE owner_user_id=? AND status='active' ORDER BY updated_at DESC",
                (owner,),
            )
            .fetchall()
        )

    def turns_for(self, owner, conversation_id):
        return (
            self._connect()
            .execute(
                "SELECT t.turn_id,t.turn_number,t.user_question,t.answer_text,t.supported,t.snapshot_id,t.citation_chunk_ids,t.created_at FROM rag_conversation_turns t JOIN rag_conversations c ON t.conversation_id=c.conversation_id WHERE c.owner_user_id=? AND c.conversation_id=? AND c.status='active' ORDER BY t.turn_number",
                (owner, conversation_id),
            )
            .fetchall()
        )

    def _owns(self, owner, i):
        return bool(
            self._connect()
            .execute(
                "SELECT 1 FROM rag_conversations WHERE conversation_id=? AND owner_user_id=? AND status='active'",
                (i, owner),
            )
            .fetchone()
        )

    def delete_conversation(self, owner, conversation_id):
        if not self._owns(owner, conversation_id):
            return False
        self._connect().execute(
            "UPDATE rag_conversations SET status='deleted',updated_at=CURRENT_TIMESTAMP WHERE conversation_id=? AND owner_user_id=?",
            (conversation_id, owner),
        )
        return True

    def append_turn(
        self, owner, conversation_id, *, question, resolved_query, answer, citation_ids, latency_ms
    ):
        if not self._owns(owner, conversation_id):
            raise PermissionError("conversation not found")
        c = self._connect()
        n = c.execute(
            "SELECT coalesce(max(turn_number),0)+1 FROM rag_conversation_turns WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()[0]
        i = uuid4().hex
        c.execute(
            "INSERT INTO rag_conversation_turns (turn_id,conversation_id,turn_number,user_question,resolved_query,answer_text,supported,provider,model,snapshot_id,citation_chunk_ids,created_at,latency_ms) VALUES (?,?,?,?,?,?,?,?,NULL,?,?,CURRENT_TIMESTAMP,?)",
            (
                i,
                conversation_id,
                n,
                question,
                resolved_query,
                answer.text,
                answer.supported,
                answer.provider,
                answer.snapshot_id,
                json.dumps(list(citation_ids)),
                latency_ms,
            ),
        )
        c.execute(
            "UPDATE rag_conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?",
            (conversation_id,),
        )
        return i

    def record_feedback(self, payload):
        self._connect().execute(
            "INSERT INTO rag_feedback (feedback_id,question,submitted_at,provider,model,snapshot_id,retrieved_chunk_ids,latency_ms,rating,comment) VALUES (?,?,CURRENT_TIMESTAMP,?,?,?,?,?,?,?)",
            (
                uuid4().hex,
                payload.get("question", ""),
                self.provider,
                self.model,
                payload.get("snapshot_id", "unknown"),
                json.dumps(list(payload.get("retrieved_chunk_ids", []))),
                payload.get("latency_ms"),
                payload["rating"],
                payload.get("comment") or None,
            ),
        )

    def record_request_trace(
        self,
        *,
        turn_id,
        conversation_id,
        owner,
        question,
        resolved_query,
        retrieval_trace,
        results,
        grounding_trace,
        answer,
        latency_ms,
        llm_usage=(),
    ):
        self._connect().execute(
            "INSERT INTO rag_request_traces (trace_id,turn_id,conversation_id,owner_user_id,user_question,resolved_query,retrieval_queries,retrieval_status,selected_evidence_json,raw_model_output,parsed_citation_labels,fallback_reason,final_answer_text,supported,provider,model,snapshot_id,latency_ms,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            (
                uuid4().hex,
                turn_id,
                conversation_id,
                owner,
                question,
                resolved_query,
                json.dumps(list(getattr(retrieval_trace, "queries", ()) or ())),
                getattr(retrieval_trace, "status", "unavailable"),
                json.dumps({}),
                getattr(grounding_trace, "raw_model_output", None),
                json.dumps(list(getattr(grounding_trace, "parsed_citation_labels", ()) or ())),
                getattr(grounding_trace, "fallback_reason", None),
                answer.text,
                answer.supported,
                self.provider,
                self.model,
                answer.snapshot_id,
                latency_ms,
            ),
        )
