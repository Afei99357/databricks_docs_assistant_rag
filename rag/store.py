"""Small Databricks SQL/Volume adapter; Delta is the system of record."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from rag.models import Chunk, Document


@dataclass(frozen=True)
class SqlResult:
    columns: list[str]
    rows: list[list]


class DatabricksStore:
    def __init__(self, warehouse_id: str, profile: str | None = None):
        self.workspace = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        self.warehouse_id = warehouse_id

    def execute(self, statement: str, timeout_seconds: int = 300) -> SqlResult:
        result = self.workspace.statement_execution.execute_statement(warehouse_id=self.warehouse_id, statement=statement, wait_timeout="30s")
        for _ in range(timeout_seconds // 3):
            if result.status.state.name not in {"PENDING", "RUNNING"}:
                break
            time.sleep(3)
            result = self.workspace.statement_execution.get_statement(result.statement_id)
        if result.status.state.name != "SUCCEEDED":
            raise RuntimeError(getattr(result.status.error, "message", "Databricks SQL statement failed"))
        return SqlResult([column.name for column in result.manifest.schema.columns], result.result.data_array or [])

    def apply_schema(self, schema_file: str | Path, *, catalog: str, schema: str, artifact_volume: str) -> None:
        sql = Path(schema_file).read_text(encoding="utf-8").replace("${catalog}", catalog).replace("${schema}", schema).replace("${artifact_volume}", artifact_volume)
        sql = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
        for statement in (item.strip() for item in sql.split(";") if item.strip()):
            self.execute(statement)

    def upload(self, local_path: str | Path, volume_path: str) -> None:
        with Path(local_path).open("rb") as handle:
            self.workspace.files.upload(volume_path, handle, overwrite=False)

    def upsert_document(self, table: str, document: Document) -> None:
        values = {"doc_id": document.doc_id, "requested_url": document.requested_url,
                  "canonical_url": document.canonical_url, "title": document.title,
                  "category": document.category, "source_last_updated": document.source_last_updated,
                  "source_content_hash": document.content_hash, "document_version": document.document_version,
                  "status": document.status, "consecutive_404_count": 0,
                  "retrieved_at": "current_timestamp()"}
        source = ", ".join((value if name == "retrieved_at" else sql_literal(value)) + f" AS {name}" for name, value in values.items())
        updates = ", ".join(f"target.{name} = source.{name}" for name in values if name != "doc_id")
        self.execute(f"MERGE INTO {table} target USING (SELECT {source}) source ON target.doc_id = source.doc_id "
                     f"WHEN MATCHED THEN UPDATE SET {updates} WHEN NOT MATCHED THEN INSERT ({', '.join(values)}) VALUES ({', '.join('source.' + name for name in values)})")

    def replace_document_chunks(self, table: str, document: Document, chunks: list[Chunk], embedding_model: str | None = None, embedding_dimension: int | None = None) -> None:
        self.execute(f"DELETE FROM {table} WHERE doc_id = {sql_literal(document.doc_id)} AND document_version = {sql_literal(document.document_version)}")
        if not chunks:
            return
        rows = []
        for chunk in chunks:
            path = "array(" + ", ".join(sql_literal(item) for item in chunk.heading_path) + ")"
            rows.append("(" + ", ".join((sql_literal(chunk.chunk_id), sql_literal(chunk.doc_id), sql_literal(chunk.document_version), str(chunk.position), sql_literal(chunk.text), path, sql_literal(chunk.source_url), sql_literal(chunk.source_title), sql_literal(embedding_model), sql_literal(embedding_dimension), "current_timestamp()")) + ")")
        self.execute(f"INSERT INTO {table} (chunk_id, doc_id, document_version, position, chunk_text, heading_path, source_url, source_title, embedding_model, embedding_dimension, embedding_created_at) VALUES {', '.join(rows)}")


def sql_literal(value: object) -> str:
    if value is None: return "NULL"
    if isinstance(value, bool): return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)): return str(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"


class DatabricksFeedbackSink:
    """Stores only request/retrieval diagnostics and user feedback, never answer text."""
    def __init__(self, store: DatabricksStore, table: str, *, provider: str, model: str):
        self.store, self.table, self.provider, self.model = store, table, provider, model

    def __call__(self, payload: dict) -> None:
        columns = {
            "feedback_id": uuid4().hex, "question": payload.get("question", ""),
            "submitted_at": datetime.now(timezone.utc).isoformat(), "provider": self.provider,
            "model": self.model, "snapshot_id": payload.get("snapshot_id", "unknown"),
            "retrieved_chunk_ids": "array(" + ",".join(sql_literal(value) for value in payload.get("retrieved_chunk_ids", [])) + ")",
            "latency_ms": payload.get("latency_ms"), "rating": payload["rating"], "comment": payload.get("comment") or None,
        }
        values = ", ".join(value if key == "retrieved_chunk_ids" else sql_literal(value) for key, value in columns.items())
        self.store.execute(f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({values})")
