"""Small Databricks SQL/Volume adapter; Delta is the system of record."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

from rag.conversation import history_title
from rag.models import Chunk, Document


@dataclass(frozen=True)
class SqlResult:
    columns: list[str]
    rows: list[list]


def to_statement_parameters(values: dict[str, object]) -> list[StatementParameterListItem]:
    """Bind values as typed parameters instead of escaping them into the statement.

    Escaping is dialect-specific and silent when wrong: Databricks discards the
    SQL-standard '' doubling, which deleted every apostrophe in the corpus on
    2026-08-27. Parameters remove the escaping decision entirely.
    """
    parameters = []
    for name, value in values.items():
        if value is None:
            kind, rendered = "STRING", None
        elif isinstance(value, bool):
            kind, rendered = "BOOLEAN", "true" if value else "false"
        elif isinstance(value, int):
            kind, rendered = "BIGINT", str(value)
        elif isinstance(value, float):
            kind, rendered = "DOUBLE", str(value)
        else:
            kind, rendered = "STRING", str(value)
        parameters.append(StatementParameterListItem(name=name, type=kind, value=rendered))
    return parameters


class DatabricksStore:
    def __init__(self, warehouse_id: str, profile: str | None = None, namespace: str | None = None):
        self.workspace = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        self.warehouse_id = warehouse_id
        self.namespace = namespace

    def execute(
        self,
        statement: str,
        timeout_seconds: int = 300,
        *,
        parameters: dict[str, object] | None = None,
    ) -> SqlResult:
        result = self.workspace.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id,
            statement=statement,
            wait_timeout="30s",
            parameters=to_statement_parameters(parameters) if parameters else None,
        )
        for _ in range(timeout_seconds // 3):
            if result.status.state.name not in {"PENDING", "RUNNING"}:
                break
            time.sleep(3)
            result = self.workspace.statement_execution.get_statement(result.statement_id)
        if result.status.state.name != "SUCCEEDED":
            raise RuntimeError(
                getattr(result.status.error, "message", "Databricks SQL statement failed")
            )
        return SqlResult(
            [column.name for column in result.manifest.schema.columns],
            result.result.data_array or [],
        )

    def apply_schema(
        self, schema_file: str | Path, *, catalog: str, schema: str, artifact_volume: str
    ) -> None:
        sql = (
            Path(schema_file)
            .read_text(encoding="utf-8")
            .replace("${catalog}", catalog)
            .replace("${schema}", schema)
            .replace("${artifact_volume}", artifact_volume)
        )
        sql = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
        for statement in (item.strip() for item in sql.split(";") if item.strip()):
            self.execute(statement)
        self._add_missing_columns(
            f"{catalog}.{schema}.rag_documents",
            {
                "source_origins": "ARRAY<STRING>",
                "indexed_content_hash": "STRING",
                "indexed_source_last_updated": "STRING",
                "last_run_action": "STRING",
            },
        )
        self._add_missing_columns(
            f"{catalog}.{schema}.rag_index_snapshots", {"corpus_fingerprint": "STRING"}
        )

    def _add_missing_columns(self, table: str, columns: dict[str, str]) -> None:
        """Upgrade existing Delta tables without relying on unsupported IF NOT EXISTS DDL."""
        existing = {
            str(row[0]).lower() for row in self.execute(f"DESCRIBE {table}").rows if row and row[0]
        }
        missing = {
            name: data_type for name, data_type in columns.items() if name.lower() not in existing
        }
        if missing:
            definition = ", ".join(f"{name} {data_type}" for name, data_type in missing.items())
            self.execute(f"ALTER TABLE {table} ADD COLUMNS ({definition})")

    def create_conversation(self, owner: str, title: str) -> str:
        conversation_id = uuid4().hex
        self.execute(
            f"INSERT INTO {self.namespace}.rag_conversations "
            "(conversation_id,owner_user_id,title,status,created_at,updated_at) VALUES "
            "(:conversation_id,:owner,:title,'active',current_timestamp(),current_timestamp())",
            parameters={
                "conversation_id": conversation_id,
                "owner": owner,
                "title": history_title(title),
            },
        )
        return conversation_id

    def list_conversations(self, owner: str):
        return self.execute(
            f"SELECT conversation_id,title,updated_at FROM {self.namespace}.rag_conversations "
            "WHERE owner_user_id=:owner AND status='active' ORDER BY updated_at DESC",
            parameters={"owner": owner},
        ).rows

    def turns_for(self, owner: str, conversation_id: str):
        return self.execute(
            "SELECT t.turn_id,t.turn_number,t.user_question,t.answer_text,t.supported,"
            "t.snapshot_id,t.citation_chunk_ids,t.created_at "
            f"FROM {self.namespace}.rag_conversation_turns t "
            f"JOIN {self.namespace}.rag_conversations c ON t.conversation_id=c.conversation_id "
            "WHERE c.owner_user_id=:owner AND c.conversation_id=:conversation_id "
            "AND c.status='active' ORDER BY t.turn_number",
            parameters={"owner": owner, "conversation_id": conversation_id},
        ).rows

    def _owns_active_conversation(self, owner: str, conversation_id: str) -> bool:
        rows = self.execute(
            f"SELECT count(*) FROM {self.namespace}.rag_conversations "
            "WHERE conversation_id=:conversation_id AND owner_user_id=:owner AND status='active'",
            parameters={"conversation_id": conversation_id, "owner": owner},
        ).rows
        return int(rows[0][0]) == 1

    def delete_conversation(self, owner: str, conversation_id: str) -> bool:
        """Hide one owned conversation without deleting its audit records."""
        if not self._owns_active_conversation(owner, conversation_id):
            return False
        self.execute(
            f"UPDATE {self.namespace}.rag_conversations "
            "SET status='deleted',updated_at=current_timestamp() "
            "WHERE conversation_id=:conversation_id AND owner_user_id=:owner",
            parameters={"conversation_id": conversation_id, "owner": owner},
        )
        return True

    def append_turn(self, owner: str, conversation_id: str, *, question: str,
                    resolved_query: str, answer, citation_ids: list[str],
                    latency_ms: int) -> str:
        if not self._owns_active_conversation(owner, conversation_id):
            raise PermissionError("conversation not found")
        number = int(self.execute(
            f"SELECT coalesce(max(turn_number),0)+1 FROM {self.namespace}.rag_conversation_turns "
            "WHERE conversation_id=:conversation_id",
            parameters={"conversation_id": conversation_id},
        ).rows[0][0])
        turn_id = uuid4().hex
        self.execute(
            f"INSERT INTO {self.namespace}.rag_conversation_turns "
            "(turn_id,conversation_id,turn_number,user_question,resolved_query,answer_text,"
            "supported,provider,model,snapshot_id,citation_chunk_ids,created_at,latency_ms) "
            "VALUES (:turn_id,:conversation_id,:number,:question,:resolved_query,:answer_text,"
            ":supported,:provider,NULL,:snapshot_id,"
            "from_json(:citation_ids,'array<string>'),current_timestamp(),:latency_ms)",
            parameters={
                "turn_id": turn_id,
                "conversation_id": conversation_id,
                "number": number,
                "question": question,
                "resolved_query": resolved_query,
                "answer_text": answer.text,
                "supported": answer.supported,
                "provider": answer.provider,
                "snapshot_id": answer.snapshot_id,
                "citation_ids": json.dumps(list(citation_ids)),
                "latency_ms": latency_ms,
            },
        )
        self.execute(
            f"UPDATE {self.namespace}.rag_conversations SET updated_at=current_timestamp() "
            "WHERE conversation_id=:conversation_id",
            parameters={"conversation_id": conversation_id},
        )
        return turn_id

    def current_chunks(self) -> list[Chunk]:
        """Chunks belonging to the version of each document that is indexed."""
        rows = self.execute(
            "SELECT c.chunk_id,c.doc_id,c.document_version,c.position,c.chunk_text,"
            "c.heading_path,c.source_url,c.source_title "
            f"FROM {self.namespace}.rag_chunks c "
            f"JOIN {self.namespace}.rag_documents d ON c.doc_id=d.doc_id "
            "AND c.document_version=d.document_version "
            "WHERE d.status IN ('ok','pending_snapshot') "
            "ORDER BY c.doc_id,c.document_version,c.position"
        ).rows

        def heading_path(value) -> tuple[str, ...]:
            return tuple(json.loads(value) if isinstance(value, str) else value or ())

        chunks = [
            Chunk(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                str(row[4]),
                heading_path(row[5]),
                str(row[6]),
                str(row[7]),
            )
            for row in rows
        ]
        if not chunks:
            raise RuntimeError(
                f"no chunks found in {self.namespace}.rag_chunks; "
                "run the source refresh before building an index"
            )
        return chunks

    def activate_snapshot(self, metadata) -> None:
        """Deactivate the previous snapshot, then record the new one as active."""
        self.execute(
            f"UPDATE {self.namespace}.rag_index_snapshots SET active=FALSE WHERE active=TRUE"
        )
        chunk_map_path = metadata.artifact_path.rsplit("/", 1)[0] + "/chunk_map.json"
        self.execute(
            f"INSERT INTO {self.namespace}.rag_index_snapshots "
            "(snapshot_id,embedding_model,embedding_dimension,chunk_count,artifact_path,"
            "chunk_map_path,status,active,corpus_fingerprint,created_at) VALUES "
            "(:snapshot_id,:embedding_model,:embedding_dimension,:chunk_count,:artifact_path,"
            ":chunk_map_path,:status,TRUE,:corpus_fingerprint,current_timestamp())",
            parameters={
                "snapshot_id": metadata.snapshot_id,
                "embedding_model": metadata.embedding_model,
                "embedding_dimension": metadata.embedding_dimension,
                "chunk_count": metadata.chunk_count,
                "artifact_path": metadata.artifact_path,
                "chunk_map_path": chunk_map_path,
                "status": metadata.status,
                "corpus_fingerprint": metadata.corpus_fingerprint,
            },
        )

    def upload(self, local_path: str | Path, volume_path: str, *, overwrite: bool = False) -> None:
        with Path(local_path).open("rb") as handle:
            self.workspace.files.upload(volume_path, handle, overwrite=overwrite)

    def documents(self, table: str) -> dict[str, Document]:
        rows = self.execute(
            f"SELECT doc_id,requested_url,canonical_url,title,category,source_last_updated,"
            f"source_content_hash,document_version,status,source_origins,indexed_content_hash,"
            f"indexed_source_last_updated,consecutive_404_count,error_message FROM {table}"
        ).rows
        result = {}
        for row in rows:
            origins = row[9]
            if isinstance(origins, str):
                origins = json.loads(origins)
            result[str(row[0])] = Document(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                row[3],
                str(row[4]),
                row[5],
                row[6],
                row[7],
                str(row[8]),
                tuple(origins or ()),
                row[10],
                row[11],
                int(row[12] or 0),
                row[13],
            )
        return result

    def upsert_document(self, table: str, document: Document, *, action: str | None = None) -> None:
        values = {
            "doc_id": document.doc_id,
            "requested_url": document.requested_url,
            "canonical_url": document.canonical_url,
            "title": document.title,
            "category": document.category,
            "source_last_updated": document.source_last_updated,
            "source_content_hash": document.content_hash,
            "document_version": document.document_version,
            "status": document.status,
            "source_origins": "array("
            + ",".join(sql_literal(item) for item in document.source_origins)
            + ")",
            "indexed_content_hash": document.indexed_content_hash,
            "indexed_source_last_updated": document.indexed_source_last_updated,
            "consecutive_404_count": document.consecutive_404_count,
            "error_message": document.error_message,
            "last_run_action": action,
            "retrieved_at": "current_timestamp()",
        }
        raw = {"retrieved_at", "source_origins"}
        source = ", ".join(
            (value if name in raw else sql_literal(value)) + f" AS {name}"
            for name, value in values.items()
        )
        updates = ", ".join(f"target.{name} = source.{name}" for name in values if name != "doc_id")
        self.execute(
            f"MERGE INTO {table} target USING (SELECT {source}) source ON target.doc_id = source.doc_id "
            f"WHEN MATCHED THEN UPDATE SET {updates} WHEN NOT MATCHED THEN INSERT ({', '.join(values)}) VALUES ({', '.join('source.' + name for name in values)})"
        )

    def replace_document_chunks(
        self,
        table: str,
        document: Document,
        chunks: list[Chunk],
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        """Write a replacement version without changing manifest selection.

        The manifest switches the active version only after this method returns,
        so an insert failure cannot make a previously usable chunk set vanish.
        """
        self.execute(
            f"DELETE FROM {table} WHERE doc_id = {sql_literal(document.doc_id)} AND document_version = {sql_literal(document.document_version)}"
        )
        if not chunks:
            return
        rows = []
        for chunk in chunks:
            path = "array(" + ", ".join(sql_literal(item) for item in chunk.heading_path) + ")"
            rows.append(
                "("
                + ", ".join(
                    (
                        sql_literal(chunk.chunk_id),
                        sql_literal(chunk.doc_id),
                        sql_literal(chunk.document_version),
                        str(chunk.position),
                        sql_literal(chunk.text),
                        path,
                        sql_literal(chunk.source_url),
                        sql_literal(chunk.source_title),
                        sql_literal(embedding_model),
                        sql_literal(embedding_dimension),
                        "current_timestamp()",
                    )
                )
                + ")"
            )
        self.execute(
            f"INSERT INTO {table} (chunk_id, doc_id, document_version, position, chunk_text, heading_path, source_url, source_title, embedding_model, embedding_dimension, embedding_created_at) VALUES {', '.join(rows)}"
        )

    def prune_document_chunks(self, table: str, document: Document) -> None:
        """Remove superseded versions only after the manifest points at the new one."""
        self.execute(
            f"DELETE FROM {table} WHERE doc_id = {sql_literal(document.doc_id)} "
            f"AND document_version <> {sql_literal(document.document_version)}"
        )

    def clear_indexed_content_hashes(self, table: str) -> int:
        """Force the next refresh to re-chunk every document.

        A refresh rewrites chunks only when the fetched page hash differs from
        indexed_content_hash, so a document whose stored chunks are wrong for a
        reason unrelated to the source -- a write-side defect, a chunker change --
        is skipped forever while the page itself stays put. Clearing the recorded
        hash is what makes that repairable.
        """
        rows = self.execute(
            f"SELECT count(*) FROM {table} WHERE indexed_content_hash IS NOT NULL"
        ).rows
        affected = int(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else 0
        self.execute(f"UPDATE {table} SET indexed_content_hash = NULL")
        return affected

    def active_snapshot_fingerprint(self, table: str) -> str | None:
        rows = self.execute(
            f"SELECT corpus_fingerprint FROM {table} WHERE active=TRUE ORDER BY created_at DESC LIMIT 1"
        ).rows
        return str(rows[0][0]) if rows and rows[0][0] else None

    def mark_documents_materialized(self, table: str, *, chunk_table: str | None = None) -> None:
        self.execute(
            f"UPDATE {table} SET indexed_content_hash=source_content_hash, "
            f"indexed_source_last_updated=source_last_updated, status='ok', "
            f"last_run_action='published' WHERE status='pending_snapshot'"
        )
        if chunk_table:
            self.execute(
                f"DELETE FROM {chunk_table} WHERE doc_id IN "
                f"(SELECT doc_id FROM {table} WHERE status='removed')"
            )


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    # Databricks SQL escapes with backslashes, not SQL-standard quote doubling:
    # a doubled '' is silently dropped, deleting every apostrophe in the value.
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


class DatabricksFeedbackSink:
    """Stores only request/retrieval diagnostics and user feedback, never answer text."""

    def __init__(self, store: DatabricksStore, table: str, *, provider: str, model: str):
        self.store, self.table, self.provider, self.model = store, table, provider, model

    def __call__(self, payload: dict) -> None:
        columns = {
            "feedback_id": uuid4().hex,
            "question": payload.get("question", ""),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "model": self.model,
            "snapshot_id": payload.get("snapshot_id", "unknown"),
            "retrieved_chunk_ids": "array("
            + ",".join(sql_literal(value) for value in payload.get("retrieved_chunk_ids", []))
            + ")",
            "latency_ms": payload.get("latency_ms"),
            "rating": payload["rating"],
            "comment": payload.get("comment") or None,
        }
        values = ", ".join(
            value if key == "retrieved_chunk_ids" else sql_literal(value)
            for key, value in columns.items()
        )
        self.store.execute(f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({values})")


class DatabricksRequestTraceSink:
    """Persist final-answer diagnostics without storing the full model prompt."""

    def __init__(
        self,
        store: DatabricksStore,
        table: str,
        *,
        provider: str,
        model: str,
        retrieval_table: str | None = None,
        agent_provider: str | None = None,
        agent_model: str | None = None,
    ):
        self.store, self.table, self.provider, self.model = store, table, provider, model
        self.retrieval_table = retrieval_table
        self.agent_provider = agent_provider or provider
        self.agent_model = agent_model or model

    def record(
        self,
        *,
        turn_id: str | None,
        conversation_id: str | None,
        owner: str | None,
        question: str,
        resolved_query: str,
        retrieval_trace,
        results,
        grounding_trace,
        answer,
        latency_ms: int,
        llm_usage=(),
    ) -> None:
        evidence = [
            {
                "chunk_id": item.chunk.chunk_id,
                "score": item.score,
                "title": item.chunk.source_title,
                "heading_path": item.chunk.heading_path,
                "source_url": item.chunk.source_url,
            }
            for item in results
        ]
        queries = tuple(getattr(retrieval_trace, "queries", ()) or ())
        selected = tuple(getattr(retrieval_trace, "selected_chunk_ids", ()) or ())
        steps = [
            {
                "turn": step.turn,
                "action": step.action,
                "status": step.status,
                "query": step.query,
                "chunk_ids": step.chunk_ids,
                "candidate_ids": step.candidate_ids,
                "selected_chunk_ids": step.selected_chunk_ids,
                "candidate_cards": step.candidate_cards,
                "detail": step.detail,
            }
            for step in tuple(getattr(retrieval_trace, "steps", ()) or ())
        ]
        values = {
            "trace_id": uuid4().hex,
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "owner_user_id": owner,
            "user_question": question,
            "resolved_query": resolved_query,
            "retrieval_queries": "array(" + ",".join(sql_literal(value) for value in queries) + ")",
            "retrieval_status": getattr(retrieval_trace, "status", "unavailable"),
            "selected_evidence_json": json.dumps(
                {
                    "selected_chunk_ids": selected,
                    "evidence": evidence,
                    "tool_steps": steps,
                    "stop_reason": getattr(retrieval_trace, "stop_reason", None),
                    "agent_provider": self.agent_provider,
                    "agent_model": self.agent_model,
                    "evidence_support": getattr(retrieval_trace, "evidence_support", ()),
                    "unverified_points": getattr(retrieval_trace, "unverified_points", ()),
                    "llm_calls": [
                        asdict(call) if is_dataclass(call) else call for call in llm_usage
                    ],
                }
            ),
            "raw_model_output": grounding_trace.raw_model_output,
            "parsed_citation_labels": "array("
            + ",".join(sql_literal(value) for value in grounding_trace.parsed_citation_labels)
            + ")",
            "fallback_reason": grounding_trace.fallback_reason,
            "final_answer_text": answer.text,
            "supported": answer.supported,
            "provider": self.provider,
            "model": self.model,
            "snapshot_id": answer.snapshot_id,
            "latency_ms": latency_ms,
            "created_at": "current_timestamp()",
        }
        raw = {"retrieval_queries", "parsed_citation_labels", "created_at"}
        rendered = ", ".join(
            value if key in raw else sql_literal(value) for key, value in values.items()
        )
        trace_id = values["trace_id"]
        self.store.execute(f"INSERT INTO {self.table} ({', '.join(values)}) VALUES ({rendered})")
        if self.retrieval_table and turn_id:
            for number, step in enumerate(steps, 1):
                columns = {
                    "trace_id": trace_id,
                    "turn_id": turn_id,
                    "search_number": number,
                    "search_query": step["query"] or step["action"],
                    "retrieved_chunk_ids": "array("
                    + ",".join(sql_literal(value) for value in step["candidate_ids"])
                    + ")",
                    "selected_chunk_ids": "array("
                    + ",".join(sql_literal(value) for value in step["selected_chunk_ids"])
                    + ")",
                    "agent_decision": json.dumps(step),
                    "created_at": "current_timestamp()",
                    "latency_ms": latency_ms,
                }
                raw = {"retrieved_chunk_ids", "selected_chunk_ids", "created_at"}
                rendered = ", ".join(
                    value if key in raw else sql_literal(value) for key, value in columns.items()
                )
                self.store.execute(
                    f"INSERT INTO {self.retrieval_table} ({', '.join(columns)}) VALUES ({rendered})"
                )
