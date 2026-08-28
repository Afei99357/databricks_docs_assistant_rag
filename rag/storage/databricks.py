"""Small Databricks SQL/Volume adapter; Delta is the system of record."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
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


# Each chunk row binds 11 parameters. The warehouse accepted 512 parameters in a
# single statement when measured directly; the largest document in the corpus has
# 109 chunks (1,199 parameters), which exceeds that. 40 rows/insert (~440 params)
# stays under the ceiling while keeping the average 12.6-chunk document to one statement.
CHUNK_INSERT_BATCH = 40


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
    def __init__(
        self,
        warehouse_id: str,
        profile: str | None = None,
        namespace: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        agent_provider: str | None = None,
        agent_model: str | None = None,
    ):
        self.workspace = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        self.warehouse_id = warehouse_id
        self.namespace = namespace
        # Diagnostics-only identity: which chat/agent model produced a recorded
        # answer. Not used by any corpus/conversation method. agent_provider and
        # agent_model default to the answer provider/model because most callers
        # run one model for both retrieval and generation.
        self.provider = provider
        self.model = model
        self.agent_provider = agent_provider or provider
        self.agent_model = agent_model or model

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
            "from_json(:citation_ids,'array<string>'),current_timestamp(),CAST(:latency_ms AS BIGINT))",
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

    def documents(self) -> dict[str, Document]:
        rows = self.execute(
            f"SELECT doc_id,requested_url,canonical_url,title,category,source_last_updated,"
            f"source_content_hash,document_version,status,source_origins,indexed_content_hash,"
            f"indexed_source_last_updated,consecutive_404_count,error_message "
            f"FROM {self.namespace}.rag_documents"
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

    def upsert_document(self, document: Document, *, action: str | None = None) -> None:
        # `values` supplies both the bound parameters and the MERGE source row/update
        # list below, so the three cannot drift out of sync with each other.
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
            "source_origins": json.dumps(list(document.source_origins)),
            "indexed_content_hash": document.indexed_content_hash,
            "indexed_source_last_updated": document.indexed_source_last_updated,
            "consecutive_404_count": document.consecutive_404_count,
            "error_message": document.error_message,
            "last_run_action": action,
        }
        # Columns whose source-row expression is not a plain bound parameter.
        raw = {
            "source_origins": "from_json(:source_origins,'array<string>')",
            "retrieved_at": "current_timestamp()",
        }
        columns = [*values, "retrieved_at"]
        source = ", ".join((raw.get(name) or f":{name}") + f" AS {name}" for name in columns)
        updates = ", ".join(f"target.{name} = source.{name}" for name in columns if name != "doc_id")
        self.execute(
            f"MERGE INTO {self.namespace}.rag_documents target USING (SELECT {source}) source "
            f"ON target.doc_id = source.doc_id "
            f"WHEN MATCHED THEN UPDATE SET {updates} WHEN NOT MATCHED THEN "
            f"INSERT ({', '.join(columns)}) VALUES ({', '.join('source.' + name for name in columns)})",
            parameters=values,
        )

    def replace_document_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        """Write a replacement version without changing manifest selection.

        The manifest switches the active version only after this method returns,
        so an insert failure cannot make a previously usable chunk set vanish.

        Inserts are batched (CHUNK_INSERT_BATCH rows/statement) because a single
        multi-row INSERT binds 11 parameters per row -- the largest document in
        the corpus would need 1,199 parameters in one statement, above the
        measured 512-parameter ceiling.
        """
        self.execute(
            f"DELETE FROM {self.namespace}.rag_chunks WHERE doc_id=:doc_id AND document_version=:version",
            parameters={"doc_id": document.doc_id, "version": document.document_version},
        )
        for start in range(0, len(chunks), CHUNK_INSERT_BATCH):
            batch = chunks[start : start + CHUNK_INSERT_BATCH]
            rows, parameters = [], {}
            for offset, chunk in enumerate(batch):
                n = f"r{offset}"
                rows.append(
                    f"(:{n}_id,:{n}_doc,:{n}_ver,:{n}_pos,:{n}_text,"
                    f"from_json(:{n}_path,'array<string>'),:{n}_url,:{n}_title,"
                    f":{n}_model,CAST(:{n}_dim AS INT),current_timestamp())"
                )
                parameters |= {
                    f"{n}_id": chunk.chunk_id,
                    f"{n}_doc": chunk.doc_id,
                    f"{n}_ver": chunk.document_version,
                    f"{n}_pos": chunk.position,
                    f"{n}_text": chunk.text,
                    f"{n}_path": json.dumps(list(chunk.heading_path)),
                    f"{n}_url": chunk.source_url,
                    f"{n}_title": chunk.source_title,
                    f"{n}_model": embedding_model,
                    f"{n}_dim": embedding_dimension,
                }
            self.execute(
                f"INSERT INTO {self.namespace}.rag_chunks (chunk_id, doc_id, document_version, position, chunk_text, "
                "heading_path, source_url, source_title, embedding_model, embedding_dimension, "
                "embedding_created_at) VALUES " + ", ".join(rows),
                parameters=parameters,
            )

    def prune_document_chunks(self, document: Document) -> None:
        """Remove superseded versions only after the manifest points at the new one."""
        self.execute(
            f"DELETE FROM {self.namespace}.rag_chunks WHERE doc_id=:doc_id AND document_version<>:version",
            parameters={"doc_id": document.doc_id, "version": document.document_version},
        )

    def clear_indexed_content_hashes(self) -> int:
        """Force the next refresh to re-chunk every document.

        A refresh rewrites chunks only when the fetched page hash differs from
        indexed_content_hash, so a document whose stored chunks are wrong for a
        reason unrelated to the source -- a write-side defect, a chunker change --
        is skipped forever while the page itself stays put. Clearing the recorded
        hash is what makes that repairable.
        """
        table = f"{self.namespace}.rag_documents"
        rows = self.execute(
            f"SELECT count(*) FROM {table} WHERE indexed_content_hash IS NOT NULL"
        ).rows
        affected = int(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else 0
        self.execute(f"UPDATE {table} SET indexed_content_hash = NULL")
        return affected

    def active_snapshot_fingerprint(self) -> str | None:
        rows = self.execute(
            f"SELECT corpus_fingerprint FROM {self.namespace}.rag_index_snapshots "
            "WHERE active=TRUE ORDER BY created_at DESC LIMIT 1"
        ).rows
        return str(rows[0][0]) if rows and rows[0][0] else None

    def mark_documents_materialized(self) -> None:
        table = f"{self.namespace}.rag_documents"
        self.execute(
            f"UPDATE {table} SET indexed_content_hash=source_content_hash, "
            f"indexed_source_last_updated=source_last_updated, status='ok', "
            f"last_run_action='published' WHERE status='pending_snapshot'"
        )
        self.execute(
            f"DELETE FROM {self.namespace}.rag_chunks WHERE doc_id IN "
            f"(SELECT doc_id FROM {table} WHERE status='removed')"
        )

    def record_request_trace(
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
        """Persist final-answer diagnostics without storing the full model prompt."""
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
        trace_id = uuid4().hex
        self.execute(
            f"INSERT INTO {self.namespace}.rag_request_traces "
            "(trace_id,turn_id,conversation_id,owner_user_id,user_question,resolved_query,"
            "retrieval_queries,retrieval_status,selected_evidence_json,raw_model_output,"
            "parsed_citation_labels,fallback_reason,final_answer_text,supported,provider,"
            "model,snapshot_id,latency_ms,created_at) VALUES "
            "(:trace_id,:turn_id,:conversation_id,:owner_user_id,:user_question,:resolved_query,"
            "from_json(:retrieval_queries,'array<string>'),:retrieval_status,"
            ":selected_evidence_json,:raw_model_output,"
            "from_json(:parsed_citation_labels,'array<string>'),:fallback_reason,"
            ":final_answer_text,:supported,:provider,:model,:snapshot_id,CAST(:latency_ms AS BIGINT),"
            "current_timestamp())",
            parameters={
                "trace_id": trace_id,
                "turn_id": turn_id,
                "conversation_id": conversation_id,
                "owner_user_id": owner,
                "user_question": question,
                "resolved_query": resolved_query,
                "retrieval_queries": json.dumps(list(queries)),
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
                "raw_model_output": getattr(grounding_trace, "raw_model_output", None),
                "parsed_citation_labels": json.dumps(
                    list(getattr(grounding_trace, "parsed_citation_labels", ()) or ())
                ),
                "fallback_reason": getattr(grounding_trace, "fallback_reason", None),
                "final_answer_text": answer.text,
                "supported": answer.supported,
                "provider": self.provider,
                "model": self.model,
                "snapshot_id": answer.snapshot_id,
                "latency_ms": latency_ms,
            },
        )
        if turn_id:
            for number, step in enumerate(steps, 1):
                self.execute(
                    f"INSERT INTO {self.namespace}.rag_retrieval_traces "
                    "(trace_id,turn_id,search_number,search_query,retrieved_chunk_ids,"
                    "selected_chunk_ids,agent_decision,created_at,latency_ms) VALUES "
                    "(:trace_id,:turn_id,:search_number,:search_query,"
                    "from_json(:retrieved_chunk_ids,'array<string>'),"
                    "from_json(:selected_chunk_ids,'array<string>'),:agent_decision,"
                    "current_timestamp(),CAST(:latency_ms AS BIGINT))",
                    parameters={
                        "trace_id": trace_id,
                        "turn_id": turn_id,
                        "search_number": number,
                        "search_query": step["query"] or step["action"],
                        "retrieved_chunk_ids": json.dumps(list(step["candidate_ids"])),
                        "selected_chunk_ids": json.dumps(list(step["selected_chunk_ids"])),
                        "agent_decision": json.dumps(step),
                        "latency_ms": latency_ms,
                    },
                )

    def record_feedback(self, payload: dict) -> None:
        """Stores only request/retrieval diagnostics and user feedback, never answer text."""
        self.execute(
            f"INSERT INTO {self.namespace}.rag_feedback "
            "(feedback_id,question,submitted_at,provider,model,snapshot_id,"
            "retrieved_chunk_ids,latency_ms,rating,comment) VALUES "
            "(:feedback_id,:question,current_timestamp(),:provider,:model,:snapshot_id,"
            "from_json(:retrieved_chunk_ids,'array<string>'),CAST(:latency_ms AS BIGINT),:rating,:comment)",
            parameters={
                "feedback_id": uuid4().hex,
                "question": payload.get("question", ""),
                "provider": self.provider,
                "model": self.model,
                "snapshot_id": payload.get("snapshot_id", "unknown"),
                "retrieved_chunk_ids": json.dumps(list(payload.get("retrieved_chunk_ids", []))),
                "latency_ms": payload.get("latency_ms"),
                "rating": payload["rating"],
                "comment": payload.get("comment") or None,
            },
        )


class VolumePublisher:
    """Publishes snapshot artifacts to a Unity Catalog Volume.

    Alongside ``index.faiss`` and ``chunk_map.json`` under the per-snapshot
    directory, this also republishes the top-level ``active_snapshot.json``
    manifest (overwriting the previous one). That manifest is how the deployed
    App's Volume-backed runtime (``rag.index.runtime``) discovers the active
    snapshot by polling a file instead of querying the database -- it has no
    other way to learn a new snapshot is live, so dropping this upload would
    silently break that sync path even though the database row is correct.
    """

    def __init__(self, store: DatabricksStore, volume_path: str):
        self.store, self.volume_path = store, volume_path

    def publish(self, local_directory: Path, snapshot_id: str) -> str:
        remote_dir = f"{self.volume_path.rstrip('/')}/snapshots/{snapshot_id}"
        self.store.upload(local_directory / "index.faiss", f"{remote_dir}/index.faiss")
        self.store.upload(local_directory / "chunk_map.json", f"{remote_dir}/chunk_map.json")
        manifest = local_directory.parent.parent / "active_snapshot.json"
        self.store.upload(
            manifest, f"{self.volume_path.rstrip('/')}/active_snapshot.json", overwrite=True
        )
        return remote_dir

