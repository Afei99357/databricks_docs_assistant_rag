"""Protocols shared code depends on instead of the concrete Databricks adapter.

Each protocol is derived from the actual methods on ``rag.storage.databricks.DatabricksStore``
as they exist today. They intentionally say nothing about how a conforming
store is implemented -- a SQLite-backed store (or any other backend) only has
to expose the same methods with the same signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.models import Chunk, Document, IndexSnapshot


@runtime_checkable
class CorpusStore(Protocol):
    """Documents, their chunks, and the active-snapshot pointer."""

    def apply_schema(
        self, schema_file: str | Path, *, catalog: str, schema: str, artifact_volume: str
    ) -> None: ...

    def documents(self) -> dict[str, Document]: ...

    def upsert_document(self, document: Document, *, action: str | None = None) -> None: ...

    def replace_document_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None: ...

    def prune_document_chunks(self, document: Document) -> None: ...

    def mark_documents_materialized(self) -> None: ...

    def clear_indexed_content_hashes(self) -> int: ...

    def current_chunks(self) -> list[Chunk]: ...

    def active_snapshot_fingerprint(self) -> str | None: ...

    def activate_snapshot(self, metadata: IndexSnapshot) -> None: ...


@runtime_checkable
class ConversationStore(Protocol):
    """Conversation history: threads and their turns."""

    def create_conversation(self, owner: str, title: str) -> str: ...

    def list_conversations(self, owner: str): ...

    def turns_for(self, owner: str, conversation_id: str): ...

    def delete_conversation(self, owner: str, conversation_id: str) -> bool: ...

    def append_turn(
        self,
        owner: str,
        conversation_id: str,
        *,
        question: str,
        resolved_query: str,
        answer,
        citation_ids: list[str],
        latency_ms: int,
    ) -> str: ...


@runtime_checkable
class DiagnosticsStore(Protocol):
    """Request traces and user feedback."""

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
    ) -> None: ...

    def record_feedback(self, payload: dict) -> None: ...


@runtime_checkable
class ArtifactPublisher(Protocol):
    """Publishes a built snapshot's local artifacts somewhere retrievers can read them.

    No implementation exists yet -- the Volume and local-filesystem publishers
    are added in a later task.
    """

    def publish(self, local_directory: Path, snapshot_id: str) -> str: ...
