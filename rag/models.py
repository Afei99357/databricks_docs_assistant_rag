"""Stable, provider-neutral data contracts for the RAG system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Document:
    doc_id: str
    requested_url: str
    canonical_url: str
    title: str | None
    category: str
    source_last_updated: str | None
    content_hash: str | None
    document_version: str | None
    status: str
    source_origins: tuple[str, ...] = ()
    indexed_content_hash: str | None = None
    indexed_source_last_updated: str | None = None
    consecutive_404_count: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    document_version: str
    position: int
    text: str
    heading_path: tuple[str, ...]
    source_url: str
    source_title: str


@dataclass(frozen=True)
class IndexSnapshot:
    snapshot_id: str
    embedding_model: str
    embedding_dimension: int
    chunk_count: int
    artifact_path: str
    status: str
    created_at: datetime
    corpus_fingerprint: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    snapshot_id: str


@dataclass(frozen=True)
class Citation:
    label: str
    title: str
    url: str
    excerpt: str
    chunk_id: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    supported: bool
    provider: str
    snapshot_id: str
