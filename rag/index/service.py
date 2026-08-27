"""Snapshot publication workflow with validation before activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag.index.embeddings import EmbeddingProvider
from rag.index.faiss_store import FaissSnapshot, write_active_manifest
from rag.models import Chunk, IndexSnapshot


@dataclass(frozen=True)
class PublishedSnapshot:
    metadata: IndexSnapshot
    local_directory: Path


def build_and_activate(
    chunks: list[Chunk],
    embedder: EmbeddingProvider,
    root: str | Path,
    *,
    corpus_fingerprint: str | None = None,
) -> PublishedSnapshot:
    root, snapshot_id = Path(root), uuid4().hex
    staging = root / "staging" / snapshot_id
    final = root / "snapshots" / snapshot_id
    snapshot = FaissSnapshot.build(chunks, embedder, snapshot_id)
    snapshot.validate(embedder.dimension or None)
    snapshot.save(staging)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final)
    write_active_manifest(root, snapshot_id, corpus_fingerprint=corpus_fingerprint)
    metadata = IndexSnapshot(
        snapshot_id,
        embedder.model_name,
        snapshot.index.d,
        len(chunks),
        str(final),
        "active",
        datetime.now(timezone.utc),
        corpus_fingerprint,
    )
    return PublishedSnapshot(metadata, final)
