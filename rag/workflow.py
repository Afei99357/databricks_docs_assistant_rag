"""Source refresh coordination and immutable snapshot publication."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from rag.index.chunking import chunk_document
from rag.index.service import (
    PublishedSnapshot,
    build_and_activate,
    build_from_embeddings_and_activate,
)
from rag.ingest.fetch import fetch_page
from rag.ingest.lifecycle import REMOVAL_THRESHOLD
from rag.ingest.pipeline import ingest_source
from rag.ingest.sources import CuratedDoc, discover_root, load_curated_docs, load_discovery_roots
from rag.models import Chunk, Document, EmbeddingSpec, StoredEmbedding
from rag.storage.protocol import ArtifactPublisher, CorpusStore, EmbeddingStore


@dataclass(frozen=True)
class RefreshSummary:
    unchanged: int = 0
    changed: int = 0
    date_triggered: int = 0
    failed: int = 0
    removed: int = 0
    pending_snapshot: int = 0
    discovered: int = 0

    @property
    def needs_snapshot(self) -> bool:
        return bool(self.changed or self.date_triggered or self.removed or self.pending_snapshot)


@dataclass(frozen=True)
class RefreshResult:
    summary: RefreshSummary
    corpus_fingerprint: str


def official_sources() -> list[CuratedDoc]:
    """Load recursive roots and manual supplements, preserving every origin."""
    config_dir = Path(__file__).parent / "ingest/config"
    sources: dict[str, CuratedDoc] = {}
    for root in load_discovery_roots(config_dir / "discovery_roots.yaml"):
        for source in discover_root(root, fetch_page):
            sources[source.doc_id] = source
    for source in load_curated_docs(config_dir / "curated_urls.yaml"):
        existing = sources.get(source.doc_id)
        if existing:
            source = replace(
                source,
                source_scope="|".join(
                    sorted({*existing.source_scope.split("|"), source.source_scope})
                ),
            )
        sources[source.doc_id] = source
    return list(sources.values())


def _origins(source: CuratedDoc) -> tuple[str, ...]:
    return tuple(sorted(part for part in source.source_scope.split("|") if part))


def _fingerprint(documents: dict[str, Document]) -> str:
    active = [
        (doc.doc_id, doc.document_version, doc.source_last_updated)
        for doc in documents.values()
        if doc.status in {"ok", "pending_snapshot"} and doc.document_version
    ]
    payload = json.dumps(sorted(active), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _failure(previous: Document | None, source: CuratedDoc, outcome: str) -> Document:
    if previous is None:
        return Document(
            source.doc_id,
            source.requested_url,
            source.canonical_requested_url,
            None,
            source.category,
            None,
            None,
            None,
            "fetch_failed",
            _origins(source),
            None,
            None,
            1 if outcome == "not_found" else 0,
            outcome,
        )
    if outcome != "not_found":
        return replace(previous, error_message=outcome, source_origins=_origins(source))
    count = previous.consecutive_404_count + 1
    if count >= REMOVAL_THRESHOLD:
        return replace(
            previous,
            status="removed",
            consecutive_404_count=count,
            error_message="confirmed 404",
            source_origins=_origins(source),
        )
    return replace(
        previous,
        consecutive_404_count=count,
        error_message="404 awaiting confirmation",
        source_origins=_origins(source),
    )


def refresh_sources(store: CorpusStore, *, chunking_revision: str = "v1") -> RefreshResult:
    """Refresh sources incrementally; snapshot publication is a separate step."""
    previous = store.documents()
    sources = official_sources()
    current_ids = {source.doc_id for source in sources}
    counts = RefreshSummary(discovered=len(sources))
    print(f"Discovered {len(sources)} documentation pages to refresh.", flush=True)

    for number, source in enumerate(sources, start=1):
        print(f"[{number}/{len(sources)}] Fetching {source.requested_url}", flush=True)
        prior = previous.get(source.doc_id)
        try:
            item = ingest_source(source)
        except Exception as exc:  # noqa: BLE001 - per-source errors must not unpublish the corpus
            failure = (
                replace(prior, error_message=str(exc))
                if prior
                else _failure(None, source, "network_error")
            )
            store.upsert_document(failure, action="failed")
            previous[source.doc_id] = failure
            counts = replace(counts, failed=counts.failed + 1)
            continue
        if item.outcome != "ok" or item.extracted is None:
            failed = _failure(prior, source, item.outcome)
            store.upsert_document(
                failed, action="removed" if failed.status == "removed" else "failed"
            )
            previous[source.doc_id] = failed
            counts = replace(
                counts,
                removed=counts.removed + (failed.status == "removed"),
                failed=counts.failed + (failed.status != "removed"),
            )
            continue

        observed = replace(
            item.document,
            document_version=(
                f"{item.document.document_version}:{chunking_revision}"
                if item.document.document_version
                else None
            ),
        )
        date_changed = bool(
            prior
            and observed.source_last_updated
            and observed.source_last_updated != prior.indexed_source_last_updated
        )
        content_changed = prior is None or observed.content_hash != prior.indexed_content_hash
        chunking_changed = bool(prior and observed.document_version != prior.document_version)
        needs_materialization = content_changed or date_changed or chunking_changed
        status = "pending_snapshot" if needs_materialization else prior.status
        document = replace(
            observed,
            status=status,
            source_origins=_origins(source),
            indexed_content_hash=prior.indexed_content_hash if prior else None,
            indexed_source_last_updated=prior.indexed_source_last_updated if prior else None,
            consecutive_404_count=0,
            error_message=None,
            chunked_content_hash=prior.chunked_content_hash if prior else None,
            chunked_source_last_updated=prior.chunked_source_last_updated if prior else None,
            chunked_document_version=prior.chunked_document_version if prior else None,
        )
        if needs_materialization:
            chunks_already_current = (
                document.chunked_content_hash == document.content_hash
                and document.chunked_source_last_updated == document.source_last_updated
                and document.chunked_document_version == document.document_version
            )
            if (content_changed or chunking_changed) and not chunks_already_current:
                chunks = chunk_document(
                    doc_id=document.doc_id,
                    document_version=document.document_version or "",
                    source_url=document.canonical_url,
                    source_title=document.title or source.slug,
                    nodes=item.extracted.nodes,
                )
                store.replace_document_chunks(document, chunks)
                document = replace(
                    document,
                    chunked_content_hash=document.content_hash,
                    chunked_source_last_updated=document.source_last_updated,
                    chunked_document_version=document.document_version,
                )
            store.upsert_document(
                document,
                action="date_changed"
                if date_changed and not content_changed and not chunking_changed
                else "changed",
            )
            if content_changed or chunking_changed:
                store.prune_document_chunks(document)
            previous[document.doc_id] = document
            counts = replace(
                counts,
                changed=counts.changed + content_changed,
                date_triggered=counts.date_triggered + (date_changed and not content_changed),
            )
        else:
            store.upsert_document(document, action="unchanged")
            previous[document.doc_id] = document
            counts = replace(counts, unchanged=counts.unchanged + 1)

    # A page removed from roots/YAML follows the same three-run safeguard as a 404.
    for doc_id, document in list(previous.items()):
        if doc_id in current_ids or document.status == "removed":
            continue
        count = document.consecutive_404_count + 1
        missing = replace(
            document, consecutive_404_count=count, error_message="not present in configured sources"
        )
        if count >= REMOVAL_THRESHOLD:
            missing = replace(
                missing, status="removed", error_message="confirmed configuration removal"
            )
            counts = replace(counts, removed=counts.removed + 1)
        else:
            counts = replace(counts, failed=counts.failed + 1)
        store.upsert_document(
            missing, action="removed" if missing.status == "removed" else "missing"
        )
        previous[doc_id] = missing

    pending = sum(doc.status == "pending_snapshot" for doc in previous.values())
    return RefreshResult(replace(counts, pending_snapshot=pending), _fingerprint(previous))


def build_snapshot(
    chunks: list[Chunk], embedder, local_root: str | Path, *, corpus_fingerprint: str | None = None
) -> PublishedSnapshot:
    return build_and_activate(chunks, embedder, local_root, corpus_fingerprint=corpus_fingerprint)


def build_snapshot_from_vectors(
    chunks: list[Chunk],
    vectors,
    embedder,
    local_root: str | Path,
    *,
    corpus_fingerprint: str | None = None,
    embedding_revision: str = "v1",
    chunking_revision: str = "v1",
) -> PublishedSnapshot:
    return build_from_embeddings_and_activate(
        chunks,
        vectors,
        embedder,
        local_root,
        corpus_fingerprint=corpus_fingerprint,
        embedding_revision=embedding_revision,
        chunking_revision=chunking_revision,
    )


def publish_snapshot(
    store: CorpusStore | EmbeddingStore,
    *,
    publisher: ArtifactPublisher,
    chunks: list[Chunk],
    embedder,
    corpus_fingerprint: str | None = None,
    materialize: bool = False,
    embedding_revision: str = "v1",
    chunking_revision: str = "v1",
    embed_missing: bool = True,
) -> PublishedSnapshot:
    """Build immutable artifacts locally, publish them, then atomically select the new snapshot."""
    with tempfile.TemporaryDirectory(prefix="rag-app-snapshot-") as temporary:
        spec = EmbeddingSpec(embedder.model_name, embedding_revision)
        missing = store.missing_embeddings(chunks, spec)
        if missing:
            if not embed_missing:
                raise RuntimeError(
                    f"{len(missing)} embeddings are missing; run the resume snapshot operation first"
                )
            # Commit each provider-sized batch immediately.  A killed job can
            # therefore resume from the durable embedding cache instead of
            # losing vectors produced earlier in the call.
            batch_size = max(1, int(getattr(embedder, "batch_size", len(missing))))
            total_batches = (len(missing) + batch_size - 1) // batch_size
            for start in range(0, len(missing), batch_size):
                batch_number = start // batch_size + 1
                batch_chunks = missing[start : start + batch_size]
                print(
                    f"Caching embedding batch {batch_number}/{total_batches} "
                    f"({len(batch_chunks)} chunks)...",
                    flush=True,
                )
                vectors = embedder.embed([chunk.text for chunk in batch_chunks])
                if len(vectors) != len(batch_chunks):
                    raise RuntimeError("embedding provider returned an unexpected number of vectors")
                store.save_embeddings(
                    [
                        StoredEmbedding(
                            chunk.chunk_id,
                            EmbeddingSpec(spec.model, spec.revision, len(vector)),
                            tuple(vector),
                        )
                        for chunk, vector in zip(batch_chunks, vectors)
                    ]
                )
                interval = float(getattr(embedder, "min_interval_seconds", 0.0))
                if interval and batch_number < total_batches:
                    time.sleep(interval)
        print("loading persisted embedding vectors...", flush=True)
        vectors = [item.vector for item in store.embeddings_for(chunks, spec)]
        print("building FAISS snapshot...", flush=True)
        published = build_from_embeddings_and_activate(
            chunks,
            vectors,
            embedder,
            temporary,
            corpus_fingerprint=corpus_fingerprint,
            embedding_revision=embedding_revision,
            chunking_revision=chunking_revision,
        )
        print("uploading snapshot artifacts...", flush=True)
        published_location = publisher.publish(
            published.local_directory, published.metadata.snapshot_id
        )
        metadata = replace(published.metadata, artifact_path=f"{published_location}/index.faiss")
        print("activating snapshot...", flush=True)
        store.activate_snapshot(metadata)
        if materialize:
            store.mark_documents_materialized()
        return published
