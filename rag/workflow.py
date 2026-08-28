"""Source refresh coordination and immutable snapshot publication."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from rag.index.chunking import chunk_document
from rag.index.service import PublishedSnapshot, build_and_activate
from rag.ingest.fetch import fetch_page
from rag.ingest.lifecycle import REMOVAL_THRESHOLD
from rag.ingest.pipeline import ingest_source
from rag.ingest.sources import CuratedDoc, discover_root, load_curated_docs, load_discovery_roots
from rag.models import Chunk, Document
from rag.store import DatabricksStore


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


def refresh_sources(store: DatabricksStore) -> RefreshResult:
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

        observed = item.document
        date_changed = bool(
            prior
            and observed.source_last_updated
            and observed.source_last_updated != prior.indexed_source_last_updated
        )
        content_changed = prior is None or observed.content_hash != prior.indexed_content_hash
        needs_materialization = content_changed or date_changed
        status = "pending_snapshot" if needs_materialization else prior.status
        document = replace(
            observed,
            status=status,
            source_origins=_origins(source),
            indexed_content_hash=prior.indexed_content_hash if prior else None,
            indexed_source_last_updated=prior.indexed_source_last_updated if prior else None,
            consecutive_404_count=0,
            error_message=None,
        )
        if needs_materialization:
            if content_changed:
                chunks = chunk_document(
                    doc_id=document.doc_id,
                    document_version=document.document_version or "",
                    source_url=document.canonical_url,
                    source_title=document.title or source.slug,
                    nodes=item.extracted.nodes,
                )
                store.replace_document_chunks(document, chunks)
            store.upsert_document(
                document,
                action="date_changed" if date_changed and not content_changed else "changed",
            )
            if content_changed:
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


def publish_volume_snapshot(
    store: DatabricksStore,
    *,
    volume_path: str,
    chunks: list[Chunk],
    embedder,
    corpus_fingerprint: str | None = None,
    materialize: bool = False,
) -> PublishedSnapshot:
    """Build/upload immutable artifacts, then atomically select the new snapshot."""
    with tempfile.TemporaryDirectory(prefix="rag-app-snapshot-") as temporary:
        published = build_and_activate(
            chunks, embedder, temporary, corpus_fingerprint=corpus_fingerprint
        )
        snapshot_id = published.metadata.snapshot_id
        remote_dir = f"{volume_path.rstrip('/')}/snapshots/{snapshot_id}"
        store.upload(published.local_directory / "index.faiss", f"{remote_dir}/index.faiss")
        store.upload(published.local_directory / "chunk_map.json", f"{remote_dir}/chunk_map.json")
        store.upload(
            Path(temporary) / "active_snapshot.json",
            f"{volume_path.rstrip('/')}/active_snapshot.json",
            overwrite=True,
        )
        metadata = replace(published.metadata, artifact_path=f"{remote_dir}/index.faiss")
        store.activate_snapshot(metadata)
        if materialize:
            store.mark_documents_materialized()
        return published
