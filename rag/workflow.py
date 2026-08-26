"""Explicit manual refresh, local snapshot build, and benchmark helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rag.index.chunking import chunk_document
from rag.index.service import PublishedSnapshot, build_and_activate
from rag.ingest.fetch import fetch_page
from rag.ingest.pipeline import IngestedDocument, ingest_source
from rag.ingest.sources import (
    GENIE_LANDING_URL,
    CuratedDoc,
    discover_genie_core,
    load_curated_docs,
)
from rag.models import Chunk
from rag.store import DatabricksStore, sql_literal


def official_sources() -> list[CuratedDoc]:
    landing = fetch_page("genie-landing", GENIE_LANDING_URL)
    if landing.outcome != "ok" or not landing.html:
        raise RuntimeError(f"failed to discover Genie documentation: {landing.error_message}")
    config_dir = Path(__file__).parent / "ingest/config"
    sources = [
        *discover_genie_core(landing.html),
        *load_curated_docs(config_dir / "curated_urls.yaml"),
    ]
    unique: dict[str, CuratedDoc] = {source.doc_id: source for source in sources}
    return list(unique.values())


def refresh_sources(
    store: DatabricksStore, *, document_table: str, chunk_table: str
) -> list[Chunk]:
    """Fetch/extract/persist official docs, returning active chunks for a local build."""
    chunks: list[Chunk] = []
    sources = official_sources()
    print(f"Discovered {len(sources)} documentation pages to refresh.", flush=True)
    for number, source in enumerate(sources, start=1):
        print(f"[{number}/{len(sources)}] Fetching {source.requested_url}", flush=True)
        try:
            item = ingest_source(source)
            store.upsert_document(document_table, item.document)
            if not item.extracted or not item.document.document_version:
                continue
            document_chunks = chunk_document(
                doc_id=item.document.doc_id,
                document_version=item.document.document_version,
                source_url=item.document.canonical_url,
                source_title=item.document.title or source.slug,
                nodes=item.extracted.nodes,
            )
            store.replace_document_chunks(chunk_table, item.document, document_chunks)
            chunks.extend(document_chunks)
            print(
                f"[{number}/{len(sources)}] Indexed {len(document_chunks)} chunks "
                f"({len(chunks)} total).",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            # A changed page structure is a per-document failure; it must not
            # cancel a refresh of every other official source.
            from rag.models import Document

            item = IngestedDocument(
                Document(
                    source.doc_id,
                    source.requested_url,
                    source.canonical_requested_url,
                    None,
                    source.category,
                    None,
                    None,
                    None,
                    "fetch_failed",
                ),
                None,
            )
            store.upsert_document(document_table, item.document)
            print(f"[{number}/{len(sources)}] Failed: {exc}", flush=True)
    return chunks


def build_snapshot(chunks: list[Chunk], embedder, local_root: str | Path) -> PublishedSnapshot:
    return build_and_activate(chunks, embedder, local_root)


def load_current_chunks(store: DatabricksStore, chunk_table: str) -> list[Chunk]:
    """Load the governed chunk map that is the source of an index build."""
    rows = store.execute(
        f"SELECT chunk_id,doc_id,document_version,position,chunk_text,heading_path,source_url,source_title "
        f"FROM {chunk_table} ORDER BY doc_id,document_version,position"
    ).rows

    def heading_path(value) -> tuple[str, ...]:
        # Statement Execution can return arrays as native lists or serialized
        # JSON strings, depending on the SDK/result format.
        if isinstance(value, str):
            value = json.loads(value)
        return tuple(value or ())

    chunks = [
        Chunk(
            chunk_id=str(row[0]),
            doc_id=str(row[1]),
            document_version=str(row[2]),
            position=int(row[3]),
            text=str(row[4]),
            heading_path=heading_path(row[5]),
            source_url=str(row[6]),
            source_title=str(row[7]),
        )
        for row in rows
    ]
    if not chunks:
        raise RuntimeError(
            f"no chunks found in {chunk_table}; run the source refresh before building an index"
        )
    return chunks


def publish_volume_snapshot(
    store: DatabricksStore, *, namespace: str, volume_path: str, chunks: list[Chunk], embedder
) -> PublishedSnapshot:
    """Build locally, upload immutable artifacts, then atomically select the new snapshot.

    ``active_snapshot.json`` is uploaded last. Thus a failure leaves the App
    reading the previous completed snapshot.
    """
    with tempfile.TemporaryDirectory(prefix="rag-app-snapshot-") as temporary:
        published = build_and_activate(chunks, embedder, temporary)
        snapshot_id = published.metadata.snapshot_id
        remote_dir = f"{volume_path.rstrip('/')}/snapshots/{snapshot_id}"
        store.upload(published.local_directory / "index.faiss", f"{remote_dir}/index.faiss")
        store.upload(published.local_directory / "chunk_map.json", f"{remote_dir}/chunk_map.json")
        # Both immutable files now exist. Only this final overwrite changes
        # what an App process will load on its next request.
        store.upload(
            Path(temporary) / "active_snapshot.json",
            f"{volume_path.rstrip('/')}/active_snapshot.json",
            overwrite=True,
        )
        table = f"{namespace}.rag_index_snapshots"
        store.execute(f"UPDATE {table} SET active=FALSE WHERE active=TRUE")
        metadata = published.metadata
        store.execute(
            f"INSERT INTO {table} (snapshot_id,embedding_model,embedding_dimension,chunk_count,artifact_path,chunk_map_path,created_at,status,active) VALUES "
            f"({sql_literal(metadata.snapshot_id)},{sql_literal(metadata.embedding_model)},{metadata.embedding_dimension},{metadata.chunk_count},"
            f"{sql_literal(remote_dir + '/index.faiss')},{sql_literal(remote_dir + '/chunk_map.json')},current_timestamp(),'active',TRUE)"
        )
        return published
