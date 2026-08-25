"""Explicit manual refresh, local snapshot build, and benchmark helpers."""
from __future__ import annotations

from pathlib import Path

from rag.ingest.fetch import fetch_page
from rag.ingest.pipeline import IngestedDocument, ingest_source
from rag.ingest.sources import GENIE_LANDING_URL, CuratedDoc, discover_genie_core, load_curated_docs
from rag.index.chunking import chunk_document
from rag.index.service import PublishedSnapshot, build_and_activate
from rag.models import Chunk
from rag.store import DatabricksStore


def official_sources() -> list[CuratedDoc]:
    landing = fetch_page("genie-landing", GENIE_LANDING_URL)
    if landing.outcome != "ok" or not landing.html:
        raise RuntimeError(f"failed to discover Genie documentation: {landing.error_message}")
    sources = [*discover_genie_core(landing.html), *load_curated_docs(Path(__file__).parent / "ingest/config/curated_urls.yaml")]
    unique: dict[str, CuratedDoc] = {source.doc_id: source for source in sources}
    return list(unique.values())


def refresh_sources(store: DatabricksStore, *, document_table: str, chunk_table: str) -> list[Chunk]:
    """Fetch/extract/persist official docs, returning active chunks for a local build."""
    chunks: list[Chunk] = []
    for source in official_sources():
        try:
            item = ingest_source(source)
            store.upsert_document(document_table, item.document)
            if not item.extracted or not item.document.document_version:
                continue
            document_chunks = chunk_document(doc_id=item.document.doc_id, document_version=item.document.document_version,
                                            source_url=item.document.canonical_url, source_title=item.document.title or source.slug,
                                            nodes=item.extracted.nodes)
            store.replace_document_chunks(chunk_table, item.document, document_chunks)
            chunks.extend(document_chunks)
        except Exception:
            # A changed page structure is a per-document failure; it must not
            # cancel a refresh of every other official source.
            from rag.models import Document
            item = IngestedDocument(Document(source.doc_id, source.requested_url,
                source.canonical_requested_url, None, source.category, None, None, None, "fetch_failed"), None)
            store.upsert_document(document_table, item.document)
    return chunks


def build_snapshot(chunks: list[Chunk], embedder, local_root: str | Path) -> PublishedSnapshot:
    return build_and_activate(chunks, embedder, local_root)
