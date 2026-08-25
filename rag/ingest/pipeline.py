"""Small, inspectable source-only ingestion orchestration.

Embedding is deliberately separate: first inspect extracted source documents,
then chunk/index only accepted document versions.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.ingest.extract import ExtractedDoc, extract_content
from rag.ingest.fetch import fetch_page
from rag.ingest.sources import CuratedDoc
from rag.models import Document


@dataclass(frozen=True)
class IngestedDocument:
    document: Document
    extracted: ExtractedDoc | None


def ingest_source(source: CuratedDoc) -> IngestedDocument:
    """Fetch and extract a source; caller decides how/when to persist it."""
    fetched = fetch_page(source.doc_id, source.requested_url)
    if fetched.outcome != "ok" or not fetched.html or not fetched.resolved_url:
        return IngestedDocument(Document(source.doc_id, source.requested_url, source.canonical_requested_url,
                                        None, source.category, None, None, None,
                                        "removed" if fetched.outcome == "not_found" else "fetch_failed"), None)
    extracted = extract_content(fetched.html)
    return IngestedDocument(Document(source.doc_id, source.requested_url, source.canonical_requested_url,
                                    extracted.title, source.category, extracted.source_last_updated,
                                    extracted.source_content_hash, extracted.source_content_hash, "ok"), extracted)
