from unittest.mock import patch

from rag.ingest.fetch import FetchResult
from rag.ingest.pipeline import ingest_source
from rag.ingest.sources import CuratedDoc


def test_ingestion_produces_document_version_from_extracted_hash():
    source = CuratedDoc("https://docs.databricks.com/x", "https://docs.databricks.com/x", "d", "x", "genie-concepts", "aws", "test")
    html = '<article><div class="theme-doc-markdown"><h1>Title</h1><p>Text</p></div></article>'
    with patch("rag.ingest.pipeline.fetch_page", return_value=FetchResult("d", source.requested_url, source.requested_url, 200, html, None, "ok")):
        result = ingest_source(source)
    assert result.document.status == "ok"
    assert result.document.document_version == result.document.content_hash

