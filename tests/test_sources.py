from pathlib import Path

from rag.ingest.sources import (
    canonicalize_url,
    compute_doc_id,
    discover_genie_core,
    load_curated_docs,
)


def test_canonicalize_url_preserves_identity_rules():
    assert (
        canonicalize_url("HTTPS://Docs.Databricks.COM/aws/en/genie/?utm_source=x&keep=1#section")
        == "https://docs.databricks.com/aws/en/genie?keep=1"
    )


def test_document_id_is_stable():
    assert compute_doc_id("https://docs.databricks.com/aws/en/genie") == compute_doc_id(
        "https://docs.databricks.com/aws/en/genie"
    )


def test_discovery_filters_to_official_genie_families():
    docs = discover_genie_core(
        '<a href="/aws/en/genie-one/">One</a><a href="https://example.com/aws/en/genie">No</a>'
    )
    assert [doc.canonical_requested_url for doc in docs] == [
        "https://docs.databricks.com/aws/en/genie-one"
    ]


def test_curated_sources_include_content_search():
    path = Path(__file__).parents[1] / "rag/ingest/config/curated_urls.yaml"
    docs = load_curated_docs(path)
    assert len(docs) == 29
    assert any(
        doc.canonical_requested_url == "https://docs.databricks.com/aws/en/volumes/content-search"
        for doc in docs
    )
