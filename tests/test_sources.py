from pathlib import Path

from rag.ingest.fetch import FetchResult
from rag.ingest.sources import (
    CrawlSource,
    canonicalize_url,
    compute_doc_id,
    discover_genie_core,
    discover_site_sections,
    load_crawl_sources,
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


def test_crawl_policy_yaml_has_no_configured_sources():
    path = Path(__file__).parents[1] / "rag/ingest/config/crawl_sources.yaml"
    sources = load_crawl_sources(path)
    assert sources == []


def test_bounded_site_crawler_discovers_allowed_new_pages_and_excludes_other_sections():
    source = CrawlSource(
        name="test",
        root_url="https://example.com/docs/",
        category="genie-concepts",
        reason="test",
        allowed_prefixes=("trust/",),
        excluded_prefixes=("private/",),
        max_depth=2,
    )
    pages = {
        "https://example.com/docs": '<a href="trust/">Trust</a><a href="other/">Other</a>',
        "https://example.com/docs/trust": '<a href="trust/new/">New</a><a href="trust/private/">Private</a>',
        "https://example.com/docs/trust/new": "<p>New content</p>",
    }

    def fetcher(doc_id, url):
        return FetchResult(doc_id, url, url, 200, pages[url], None, "ok")

    docs = discover_site_sections([source], fetcher=fetcher)
    assert {doc.requested_url for doc in docs} == {
        "https://example.com/docs",
        "https://example.com/docs/trust",
        "https://example.com/docs/trust/new",
    }
