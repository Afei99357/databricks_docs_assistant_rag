from pathlib import Path

import pytest

from rag.ingest.sources import (
    DiscoveryRoot,
    canonicalize_url,
    compute_doc_id,
    discover_genie_core,
    discover_root,
    load_curated_docs,
    load_discovery_roots,
    load_sitemap_urls,
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
    assert len(docs) == 12
    assert any(
        doc.canonical_requested_url == "https://docs.databricks.com/aws/en/volumes/content-search"
        for doc in docs
    )


def test_discovery_roots_are_bounded_and_configured():
    path = Path(__file__).parents[1] / "rag/ingest/config/discovery_roots.yaml"
    roots = load_discovery_roots(path)
    assert {root.root_id for root in roots} == {
        "genie",
        "databricks-apps",
        "unity-catalog",
        "ai-gateway",
        "omnigent",
        "security",
        "admin",
        "agents",
        "machine-learning",
        "ai-bi",
        "dashboards",
        "get-started",
        "architecture",
    }
    assert all(root.max_pages == 250 for root in roots)


def test_sitemap_loader_keeps_only_canonical_databricks_documentation_urls():
    class Result:
        outcome, error_message = "ok", None
        html = """<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://docs.databricks.com/aws/en/ai-bi/admin/themes/</loc></url>
          <url><loc>https://docs.databricks.com/aws/en/ai-bi/admin/themes/</loc></url>
          <url><loc>https://docs.databricks.com/gcp/en/ai-bi/admin/themes</loc></url>
          <url><loc>https://example.com/not-a-doc</loc></url>
        </urlset>"""

    urls = load_sitemap_urls(lambda _doc_id, _url: Result())

    assert urls == [
        "https://docs.databricks.com/aws/en/ai-bi/admin/themes",
        "https://docs.databricks.com/gcp/en/ai-bi/admin/themes",
    ]


def test_sitemap_loader_fails_closed_when_the_sitemap_cannot_be_fetched():
    class Result:
        outcome, html, error_message = "network_error", None, "connection reset"

    with pytest.raises(RuntimeError, match="official sitemap fetch failed"):
        load_sitemap_urls(lambda _doc_id, _url: Result())


def test_sitemap_loader_fails_closed_when_the_sitemap_is_malformed():
    class Result:
        outcome, error_message = "ok", None
        html = "<urlset><url><loc>https://docs.databricks.com/aws/en/ai-bi</loc></urlset>"

    with pytest.raises(RuntimeError, match="official sitemap is not valid XML"):
        load_sitemap_urls(lambda _doc_id, _url: Result())


def test_sitemap_discovery_stays_in_its_allowed_path_family():
    root = DiscoveryRoot(
        "test", "https://docs.databricks.com/aws/en/test", ("/aws/en/test",), "genie-concepts", 3
    )
    sitemap_urls = [
        "https://docs.databricks.com/aws/en/test",
        "https://docs.databricks.com/aws/en/test/one",
        "https://docs.databricks.com/aws/en/other",
    ]
    progress = []

    docs = discover_root(root, sitemap_urls, on_progress=progress.append)

    assert [doc.canonical_requested_url for doc in docs] == sitemap_urls[:2]
    assert progress == [1, 2]
    assert {doc.source_scope for doc in docs} == {"sitemap:test"}


def test_sitemap_discovery_covers_ai_bi_admin_and_dashboard_families():
    sitemap_urls = [
        "https://docs.databricks.com/aws/en/ai-bi/admin/themes",
        "https://docs.databricks.com/aws/en/ai-bi/admin/use-apis",
        "https://docs.databricks.com/aws/en/dashboards/",
    ]
    ai_bi = DiscoveryRoot(
        "ai-bi", "https://docs.databricks.com/aws/en/ai-bi/", ("/aws/en/ai-bi/",), "ai-bi", 250
    )
    dashboards = DiscoveryRoot(
        "dashboards", "https://docs.databricks.com/aws/en/dashboards/", ("/aws/en/dashboards/",), "ai-bi", 250
    )

    assert [doc.canonical_requested_url for doc in discover_root(ai_bi, sitemap_urls)] == sitemap_urls[:2]
    assert [doc.canonical_requested_url for doc in discover_root(dashboards, sitemap_urls)] == [
        "https://docs.databricks.com/aws/en/dashboards"
    ]


def test_every_configured_root_has_a_selectable_sitemap_scope():
    path = Path(__file__).parents[1] / "rag/ingest/config/discovery_roots.yaml"
    roots = load_discovery_roots(path)

    for root in roots:
        sitemap_url = f"https://docs.databricks.com{root.allowed_path_prefixes[0].rstrip('/')}/coverage"
        assert discover_root(root, [sitemap_url])


def test_sitemap_discovery_fails_at_the_configured_page_cap():
    root = DiscoveryRoot(
        "test", "https://docs.databricks.com/aws/en/test", ("/aws/en/test",), "genie-concepts", 1
    )

    with pytest.raises(RuntimeError, match="1-page limit"):
        discover_root(
            root,
            [
                "https://docs.databricks.com/aws/en/test",
                "https://docs.databricks.com/aws/en/test/one",
            ],
        )
