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
    assert len(docs) == 16
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
        "get-started",
        "architecture",
    }
    assert all(root.max_pages == 250 for root in roots)


def test_recursive_discovery_stays_in_its_allowed_path_family():
    root = DiscoveryRoot(
        "test", "https://docs.databricks.com/aws/en/test", ("/aws/en/test",), "genie-concepts", 3
    )

    class Result:
        outcome, error_message = "ok", None

        def __init__(self, html):
            self.html = html

    pages = {
        "https://docs.databricks.com/aws/en/test": '<a href="/aws/en/test/one">One</a><a href="/aws/en/other">No</a>',
        "https://docs.databricks.com/aws/en/test/one": '<a href="/aws/en/test/two">Two</a>',
        "https://docs.databricks.com/aws/en/test/two": "<article>Done</article>",
    }
    docs = discover_root(root, lambda _doc_id, url: Result(pages[url]))
    assert [doc.canonical_requested_url for doc in docs] == list(pages)


def test_recursive_discovery_skips_a_linked_not_found_page():
    root = DiscoveryRoot(
        "test", "https://docs.databricks.com/aws/en/test", ("/aws/en/test",), "genie-concepts", 3
    )

    class Result:
        def __init__(self, outcome, html=None):
            self.outcome, self.html = outcome, html
            self.error_message = "HTTP 404" if outcome == "not_found" else None

    pages = {
        "https://docs.databricks.com/aws/en/test": Result(
            "ok", '<a href="/aws/en/test/missing">Missing</a><a href="/aws/en/test/live">Live</a>'
        ),
        "https://docs.databricks.com/aws/en/test/missing": Result("not_found"),
        "https://docs.databricks.com/aws/en/test/live": Result("ok", "<article>Live</article>"),
    }

    docs = discover_root(root, lambda _doc_id, url: pages[url])
    assert [doc.canonical_requested_url for doc in docs] == [
        "https://docs.databricks.com/aws/en/test",
        "https://docs.databricks.com/aws/en/test/live",
    ]


def test_recursive_discovery_fails_at_the_configured_page_cap():
    root = DiscoveryRoot(
        "test", "https://docs.databricks.com/aws/en/test", ("/aws/en/test",), "genie-concepts", 1
    )

    class Result:
        outcome, error_message = "ok", None
        html = '<a href="/aws/en/test/one">One</a>'

    with pytest.raises(RuntimeError, match="1-page limit"):
        discover_root(root, lambda _doc_id, _url: Result())
