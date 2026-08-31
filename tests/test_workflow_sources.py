from rag.ingest.sources import CuratedDoc, DiscoveryRoot, compute_doc_id
from rag.workflow import official_sources


def test_official_sources_downloads_the_sitemap_once_then_selects_each_root(monkeypatch):
    roots = [
        DiscoveryRoot("first", "https://docs.databricks.com/aws/en/first/", ("/aws/en/first/",), "first", 250),
        DiscoveryRoot("second", "https://docs.databricks.com/aws/en/second/", ("/aws/en/second/",), "second", 250),
    ]
    sitemap_urls = [
        "https://docs.databricks.com/aws/en/first/page",
        "https://docs.databricks.com/aws/en/second/page",
    ]
    sitemap_calls = []

    monkeypatch.setattr("rag.workflow.load_discovery_roots", lambda _path: roots)
    monkeypatch.setattr(
        "rag.workflow.load_sitemap_urls",
        lambda fetch: sitemap_calls.append(fetch) or sitemap_urls,
    )
    monkeypatch.setattr("rag.workflow.load_curated_docs", lambda _path: [])

    sources = official_sources()

    assert len(sitemap_calls) == 1
    assert [source.canonical_requested_url for source in sources] == sitemap_urls
    assert [source.source_scope for source in sources] == ["sitemap:first", "sitemap:second"]


def test_official_sources_preserves_a_manual_source_origin_alongside_sitemap(monkeypatch):
    root = DiscoveryRoot(
        "ai-bi", "https://docs.databricks.com/aws/en/ai-bi/", ("/aws/en/ai-bi/",), "ai-bi", 250
    )
    url = "https://docs.databricks.com/aws/en/ai-bi/admin/themes"
    manual = CuratedDoc(
        url, url, compute_doc_id(url), "themes", "ai-bi", "aws", "manual", "curated:ai-bi"
    )

    monkeypatch.setattr("rag.workflow.load_discovery_roots", lambda _path: [root])
    monkeypatch.setattr("rag.workflow.load_sitemap_urls", lambda _fetch: [url])
    monkeypatch.setattr("rag.workflow.load_curated_docs", lambda _path: [manual])

    [source] = official_sources()

    assert source.source_scope == "curated:ai-bi|sitemap:ai-bi"
