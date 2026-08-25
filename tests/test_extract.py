from rag.ingest.extract import extract_content


def test_extracts_only_article_content_and_produces_stable_hash():
    html = '''<nav>Ignore me</nav><article><div class="theme-doc-markdown markdown">
      <h1>A \u200btitle</h1><h2>Setup</h2><p>Use <code>Genie</code> : now.</p>
      <ul><li>first</li><li>second</li></ul><span class="theme-last-updated"><time datetime="2026-08-24" /></span>
    </div></article>'''
    doc = extract_content(html)
    assert doc.title == "A title"
    assert doc.source_last_updated == "2026-08-24"
    assert "Use Genie: now." in doc.markdown_body
    assert "Ignore me" not in doc.markdown_body
    assert len(doc.source_content_hash) == 64

