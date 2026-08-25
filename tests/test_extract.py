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


def test_extracts_semantic_article_for_supplemental_documentation_site():
    html = '''<main><nav>Ignore navigation</nav><article class="md-content__inner md-typeset">
      <h1>Running the Assessment</h1><p>Run the readiness assessment in your workspace.</p>
      <h2>How you get it</h2><ol><li>Request access.</li><li>Install the listing.</li></ol>
    </article></main>'''
    doc = extract_content(html)
    assert doc.title == "Running the Assessment"
    assert "Run the readiness assessment" in doc.markdown_body
    assert "Request access." in doc.markdown_body
    assert "Ignore navigation" not in doc.markdown_body
