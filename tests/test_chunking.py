from rag.index.chunking import ChunkingConfig, chunk_document


def test_chunking_preserves_heading_context_and_stable_ids():
    nodes = [{"type": "heading", "level": 1, "text": "Genie"}, {"type": "heading", "level": 2, "text": "Permissions"}, {"type": "paragraph", "text": "Use Unity Catalog permissions."}]
    one = chunk_document(doc_id="doc", document_version="v1", source_url="https://docs.databricks.com/x", source_title="Title", nodes=nodes, config=ChunkingConfig(max_chars=100))
    two = chunk_document(doc_id="doc", document_version="v1", source_url="https://docs.databricks.com/x", source_title="Title", nodes=nodes, config=ChunkingConfig(max_chars=100))
    assert one[0].chunk_id == two[0].chunk_id
    assert one[0].heading_path == ("Genie", "Permissions")
    assert "Unity Catalog" in one[0].text


def test_chunking_keeps_paragraphs_and_tables_together_before_fallback_split():
    table = "column | value"
    nodes = [{"type": "heading", "level": 1, "text": "Reference"},
             {"type": "paragraph", "text": "First complete paragraph."},
             {"type": "paragraph", "text": "Second complete paragraph."},
             {"type": "table", "rows": [["column", "value"], ["one", "two"]]}]
    chunks = chunk_document(doc_id="d", document_version="v", source_url="https://docs.databricks.com/x", source_title="X", nodes=nodes, config=ChunkingConfig(max_chars=55, overlap_chars=0))
    assert any("First complete paragraph." in item.text for item in chunks)
    assert any(table in item.text and "one | two" in item.text for item in chunks)
