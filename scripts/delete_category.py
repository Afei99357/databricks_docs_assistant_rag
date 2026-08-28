"""One-off maintenance: delete all documents/chunks for a given source category.

Usage:
    uv run python scripts/delete_category.py <category>

Removes rows from rag_chunks first (foreign-key-shaped: chunks reference doc_id),
then rag_documents, for every document tagged with the given category. Intended
for cleaning up a source that was dropped from the crawl/curated config so its
previously-ingested content doesn't linger in the governed tables or get picked
back up by a snapshot rebuild.
"""

from __future__ import annotations

import sys

from rag.config import Settings
from rag.storage.databricks import DatabricksStore, sql_literal


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <category>")
    category = sql_literal(sys.argv[1])

    settings = Settings.from_env()
    store = DatabricksStore(settings.warehouse_id, settings.databricks_profile)
    doc_table = f"{settings.namespace}.rag_documents"
    chunk_table = f"{settings.namespace}.rag_chunks"

    before_docs = store.execute(
        f"SELECT count(*) FROM {doc_table} WHERE category = {category}"
    ).rows[0][0]
    before_chunks = store.execute(
        f"SELECT count(*) FROM {chunk_table} c JOIN {doc_table} d ON c.doc_id = d.doc_id "
        f"WHERE d.category = {category}"
    ).rows[0][0]
    print(f"category {category}: {before_docs} documents, {before_chunks} chunks to delete")

    store.execute(
        f"DELETE FROM {chunk_table} WHERE doc_id IN "
        f"(SELECT doc_id FROM {doc_table} WHERE category = {category})"
    )
    print("deleted chunks")
    store.execute(f"DELETE FROM {doc_table} WHERE category = {category}")
    print("deleted documents")

    total_docs = store.execute(f"SELECT count(*) FROM {doc_table}").rows[0][0]
    total_chunks = store.execute(f"SELECT count(*) FROM {chunk_table}").rows[0][0]
    print(f"remaining totals: {total_docs} documents, {total_chunks} chunks")


if __name__ == "__main__":
    main()
