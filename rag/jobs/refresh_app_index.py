"""Initialize the governed store, refresh sources, and publish an App snapshot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag.config import Settings
from rag.index.embeddings import DatabricksEmbeddingProvider
from rag.store import DatabricksStore
from rag.workflow import publish_volume_snapshot, refresh_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--artifact-volume", default="rag_artifacts")
    parser.add_argument("--embedding-endpoint", default="databricks-qwen3-embedding-0-6b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(
        RAG_CATALOG=args.catalog,
        RAG_SCHEMA=args.schema,
        RAG_WAREHOUSE_ID=args.warehouse_id,
        RAG_ARTIFACT_VOLUME=args.artifact_volume,
        RAG_DATABRICKS_EMBEDDING_ENDPOINT=args.embedding_endpoint,
    )
    settings = Settings.from_env()
    store = DatabricksStore(settings.warehouse_id)
    store.apply_schema(
        Path(__file__).parents[2] / "sql/001_rag_schema.sql",
        catalog=settings.catalog,
        schema=settings.schema,
        artifact_volume=settings.artifact_volume,
    )
    print("refreshing configured documentation sources...", flush=True)
    chunks = refresh_sources(
        store,
        document_table=f"{settings.namespace}.rag_documents",
        chunk_table=f"{settings.namespace}.rag_chunks",
    )
    if not chunks:
        raise RuntimeError("source refresh produced no chunks")
    print(f"publishing App snapshot from {len(chunks)} chunks...", flush=True)
    embedder = DatabricksEmbeddingProvider(args.embedding_endpoint)
    published = publish_volume_snapshot(
        store,
        namespace=settings.namespace,
        volume_path=f"{settings.volume_path}/app-qwen3-embedding-0-6b",
        chunks=chunks,
        embedder=embedder,
    )
    print(
        f"published App snapshot {published.metadata.snapshot_id} "
        f"({published.metadata.chunk_count} chunks)",
        flush=True,
    )


if __name__ == "__main__":
    main()
