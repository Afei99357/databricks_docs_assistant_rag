"""Initialize the governed store, refresh sources, and publish an App snapshot."""

from __future__ import annotations

import argparse
import os

from rag.config import Settings
from rag.index.embeddings import DatabricksEmbeddingProvider
from rag.index.runtime import app_snapshot_root
from rag.store import DatabricksStore
from rag.workflow import publish_volume_snapshot, refresh_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--artifact-volume", default="rag_artifacts")
    parser.add_argument("--embedding-endpoint", default="databricks-qwen3-embedding-0-6b")
    parser.add_argument("--embedding-batch-size", type=int, default=10)
    parser.add_argument("--embedding-min-interval-seconds", type=float, default=2.0)
    parser.add_argument("--schema-sql-path", required=True)
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
        args.schema_sql_path,
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
    embedder = DatabricksEmbeddingProvider(
        args.embedding_endpoint,
        batch_size=args.embedding_batch_size,
        min_interval_seconds=args.embedding_min_interval_seconds,
    )
    published = publish_volume_snapshot(
        store,
        namespace=settings.namespace,
        volume_path=app_snapshot_root(settings.volume_path),
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
