"""Explicit operator commands; refreshes are intentionally manual in v1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag.agent.retrieval import RetrievalAgent
from rag.config import Settings
from rag.history import ConversationRepository
from rag.identity import LocalTestIdentityProvider
from rag.index.embeddings import OllamaEmbeddingProvider
from rag.index.runtime import ActiveSnapshotRetriever
from rag.ingest.fetch import fetch_page
from rag.ingest.sources import (
    GENIE_LANDING_URL,
    discover_genie_core,
    discover_site_sections,
    load_crawl_sources,
    load_curated_docs,
)
from rag.llm.providers import DatabricksEndpointProvider, OllamaProvider
from rag.store import DatabricksFeedbackSink, DatabricksStore
from rag.workflow import (
    build_snapshot,
    load_current_chunks,
    publish_volume_snapshot,
    refresh_sources,
)


def discover() -> None:
    landing = fetch_page("genie-landing", GENIE_LANDING_URL)
    if landing.outcome != "ok" or not landing.html:
        raise RuntimeError(f"landing-page discovery failed: {landing.error_message}")
    docs = discover_genie_core(landing.html)
    config_dir = Path(__file__).parent / "ingest/config"
    supplemental = load_curated_docs(config_dir / "curated_urls.yaml")
    crawled = discover_site_sections(load_crawl_sources(config_dir / "crawl_sources.yaml"))
    print(
        f"discovered {len(docs)} Genie-core pages, loaded {len(supplemental)} curated pages, and discovered {len(crawled)} crawl pages"
    )
    for document in [*docs, *supplemental, *crawled]:
        print(f"{document.doc_id}\t{document.canonical_requested_url}")


def setup_db() -> None:
    settings = Settings.from_env()
    DatabricksStore(settings.warehouse_id, settings.databricks_profile).apply_schema(
        Path(__file__).parents[1] / "sql/001_rag_schema.sql",
        catalog=settings.catalog,
        schema=settings.schema,
        artifact_volume=settings.artifact_volume,
    )
    print(f"initialized {settings.namespace}")


def serve() -> None:
    settings = Settings.from_env()
    root = os.getenv("RAG_LOCAL_INDEX_DIR")
    if not root:
        raise ValueError("RAG_LOCAL_INDEX_DIR must point to downloaded local snapshot artifacts")
    embedder = OllamaEmbeddingProvider(settings.embedding_model, base_url=settings.ollama_base_url)
    retriever = ActiveSnapshotRetriever(root, embedder, settings.top_k)
    provider = (
        OllamaProvider(settings.ollama_base_url, settings.ollama_model)
        if settings.answer_provider == "ollama"
        else DatabricksEndpointProvider(
            settings.databricks_chat_endpoint, profile=settings.databricks_profile
        )
    )
    from rag.app.web import create_app

    store = DatabricksStore(settings.warehouse_id, settings.databricks_profile)
    feedback = DatabricksFeedbackSink(
        store, f"{settings.namespace}.rag_feedback", provider=provider.name, model=provider.model
    )
    history = ConversationRepository(store, settings.namespace)
    identity = LocalTestIdentityProvider()
    agent = RetrievalAgent(retriever.retrieve, provider)
    create_app(
        retrieve=agent.retrieve,
        provider=provider,
        threshold=settings.relevance_threshold,
        feedback_sink=feedback,
        history=history,
        identity=identity,
    ).run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)


def build_app_snapshot() -> None:
    """Build the App-compatible FAISS snapshot with Databricks Qwen embeddings."""
    settings = Settings.from_env()
    from rag.index.embeddings import DatabricksEmbeddingProvider

    store = DatabricksStore(settings.warehouse_id, settings.databricks_profile)
    embedder = DatabricksEmbeddingProvider(
        os.getenv("RAG_DATABRICKS_EMBEDDING_ENDPOINT", "databricks-qwen3-embedding-0-6b"),
        profile=settings.databricks_profile,
    )
    chunks = load_current_chunks(store, f"{settings.namespace}.rag_chunks")
    app_index_root = (
        os.getenv("RAG_APP_INDEX_ROOT") or f"{settings.volume_path}/app-qwen3-embedding-0-6b"
    )
    published = publish_volume_snapshot(
        store,
        namespace=settings.namespace,
        volume_path=app_index_root,
        chunks=chunks,
        embedder=embedder,
    )
    print(
        f"published App snapshot {published.metadata.snapshot_id} ({published.metadata.chunk_count} chunks)"
    )


def build_local_snapshot() -> None:
    """Refresh governed chunks and build a local Ollama-backed FAISS snapshot."""
    settings = Settings.from_env()
    root = os.getenv("RAG_LOCAL_INDEX_DIR")
    if not root:
        raise ValueError("RAG_LOCAL_INDEX_DIR must point to the local snapshot directory")
    store = DatabricksStore(settings.warehouse_id, settings.databricks_profile)
    print("refreshing configured sources and chunks...", flush=True)
    chunks = refresh_sources(
        store,
        document_table=f"{settings.namespace}.rag_documents",
        chunk_table=f"{settings.namespace}.rag_chunks",
    )
    if not chunks:
        raise RuntimeError("source refresh produced no chunks")
    print(f"building local FAISS snapshot from {len(chunks)} chunks...", flush=True)
    embedder = OllamaEmbeddingProvider(settings.embedding_model, base_url=settings.ollama_base_url)
    published = build_snapshot(chunks, embedder, root)
    print(
        f"published local snapshot {published.metadata.snapshot_id} "
        f"({published.metadata.chunk_count} chunks)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom Databricks Docs RAG operations")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("discover", help="fetch and list official source URLs")
    commands.add_parser("setup-db", help="create configured Delta tables and artifact Volume")
    commands.add_parser("serve", help="serve the active local snapshot through Flask")
    commands.add_parser(
        "build-app-snapshot",
        help="build and activate a Databricks-embedding snapshot in the artifact Volume",
    )
    commands.add_parser(
        "build-local-snapshot",
        help="refresh sources and build an Ollama-embedding local FAISS snapshot",
    )
    args = parser.parse_args()
    {
        "discover": discover,
        "setup-db": setup_db,
        "serve": serve,
        "build-app-snapshot": build_app_snapshot,
        "build-local-snapshot": build_local_snapshot,
    }[args.command]()


if __name__ == "__main__":
    main()
