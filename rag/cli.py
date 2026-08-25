"""Explicit operator commands; refreshes are intentionally manual in v1."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag.config import Settings
from rag.ingest.fetch import fetch_page
from rag.ingest.sources import GENIE_LANDING_URL, discover_genie_core, load_curated_docs
from rag.index.embeddings import OllamaEmbeddingProvider
from rag.index.runtime import ActiveSnapshotRetriever
from rag.llm.providers import DatabricksEndpointProvider, OllamaProvider
from rag.store import DatabricksFeedbackSink, DatabricksStore


def discover() -> None:
    landing = fetch_page("genie-landing", GENIE_LANDING_URL)
    if landing.outcome != "ok" or not landing.html:
        raise RuntimeError(f"landing-page discovery failed: {landing.error_message}")
    docs = discover_genie_core(landing.html)
    supplemental = load_curated_docs(Path(__file__).parent / "ingest/config/curated_urls.yaml")
    print(f"discovered {len(docs)} Genie-core pages and loaded {len(supplemental)} supplemental pages")
    for document in [*docs, *supplemental]:
        print(f"{document.doc_id}\t{document.canonical_requested_url}")


def setup_db() -> None:
    settings = Settings.from_env()
    DatabricksStore(settings.warehouse_id, settings.databricks_profile).apply_schema(
        Path(__file__).parents[1] / "sql/001_rag_schema.sql", catalog=settings.catalog,
        schema=settings.schema, artifact_volume=settings.artifact_volume)
    print(f"initialized {settings.namespace}")


def serve() -> None:
    settings = Settings.from_env()
    root = os.getenv("RAG_LOCAL_INDEX_DIR")
    if not root:
        raise ValueError("RAG_LOCAL_INDEX_DIR must point to downloaded local snapshot artifacts")
    embedder = OllamaEmbeddingProvider(settings.embedding_model)
    retriever = ActiveSnapshotRetriever(root, embedder, settings.top_k)
    provider = (OllamaProvider(settings.ollama_base_url, settings.ollama_model)
                if settings.answer_provider == "ollama"
                else DatabricksEndpointProvider(settings.databricks_chat_endpoint, profile=settings.databricks_profile))
    from rag.app.web import create_app
    store = DatabricksStore(settings.warehouse_id, settings.databricks_profile)
    feedback = DatabricksFeedbackSink(store, f"{settings.namespace}.rag_feedback", provider=provider.name, model=provider.model)
    create_app(retrieve=retriever.retrieve, provider=provider, threshold=settings.relevance_threshold, feedback_sink=feedback).run(
        host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom Databricks Docs RAG operations")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("discover", help="fetch and list official source URLs")
    commands.add_parser("setup-db", help="create configured Delta tables and artifact Volume")
    commands.add_parser("serve", help="serve the active local snapshot through Flask")
    args = parser.parse_args()
    {"discover": discover, "setup-db": setup_db, "serve": serve}[args.command]()


if __name__ == "__main__":
    main()
