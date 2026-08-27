"""Explicit operator commands; refreshes are intentionally manual in v1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag.agent.retrieval import RetrievalAgent
from rag.config import Settings
from rag.evaluate import evaluate as run_evaluation
from rag.evaluate import format_header, format_row, format_summary, load_cases
from rag.history import ConversationRepository
from rag.identity import LocalTestIdentityProvider
from rag.index.embeddings import OllamaEmbeddingProvider
from rag.index.runtime import ActiveSnapshotRetriever, app_snapshot_root
from rag.ingest.fetch import fetch_page
from rag.ingest.sources import (
    GENIE_LANDING_URL,
    discover_genie_core,
    load_curated_docs,
)
from rag.llm.providers import OpenAICompatibleProvider
from rag.store import DatabricksFeedbackSink, DatabricksRequestTraceSink, DatabricksStore
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
    print(f"discovered {len(docs)} Genie-core pages and loaded {len(supplemental)} curated pages")
    for document in [*docs, *supplemental]:
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


def _local_stack(settings: Settings):
    """The retriever and answer provider the local commands share."""
    root = os.getenv("RAG_LOCAL_INDEX_DIR")
    if not root:
        raise ValueError("RAG_LOCAL_INDEX_DIR must point to downloaded local snapshot artifacts")
    embedder = OllamaEmbeddingProvider(settings.embedding_model, base_url=settings.embedding_base_url)
    if not settings.chat_model:
        raise ValueError("RAG_CHAT_MODEL must name the chat model or Databricks serving endpoint")
    if not settings.chat_base_url:
        raise ValueError("RAG_CHAT_BASE_URL must name the local OpenAI-compatible chat endpoint")
    provider = OpenAICompatibleProvider(settings.chat_base_url, settings.chat_model,
                                        api_key=settings.chat_api_key or "local")
    if bool(settings.agent_base_url) != bool(settings.agent_model):
        raise ValueError("set both RAG_AGENT_BASE_URL and RAG_AGENT_MODEL, or neither")
    agent_provider = (OpenAICompatibleProvider(settings.agent_base_url, settings.agent_model,
                                                api_key=settings.agent_api_key or "local")
                      if settings.agent_base_url else provider)
    return ActiveSnapshotRetriever(root, embedder), provider, agent_provider


def evaluate(mode: str) -> None:
    """Run the question battery and print the per-case table."""
    settings = Settings.from_env()
    retriever, _provider, agent_provider = _local_stack(settings)
    retrieve = (retriever.retrieve if mode == "plain"
                else RetrievalAgent(retriever, agent_provider).retrieve)
    cases = load_cases()
    print(f"evaluating {len(cases)} questions against {mode} retrieval", flush=True)
    print(format_header(), flush=True)
    # Cases are printed as they finish: an agent run takes minutes, and a table
    # that only appears at the end is a table nobody watches.
    counter = iter(range(1, len(cases) + 1))
    report = run_evaluation(cases, retrieve,
                            on_case=lambda outcome: print(format_row(next(counter), outcome), flush=True))
    print(format_summary(report))


def serve() -> None:
    settings = Settings.from_env()
    retriever, provider, agent_provider = _local_stack(settings)
    from rag.app.web import create_app

    store = DatabricksStore(settings.warehouse_id, settings.databricks_profile)
    feedback = DatabricksFeedbackSink(
        store, f"{settings.namespace}.rag_feedback", provider=provider.name, model=provider.model
    )
    agent = RetrievalAgent(retriever, agent_provider, candidates_per_search=settings.agent_candidates_per_search)
    history = ConversationRepository(store, settings.namespace)
    identity = LocalTestIdentityProvider()
    create_app(
        retrieve=agent.retrieve,
        provider=provider,
        threshold=settings.relevance_threshold,
        feedback_sink=feedback,
        history=history,
        identity=identity,
        trace_getter=lambda: agent.last_trace,
        trace_sink=DatabricksRequestTraceSink(
            store, f"{settings.namespace}.rag_request_traces", provider=provider.name, model=provider.model,
            retrieval_table=f"{settings.namespace}.rag_retrieval_traces",
            agent_provider=agent_provider.name, agent_model=agent_provider.model,
        ),
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
    app_index_root = app_snapshot_root(settings.volume_path)
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
    embedder = OllamaEmbeddingProvider(settings.embedding_model, base_url=settings.embedding_base_url)
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
    evaluation = commands.add_parser("evaluate", help="run the question battery and report retrieval quality")
    evaluation.add_argument("--mode", choices=("agent", "plain"), default="agent",
                            help="investigate with the retrieval agent, or run one plain search per question")
    commands.add_parser(
        "build-app-snapshot",
        help="build and activate a Databricks-embedding snapshot in the artifact Volume",
    )
    commands.add_parser(
        "build-local-snapshot",
        help="refresh sources and build an Ollama-embedding local FAISS snapshot",
    )
    args = parser.parse_args()
    if args.command == "evaluate":
        return evaluate(args.mode)
    {
        "discover": discover,
        "setup-db": setup_db,
        "serve": serve,
        "build-app-snapshot": build_app_snapshot,
        "build-local-snapshot": build_local_snapshot,
    }[args.command]()


if __name__ == "__main__":
    main()
