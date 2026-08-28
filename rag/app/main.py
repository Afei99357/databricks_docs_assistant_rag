"""Databricks Apps entry point.

This module is intentionally separate from ``rag.cli serve``: importing it
selects Databricks-managed embeddings and generation, while local serving
continues to use Ollama and a locally downloaded snapshot.
"""
from __future__ import annotations

import os

from rag.agent.retrieval import RetrievalAgent
from rag.app.web import create_app
from rag.config import Settings
from rag.identity import DatabricksAppIdentityProvider
from rag.index.embeddings import DatabricksEmbeddingProvider
from rag.index.runtime import VolumeSnapshotRetriever, app_snapshot_root
from rag.llm.providers import DatabricksEndpointProvider
from rag.store import DatabricksStore


def create_databricks_app():
    settings = Settings.from_env()
    artifact_volume = os.getenv("RAG_ARTIFACT_ROOT")
    if not artifact_volume:
        raise RuntimeError("RAG_ARTIFACT_ROOT must be supplied by the Databricks App volume resource.")
    embedding_endpoint = os.getenv("RAG_DATABRICKS_EMBEDDING_ENDPOINT", "databricks-qwen3-embedding-0-6b")
    # Never share an active manifest across embedding spaces. Local Ollama
    # snapshots can remain in the Volume root while this App reads only its
    # Databricks-Qwen namespace.
    artifact_root = app_snapshot_root(artifact_volume)
    embedder = DatabricksEmbeddingProvider(embedding_endpoint)
    retriever = VolumeSnapshotRetriever(artifact_root, embedder)
    if not settings.chat_model:
        raise RuntimeError("the Databricks App requires RAG_CHAT_MODEL from its serving-endpoint resource")
    provider = DatabricksEndpointProvider(settings.chat_model)
    store = DatabricksStore(settings.warehouse_id, namespace=settings.namespace)
    agent = RetrievalAgent(retriever, provider, candidates_per_search=settings.agent_candidates_per_search)
    return create_app(
        retrieve=agent.retrieve,
        provider=provider,
        threshold=settings.relevance_threshold,
        history=store,
        identity=DatabricksAppIdentityProvider(),
        diagnostics=store,
        trace_getter=lambda: agent.last_trace,
        progress_retrieve=agent.retrieve,
    )


app = create_databricks_app()
