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
from rag.history import ConversationRepository
from rag.identity import DatabricksAppIdentityProvider
from rag.index.embeddings import DatabricksEmbeddingProvider
from rag.index.runtime import ActiveSnapshotRetriever
from rag.llm.providers import DatabricksEndpointProvider
from rag.store import DatabricksFeedbackSink, DatabricksStore


def create_databricks_app():
    settings = Settings.from_env()
    artifact_root = os.getenv("RAG_ARTIFACT_ROOT")
    if not artifact_root:
        raise RuntimeError("RAG_ARTIFACT_ROOT must be supplied by the Databricks App volume resource.")
    embedding_endpoint = os.getenv("RAG_DATABRICKS_EMBEDDING_ENDPOINT", "databricks-qwen3-embedding-0-6b")
    embedder = DatabricksEmbeddingProvider(embedding_endpoint)
    retriever = ActiveSnapshotRetriever(artifact_root, embedder, settings.top_k)
    provider = DatabricksEndpointProvider(settings.databricks_chat_endpoint)
    store = DatabricksStore(settings.warehouse_id)
    feedback = DatabricksFeedbackSink(
        store, f"{settings.namespace}.rag_feedback", provider=provider.name, model=provider.model,
    )
    return create_app(
        retrieve=RetrievalAgent(retriever.retrieve, provider).retrieve,
        provider=provider,
        threshold=settings.relevance_threshold,
        feedback_sink=feedback,
        history=ConversationRepository(store, settings.namespace),
        identity=DatabricksAppIdentityProvider(),
    )


app = create_databricks_app()
