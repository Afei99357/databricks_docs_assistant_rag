"""Environment-only configuration; no workspace values are checked in."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    catalog: str
    schema: str
    warehouse_id: str
    artifact_volume: str
    embedding_model: str
    agent_candidates_per_search: int
    relevance_threshold: float
    chat_base_url: str | None
    chat_model: str | None
    chat_api_key: str | None
    embedding_base_url: str
    databricks_profile: str | None
    storage_backend: str = "databricks"
    sqlite_path: str = "./data/local.sqlite"
    embedding_revision: str = "v1"
    chunking_revision: str = "v1"

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.artifact_volume}"

    @classmethod
    def from_env(cls) -> Settings:
        backend = os.getenv("RAG_STORAGE_BACKEND", "databricks")
        if backend == "databricks":
            required = ("RAG_CATALOG", "RAG_SCHEMA", "RAG_WAREHOUSE_ID")
            missing = [name for name in required if not os.getenv(name)]
            if missing:
                raise ValueError("missing required configuration: " + ", ".join(missing))
        # Local chat servers all use the OpenAI-compatible /v1 API. The old
        # names remain fallbacks so existing private .env files still start.
        legacy_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
        legacy_model = os.getenv("OPENAI_MODEL") or os.getenv("OLLAMA_MODEL")
        return cls(
            os.getenv("RAG_CATALOG", ""),
            os.getenv("RAG_SCHEMA", ""),
            os.getenv("RAG_WAREHOUSE_ID", ""),
            os.getenv("RAG_ARTIFACT_VOLUME", "rag_artifacts"),
            os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding:4b"),
            int(os.getenv("RAG_AGENT_CANDIDATES_PER_SEARCH", "10")),
            float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.35")),
            os.getenv("RAG_CHAT_BASE_URL") or legacy_base_url or None,
            os.getenv("RAG_CHAT_MODEL") or legacy_model or None,
            os.getenv("RAG_CHAT_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            os.getenv("RAG_EMBEDDING_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("RAG_DATABRICKS_PROFILE") or None,
            backend,
            os.getenv("RAG_SQLITE_PATH", "./data/local.sqlite"),
            os.getenv("RAG_EMBEDDING_REVISION", "v1"),
            os.getenv("RAG_CHUNKING_REVISION", "v1"),
        )
