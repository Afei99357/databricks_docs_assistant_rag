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
    top_k: int
    relevance_threshold: float
    answer_provider: str
    ollama_base_url: str
    ollama_model: str
    databricks_chat_endpoint: str
    databricks_profile: str | None

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.namespace}/{self.artifact_volume}"

    @classmethod
    def from_env(cls) -> "Settings":
        required = ("RAG_CATALOG", "RAG_SCHEMA", "RAG_WAREHOUSE_ID")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError("missing required configuration: " + ", ".join(missing))
        return cls(os.environ["RAG_CATALOG"], os.environ["RAG_SCHEMA"], os.environ["RAG_WAREHOUSE_ID"],
                   os.getenv("RAG_ARTIFACT_VOLUME", "rag_artifacts"),
                   os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding:4b"),
                   int(os.getenv("RAG_RETRIEVAL_TOP_K", "25")), float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.35")),
                   os.getenv("ANSWER_PROVIDER", "ollama"), os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                   os.getenv("OLLAMA_MODEL", "qwen3.5"), os.getenv("DATABRICKS_CHAT_ENDPOINT", "databricks-gpt-oss-20b"),
                   os.getenv("RAG_DATABRICKS_PROFILE") or None)
