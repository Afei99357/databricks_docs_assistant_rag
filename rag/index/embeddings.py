"""Local embedding adapters. Production models are loaded lazily on first use."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Protocol

import requests
from databricks.sdk import WorkspaceClient


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class QwenEmbeddingProvider:
    def __init__(self, model_name: str, *, batch_size: int = 16, device: str | None = None):
        self.model_name, self.batch_size, self.device = model_name, batch_size, device
        self._model = None
        self.dimension = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("install the 'embedding' extra to use Qwen embeddings") from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        vectors = self._model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)
        result = vectors.tolist()
        self.dimension = len(result[0]) if result else self.dimension
        return result


class HashEmbeddingProvider:
    """Deterministic test-only embedder; never select this in application configuration."""
    model_name = "test/hash-embedding"
    dimension = 16

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[(byte - 127.5) / 127.5 for byte in hashlib.sha256(text.encode()).digest()[:self.dimension]] for text in texts]


class OllamaEmbeddingProvider:
    """Local Qwen embeddings through Ollama; useful when no Python GPU stack is installed."""
    def __init__(self, model_name: str = "qwen3-embedding:0.6b", base_url: str = "http://localhost:11434", timeout: float = 120, batch_size: int = 32):
        self.model_name, self.base_url, self.timeout, self.batch_size, self.dimension = model_name, base_url.rstrip("/"), timeout, batch_size, 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = []
        for start in range(0, len(texts), self.batch_size):
            response = requests.post(f"{self.base_url}/api/embed", json={"model": self.model_name, "input": list(texts[start:start + self.batch_size])}, timeout=self.timeout)
            response.raise_for_status()
            values.extend(response.json()["embeddings"])
        if values: self.dimension = len(values[0])
        return values


class DatabricksEmbeddingProvider:
    """Embedding adapter for a Databricks Model Serving endpoint."""
    def __init__(self, endpoint: str, *, profile: str | None = None, batch_size: int = 10,
                 min_interval_seconds: float = 2.0, max_rate_limit_retries: int = 8):
        if not 1 <= batch_size <= 150:
            raise ValueError("Databricks embedding batch_size must be between 1 and 150")
        self.model_name, self.profile, self.batch_size, self.dimension = endpoint, profile, batch_size, 0
        self.min_interval_seconds = min_interval_seconds
        self.max_rate_limit_retries = max_rate_limit_retries

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = WorkspaceClient(profile=self.profile) if self.profile else WorkspaceClient()
        values: list[list[float]] = []
        batches = range(0, len(texts), self.batch_size)
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_number, start in enumerate(batches, start=1):
            batch = list(texts[start:start + self.batch_size])
            print(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)...", flush=True)
            for attempt in range(self.max_rate_limit_retries + 1):
                try:
                    response = client.serving_endpoints.query(name=self.model_name, input=batch)
                    break
                except Exception as exc:
                    if "REQUEST_LIMIT_EXCEEDED" not in str(exc) or attempt == self.max_rate_limit_retries:
                        raise
                    wait_seconds = min(60.0, 5.0 * (2 ** attempt))
                    print(
                        f"Embedding rate limit reached; waiting {wait_seconds:.0f}s before retry "
                        f"{attempt + 1}/{self.max_rate_limit_retries}...",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
            values.extend(list(item.embedding) for item in response.data or [])
            if batch_number < total_batches:
                time.sleep(self.min_interval_seconds)
        if len(values) != len(texts):
            raise RuntimeError("Databricks embedding endpoint returned an unexpected number of vectors")
        if values:
            self.dimension = len(values[0])
        return values
