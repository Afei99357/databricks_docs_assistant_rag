"""Local embedding adapters. Production models are loaded lazily on first use."""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import requests


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
