"""FAISS snapshot construction and read-only retrieval."""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from rag.index.embeddings import EmbeddingProvider
from rag.models import Chunk, RetrievalResult


def _faiss():
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("install faiss-cpu and numpy to build or load a FAISS snapshot") from exc
    return faiss, np


def _chunk_from_dict(value: dict) -> Chunk:
    value["heading_path"] = tuple(value["heading_path"])
    return Chunk(**value)


class FaissSnapshot:
    """A completed local snapshot. Files are immutable once published."""
    def __init__(self, index, chunks: list[Chunk], snapshot_id: str):
        self.index, self.chunks, self.snapshot_id = index, chunks, snapshot_id

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: EmbeddingProvider, snapshot_id: str) -> "FaissSnapshot":
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        faiss, np = _faiss()
        vectors = np.asarray(embedder.embed([chunk.text for chunk in chunks]), dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("embedding provider returned an invalid vector matrix")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, list(chunks), snapshot_id)

    def validate(self, expected_dimension: int | None = None) -> None:
        if self.index.ntotal != len(self.chunks):
            raise ValueError("FAISS vector count does not match chunk map")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("chunk map has duplicate chunk IDs")
        if expected_dimension is not None and self.index.d != expected_dimension:
            raise ValueError("FAISS dimension does not match snapshot metadata")

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        faiss, _ = _faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_path, map_path = directory / "index.faiss", directory / "chunk_map.json"
        faiss.write_index(self.index, str(index_path))
        map_path.write_text(json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False), encoding="utf-8")
        return index_path, map_path

    @classmethod
    def load(cls, directory: str | Path, snapshot_id: str) -> "FaissSnapshot":
        faiss, _ = _faiss()
        directory = Path(directory)
        chunks = [_chunk_from_dict(value) for value in json.loads((directory / "chunk_map.json").read_text(encoding="utf-8"))]
        snapshot = cls(faiss.read_index(str(directory / "index.faiss")), chunks, snapshot_id)
        snapshot.validate()
        return snapshot

    def search(self, question: str, embedder: EmbeddingProvider, top_k: int) -> list[RetrievalResult]:
        _, np = _faiss()
        vector = np.asarray(embedder.embed([question]), dtype="float32")
        if vector.shape != (1, self.index.d):
            raise ValueError("query embedding dimension does not match index")
        import faiss
        faiss.normalize_L2(vector)
        scores, positions = self.index.search(vector, min(top_k, len(self.chunks)))
        return [RetrievalResult(self.chunks[position], float(score), self.snapshot_id)
                for score, position in zip(scores[0], positions[0]) if position >= 0]


def write_active_manifest(root: str | Path, snapshot_id: str) -> None:
    """Atomically select an already-validated snapshot; never point at a partial build."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=root, delete=False, encoding="utf-8") as handle:
        json.dump({"snapshot_id": snapshot_id}, handle)
        temporary = Path(handle.name)
    temporary.replace(root / "active_snapshot.json")


def read_active_manifest(root: str | Path) -> str | None:
    path = Path(root) / "active_snapshot.json"
    return json.loads(path.read_text(encoding="utf-8"))["snapshot_id"] if path.exists() else None

