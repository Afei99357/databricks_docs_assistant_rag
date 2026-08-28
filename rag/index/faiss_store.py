"""FAISS snapshot construction and read-only retrieval."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

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


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "difference",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


def _terms(text: str) -> set[str]:
    return {
        word for word in _WORD.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS
    }


def _lexical_score(question: str, chunk: Chunk) -> float:
    """Small exact-term signal to complement semantic similarity."""
    query_terms = _terms(question)
    if not query_terms:
        return 0.0
    text = f"{chunk.source_title} {chunk.text}".lower()
    matched = sum(term in text for term in query_terms) / len(query_terms)
    phrases = [
        " ".join(pair)
        for pair in zip(_WORD.findall(question.lower()), _WORD.findall(question.lower())[1:])
    ]
    phrase_bonus = sum(phrase in text for phrase in phrases) / max(1, len(phrases))
    return min(1.0, 0.75 * matched + 0.25 * phrase_bonus)


class FaissSnapshot:
    """A completed local snapshot. Files are immutable once published."""

    def __init__(self, index, chunks: list[Chunk], snapshot_id: str):
        self.index, self.chunks, self.snapshot_id = index, chunks, snapshot_id

    @classmethod
    def build(
        cls, chunks: Sequence[Chunk], embedder: EmbeddingProvider, snapshot_id: str
    ) -> FaissSnapshot:
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        return cls.build_from_vectors(
            chunks, embedder.embed([chunk.text for chunk in chunks]), snapshot_id
        )

    @classmethod
    def build_from_vectors(
        cls, chunks: Sequence[Chunk], vectors, snapshot_id: str
    ) -> FaissSnapshot:
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        faiss, np = _faiss()
        vectors = np.asarray(vectors, dtype="float32")
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
        map_path.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        return index_path, map_path

    @classmethod
    def load(cls, directory: str | Path, snapshot_id: str) -> FaissSnapshot:
        faiss, _ = _faiss()
        directory = Path(directory)
        chunks = [
            _chunk_from_dict(value)
            for value in json.loads((directory / "chunk_map.json").read_text(encoding="utf-8"))
        ]
        snapshot = cls(faiss.read_index(str(directory / "index.faiss")), chunks, snapshot_id)
        snapshot.validate()
        return snapshot

    def search(
        self, question: str, embedder: EmbeddingProvider, top_k: int
    ) -> list[RetrievalResult]:
        _, np = _faiss()
        vector = np.asarray(embedder.embed([question]), dtype="float32")
        if vector.shape != (1, self.index.d):
            raise ValueError("query embedding dimension does not match index")
        import faiss

        faiss.normalize_L2(vector)
        # Retrieve a broader semantic pool, then blend it with lexical matches
        # over the immutable chunk map. This preserves semantic recall while
        # making exact product terminology discoverable.
        scores, positions = self.index.search(vector, min(max(top_k * 6, 60), len(self.chunks)))
        semantic = {
            int(position): float(score)
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        }
        lexical = {
            position: _lexical_score(question, chunk) for position, chunk in enumerate(self.chunks)
        }
        candidates = set(semantic) | {position for position, score in lexical.items() if score > 0}
        ranked = sorted(
            candidates,
            key=lambda position: 0.65 * semantic.get(position, 0.0) + 0.35 * lexical[position],
            reverse=True,
        )[:top_k]
        return [
            RetrievalResult(
                self.chunks[position],
                0.65 * semantic.get(position, 0.0) + 0.35 * lexical[position],
                self.snapshot_id,
            )
            for position in ranked
        ]


def write_active_manifest(
    root: str | Path,
    snapshot_id: str,
    *,
    corpus_fingerprint: str | None = None,
    embedding_model: str | None = None,
    embedding_revision: str = "v1",
    chunking_revision: str = "v1",
) -> None:
    """Atomically select an already-validated snapshot; never point at a partial build."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=root, delete=False, encoding="utf-8") as handle:
        json.dump(
            {
                "snapshot_id": snapshot_id,
                "corpus_fingerprint": corpus_fingerprint,
                "embedding_model": embedding_model,
                "embedding_revision": embedding_revision,
                "chunking_revision": chunking_revision,
            },
            handle,
        )
        temporary = Path(handle.name)
    temporary.replace(root / "active_snapshot.json")


def read_active_manifest(root: str | Path) -> str | None:
    path = Path(root) / "active_snapshot.json"
    return json.loads(path.read_text(encoding="utf-8"))["snapshot_id"] if path.exists() else None


def read_active_fingerprint(root: str | Path) -> str | None:
    path = Path(root) / "active_snapshot.json"
    return (
        json.loads(path.read_text(encoding="utf-8")).get("corpus_fingerprint")
        if path.exists()
        else None
    )


def active_manifest_matches(
    root: str | Path,
    *,
    corpus_fingerprint: str,
    embedding_model: str,
    embedding_revision: str,
    chunking_revision: str,
) -> bool:
    path = Path(root) / "active_snapshot.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return (
        manifest.get("corpus_fingerprint") == corpus_fingerprint
        and manifest.get("embedding_model") == embedding_model
        and manifest.get("embedding_revision") == embedding_revision
        and manifest.get("chunking_revision") == chunking_revision
    )
