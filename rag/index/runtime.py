"""Read-only active-snapshot retrieval, refreshing only after activation changes."""
from __future__ import annotations

from pathlib import Path

from rag.index.embeddings import EmbeddingProvider
from rag.index.faiss_store import FaissSnapshot, read_active_manifest
from rag.models import RetrievalResult


class ActiveSnapshotRetriever:
    def __init__(self, root: str | Path, embedder: EmbeddingProvider, top_k: int):
        self.root, self.embedder, self.top_k = Path(root), embedder, top_k
        self._snapshot: FaissSnapshot | None = None

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        snapshot_id = read_active_manifest(self.root)
        if not snapshot_id:
            raise RuntimeError("no active retrieval snapshot is available")
        if self._snapshot is None or self._snapshot.snapshot_id != snapshot_id:
            self._snapshot = FaissSnapshot.load(self.root / "snapshots" / snapshot_id, snapshot_id)
        return self._snapshot.search(question, self.embedder, top_k or self.top_k)
