"""Read-only active-snapshot retrieval, refreshing only after activation changes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rag.index.embeddings import EmbeddingProvider
from rag.index.faiss_store import FaissSnapshot, read_active_manifest
from rag.models import RetrievalResult


class ActiveSnapshotRetriever:
    def __init__(self, root: str | Path, embedder: EmbeddingProvider, top_k: int):
        self.root, self.embedder, self.top_k = Path(root), embedder, top_k
        self._snapshot: FaissSnapshot | None = None

    def _active_snapshot(self) -> FaissSnapshot:
        snapshot_id = read_active_manifest(self.root)
        if not snapshot_id:
            raise RuntimeError("no active retrieval snapshot is available")
        if self._snapshot is None or self._snapshot.snapshot_id != snapshot_id:
            self._snapshot = FaissSnapshot.load(self.root / "snapshots" / snapshot_id, snapshot_id)
        return self._snapshot

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        return self._active_snapshot().search(question, self.embedder, top_k or self.top_k)

    def read_chunks(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        snapshot = self._active_snapshot()
        by_id = {chunk.chunk_id: chunk for chunk in snapshot.chunks}
        return [RetrievalResult(by_id[chunk_id], 1.0, snapshot.snapshot_id) for chunk_id in chunk_ids if chunk_id in by_id]

    def related_chunks(self, chunk_id: str, *, radius: int = 1) -> list[RetrievalResult]:
        snapshot = self._active_snapshot()
        target = next((chunk for chunk in snapshot.chunks if chunk.chunk_id == chunk_id), None)
        if target is None:
            return []
        related = [
            chunk for chunk in snapshot.chunks
            if chunk.doc_id == target.doc_id
            and chunk.document_version == target.document_version
            and abs(chunk.position - target.position) <= radius
        ]
        return [RetrievalResult(chunk, 1.0, snapshot.snapshot_id) for chunk in sorted(related, key=lambda chunk: chunk.position)]

    def search_within_document(self, source_url: str, question: str, top_k: int) -> list[RetrievalResult]:
        snapshot = self._active_snapshot()
        ranked = snapshot.search(question, self.embedder, len(snapshot.chunks))
        return [item for item in ranked if item.chunk.source_url == source_url][:top_k]


class VolumeSnapshotRetriever:
    """Read an active FAISS snapshot from a UC Volume in Databricks Apps.

    Apps service containers do not reliably mount ``/Volumes`` as a local
    filesystem. This adapter reads artifacts through the Files API (authorized
    by the App service principal) and caches the immutable active snapshot in
    the container's ephemeral filesystem.
    """
    def __init__(self, volume_root: str, embedder: EmbeddingProvider, top_k: int, *, workspace=None,
                 cache_root: str | Path | None = None):
        self.volume_root = volume_root.rstrip("/")
        self.embedder, self.top_k = embedder, top_k
        if workspace is None:
            from databricks.sdk import WorkspaceClient
            workspace = WorkspaceClient()
        self.workspace = workspace
        self.cache_root = Path(cache_root or Path(tempfile.gettempdir()) / "databricks-docs-rag-snapshots")
        self.local = ActiveSnapshotRetriever(self.cache_root, embedder, top_k)
        self._downloaded_snapshot_id: str | None = None

    def _download(self, remote_path: str) -> bytes:
        return self.workspace.files.download(remote_path).contents.read()

    def _sync_active_snapshot(self) -> None:
        manifest = json.loads(self._download(f"{self.volume_root}/active_snapshot.json"))
        snapshot_id = manifest.get("snapshot_id")
        if not snapshot_id:
            raise RuntimeError("the artifact Volume has no active retrieval snapshot")
        target = self.cache_root / "snapshots" / snapshot_id
        if self._downloaded_snapshot_id == snapshot_id and (target / "index.faiss").exists():
            return
        if (target / "index.faiss").exists() and (target / "chunk_map.json").exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
            (self.cache_root / "active_snapshot.json").write_text(
                json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8",
            )
            self._downloaded_snapshot_id = snapshot_id
            return
        staging = self.cache_root / "staging" / snapshot_id
        staging.mkdir(parents=True, exist_ok=True)
        remote = f"{self.volume_root}/snapshots/{snapshot_id}"
        (staging / "index.faiss").write_bytes(self._download(f"{remote}/index.faiss"))
        (staging / "chunk_map.json").write_bytes(self._download(f"{remote}/chunk_map.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        (self.cache_root / "active_snapshot.json").write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")
        self._downloaded_snapshot_id = snapshot_id

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        self._sync_active_snapshot()
        return self.local.retrieve(question, top_k)

    def read_chunks(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        self._sync_active_snapshot()
        return self.local.read_chunks(chunk_ids)

    def related_chunks(self, chunk_id: str, *, radius: int = 1) -> list[RetrievalResult]:
        self._sync_active_snapshot()
        return self.local.related_chunks(chunk_id, radius=radius)

    def search_within_document(self, source_url: str, question: str, top_k: int) -> list[RetrievalResult]:
        self._sync_active_snapshot()
        return self.local.search_within_document(source_url, question, top_k)
