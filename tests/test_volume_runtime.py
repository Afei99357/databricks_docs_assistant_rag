import io

from rag.index.embeddings import HashEmbeddingProvider
from rag.index.runtime import VolumeSnapshotRetriever
from rag.index.service import build_and_activate
from rag.models import Chunk


class Download:
    def __init__(self, contents): self.contents = io.BytesIO(contents)


class Files:
    def __init__(self, data): self.data = data
    def download(self, path): return Download(self.data[path])


class Workspace:
    def __init__(self, data): self.files = Files(data)


def test_volume_runtime_downloads_active_snapshot_to_ephemeral_cache(tmp_path):
    source = tmp_path / "source"
    chunk = Chunk("id", "d", "v", 0, "Genie concepts evidence", (), "https://docs.databricks.com/x", "Docs")
    snapshot = build_and_activate([chunk], HashEmbeddingProvider(), source)
    snapshot_id = snapshot.metadata.snapshot_id
    root = "/Volumes/catalog/schema/volume/app-qwen"
    workspace = Workspace({
        f"{root}/active_snapshot.json": (source / "active_snapshot.json").read_bytes(),
        f"{root}/snapshots/{snapshot_id}/index.faiss": (snapshot.local_directory / "index.faiss").read_bytes(),
        f"{root}/snapshots/{snapshot_id}/chunk_map.json": (snapshot.local_directory / "chunk_map.json").read_bytes(),
    })
    retriever = VolumeSnapshotRetriever(root, HashEmbeddingProvider(), 1, workspace=workspace, cache_root=tmp_path / "cache")
    assert retriever.retrieve("Genie")[0].chunk.chunk_id == "id"
