from pathlib import Path

from rag.storage.artifacts import LocalDirectoryPublisher


def test_local_publisher_copies_both_snapshot_files(tmp_path):
    source = tmp_path / "build"
    source.mkdir()
    (source / "index.faiss").write_bytes(b"idx")
    (source / "chunk_map.json").write_text("[]", encoding="utf-8")
    destination = LocalDirectoryPublisher(tmp_path / "published").publish(source, "abc")
    assert (Path(destination) / "index.faiss").read_bytes() == b"idx"
    assert (Path(destination) / "chunk_map.json").read_text(encoding="utf-8") == "[]"


def test_local_publisher_nests_under_snapshots_and_snapshot_id(tmp_path):
    source = tmp_path / "build"
    source.mkdir()
    (source / "index.faiss").write_bytes(b"idx")
    (source / "chunk_map.json").write_text("[]", encoding="utf-8")
    destination = LocalDirectoryPublisher(tmp_path / "published").publish(source, "snap-1")
    assert Path(destination) == tmp_path / "published" / "snapshots" / "snap-1"
