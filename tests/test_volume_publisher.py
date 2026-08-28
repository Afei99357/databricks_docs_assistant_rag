from pathlib import Path

from rag.store import VolumePublisher


class FakeUploader:
    """Stands in for DatabricksStore.upload without touching a real workspace."""

    def __init__(self):
        self.calls = []

    def upload(self, local_path, volume_path, *, overwrite=False):
        self.calls.append((Path(local_path), volume_path, overwrite))


def test_volume_publisher_uploads_index_and_chunk_map_under_snapshots(tmp_path):
    local_directory = tmp_path / "snapshots" / "snap-1"
    local_directory.mkdir(parents=True)
    (local_directory / "index.faiss").write_bytes(b"idx")
    (local_directory / "chunk_map.json").write_text("[]", encoding="utf-8")

    uploader = FakeUploader()
    destination = VolumePublisher(uploader, "/Volumes/cat/sch/vol").publish(
        local_directory, "snap-1"
    )

    assert destination == "/Volumes/cat/sch/vol/snapshots/snap-1"
    uploads = {volume_path: (local_path, overwrite) for local_path, volume_path, overwrite in uploader.calls}
    assert uploads["/Volumes/cat/sch/vol/snapshots/snap-1/index.faiss"] == (
        local_directory / "index.faiss",
        False,
    )
    assert uploads["/Volumes/cat/sch/vol/snapshots/snap-1/chunk_map.json"] == (
        local_directory / "chunk_map.json",
        False,
    )


def test_volume_publisher_strips_a_trailing_slash_from_the_volume_path(tmp_path):
    local_directory = tmp_path / "snapshots" / "snap-1"
    local_directory.mkdir(parents=True)
    (local_directory / "index.faiss").write_bytes(b"idx")
    (local_directory / "chunk_map.json").write_text("[]", encoding="utf-8")

    destination = VolumePublisher(FakeUploader(), "/Volumes/cat/sch/vol/").publish(
        local_directory, "snap-1"
    )

    assert destination == "/Volumes/cat/sch/vol/snapshots/snap-1"


def test_volume_publisher_republishes_the_active_snapshot_manifest_when_present(tmp_path):
    build_root = tmp_path / "build"
    local_directory = build_root / "snapshots" / "snap-1"
    local_directory.mkdir(parents=True)
    (local_directory / "index.faiss").write_bytes(b"idx")
    (local_directory / "chunk_map.json").write_text("[]", encoding="utf-8")
    (build_root / "active_snapshot.json").write_text('{"snapshot_id": "snap-1"}', encoding="utf-8")

    uploader = FakeUploader()
    VolumePublisher(uploader, "/Volumes/cat/sch/vol").publish(local_directory, "snap-1")

    manifest_uploads = [
        (local_path, overwrite)
        for local_path, volume_path, overwrite in uploader.calls
        if volume_path == "/Volumes/cat/sch/vol/active_snapshot.json"
    ]
    assert manifest_uploads == [(build_root / "active_snapshot.json", True)]


def test_volume_publisher_skips_the_manifest_upload_when_it_does_not_exist(tmp_path):
    local_directory = tmp_path / "snapshots" / "snap-1"
    local_directory.mkdir(parents=True)
    (local_directory / "index.faiss").write_bytes(b"idx")
    (local_directory / "chunk_map.json").write_text("[]", encoding="utf-8")

    uploader = FakeUploader()
    VolumePublisher(uploader, "/Volumes/cat/sch/vol").publish(local_directory, "snap-1")

    assert all(
        volume_path != "/Volumes/cat/sch/vol/active_snapshot.json"
        for _, volume_path, _ in uploader.calls
    )
