"""Publishers that hand a built snapshot's local artifacts to wherever retrievers read them.

Both publishers implement ``ArtifactPublisher.publish(local_directory, snapshot_id) -> str``
(see ``rag.storage.protocol``). The returned string is the *directory* holding the
published ``index.faiss`` and ``chunk_map.json`` -- callers append the filename
themselves, so the contract is identical whether the destination is local or remote.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class LocalDirectoryPublisher:
    """Publishes snapshot artifacts to a directory on this machine."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def publish(self, local_directory: Path, snapshot_id: str) -> str:
        destination = self.root / "snapshots" / snapshot_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_directory, destination, dirs_exist_ok=True)
        return str(destination)
