import subprocess
import sys
from pathlib import Path

import pytest

from rag.config import Settings


def _settings(**overrides) -> Settings:
    values = {
        "catalog": "catalog", "schema": "schema", "warehouse_id": "warehouse",
        "artifact_volume": "artifacts", "embedding_model": "embedding",
        "agent_candidates_per_search": 10, "relevance_threshold": 0.35,
        "chat_base_url": None, "chat_model": None, "chat_api_key": None,
        "embedding_base_url": "http://localhost", "databricks_profile": None,
        "agent_base_url": None, "agent_model": None, "agent_api_key": None,
        "storage_backend": "databricks",
    }
    values.update(overrides)
    return Settings(**values)


def test_create_store_selects_databricks_backend_by_default(monkeypatch):
    from rag.storage import create_store
    from rag.storage import databricks as databricks_module

    captured = {}

    class FakeDatabricksStore:
        def __init__(self, warehouse_id, profile=None, *, namespace=None, **_kwargs):
            captured["warehouse_id"] = warehouse_id
            captured["profile"] = profile
            captured["namespace"] = namespace

    monkeypatch.setattr(databricks_module, "DatabricksStore", FakeDatabricksStore)

    store = create_store(_settings())

    assert isinstance(store, FakeDatabricksStore)
    assert captured == {"warehouse_id": "warehouse", "profile": None, "namespace": "catalog.schema"}


def test_create_store_selects_sqlite_backend(tmp_path):
    from rag.storage import create_store
    from rag.storage.sqlite import SQLiteStore

    assert isinstance(create_store(_settings(storage_backend="sqlite", sqlite_path=str(tmp_path / "store.sqlite"))), SQLiteStore)


def test_create_store_unknown_backend_raises_value_error():
    from rag.storage import create_store

    with pytest.raises(ValueError):
        create_store(_settings(storage_backend="bogus"))


def test_importing_storage_package_does_not_import_databricks_sdk():
    # A fresh interpreter, not the pytest process (which has already imported
    # rag.storage.databricks via other test modules), is the only reliable way
    # to prove `import rag.storage` alone never pulls in the SDK.
    script = "import sys; import rag.storage; assert 'databricks' not in sys.modules, sorted(sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_local_cli_import_and_store_construction_stays_sdk_free(tmp_path):
    script = f"""
import os
import sys
sys.path.insert(0, {str(Path.cwd())!r})
import rag.cli
os.environ['RAG_STORAGE_BACKEND'] = 'sqlite'
os.environ['RAG_SQLITE_PATH'] = {str(tmp_path / 'store.sqlite')!r}
from rag.config import Settings
from rag.storage import create_store
create_store(Settings.from_env())
assert 'databricks' not in sys.modules, sorted(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
