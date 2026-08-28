import subprocess
import sys

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
        def __init__(self, warehouse_id, profile=None, *, namespace=None):
            captured["warehouse_id"] = warehouse_id
            captured["profile"] = profile
            captured["namespace"] = namespace

    monkeypatch.setattr(databricks_module, "DatabricksStore", FakeDatabricksStore)

    store = create_store(_settings())

    assert isinstance(store, FakeDatabricksStore)
    assert captured == {"warehouse_id": "warehouse", "profile": None, "namespace": "catalog.schema"}


def test_create_store_sqlite_backend_raises_not_implemented():
    from rag.storage import create_store

    with pytest.raises(NotImplementedError):
        create_store(_settings(storage_backend="sqlite"))


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
