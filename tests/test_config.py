from rag.config import Settings


def test_volume_path_uses_catalog_and_schema_path_segments():
    settings = Settings(
        "catalog", "schema", "warehouse", "artifacts", "embedding", 10, 0.5,
        "http://localhost", "model", None, "http://localhost", None,
    )

    assert settings.volume_path == "/Volumes/catalog/schema/artifacts"


def test_new_chat_variables_take_precedence_over_legacy_variables(monkeypatch):
    values = {
        "RAG_CATALOG": "catalog", "RAG_SCHEMA": "schema", "RAG_WAREHOUSE_ID": "warehouse",
        "RAG_CHAT_BASE_URL": "http://muse:1234/v1",
        "RAG_CHAT_MODEL": "muse", "RAG_CHAT_API_KEY": "local", "OLLAMA_MODEL": "old-qwen",
    }
    for name, value in values.items(): monkeypatch.setenv(name, value)

    settings = Settings.from_env()

    assert (settings.chat_base_url, settings.chat_model, settings.chat_api_key) == (
        "http://muse:1234/v1", "muse", "local",
    )


def test_storage_backend_defaults_to_databricks_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_STORAGE_BACKEND", raising=False)
    for name, value in {"RAG_CATALOG": "catalog", "RAG_SCHEMA": "schema",
                        "RAG_WAREHOUSE_ID": "warehouse"}.items(): monkeypatch.setenv(name, value)

    settings = Settings.from_env()

    assert settings.storage_backend == "databricks"


def test_sqlite_backend_does_not_require_warehouse_config(monkeypatch):
    for name in ("RAG_CATALOG", "RAG_SCHEMA", "RAG_WAREHOUSE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_STORAGE_BACKEND", "sqlite")

    settings = Settings.from_env()

    assert settings.storage_backend == "sqlite"
    assert settings.catalog == ""
    assert settings.warehouse_id == ""


def test_sqlite_path_defaults_under_data(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE_BACKEND", "sqlite")
    monkeypatch.delenv("RAG_SQLITE_PATH", raising=False)
    for name in ("RAG_CATALOG", "RAG_SCHEMA", "RAG_WAREHOUSE_ID"):
        monkeypatch.delenv(name, raising=False)

    assert Settings.from_env().sqlite_path == "./data/local.sqlite"


def test_sqlite_path_is_overridable(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("RAG_SQLITE_PATH", "/tmp/custom.sqlite")
    for name in ("RAG_CATALOG", "RAG_SCHEMA", "RAG_WAREHOUSE_ID"):
        monkeypatch.delenv(name, raising=False)

    assert Settings.from_env().sqlite_path == "/tmp/custom.sqlite"


def test_databricks_backend_still_requires_warehouse_config(monkeypatch):
    monkeypatch.delenv("RAG_STORAGE_BACKEND", raising=False)
    for name in ("RAG_CATALOG", "RAG_SCHEMA", "RAG_WAREHOUSE_ID"):
        monkeypatch.delenv(name, raising=False)

    import pytest

    with pytest.raises(ValueError, match="missing required configuration"):
        Settings.from_env()


def test_legacy_ollama_variables_remain_supported(monkeypatch):
    for name in ("RAG_CHAT_BASE_URL", "RAG_CHAT_MODEL", "RAG_CHAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {"RAG_CATALOG": "catalog", "RAG_SCHEMA": "schema", "RAG_WAREHOUSE_ID": "warehouse",
                        "OLLAMA_BASE_URL": "http://ollama:11434",
                        "OLLAMA_MODEL": "qwen"}.items(): monkeypatch.setenv(name, value)

    settings = Settings.from_env()

    assert (settings.chat_base_url, settings.chat_model) == (
        "http://ollama:11434", "qwen",
    )
