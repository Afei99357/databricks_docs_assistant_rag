from rag.config import Settings


def test_volume_path_uses_catalog_and_schema_path_segments():
    settings = Settings(
        "catalog", "schema", "warehouse", "artifacts", "embedding", 10, 0.35,
        "http://localhost", "model", None, "http://localhost", None,
        None, None, None,
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
