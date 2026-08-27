from rag.config import Settings


def test_volume_path_uses_catalog_and_schema_path_segments():
    settings = Settings(
        "catalog", "schema", "warehouse", "artifacts", "embedding", 25, 0.35,
        "ollama", "http://localhost", "qwen", "endpoint", None,
        None, None, None, None, None, None,
    )

    assert settings.volume_path == "/Volumes/catalog/schema/artifacts"
