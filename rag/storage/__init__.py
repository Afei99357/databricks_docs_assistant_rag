"""Storage-layer interfaces: protocols that shared code depends on instead of a concrete adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import Settings


def create_store(settings: Settings, *, provider=None, model=None, agent_provider=None, agent_model=None):
    """Build the storage adapter selected by ``settings.storage_backend``.

    Each branch imports its adapter module lazily -- importing at module level
    would pull in that adapter's dependencies (e.g. ``databricks-sdk``)
    regardless of which backend is actually selected.
    """
    backend = settings.storage_backend
    if backend == "databricks":
        from rag.storage.databricks import DatabricksStore

        return DatabricksStore(settings.warehouse_id, settings.databricks_profile, namespace=settings.namespace, provider=provider, model=model, agent_provider=agent_provider, agent_model=agent_model)
    if backend == "sqlite":
        from rag.storage.sqlite import SQLiteStore
        return SQLiteStore(settings.sqlite_path, provider=provider, model=model, agent_provider=agent_provider, agent_model=agent_model)
    raise ValueError(f"unknown storage_backend: {backend!r}")
