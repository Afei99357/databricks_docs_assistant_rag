"""Identity boundary: never trust a browser-supplied user ID."""
from __future__ import annotations

import os


class LocalTestIdentityProvider:
    def __init__(self, user_id: str | None = None):
        self.user_id = user_id or os.getenv("RAG_LOCAL_TEST_USER_ID")
        if not self.user_id:
            raise ValueError("RAG_LOCAL_TEST_USER_ID is required in local history mode")

    def current_user_id(self, request) -> str:
        return self.user_id


class DatabricksAppIdentityProvider:
    """Resolve a real Databricks App caller from the forwarded OBO token."""
    def current_user_id(self, request) -> str:
        from rag.app.auth import get_user_workspace_client
        user = get_user_workspace_client(request.headers).current_user.me()
        user_id = getattr(user, "user_name", None) or getattr(user, "id", None)
        if not user_id:
            raise PermissionError("Databricks App caller identity is unavailable")
        return str(user_id)
