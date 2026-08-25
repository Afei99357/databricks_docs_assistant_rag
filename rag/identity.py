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
    """Placeholder for the App-auth implementation; never falls back to request JSON."""
    def current_user_id(self, request) -> str:
        user_id = request.headers.get("X-Forwarded-User")
        if not user_id:
            raise PermissionError("authenticated Databricks App user identity is required")
        return user_id
