"""Databricks Apps OBO boundary; never fall back to app credentials for identity."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

_FORWARDED_TOKEN_HEADER = "x-forwarded-access-token"


class MissingForwardedTokenError(RuntimeError):
    pass


def get_user_workspace_client(headers: Mapping[str, str]) -> WorkspaceClient:
    from databricks.sdk import WorkspaceClient

    token = headers.get(_FORWARDED_TOKEN_HEADER)
    if not token:
        raise MissingForwardedTokenError("Databricks App user authorization is required.")
    host = os.getenv("DATABRICKS_HOST", "")
    if not host:
        raise RuntimeError("DATABRICKS_HOST is required in the Databricks App runtime.")
    return WorkspaceClient(host=host, token=token, auth_type="pat")
