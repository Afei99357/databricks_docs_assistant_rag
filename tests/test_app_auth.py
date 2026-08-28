import pytest
from databricks import sdk

from rag.app import auth


class FakeWorkspaceClient:
    def __init__(self, **kwargs): self.kwargs = kwargs


def test_obo_client_requires_forwarded_token(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace")
    with pytest.raises(auth.MissingForwardedTokenError):
        auth.get_user_workspace_client({})


def test_obo_client_uses_only_forwarded_token(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace")
    monkeypatch.setattr(sdk, "WorkspaceClient", FakeWorkspaceClient)
    client = auth.get_user_workspace_client({"x-forwarded-access-token": "user-token"})
    assert client.kwargs == {"host": "https://workspace", "token": "user-token", "auth_type": "pat"}
