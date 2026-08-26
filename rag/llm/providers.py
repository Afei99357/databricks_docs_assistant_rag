"""Answer-provider interface for local Ollama and Databricks serving endpoints."""
from __future__ import annotations

from typing import Protocol

import requests


def _response_text(content) -> str:
    """Normalize classic chat text and Responses-style content parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif getattr(item, "type", None) == "text" and isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return "\n".join(parts).strip()
    raise RuntimeError("Databricks chat endpoint returned an unsupported message-content format")


def _client_config_kwargs(profile: str | None, timeout: float) -> dict:
    """Bound a serving-endpoint call in time.

    The SDK otherwise applies its own generous defaults, so a stalled endpoint
    holds a Databricks App request open. The retrieval agent's own deadline
    cannot help: it is only checked between provider calls.
    """
    kwargs = {"http_timeout_seconds": timeout, "retry_timeout_seconds": timeout}
    if profile:
        kwargs["profile"] = profile
    return kwargs


class AnswerProvider(Protocol):
    name: str
    model: str
    def complete(self, prompt: str) -> str: ...


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 90):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def complete(self, prompt: str) -> str:
        response = requests.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False, "think": False}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"].strip()


class DatabricksEndpointProvider:
    name = "databricks"

    def __init__(self, endpoint: str, *, profile: str | None = None, timeout: float = 90):
        self.model, self.profile, self.timeout = endpoint, profile, timeout

    def complete(self, prompt: str) -> str:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        client = WorkspaceClient(config=Config(**_client_config_kwargs(self.profile, self.timeout)))
        response = client.serving_endpoints.query(
            name=self.model,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        )
        return _response_text(response.choices[0].message.content)
