"""Answer-provider interface for local Ollama and Databricks serving endpoints.

Two call shapes, deliberately separate:

``complete`` asks for prose and is used for grounded answers, where free text
is the point. ``call_tool`` continues a tool-calling conversation and is used
by the retrieval agent. The tool call arrives as structured data from the
serving runtime, so nothing downstream scavenges JSON out of free-form text.

``call_tool`` takes the whole conversation so far and returns the assistant
turn alongside the parsed call. The caller appends that turn verbatim, which
keeps each request a strict extension of the last one and lets the serving
runtime reuse its cached prefix instead of re-reading the transcript.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None
    message: dict | None = None
    """The assistant turn exactly as the runtime produced it, for appending."""


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


def _tool_call_from_openai(message) -> ToolCall:
    """Normalize an OpenAI-shaped assistant turn.

    The OpenAI wire format carries ``arguments`` as a JSON string produced by a
    constrained decoder; Ollama hands back a dict. Each provider normalizes to a
    dict so callers never decode anything themselves.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise RuntimeError("the serving endpoint returned no tool call")
    call = tool_calls[0]
    return ToolCall(call.function.name, json.loads(call.function.arguments),
                    call_id=call.id, message=message.model_dump(exclude_none=True))


class AnswerProvider(Protocol):
    name: str
    model: str
    def complete(self, prompt: str) -> str: ...
    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall: ...


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 90):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def complete(self, prompt: str) -> str:
        response = requests.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False, "think": False}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"].strip()

    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall:
        response = requests.post(f"{self.base_url}/api/chat", json={
            "model": self.model, "stream": False, "think": False, "tools": tools,
            "messages": messages,
        }, timeout=self.timeout)
        response.raise_for_status()
        message = response.json().get("message") or {}
        calls = message.get("tool_calls")
        if not calls:
            raise RuntimeError(f"{self.model} returned prose instead of a tool call")
        function = calls[0]["function"]
        return ToolCall(function["name"], function.get("arguments") or {},
                        call_id=calls[0].get("id"), message=message)


class DatabricksEndpointProvider:
    name = "databricks"

    def __init__(self, endpoint: str, *, profile: str | None = None, timeout: float = 90):
        self.model, self.profile, self.timeout = endpoint, profile, timeout

    def _workspace(self):
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
        return WorkspaceClient(config=Config(**_client_config_kwargs(self.profile, self.timeout)))

    def complete(self, prompt: str) -> str:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        response = self._workspace().serving_endpoints.query(
            name=self.model,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        )
        return _response_text(response.choices[0].message.content)

    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall:
        # The typed serving_endpoints.query API exposes no tools/response_format
        # parameter, so tool calling goes through the OpenAI-compatible client
        # the SDK builds against the same workspace credentials.
        client = self._workspace().serving_endpoints.get_open_ai_client(timeout=self.timeout)
        completion = client.chat.completions.create(
            model=self.model, tools=tools, tool_choice="required", messages=messages,
        )
        return _tool_call_from_openai(completion.choices[0].message)
