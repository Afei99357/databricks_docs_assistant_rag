"""Answer-provider interface for local Ollama and Databricks serving endpoints.

Both providers speak the OpenAI chat-completions protocol, so the request and
response handling lives in one place and only the client construction differs:
Ollama exposes ``/v1`` directly, while Databricks builds a client against the
workspace credentials. Talking one protocol keeps the two runtimes at parity —
a change to the conversation or the tool schemas cannot work on one and quietly
break the other.

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


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None
    message: dict | None = None
    """The assistant turn exactly as the runtime produced it, for appending."""


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

    ``arguments`` arrives as a JSON string produced by the runtime's constrained
    decoder, so it is well-formed by construction rather than scraped out of
    prose. Decoding it here means callers never parse anything themselves.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise RuntimeError("the serving endpoint returned prose instead of a tool call")
    call = tool_calls[0]
    return ToolCall(call.function.name, json.loads(call.function.arguments),
                    call_id=call.id, message=message.model_dump(exclude_none=True))


class AnswerProvider(Protocol):
    name: str
    model: str
    def complete(self, prompt: str) -> str: ...
    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall: ...


class _OpenAIChatProvider:
    """Chat-completions calls shared by every runtime that speaks the protocol."""

    model: str
    extra_body: dict

    def _client(self):
        raise NotImplementedError

    def _create(self, messages: list[dict], **kwargs):
        return self._client().chat.completions.create(
            model=self.model, messages=messages, extra_body=self.extra_body, **kwargs,
        )

    def complete(self, prompt: str) -> str:
        completion = self._create([{"role": "user", "content": prompt}])
        return (completion.choices[0].message.content or "").strip()

    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall:
        # tool_choice="required" is honoured by Databricks and ignored by
        # Ollama, which is why the agent's system prompt also states the
        # contract in words. Asking for it costs nothing where it works.
        completion = self._create(messages, tools=tools, tool_choice="required")
        return _tool_call_from_openai(completion.choices[0].message)


class OllamaProvider(_OpenAIChatProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 90):
        self.model, self.timeout = model, timeout
        self.base_url = base_url.rstrip("/") + "/v1"
        # Ollama exposes its reasoning toggle as a vendor extension rather than
        # an OpenAI parameter. Leaving it on roughly doubles per-turn latency
        # without measurably improving tool selection.
        self.extra_body = {"think": False}
        self._cached = None

    def _client(self):
        if self._cached is None:
            from openai import OpenAI
            # Ollama ignores the key but the client insists on one.
            self._cached = OpenAI(base_url=self.base_url, api_key="ollama", timeout=self.timeout)
        return self._cached


class OpenAICompatibleProvider(_OpenAIChatProvider):
    """A local or self-hosted OpenAI-compatible chat endpoint.

    This is intentionally separate from Ollama: it lets the retrieval agent
    use a stronger tool-calling model while the local answer generator remains
    on the configured Ollama model.
    """

    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, *, api_key: str = "local", timeout: float = 120):
        self.model, self.timeout = model, timeout
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key, self.extra_body, self._cached = api_key, {}, None

    def _client(self):
        if self._cached is None:
            from openai import OpenAI
            self._cached = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._cached


class DatabricksEndpointProvider(_OpenAIChatProvider):
    name = "databricks"

    def __init__(self, endpoint: str, *, profile: str | None = None, timeout: float = 90):
        self.model, self.profile, self.timeout = endpoint, profile, timeout
        self.extra_body = {}
        self._cached = None

    def _client(self):
        if self._cached is None:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.core import Config
            workspace = WorkspaceClient(config=Config(**_client_config_kwargs(self.profile, self.timeout)))
            self._cached = workspace.serving_endpoints.get_open_ai_client(timeout=self.timeout)
        return self._cached
