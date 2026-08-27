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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None
    message: dict | None = None
    """The assistant turn exactly as the runtime produced it, for appending."""


@dataclass(frozen=True)
class LLMCallUsage:
    """Usage reported by one endpoint call; unknown token counts remain null."""
    provider: str
    model: str
    operation: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int


_USAGE_CAPTURE: ContextVar[list[LLMCallUsage] | None] = ContextVar("llm_usage_capture", default=None)


@contextmanager
def capture_llm_usage():
    """Collect provider-call usage for one request without shared mutable state."""
    calls: list[LLMCallUsage] = []
    token = _USAGE_CAPTURE.set(calls)
    try:
        yield calls
    finally:
        _USAGE_CAPTURE.reset(token)


def _usage_value(usage, *names: str) -> int | None:
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return None


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


def _tool_calls_from_openai(message) -> tuple[ToolCall, ...]:
    """Normalize an OpenAI-shaped assistant turn.

    ``arguments`` arrives as a JSON string produced by the runtime's constrained
    decoder, so it is well-formed by construction rather than scraped out of
    prose. Decoding it here means callers never parse anything themselves.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise RuntimeError("the serving endpoint returned prose instead of a tool call")
    assistant = message.model_dump(exclude_none=True)
    return tuple(
        ToolCall(call.function.name, json.loads(call.function.arguments), call_id=call.id, message=assistant)
        for call in tool_calls
    )


def _tool_call_from_openai(message) -> ToolCall:
    """Return one call for single-tool consumers such as follow-up rewriting."""
    return _tool_calls_from_openai(message)[0]


class AnswerProvider(Protocol):
    name: str
    model: str
    def complete(self, prompt: str) -> str: ...
    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall: ...
    def call_tools(self, messages: list[dict], tools: list[dict]) -> tuple[ToolCall, ...]: ...


class _OpenAIChatProvider:
    """Chat-completions calls shared by every runtime that speaks the protocol."""

    model: str
    extra_body: dict

    def _client(self):
        raise NotImplementedError

    def _create(self, messages: list[dict], *, operation: str, **kwargs):
        started = perf_counter()
        completion = self._client().chat.completions.create(
            model=self.model, messages=messages, extra_body=self.extra_body, **kwargs,
        )
        captured = _USAGE_CAPTURE.get()
        if captured is not None:
            usage = getattr(completion, "usage", None)
            input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens", "prompt_eval_count")
            output_tokens = _usage_value(usage, "completion_tokens", "output_tokens", "eval_count")
            total_tokens = _usage_value(usage, "total_tokens")
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            captured.append(LLMCallUsage(
                self.name, self.model, operation, input_tokens, output_tokens, total_tokens,
                round((perf_counter() - started) * 1000),
            ))
        return completion

    def complete(self, prompt: str) -> str:
        completion = self._create([{"role": "user", "content": prompt}], operation="completion")
        return (completion.choices[0].message.content or "").strip()

    def call_tools(self, messages: list[dict], tools: list[dict]) -> tuple[ToolCall, ...]:
        # tool_choice="required" is honoured by Databricks and ignored by
        # Ollama, which is why the agent's system prompt also states the
        # contract in words. Asking for it costs nothing where it works.
        completion = self._create(messages, operation="tool_call", tools=tools,
                                  tool_choice="required", parallel_tool_calls=True)
        return _tool_calls_from_openai(completion.choices[0].message)

    def call_tool(self, messages: list[dict], tools: list[dict]) -> ToolCall:
        return self.call_tools(messages, tools)[0]


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
