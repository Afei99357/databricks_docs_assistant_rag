"""HTTP retrieval for official Databricks documentation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

# One pooled session keeps TCP/TLS handshakes out of a few hundred sequential
# page fetches, which is also why the docs host stops resetting connections.
_SESSION = requests.Session()

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class FetchResult:
    doc_id: str
    requested_url: str
    resolved_url: str | None
    http_status: int | None
    html: str | None
    error_message: str | None
    outcome: str


def fetch_page(
    doc_id: str,
    requested_url: str,
    timeout: float = 15.0,
    *,
    attempts: int = 4,
    backoff: float = 1.0,
) -> FetchResult:
    """Fetch one page, retrying only failures that are plausibly transient.

    A reset connection or a 503 says nothing about whether the page exists, so
    retrying is safe. A 404 is a real answer about the source and is returned
    immediately, leaving the removal decision to the lifecycle rules.
    """
    last_error: str | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(backoff * 2 ** (attempt - 1))
        try:
            response = _SESSION.get(requested_url, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
            continue
        if response.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
            last_error = f"HTTP {response.status_code}"
            continue
        outcome = (
            "ok"
            if 200 <= response.status_code < 300
            else "not_found"
            if response.status_code == 404
            else "http_error"
        )
        if outcome == "ok" and "charset" not in response.headers.get("content-type", "").lower():
            response.encoding = "utf-8"
        return FetchResult(
            doc_id,
            requested_url,
            response.url,
            response.status_code,
            response.text if outcome == "ok" else None,
            None if outcome == "ok" else f"HTTP {response.status_code}",
            outcome,
        )
    return FetchResult(doc_id, requested_url, None, None, None, last_error, "network_error")
