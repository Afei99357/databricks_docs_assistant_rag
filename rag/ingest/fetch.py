"""HTTP retrieval for official Databricks documentation."""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class FetchResult:
    doc_id: str
    requested_url: str
    resolved_url: str | None
    http_status: int | None
    html: str | None
    error_message: str | None
    outcome: str


def fetch_page(doc_id: str, requested_url: str, timeout: float = 15.0) -> FetchResult:
    try:
        response = requests.get(requested_url, timeout=timeout)
    except requests.RequestException as exc:
        return FetchResult(doc_id, requested_url, None, None, None, str(exc), "network_error")
    outcome = "ok" if 200 <= response.status_code < 300 else "not_found" if response.status_code == 404 else "http_error"
    if outcome == "ok" and "charset" not in response.headers.get("content-type", "").lower():
        response.encoding = "utf-8"
    return FetchResult(doc_id, requested_url, response.url, response.status_code,
                       response.text if outcome == "ok" else None,
                       None if outcome == "ok" else f"HTTP {response.status_code}", outcome)

