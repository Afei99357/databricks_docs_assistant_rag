"""Curated source loading and stable official-document identity."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import yaml
from bs4 import BeautifulSoup

CATEGORIES = frozenset(
    {
        "genie-concepts",
        "agent-mode",
        "unstructured-data-volumes",
        "knowledge-store-instructions",
        "benchmarks-evaluation",
        "genie-apis",
        "genie-one",
        "genie-code",
        "agent-framework",
        "managed-mcp",
        "multi-agent-apps",
        "uc-permissions-security",
        "databricks-apps",
    }
)
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
}
SLUG_STRIP_PREFIXES = ("aws-en-", "gcp-en-", "api-workspace-")
GENIE_LANDING_URL = "https://docs.databricks.com/aws/en/genie/"


class SourcesValidationError(ValueError):
    """Raised when the curated source configuration is invalid."""


@dataclass(frozen=True)
class CuratedDoc:
    requested_url: str
    canonical_requested_url: str
    doc_id: str
    slug: str
    category: str
    cloud: str
    reason: str
    source_scope: str = "supplemental"


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") if len(parts.path) > 1 else parts.path
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMS
        )
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def compute_doc_id(canonical_url: str) -> str:
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]


def compute_slug(canonical_url: str) -> str:
    path = urlsplit(canonical_url).path.strip("/")
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-") if path else "index"
    for prefix in SLUG_STRIP_PREFIXES:
        if slug.startswith(prefix):
            return slug[len(prefix) :] or "index"
    return slug or "index"


def _genie_category(url: str) -> str:
    path = urlsplit(canonicalize_url(url)).path
    if path.startswith("/aws/en/genie-one"):
        return "genie-one"
    if path.startswith("/aws/en/genie-code"):
        return "genie-code"
    if path in {"/aws/en/genie-agents/set-up", "/aws/en/genie-agents/tune-quality"}:
        return "knowledge-store-instructions"
    return "genie-concepts"


def discover_genie_core(html: str, *, landing_url: str = GENIE_LANDING_URL) -> list[CuratedDoc]:
    docs, seen = [], set()
    for anchor in BeautifulSoup(html, "lxml").find_all("a", href=True):
        if anchor["href"].startswith("#"):
            continue
        resolved = canonicalize_url(urljoin(landing_url, anchor["href"]))
        parts = urlsplit(resolved)
        if parts.netloc != "docs.databricks.com" or not parts.path.startswith("/aws/en/genie"):
            continue
        if parts.path != "/aws/en/genie" and not parts.path.startswith(
            ("/aws/en/genie/", "/aws/en/genie-agents", "/aws/en/genie-one", "/aws/en/genie-code")
        ):
            continue
        doc_id = compute_doc_id(resolved)
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(
                CuratedDoc(
                    resolved,
                    resolved,
                    doc_id,
                    compute_slug(resolved),
                    _genie_category(resolved),
                    "aws",
                    "Automatically discovered from the official Genie landing page",
                    "genie-core",
                )
            )
    return docs


def load_curated_docs(yaml_path: str | Path) -> list[CuratedDoc]:
    raw_entries = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise SourcesValidationError(f"{yaml_path} must contain a YAML list")
    docs, seen = [], {}
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise SourcesValidationError(f"entry {index} is not a mapping")
        url, category, reason = entry.get("url"), entry.get("category"), entry.get("reason")
        if not isinstance(url, str) or urlsplit(url).scheme not in {"http", "https"}:
            raise SourcesValidationError(f"entry {index} missing an http(s) url")
        if category not in CATEGORIES:
            raise SourcesValidationError(f"entry {index} has unknown category {category!r}")
        if not isinstance(reason, str) or not reason:
            raise SourcesValidationError(f"entry {index} missing a non-empty reason")
        canonical = canonicalize_url(url)
        doc_id = compute_doc_id(canonical)
        if doc_id in seen:
            raise SourcesValidationError(
                f"duplicate doc_id {doc_id} for {url!r} and {seen[doc_id]!r}"
            )
        seen[doc_id] = url
        docs.append(
            CuratedDoc(
                url,
                canonical,
                doc_id,
                compute_slug(canonical),
                category,
                entry.get("cloud", "aws"),
                reason,
            )
        )
    return docs
