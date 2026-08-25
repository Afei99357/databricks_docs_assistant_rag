"""Curated source loading and stable official-document identity."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import yaml
from bs4 import BeautifulSoup

from rag.ingest.fetch import fetch_page

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
        "blueprint-genie",
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


@dataclass(frozen=True)
class CrawlSource:
    name: str
    root_url: str
    category: str
    reason: str
    allowed_prefixes: tuple[str, ...]
    excluded_prefixes: tuple[str, ...]
    max_depth: int = 2
    cloud: str = "external"


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


def load_crawl_sources(yaml_path: str | Path) -> list[CrawlSource]:
    raw_entries = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise SourcesValidationError(f"{yaml_path} must contain a YAML list")
    sources = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise SourcesValidationError(f"entry {index} is not a mapping")
        name, root_url = entry.get("name"), entry.get("root_url")
        category, reason = entry.get("category"), entry.get("reason")
        allowed = entry.get("allowed_prefixes")
        excluded = entry.get("excluded_prefixes", [])
        max_depth = entry.get("max_depth", 2)
        if not isinstance(name, str) or not name:
            raise SourcesValidationError(f"entry {index} missing a name")
        if not isinstance(root_url, str) or urlsplit(root_url).scheme not in {"http", "https"}:
            raise SourcesValidationError(f"entry {index} missing an http(s) root_url")
        if category not in CATEGORIES:
            raise SourcesValidationError(f"entry {index} has unknown category {category!r}")
        if not isinstance(reason, str) or not reason:
            raise SourcesValidationError(f"entry {index} missing a non-empty reason")
        if (
            not isinstance(allowed, list)
            or not allowed
            or not all(isinstance(item, str) for item in allowed)
        ):
            raise SourcesValidationError(f"entry {index} needs a non-empty allowed_prefixes list")
        if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
            raise SourcesValidationError(f"entry {index} has invalid excluded_prefixes")
        if not isinstance(max_depth, int) or max_depth < 0 or max_depth > 5:
            raise SourcesValidationError(f"entry {index} max_depth must be an integer from 0 to 5")
        sources.append(
            CrawlSource(
                name=name,
                root_url=canonicalize_url(root_url),
                category=category,
                reason=reason,
                allowed_prefixes=tuple(allowed),
                excluded_prefixes=tuple(excluded),
                max_depth=max_depth,
                cloud=entry.get("cloud", "external"),
            )
        )
    return sources


def _crawl_path_allowed(path: str, source: CrawlSource) -> bool:
    root_path = urlsplit(source.root_url).path.rstrip("/")
    relative = path.rstrip("/")
    if relative == root_path:
        return True
    if not relative.startswith(root_path + "/"):
        return False
    suffix = relative[len(root_path) + 1 :]
    if any(
        suffix == prefix.rstrip("/") or suffix.startswith(prefix.rstrip("/") + "/")
        for prefix in source.excluded_prefixes
    ):
        return False
    return any(
        suffix == prefix.rstrip("/") or suffix.startswith(prefix.rstrip("/") + "/")
        for prefix in source.allowed_prefixes
    )


def discover_site_sections(sources: list[CrawlSource], *, fetcher=fetch_page) -> list[CuratedDoc]:
    """Discover bounded same-site documentation pages from configured roots."""
    discovered: dict[str, CuratedDoc] = {}
    for source in sources:
        root = canonicalize_url(source.root_url)
        queue: list[tuple[str, int]] = [(root, 0)]
        visited: set[str] = set()
        while queue:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            parts = urlsplit(url)
            root_parts = urlsplit(root)
            if parts.netloc != root_parts.netloc or not _crawl_path_allowed(parts.path, source):
                continue
            doc_id = compute_doc_id(url)
            discovered[doc_id] = CuratedDoc(
                requested_url=url,
                canonical_requested_url=url,
                doc_id=doc_id,
                slug=compute_slug(url),
                category=source.category,
                cloud=source.cloud,
                reason=source.reason,
                source_scope=f"crawl:{source.name}",
            )
            if depth >= source.max_depth:
                continue
            fetched = fetcher(doc_id, url)
            if fetched.outcome != "ok" or not fetched.html:
                continue
            for anchor in BeautifulSoup(fetched.html, "lxml").find_all("a", href=True):
                # Most links resolve with normal URL semantics. At a root
                # directory index, some sites omit the slash in the canonical
                # URL, so also try slash-based resolution when needed.
                standard = canonicalize_url(urljoin(url, anchor["href"]))
                standard_parts = urlsplit(standard)
                candidates = [standard]
                if not (
                    standard_parts.netloc == root_parts.netloc
                    and standard_parts.path.startswith(root_parts.path.rstrip("/") + "/")
                ):
                    candidates.append(
                        canonicalize_url(urljoin(url.rstrip("/") + "/", anchor["href"]))
                    )
                for candidate in dict.fromkeys(candidates):
                    candidate_parts = urlsplit(candidate)
                    if (
                        candidate_parts.scheme not in {"http", "https"}
                        or candidate_parts.netloc != root_parts.netloc
                    ):
                        continue
                    if candidate_parts.path.lower().endswith(
                        (
                            ".css",
                            ".js",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".gif",
                            ".svg",
                            ".pdf",
                            ".xml",
                            ".json",
                        )
                    ):
                        continue
                    if (
                        _crawl_path_allowed(candidate_parts.path, source)
                        and candidate not in visited
                    ):
                        queue.append((candidate, depth + 1))
                        break
    return list(discovered.values())
