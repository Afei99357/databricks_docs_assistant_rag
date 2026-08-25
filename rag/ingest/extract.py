"""Extract readable, reproducible text from Databricks' server-rendered docs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

ZERO_WIDTH_SPACE = "\u200b"
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([:,.;!?])")
LANGUAGE_CLASS = re.compile(r"^language-(\S+)$")


class ExtractionError(ValueError):
    """The page does not match the expected official-document structure."""


@dataclass(frozen=True)
class ExtractedDoc:
    title: str
    source_last_updated: str | None
    nodes: list[dict]
    markdown_body: str
    source_content_hash: str


def clean_text(element: Tag) -> str:
    text = re.sub(r"\s+", " ", element.get_text(separator=" ", strip=True)).strip()
    return SPACE_BEFORE_PUNCTUATION.sub(r"\1", text.replace(ZERO_WIDTH_SPACE, ""))


def _article_root(soup: BeautifulSoup) -> Tag:
    article = soup.find("article")
    root = article.find("div", class_="theme-doc-markdown") if article else None
    if root is not None:
        return root
    # Supplemental documentation sites can use a conventional semantic article
    # rather than the Docusaurus-specific Databricks container. Keep the
    # fallback narrow: it must still be an article with an actual page title,
    # which excludes navigation and page chrome.
    if article is not None and article.find("h1") is not None:
        return article
    raise ExtractionError("no supported article content container found")


def _code_text(code: Tag) -> str:
    parts: list[str] = []
    for node in code.descendants:
        if isinstance(node, Tag) and node.name == "br":
            parts.append("\n")
        elif isinstance(node, NavigableString):
            parts.append(str(node))
    return "".join(parts).rstrip("\n")


def _code_node(container: Tag, tab_label: str | None) -> dict:
    language = next((LANGUAGE_CLASS.match(c).group(1) for c in container.get("class", []) if LANGUAGE_CLASS.match(c)), None)
    code = container.find("code")
    return {"type": "code", "language": language, "tab_label": tab_label, "text": _code_text(code) if code else ""}


def _walk_tabs(container: Tag) -> Iterator[dict]:
    labels = [clean_text(tab) for tab in container.find_all("li", attrs={"role": "tab"})]
    panels = container.find_all("div", attrs={"role": "tabpanel"})
    for label, panel in zip(labels, panels) if len(labels) == len(panels) else ((None, panel) for panel in panels):
        yield from walk_blocks(panel, label)


def walk_blocks(element: Tag, tab_label: str | None = None) -> Iterator[dict]:
    """Preserve structural blocks while ignoring site chrome and decorative images."""
    for child in element.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class", [])
        if child.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            yield {"type": "heading", "level": int(child.name[1]), "text": clean_text(child)}
        elif child.name == "p" and (text := clean_text(child)):
            yield {"type": "paragraph", "text": text}
        elif "tabs-container" in classes:
            yield from _walk_tabs(child)
        elif child.name == "div" and any("codeBlockContainer" in c for c in classes):
            yield _code_node(child, tab_label)
        elif child.name == "table":
            rows = [[clean_text(cell) for cell in row.find_all(["td", "th"])] for row in child.find_all("tr")]
            yield {"type": "table", "rows": [row for row in rows if row]}
        elif child.name in {"ul", "ol"}:
            yield {"type": "list", "ordered": child.name == "ol", "items": [clean_text(item) for item in child.find_all("li", recursive=False)]}
        elif child.name != "img":
            yield from walk_blocks(child, tab_label)


def render_markdown(nodes: list[dict]) -> str:
    lines: list[str] = []
    for node in nodes:
        if node["type"] == "heading":
            lines.extend(("#" * node["level"] + " " + node["text"], ""))
        elif node["type"] == "paragraph":
            lines.extend((node["text"], ""))
        elif node["type"] == "code":
            if node["tab_label"]:
                lines.extend((f"**{node['tab_label']}**", ""))
            lines.extend((f"```{node['language'] or ''}", node["text"], "```", ""))
        elif node["type"] == "table" and node["rows"]:
            header, *body = node["rows"]
            lines.extend(("| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"))
            lines.extend("| " + " | ".join(row) + " |" for row in body)
            lines.append("")
        elif node["type"] == "list":
            lines.extend(f"{i}. {value}" if node["ordered"] else f"• {value}" for i, value in enumerate(node["items"], 1))
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def extract_content(html: str) -> ExtractedDoc:
    soup = BeautifulSoup(html, "lxml")
    root = _article_root(soup)
    title_element = root.find("h1")
    if title_element is None:
        raise ExtractionError("no h1 title found in article body")
    updated = soup.select_one("span.theme-last-updated time[datetime]")
    nodes = list(walk_blocks(root))
    payload = json.dumps(nodes, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return ExtractedDoc(clean_text(title_element), updated["datetime"] if updated else None, nodes,
                        render_markdown(nodes), hashlib.sha256(payload.encode()).hexdigest())
