"""Deterministic document chunking that retains source and heading context."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag.models import Chunk


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1_600
    overlap_chars: int = 240


def chunk_id(doc_id: str, document_version: str, position: int) -> str:
    value = f"{doc_id}:{document_version}:{position}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _split_text(text: str, config: ChunkingConfig) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + config.max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + config.max_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - config.overlap_chars, start + 1)
    return chunks


DEFAULT_CHUNKING = ChunkingConfig()


def chunk_document(*, doc_id: str, document_version: str, source_url: str, source_title: str,
                   nodes: list[dict], config: ChunkingConfig = DEFAULT_CHUNKING) -> list[Chunk]:
    """Chunk by section, keeping headings with each body block where possible."""
    headings: list[str] = []
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    buffer: list[str] = []
    current_path: tuple[str, ...] = ()
    for node in nodes:
        if node["type"] == "heading":
            if buffer:
                sections.append((current_path, buffer))
                buffer = []
            level = node["level"]
            headings = headings[:level - 1] + [node["text"]]
            current_path = tuple(headings)
        elif node["type"] == "paragraph":
            buffer.append(node["text"])
        elif node["type"] == "list":
            buffer.extend((f"{i}. {item}" if node["ordered"] else f"• {item}") for i, item in enumerate(node["items"], 1))
        elif node["type"] == "table":
            # A table is one semantic unit: splitting its rows across chunks
            # makes headings/columns much harder to interpret at retrieval.
            buffer.append("\n".join(" | ".join(row) for row in node["rows"]))
        elif node["type"] == "code":
            buffer.append(node["text"])
    if buffer:
        sections.append((current_path, buffer))
    output: list[Chunk] = []
    for path, units in sections:
        prefix = "\n".join(path)
        packed: list[str] = []
        packed_length = len(prefix) + 1 if prefix else 0
        texts: list[str] = []
        for unit in units:
            # Preserve paragraph/table/code units. Only a unit that cannot
            # fit on its own is split at sentence/newline/word boundaries.
            unit_parts = _split_text(unit, config) if len(unit) > config.max_chars else [unit]
            for part in unit_parts:
                added = len(part) + (1 if packed else 0)
                if packed and packed_length + added > config.max_chars:
                    texts.append("\n".join(packed))
                    overlap = _split_text(packed[-1], ChunkingConfig(config.overlap_chars, 0))[-1:] if config.overlap_chars else []
                    packed = overlap
                    packed_length = sum(len(value) for value in packed) + len(prefix) + (1 if prefix else 0)
                packed.append(part)
                packed_length += added
        if packed:
            texts.append("\n".join(packed))
        for section in texts:
            text = f"{prefix}\n{section}" if prefix else section
            position = len(output)
            output.append(Chunk(chunk_id(doc_id, document_version, position), doc_id, document_version, position,
                                text, path, source_url, source_title))
    return output
