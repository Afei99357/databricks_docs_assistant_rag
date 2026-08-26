"""Tool-using evidence retrieval agent.

The agent may investigate the indexed documentation, but it never writes the
user-facing answer. Evidence selection stays separate from final grounding.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from rag.models import RetrievalResult


class RetrievalTools(Protocol):
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]: ...
    def read_chunks(self, chunk_ids: list[str]) -> list[RetrievalResult]: ...
    def related_chunks(self, chunk_id: str, *, radius: int = 1) -> list[RetrievalResult]: ...
    def search_within_document(self, source_url: str, question: str, top_k: int) -> list[RetrievalResult]: ...


@dataclass(frozen=True)
class ToolStep:
    action: str
    status: str
    query: str | None = None
    chunk_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    selected_chunk_ids: tuple[str, ...] = ()
    candidate_cards: tuple[dict, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class RetrievalTrace:
    queries: tuple[str, ...]
    selected_chunk_ids: tuple[str, ...]
    status: str
    steps: tuple[ToolStep, ...] = ()
    stop_reason: str | None = None


_WORDS = re.compile(r"[a-z0-9]+")


def _normal_query(value: str) -> str:
    return " ".join(_WORDS.findall(value.lower()))


def _result_card(item: RetrievalResult, rank: int) -> dict:
    return {
        "rank": rank,
        "score": round(item.score, 4),
        "chunk_id": item.chunk.chunk_id,
        "title": item.chunk.source_title,
        "url": item.chunk.source_url,
        "heading": " > ".join(item.chunk.heading_path),
        "excerpt": item.chunk.text[:520],
    }


class RetrievalAgent:
    """A provider-neutral, iterative documentation retrieval harness."""

    def __init__(self, tools: RetrievalTools | Callable[[str, int], list[RetrievalResult]], provider,
                 *, candidates_per_search: int = 10, deadline_seconds: float = 240):
        # The callable adapter keeps small unit-test fakes usable. Production
        # supplies an ActiveSnapshotRetriever or VolumeSnapshotRetriever.
        if callable(tools) and not hasattr(tools, "retrieve"):
            search = tools

            class _SearchOnlyTools:
                def retrieve(self, question, top_k=None): return search(question, top_k or candidates_per_search)
                def read_chunks(self, chunk_ids): return []
                def related_chunks(self, chunk_id, *, radius=1): return []
                def search_within_document(self, source_url, question, top_k): return []

            tools = _SearchOnlyTools()
        self.tools, self.provider = tools, provider
        self.candidates_per_search, self.deadline_seconds = candidates_per_search, deadline_seconds
        self.last_trace: RetrievalTrace | None = None

    def _prompt(self, question: str, observations: list[dict]) -> str:
        return f'''You are a documentation research agent. You never answer the user directly. Investigate only through the tools below, then return a final evidence selection.

Question: {question}

Tools (return JSON only, with exactly one action):
1. {{"action":"search_docs","query":"specific documentation search"}}
2. {{"action":"read_chunks","chunk_ids":["retrieved chunk id"]}}
3. {{"action":"get_related_chunks","chunk_id":"opened chunk id"}}
4. {{"action":"search_within_document","source_url":"URL from retrieved evidence","query":"specific section search"}}
5. {{"action":"final","selected_chunk_ids":["opened chunk id"],"unverified_points":["optional unsupported part"]}}

Rules:
- Begin by calling search_docs; do not finalize without opened evidence.
- Search results are ranked. Prefer high-ranked, direct evidence. Use lower-ranked evidence only when it directly covers a gap.
- Read a chunk before selecting it as final evidence. Search snippets are leads, not proof.
- If evidence is incomplete, investigate the missing fact with a more specific query, inspect related chunks, or search within the relevant document.
- Never cite a chunk ID that was not opened through a tool.
- When remaining documentation cannot be found, finalize the supported evidence and list the missing item in unverified_points.
- Do not repeat a rejected or already-completed search. Return JSON only.

Tool observations so far:
{json.dumps(observations, ensure_ascii=False)}'''

    @staticmethod
    def _parse_action(response: str) -> dict:
        response = response.strip()
        start, end = response.find("{"), response.rfind("}")
        value = json.loads(response[start:end + 1] if start >= 0 and end >= start else response)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _result_map(results: list[RetrievalResult]) -> dict[str, RetrievalResult]:
        return {item.chunk.chunk_id: item for item in results}

    def retrieve(self, question: str) -> list[RetrievalResult]:
        started = perf_counter()
        queries: list[str] = []
        query_keys: set[str] = set()
        observations: list[dict] = []
        discovered: dict[str, RetrievalResult] = {}
        opened: dict[str, RetrievalResult] = {}
        steps: list[ToolStep] = []
        stop_reason = "agent_unavailable"

        while perf_counter() - started < self.deadline_seconds:
            try:
                action = self._parse_action(self.provider.complete(self._prompt(question, observations)))
            except Exception as exc:  # noqa: BLE001 - provider implementations expose no common error base.
                stop_reason = "provider_error"
                steps.append(ToolStep("agent", "error", detail=str(exc)))
                break

            name = action.get("action")
            if name == "search_docs":
                query = action.get("query")
                normalized = _normal_query(query) if isinstance(query, str) else ""
                if not normalized:
                    observations.append({"tool": name, "status": "rejected", "message": "query must be a non-empty string"})
                    steps.append(ToolStep(name, "rejected", detail="empty query"))
                    continue
                if normalized in query_keys:
                    observations.append({"tool": name, "status": "rejected_duplicate", "query": query,
                                         "message": "This search repeats a prior query. Make it more specific or finalize."})
                    steps.append(ToolStep(name, "rejected_duplicate", query=query))
                    continue
                results = self.tools.retrieve(query, self.candidates_per_search)
                query_keys.add(normalized)
                queries.append(query)
                new = [item for item in results if item.chunk.chunk_id not in discovered]
                discovered.update(self._result_map(results))
                observations.append({"tool": name, "status": "ok" if new else "no_new_evidence", "query": query,
                                     "results": [_result_card(item, rank) for rank, item in enumerate(results, 1)],
                                     "message": None if new else "No new chunks were found beyond earlier searches."})
                steps.append(ToolStep(name, "ok" if new else "no_new_evidence", query=query,
                                      candidate_ids=tuple(item.chunk.chunk_id for item in results),
                                      candidate_cards=tuple(_result_card(item, rank) for rank, item in enumerate(results, 1))))
                continue

            if name == "read_chunks":
                requested = action.get("chunk_ids")
                ids = [value for value in requested if isinstance(value, str) and value in discovered] if isinstance(requested, list) else []
                if not ids:
                    observations.append({"tool": name, "status": "rejected", "message": "chunk_ids must be previously retrieved IDs"})
                    steps.append(ToolStep(name, "rejected", detail="unknown chunk IDs"))
                    continue
                results = self.tools.read_chunks(ids)
                # Keep the search score/rank attached to the discovered result;
                # reading exposes full text but must not make a low-ranked chunk
                # look artificially stronger than its retrieval evidence.
                for item in results:
                    opened[item.chunk.chunk_id] = discovered.get(item.chunk.chunk_id, item)
                observations.append({"tool": name, "status": "ok", "chunks": [
                    {"chunk_id": item.chunk.chunk_id, "title": item.chunk.source_title, "url": item.chunk.source_url,
                     "heading": " > ".join(item.chunk.heading_path), "text": item.chunk.text}
                    for item in results
                ]})
                steps.append(ToolStep(name, "ok", chunk_ids=tuple(ids), candidate_ids=tuple(item.chunk.chunk_id for item in results)))
                continue

            if name == "get_related_chunks":
                chunk_id = action.get("chunk_id")
                if not isinstance(chunk_id, str) or chunk_id not in opened:
                    observations.append({"tool": name, "status": "rejected", "message": "chunk_id must be an opened chunk"})
                    steps.append(ToolStep(name, "rejected", detail="chunk not opened"))
                    continue
                results = self.tools.related_chunks(chunk_id)
                discovered.update(self._result_map(results))
                opened.update(self._result_map(results))
                observations.append({"tool": name, "status": "ok", "chunk_id": chunk_id, "chunks": [
                    {"chunk_id": item.chunk.chunk_id, "heading": " > ".join(item.chunk.heading_path), "text": item.chunk.text}
                    for item in results
                ]})
                steps.append(ToolStep(name, "ok", chunk_ids=(chunk_id,), candidate_ids=tuple(item.chunk.chunk_id for item in results)))
                continue

            if name == "search_within_document":
                source_url, query = action.get("source_url"), action.get("query")
                valid_url = isinstance(source_url, str) and any(item.chunk.source_url == source_url for item in discovered.values())
                normalized = _normal_query(query) if isinstance(query, str) else ""
                key = f"document:{source_url}:{normalized}" if valid_url else ""
                if not valid_url or not normalized:
                    observations.append({"tool": name, "status": "rejected", "message": "use a retrieved source_url and a non-empty query"})
                    steps.append(ToolStep(name, "rejected", detail="invalid document search"))
                    continue
                if key in query_keys:
                    observations.append({"tool": name, "status": "rejected_duplicate", "message": "This in-document search repeats a prior query."})
                    steps.append(ToolStep(name, "rejected_duplicate", query=query))
                    continue
                results = self.tools.search_within_document(source_url, query, self.candidates_per_search)
                query_keys.add(key)
                queries.append(f"within {source_url}: {query}")
                new = [item for item in results if item.chunk.chunk_id not in discovered]
                discovered.update(self._result_map(results))
                observations.append({"tool": name, "status": "ok" if new else "no_new_evidence", "query": query,
                                     "source_url": source_url, "results": [_result_card(item, rank) for rank, item in enumerate(results, 1)]})
                steps.append(ToolStep(name, "ok" if new else "no_new_evidence", query=query,
                                      candidate_ids=tuple(item.chunk.chunk_id for item in results),
                                      candidate_cards=tuple(_result_card(item, rank) for rank, item in enumerate(results, 1))))
                continue

            if name == "final":
                requested = action.get("selected_chunk_ids")
                ids = [value for value in requested if isinstance(value, str) and value in opened] if isinstance(requested, list) else []
                if not ids:
                    observations.append({"tool": name, "status": "rejected", "message": "final requires one or more opened chunk IDs"})
                    steps.append(ToolStep(name, "rejected", detail="no opened evidence selected"))
                    continue
                unique_ids = list(dict.fromkeys(ids))
                unverified = action.get("unverified_points")
                status = "partial" if isinstance(unverified, list) and unverified else "answered"
                steps.append(ToolStep(name, "ok", selected_chunk_ids=tuple(unique_ids)))
                self.last_trace = RetrievalTrace(tuple(queries), tuple(unique_ids), status, tuple(steps), "agent_satisfied")
                return [opened[chunk_id] for chunk_id in unique_ids]

            observations.append({"tool": "agent", "status": "rejected", "message": "unknown action; use one documented tool or final"})
            steps.append(ToolStep("agent", "rejected", detail=f"unknown action {name!r}"))

        if perf_counter() - started >= self.deadline_seconds:
            stop_reason = "request_deadline_reached"
        selected = list(opened.values())
        status = "partial" if selected else "fallback"
        self.last_trace = RetrievalTrace(tuple(queries), tuple(item.chunk.chunk_id for item in selected), status,
                                          tuple(steps), stop_reason)
        return selected
