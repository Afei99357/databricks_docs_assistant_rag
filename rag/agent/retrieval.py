"""Tool-using evidence retrieval agent.

The agent may investigate the indexed documentation, but it never writes the
user-facing answer. Evidence selection stays separate from final grounding.

The model drives the loop through native tool calls, so the serving runtime
returns a tool name and an arguments object directly. Nothing here scavenges
JSON out of prose, and there is no repair path for malformed output: a provider
that cannot produce a tool call raises, rather than degrading quietly.

Evidence is addressed by short per-request labels (``S1``, ``S2``) rather than
by chunk ID, so a slip is an invalid *reference* the harness rejects precisely
rather than a corrupted identifier.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
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


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


LABEL = {"type": "string", "description": "A label of retrieved evidence, such as S1."}
LABELS = {"type": "array", "items": LABEL}

TOOLS = [
    _tool("search_docs", "Search the indexed documentation for evidence.",
          {"query": {"type": "string", "description": "A specific documentation search."}}, ["query"]),
    _tool("read_chunks", "Open the full text of retrieved evidence before selecting it.",
          {"labels": LABELS}, ["labels"]),
    _tool("get_related_chunks", "Read the sections immediately around an opened chunk.",
          {"label": LABEL}, ["label"]),
    _tool("search_within_document", "Search inside the document an opened chunk came from.",
          {"source": LABEL, "query": {"type": "string", "description": "A specific section search."}},
          ["source", "query"]),
    _tool("final", "Select the opened evidence that answers the question.",
          {"selected": LABELS,
           "unverified_points": {"type": "array", "items": {"type": "string"},
                                 "description": "Parts of the question the evidence does not establish."}},
          ["selected"]),
]

_WORDS = re.compile(r"[a-z0-9]+")
_SEARCH_TOOLS = frozenset({"search_docs", "search_within_document"})
_SEARCH_RESULT_STATUSES = frozenset({"ok", "low_novelty", "no_new_evidence"})


def _normal_query(value: str) -> str:
    """Order-insensitive query key.

    Reordering the same terms is a reformulation, not a new search. Ranking is
    order-sensitive enough that the model can spend several turns shuffling
    words for near-identical result sets.
    """
    return " ".join(sorted(_WORDS.findall(value.lower())))


class _Ledger:
    """Per-request label registry and score-of-record for retrieved chunks."""

    def __init__(self):
        self.labels: dict[str, str] = {}
        self.results: dict[str, RetrievalResult] = {}

    def register(self, results: list[RetrievalResult], *, ranked: bool,
                 score: float | None = None) -> list[str]:
        """Record results and return their labels, in the order supplied.

        ``ranked`` results carry a real similarity score and become the score of
        record. Unranked results (opened chunks, positional neighbours) arrive
        from the retriever with a placeholder ``1.0``; that placeholder must
        never overwrite a real score, nor enter evidence as if it outranked one.
        """
        labels = []
        for item in results:
            chunk_id = item.chunk.chunk_id
            if ranked:
                self.results[chunk_id] = item
            elif chunk_id not in self.results:
                self.results[chunk_id] = replace(item, score=item.score if score is None else score)
            if chunk_id not in self.labels:
                self.labels[chunk_id] = f"S{len(self.labels) + 1}"
            labels.append(self.labels[chunk_id])
        return labels

    def label(self, chunk_id: str) -> str:
        return self.labels[chunk_id]

    def resolve(self, value) -> str | None:
        """Map a model-supplied label back to a chunk ID."""
        if not isinstance(value, str):
            return None
        return next((chunk_id for chunk_id, label in self.labels.items() if label == value.strip()), None)


def _card(item: RetrievalResult, rank: int, label: str) -> dict:
    return {
        "label": label,
        "rank": rank,
        "score": round(item.score, 4),
        "title": item.chunk.source_title,
        "url": item.chunk.source_url,
        "heading": " > ".join(item.chunk.heading_path),
        "excerpt": item.chunk.text[:520],
    }


def _compact_card(card: dict) -> dict:
    return {"label": card["label"], "title": card["title"], "heading": card["heading"]}


class RetrievalAgent:
    """A provider-neutral, iterative documentation retrieval harness."""

    def __init__(self, tools: RetrievalTools | Callable[[str, int], list[RetrievalResult]], provider,
                 *, candidates_per_search: int = 10, deadline_seconds: float = 240,
                 max_steps: int = 12, max_evidence: int = 8, min_new_chunks: int = 2):
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
        self.max_steps, self.max_evidence = max_steps, max_evidence
        self.min_new_chunks = min_new_chunks
        self.last_trace: RetrievalTrace | None = None

    def _prompt(self, question: str, observations: list[dict]) -> str:
        return f'''You are a documentation research agent. You never answer the user directly. Investigate only through your tools, then select the final evidence.

Question: {question}

Every retrieved excerpt has a short label such as S1. Refer to evidence only by label.

Rules:
- Begin by calling search_docs; do not finalize without opened evidence.
- Search results are ranked. Prefer high-ranked, direct evidence. Use lower-ranked evidence only when it directly covers a gap.
- Read a chunk before selecting it as final evidence. Search snippets are leads, not proof.
- If evidence is incomplete, investigate the missing fact with a more specific query, inspect related chunks, or search within the relevant document.
- Never select a label that was not opened through a tool.
- Finalize as soon as the opened evidence covers the question. Reformulating a search that returns the same excerpts wastes the step budget.
- When remaining documentation cannot be found, finalize the supported evidence and list the missing item in unverified_points.
- Do not repeat a rejected or already-completed search.

Tool observations so far:
{json.dumps(self._compact(observations), ensure_ascii=False)}'''

    @staticmethod
    def _compact(observations: list[dict]) -> list[dict]:
        """Drop excerpt bodies from every search except the most recent one.

        Ranked excerpts are leads. Once the model has moved past a search, the
        excerpt text of chunks it never opened is dead weight that grows the
        prompt on every turn. Labels, titles and headings stay so earlier
        results remain addressable; opened chunk text is never touched.
        """
        latest = max((index for index, item in enumerate(observations)
                      if item.get("tool") in _SEARCH_TOOLS and item.get("status") in _SEARCH_RESULT_STATUSES),
                     default=-1)
        return [
            {**item, "results": [_compact_card(card) for card in item["results"]]}
            if index != latest and item.get("tool") in _SEARCH_TOOLS and "results" in item else item
            for index, item in enumerate(observations)
        ]

    def retrieve(self, question: str) -> list[RetrievalResult]:
        started = perf_counter()
        queries: list[str] = []
        query_keys: set[str] = set()
        observations: list[dict] = []
        ledger = _Ledger()
        opened: dict[str, RetrievalResult] = {}
        steps: list[ToolStep] = []
        stop_reason = "agent_unavailable"

        while True:
            if len(steps) >= self.max_steps:
                stop_reason = "step_budget_exhausted"
                break
            if perf_counter() - started >= self.deadline_seconds:
                stop_reason = "request_deadline_reached"
                break
            if len(steps) == self.max_steps - 1:
                # Left alone, the model keeps reformulating searches until the
                # budget runs out and the harness has to guess its evidence.
                # Spend the last step on a deliberate selection instead.
                observations.append({
                    "tool": "agent_protocol", "status": "final_step",
                    "message": "This is your last step. Return a final action selecting the opened evidence "
                               "that answers the question, and list anything still unverified.",
                })

            # A provider that cannot produce a tool call raises. There is no
            # degraded path here: a broken protocol is an outage, not an answer.
            call = self.provider.call_tool(self._prompt(question, observations), TOOLS)
            name, action = call.name, call.arguments
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
                had_prior_search = bool(queries)
                results = self.tools.retrieve(query, self.candidates_per_search)
                new = [item for item in results if item.chunk.chunk_id not in ledger.results]
                labels = ledger.register(results, ranked=True)
                query_keys.add(normalized)
                queries.append(query)
                status, message = self._novelty(new, had_prior_search)
                cards = [_card(item, rank, label) for rank, (item, label) in enumerate(zip(results, labels), 1)]
                observations.append({"tool": name, "status": status, "query": query, "results": cards, "message": message})
                steps.append(ToolStep(name, status, query=query,
                                      candidate_ids=tuple(item.chunk.chunk_id for item in results),
                                      candidate_cards=tuple(self._trace_cards(cards, results))))
                continue

            if name == "read_chunks":
                requested = action.get("labels")
                ids = self._resolve_all(ledger, requested)
                if not ids:
                    observations.append({"tool": name, "status": "rejected",
                                         "message": "labels must be labels of previously retrieved evidence, such as S1"})
                    steps.append(ToolStep(name, "rejected", detail="unknown labels"))
                    continue
                results = self.tools.read_chunks(ids)
                # Reading exposes full text but must not make a low-ranked chunk
                # look stronger than its retrieval evidence, so the ledger's
                # score of record wins over the retriever's placeholder.
                for item in results:
                    opened[item.chunk.chunk_id] = ledger.results.get(item.chunk.chunk_id, item)
                observations.append({"tool": name, "status": "ok", "chunks": [
                    {"label": ledger.label(item.chunk.chunk_id), "title": item.chunk.source_title,
                     "url": item.chunk.source_url, "heading": " > ".join(item.chunk.heading_path),
                     "text": item.chunk.text}
                    for item in results
                ]})
                steps.append(ToolStep(name, "ok", chunk_ids=tuple(ids),
                                      candidate_ids=tuple(item.chunk.chunk_id for item in results)))
                continue

            if name == "get_related_chunks":
                chunk_id = ledger.resolve(action.get("label"))
                if chunk_id is None or chunk_id not in opened:
                    observations.append({"tool": name, "status": "rejected", "message": "label must refer to an opened chunk"})
                    steps.append(ToolStep(name, "rejected", detail="chunk not opened"))
                    continue
                results = self.tools.related_chunks(chunk_id)
                # A positional neighbour has no similarity score of its own. It
                # inherits the anchor's, which is honest about why it is here.
                anchor_score = ledger.results[chunk_id].score
                ledger.register(results, ranked=False, score=anchor_score)
                for item in results:
                    opened[item.chunk.chunk_id] = ledger.results[item.chunk.chunk_id]
                observations.append({"tool": name, "status": "ok", "label": ledger.label(chunk_id), "chunks": [
                    {"label": ledger.label(item.chunk.chunk_id), "heading": " > ".join(item.chunk.heading_path),
                     "text": item.chunk.text}
                    for item in results
                ]})
                steps.append(ToolStep(name, "ok", chunk_ids=(chunk_id,),
                                      candidate_ids=tuple(item.chunk.chunk_id for item in results)))
                continue

            if name == "search_within_document":
                anchor = ledger.resolve(action.get("source"))
                query = action.get("query")
                normalized = _normal_query(query) if isinstance(query, str) else ""
                if anchor is None or not normalized:
                    observations.append({"tool": name, "status": "rejected",
                                         "message": "use the label of retrieved evidence as source and a non-empty query"})
                    steps.append(ToolStep(name, "rejected", detail="invalid document search"))
                    continue
                source_url = ledger.results[anchor].chunk.source_url
                key = f"document:{source_url}:{normalized}"
                if key in query_keys:
                    observations.append({"tool": name, "status": "rejected_duplicate",
                                         "message": "This in-document search repeats a prior query."})
                    steps.append(ToolStep(name, "rejected_duplicate", query=query))
                    continue
                had_prior_search = bool(queries)
                results = self.tools.search_within_document(source_url, query, self.candidates_per_search)
                new = [item for item in results if item.chunk.chunk_id not in ledger.results]
                labels = ledger.register(results, ranked=True)
                query_keys.add(key)
                queries.append(f"within {source_url}: {query}")
                status, message = self._novelty(new, had_prior_search)
                cards = [_card(item, rank, label) for rank, (item, label) in enumerate(zip(results, labels), 1)]
                observations.append({"tool": name, "status": status, "query": query, "source_url": source_url,
                                     "results": cards, "message": message})
                steps.append(ToolStep(name, status, query=query,
                                      candidate_ids=tuple(item.chunk.chunk_id for item in results),
                                      candidate_cards=tuple(self._trace_cards(cards, results))))
                continue

            if name == "final":
                requested = action.get("selected")
                ids = [chunk_id for chunk_id in self._resolve_all(ledger, requested) if chunk_id in opened]
                if not ids:
                    observations.append({"tool": name, "status": "rejected",
                                         "message": "final requires the labels of one or more opened chunks"})
                    steps.append(ToolStep(name, "rejected", detail="no opened evidence selected"))
                    continue
                selected = self._rank(opened[chunk_id] for chunk_id in dict.fromkeys(ids))
                unverified = action.get("unverified_points")
                status = "partial" if isinstance(unverified, list) and unverified else "answered"
                chosen_ids = tuple(item.chunk.chunk_id for item in selected)
                steps.append(ToolStep(name, "ok", selected_chunk_ids=chosen_ids))
                self.last_trace = RetrievalTrace(tuple(queries), chosen_ids, status, tuple(steps), "agent_satisfied")
                return selected

            observations.append({"tool": "agent", "status": "rejected",
                                 "message": "unknown tool; call one of the declared tools"})
            steps.append(ToolStep("agent", "rejected", detail=f"unknown tool {name!r}"))

        selected = self._rank(opened.values())
        status = "partial" if selected else "fallback"
        self.last_trace = RetrievalTrace(tuple(queries), tuple(item.chunk.chunk_id for item in selected), status,
                                          tuple(steps), stop_reason)
        return selected

    def _novelty(self, new: list[RetrievalResult], had_prior_search: bool) -> tuple[str, str | None]:
        if not new:
            return "no_new_evidence", "No new chunks were found beyond earlier searches."
        if had_prior_search and len(new) < self.min_new_chunks:
            return "low_novelty", ("This search mostly repeats evidence you already have. "
                                   "Investigate a different aspect of the question or finalize.")
        return "ok", None

    def _rank(self, results) -> list[RetrievalResult]:
        """Order evidence by score and cap it.

        The grounding layer gates on the leading result's score, so the model's
        listing order must never decide which chunk that is.
        """
        return sorted(results, key=lambda item: item.score, reverse=True)[:self.max_evidence]

    @staticmethod
    def _resolve_all(ledger: _Ledger, requested) -> list[str]:
        if not isinstance(requested, list):
            return []
        resolved = (ledger.resolve(value) for value in requested)
        return [chunk_id for chunk_id in resolved if chunk_id is not None]

    @staticmethod
    def _trace_cards(cards: list[dict], results: list[RetrievalResult]) -> list[dict]:
        """Trace records keep the chunk ID that the prompt deliberately omits."""
        return [{**card, "chunk_id": item.chunk.chunk_id} for card, item in zip(cards, results)]
