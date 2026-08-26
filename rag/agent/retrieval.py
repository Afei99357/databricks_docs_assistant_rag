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


def _opened_card(ledger: _Ledger, item: RetrievalResult) -> dict:
    return {"label": ledger.label(item.chunk.chunk_id), "title": item.chunk.source_title,
            "url": item.chunk.source_url, "heading": " > ".join(item.chunk.heading_path),
            "text": item.chunk.text}


def _trace_cards(cards: list[dict], results: list[RetrievalResult]) -> list[dict]:
    """Trace records keep the chunk ID that the prompt deliberately omits."""
    return [{**card, "chunk_id": item.chunk.chunk_id} for card, item in zip(cards, results)]


def _resolve_all(ledger: _Ledger, requested) -> list[str]:
    if not isinstance(requested, list):
        return []
    resolved = (ledger.resolve(value) for value in requested)
    return [chunk_id for chunk_id in resolved if chunk_id is not None]


def _compact(observations: list[dict]) -> list[dict]:
    """Drop excerpt bodies from every search except the most recent one.

    Ranked excerpts are leads. Once the model has moved past a search, the
    excerpt text of chunks it never opened is dead weight that grows the prompt
    on every turn. Labels, titles and headings stay so earlier results remain
    addressable; opened chunk text is never touched.
    """
    latest = max((index for index, item in enumerate(observations)
                  if item.get("tool") in _SEARCH_TOOLS and item.get("status") in _SEARCH_RESULT_STATUSES),
                 default=-1)
    return [
        {**item, "results": [_compact_card(card) for card in item["results"]]}
        if index != latest and item.get("tool") in _SEARCH_TOOLS and "results" in item else item
        for index, item in enumerate(observations)
    ]


def _reject(name: str, message: str, detail: str) -> Outcome:
    """Refuse a tool call and tell the model why, in one place."""
    return Outcome(ToolStep(name, "rejected", detail=detail),
                   {"tool": name, "status": "rejected", "message": message})


def _reject_duplicate(name: str, query, message: str) -> Outcome:
    return Outcome(ToolStep(name, "rejected_duplicate", query=query),
                   {"tool": name, "status": "rejected_duplicate", "query": query, "message": message})


FINAL_STEP_MESSAGE = ("This is your last step. Return a final action selecting the opened evidence "
                      "that answers the question, and list anything still unverified.")



@dataclass(frozen=True)
class Outcome:
    """What executing one tool call produced.

    Handlers return this instead of mutating shared state and calling
    ``continue``. The loop records every outcome the same way, so a handler
    cannot forget to leave a trace step or an observation behind.

    ``evidence`` is ``None`` while the investigation continues; a non-``None``
    value ends it, so the loop never infers termination from a tool name.
    """
    step: ToolStep
    observation: dict | None = None
    evidence: list[RetrievalResult] | None = None
    trace_status: str | None = None


class _Session:
    """Everything one investigation accumulates.

    The loop and every handler share exactly this object. Adding a tool means
    writing a handler, not threading another local through the loop.
    """

    def __init__(self):
        self.ledger = _Ledger()
        self.opened: dict[str, RetrievalResult] = {}
        self.queries: list[str] = []
        self.observations: list[dict] = []
        self.steps: list[ToolStep] = []
        self._query_keys: set[str] = set()

    @property
    def has_searched(self) -> bool:
        return bool(self.queries)

    def repeats_search(self, key: str) -> bool:
        return key in self._query_keys

    def note_search(self, key: str, query: str) -> None:
        self._query_keys.add(key)
        self.queries.append(query)

    def mark_opened(self, results: list[RetrievalResult]) -> None:
        """Open chunks at their score of record.

        Reading exposes full text but must not make a low-ranked chunk look
        stronger than its retrieval evidence, so the ledger's score wins over
        the retriever's placeholder.
        """
        for item in results:
            self.opened[item.chunk.chunk_id] = self.ledger.results.get(item.chunk.chunk_id, item)

    def notice(self, status: str, message: str) -> None:
        self.observations.append({"tool": "agent_protocol", "status": status, "message": message})

    def record(self, outcome: Outcome) -> None:
        self.steps.append(outcome.step)
        if outcome.observation is not None:
            self.observations.append(outcome.observation)


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
        self._handlers = {
            "search_docs": self._search_docs,
            "read_chunks": self._read_chunks,
            "get_related_chunks": self._get_related_chunks,
            "search_within_document": self._search_within_document,
            "final": self._final,
        }

    # -- the loop -----------------------------------------------------------

    def retrieve(self, question: str) -> list[RetrievalResult]:
        session = _Session()
        started = perf_counter()
        while True:
            stop_reason = self._stop_reason(session, started)
            if stop_reason:
                return self._conclude(session, stop_reason)
            if len(session.steps) == self.max_steps - 1:
                session.notice("final_step", FINAL_STEP_MESSAGE)
            # A provider that cannot produce a tool call raises. There is no
            # degraded path here: a broken protocol is an outage, not an answer.
            call = self.provider.call_tool(self._prompt(question, session.observations), TOOLS)
            outcome = self._dispatch(session, call)
            session.record(outcome)
            if outcome.evidence is not None:
                return self._conclude(session, "agent_satisfied", outcome)

    def _stop_reason(self, session: _Session, started: float) -> str | None:
        if len(session.steps) >= self.max_steps:
            return "step_budget_exhausted"
        if perf_counter() - started >= self.deadline_seconds:
            return "request_deadline_reached"
        return None

    def _dispatch(self, session: _Session, call) -> Outcome:
        handler = self._handlers.get(call.name)
        if handler is None:
            return _reject("agent", "unknown tool; call one of the declared tools",
                           f"unknown tool {call.name!r}")
        return handler(session, call.arguments)

    def _conclude(self, session: _Session, stop_reason: str, outcome: Outcome | None = None) -> list[RetrievalResult]:
        """Build the trace and return evidence from the one place that does so."""
        if outcome is not None:
            selected, status = outcome.evidence, outcome.trace_status
        else:
            selected = self._rank(session.opened.values())
            status = "partial" if selected else "fallback"
        self.last_trace = RetrievalTrace(
            tuple(session.queries), tuple(item.chunk.chunk_id for item in selected),
            status, tuple(session.steps), stop_reason,
        )
        return selected

    # -- tool handlers ------------------------------------------------------

    def _search_docs(self, session: _Session, arguments: dict) -> Outcome:
        query = arguments.get("query")
        key = _normal_query(query) if isinstance(query, str) else ""
        if not key:
            return _reject("search_docs", "query must be a non-empty string", "empty query")
        if session.repeats_search(key):
            return _reject_duplicate("search_docs", query,
                                     "This search repeats a prior query. Make it more specific or finalize.")
        return self._searched(session, "search_docs", query, key,
                              self.tools.retrieve(query, self.candidates_per_search))

    def _search_within_document(self, session: _Session, arguments: dict) -> Outcome:
        anchor = session.ledger.resolve(arguments.get("source"))
        query = arguments.get("query")
        normalized = _normal_query(query) if isinstance(query, str) else ""
        if anchor is None or not normalized:
            return _reject("search_within_document",
                           "use the label of retrieved evidence as source and a non-empty query",
                           "invalid document search")
        source_url = session.ledger.results[anchor].chunk.source_url
        key = f"document:{source_url}:{normalized}"
        if session.repeats_search(key):
            return _reject_duplicate("search_within_document", query,
                                     "This in-document search repeats a prior query.")
        results = self.tools.search_within_document(source_url, query, self.candidates_per_search)
        return self._searched(session, "search_within_document", query, key, results,
                              recorded_query=f"within {source_url}: {query}", source_url=source_url)

    def _searched(self, session: _Session, name: str, query: str, key: str,
                  results: list[RetrievalResult], *, recorded_query: str | None = None, **fields) -> Outcome:
        """Shared tail of both searches: register, label, judge novelty, report."""
        had_prior_search = session.has_searched
        new = [item for item in results if item.chunk.chunk_id not in session.ledger.results]
        labels = session.ledger.register(results, ranked=True)
        session.note_search(key, recorded_query or query)
        status, message = self._novelty(new, had_prior_search)
        cards = [_card(item, rank, label) for rank, (item, label) in enumerate(zip(results, labels), 1)]
        return Outcome(
            ToolStep(name, status, query=query,
                     candidate_ids=tuple(item.chunk.chunk_id for item in results),
                     candidate_cards=tuple(_trace_cards(cards, results))),
            {"tool": name, "status": status, "query": query, "results": cards, "message": message, **fields},
        )

    def _read_chunks(self, session: _Session, arguments: dict) -> Outcome:
        ids = _resolve_all(session.ledger, arguments.get("labels"))
        if not ids:
            return _reject("read_chunks",
                           "labels must be labels of previously retrieved evidence, such as S1",
                           "unknown labels")
        results = self.tools.read_chunks(ids)
        session.mark_opened(results)
        return Outcome(
            ToolStep("read_chunks", "ok", chunk_ids=tuple(ids),
                     candidate_ids=tuple(item.chunk.chunk_id for item in results)),
            {"tool": "read_chunks", "status": "ok",
             "chunks": [_opened_card(session.ledger, item) for item in results]},
        )

    def _get_related_chunks(self, session: _Session, arguments: dict) -> Outcome:
        chunk_id = session.ledger.resolve(arguments.get("label"))
        if chunk_id is None or chunk_id not in session.opened:
            return _reject("get_related_chunks", "label must refer to an opened chunk", "chunk not opened")
        results = self.tools.related_chunks(chunk_id)
        # A positional neighbour has no similarity score of its own. It inherits
        # the anchor's, which is honest about why it is here.
        session.ledger.register(results, ranked=False, score=session.ledger.results[chunk_id].score)
        session.mark_opened(results)
        return Outcome(
            ToolStep("get_related_chunks", "ok", chunk_ids=(chunk_id,),
                     candidate_ids=tuple(item.chunk.chunk_id for item in results)),
            {"tool": "get_related_chunks", "status": "ok", "label": session.ledger.label(chunk_id),
             "chunks": [_opened_card(session.ledger, item) for item in results]},
        )

    def _final(self, session: _Session, arguments: dict) -> Outcome:
        ids = [chunk_id for chunk_id in _resolve_all(session.ledger, arguments.get("selected"))
               if chunk_id in session.opened]
        if not ids:
            return _reject("final", "final requires the labels of one or more opened chunks",
                           "no opened evidence selected")
        selected = self._rank(session.opened[chunk_id] for chunk_id in dict.fromkeys(ids))
        unverified = arguments.get("unverified_points")
        return Outcome(
            ToolStep("final", "ok", selected_chunk_ids=tuple(item.chunk.chunk_id for item in selected)),
            evidence=selected,
            trace_status="partial" if isinstance(unverified, list) and unverified else "answered",
        )

    # -- policy -------------------------------------------------------------

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

    # -- prompt -------------------------------------------------------------

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
{json.dumps(_compact(observations), ensure_ascii=False)}'''
