"""A bounded local retrieval agent; it may search but never answers users."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from rag.models import RetrievalResult


@dataclass(frozen=True)
class RetrievalTrace:
    queries: tuple[str, ...]
    selected_chunk_ids: tuple[str, ...]
    status: str


class RetrievalAgent:
    def __init__(self, search: Callable[[str, int], list[RetrievalResult]], provider, *, max_searches: int = 5, candidates_per_search: int = 12):
        self.search, self.provider = search, provider
        self.max_searches, self.candidates_per_search = max_searches, candidates_per_search
        self.last_trace: RetrievalTrace | None = None

    def _plan_aspect_queries(self, question: str) -> list[str]:
        """Create bounded, separately searchable aspects for compound questions."""
        prompt = f'''You are a retrieval query planner. Never answer the user. Return JSON only.
Question: {question}

Return {{"queries":["..."]}} with 0-4 concise searches for distinct factual aspects of the question.
For example, a question asking what a feature is and its limitations needs one definition query and one limitations query. Return [] for a single-aspect question.'''
        try:
            value = json.loads(self.provider.complete(prompt))
            queries = value.get("queries", []) if isinstance(value, dict) else []
            return [query.strip() for query in queries if isinstance(query, str) and query.strip() and query.strip() != question][:4]
        except (json.JSONDecodeError, TypeError):
            return []

    def _decide(self, question: str, evidence: list[RetrievalResult], queries: list[str]) -> dict:
        candidates = "\n".join(f"{item.chunk.chunk_id}: {item.chunk.source_title} | {item.chunk.heading_path} | {item.chunk.text[:700]}" for item in evidence)
        prompt = f'''You are a retrieval evidence reviewer. Never answer the user. Return JSON only.
Question: {question}
Searches so far: {json.dumps(queries)}
Evidence:\n{candidates}

First identify every factual aspect the user requested. For comparisons, each named side and each requested difference is a required aspect. The search plan above includes separate aspect queries; you may return {{"action":"answer","selected_chunk_ids":["id"]}} only when the selected chunks directly support every required aspect and each planned aspect query. Include evidence for every side of a comparison, not just its dominant term. Exclude chunks that mention a term only in a narrow context (for example, benchmark evaluation, API behavior, or trusted assets) unless the user asked about that context. If any aspect lacks direct evidence, return {{"action":"search","query":"precise query for only the missing aspect"}}. Return {{"action":"refuse"}} only if the question cannot be supported. Select 2-10 chunks, preferring a small complete set.'''
        try:
            response = self.provider.complete(prompt).strip()
            start, end = response.find("{"), response.rfind("}")
            value = json.loads(response[start:end + 1] if start >= 0 and end >= start else response)
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def retrieve(self, question: str) -> list[RetrievalResult]:
        queries, evidence, seen = [question, *self._plan_aspect_queries(question)], [], set()
        query_index = 0
        for _ in range(self.max_searches):
            query = queries[query_index]
            query_index += 1
            for item in self.search(query, self.candidates_per_search):
                if item.chunk.chunk_id not in seen:
                    seen.add(item.chunk.chunk_id)
                    evidence.append(item)
            # Search every planned aspect before accepting a premature answer.
            if query_index < len(queries):
                continue
            decision = self._decide(question, evidence, queries)
            if decision.get("action") == "answer":
                ids = set(decision.get("selected_chunk_ids", []))
                selected = [item for item in evidence if item.chunk.chunk_id in ids][:10]
                if selected:
                    self.last_trace = RetrievalTrace(tuple(queries), tuple(item.chunk.chunk_id for item in selected), "answered")
                    return selected
            next_query = decision.get("query") if decision.get("action") == "search" else None
            if not isinstance(next_query, str) or not next_query.strip() or next_query in queries:
                break
            queries.append(next_query.strip())
        selected = evidence[:min(4, len(evidence))]
        status = "partial" if evidence else "fallback"
        self.last_trace = RetrievalTrace(tuple(queries), tuple(item.chunk.chunk_id for item in selected), status)
        return selected
