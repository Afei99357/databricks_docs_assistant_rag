"""A bounded local retrieval agent; it may search but never answers users."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

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

    def _decide(self, question: str, evidence: list[RetrievalResult], queries: list[str]) -> dict:
        candidates = "\n".join(f"{item.chunk.chunk_id}: {item.chunk.source_title} | {item.chunk.heading_path} | {item.chunk.text[:700]}" for item in evidence)
        prompt = f'''You are a retrieval evidence reviewer. Never answer the user. Return JSON only.
Question: {question}
Searches so far: {json.dumps(queries)}
Evidence:\n{candidates}

First identify every factual aspect the user requested. For comparisons, each named side and each requested difference is a required aspect. You may return {{"action":"answer","selected_chunk_ids":["id"]}} only when the selected chunks directly support every required aspect. Include evidence for every side of a comparison, not just its dominant term. If any aspect lacks direct evidence, return {{"action":"search","query":"precise query for only the missing aspect"}}. Return {{"action":"refuse"}} only if the question cannot be supported. Select 2-10 chunks, preferring a small complete set.'''
        try:
            response = self.provider.complete(prompt).strip()
            start, end = response.find("{"), response.rfind("}")
            value = json.loads(response[start:end + 1] if start >= 0 and end >= start else response)
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def retrieve(self, question: str) -> list[RetrievalResult]:
        queries, evidence, seen = [question], [], set()
        for _ in range(self.max_searches):
            for item in self.search(queries[-1], self.candidates_per_search):
                if item.chunk.chunk_id not in seen:
                    seen.add(item.chunk.chunk_id)
                    evidence.append(item)
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
        self.last_trace = RetrievalTrace(tuple(queries), tuple(item.chunk.chunk_id for item in selected), "fallback")
        return selected
