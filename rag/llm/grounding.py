"""Evidence-only prompting and conservative unsupported-answer behavior."""
from __future__ import annotations

import re

from rag.llm.providers import AnswerProvider
from rag.models import Answer, Citation, RetrievalResult

LABEL = re.compile(r"\[(S\d+)\]")
SELECTED_LABEL = re.compile(r"\b(S\d+)\b")


def select_evidence(question: str, candidates: list[RetrievalResult], provider: AnswerProvider,
                    *, minimum: int = 4, maximum: int = 6) -> list[RetrievalResult]:
    """Use the local LLM only to select evidence IDs, never to answer the question."""
    labelled = "\n\n".join(
        f"S{i}: {item.chunk.source_title}\n{item.chunk.text}" for i, item in enumerate(candidates, 1)
    )
    prompt = f"""Select the {minimum} to {maximum} most relevant evidence IDs for this question.
Return only a comma-separated list like S2,S5,S7. Do not answer the question, explain, or add text.

Question: {question}

Candidates:
{labelled}"""
    labels = SELECTED_LABEL.findall(provider.complete(prompt))
    selected: list[RetrievalResult] = []
    for label in labels:
        position = int(label[1:]) - 1
        if 0 <= position < len(candidates) and candidates[position] not in selected:
            selected.append(candidates[position])
        if len(selected) == maximum:
            break
    # Invalid/insufficient selector output must not prevent a grounded answer.
    return selected if len(selected) >= minimum else candidates[:maximum]


def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    evidence = "\n\n".join(f"[S{i}] {result.chunk.source_title}\nURL: {result.chunk.source_url}\nExcerpt: {result.chunk.text}" for i, result in enumerate(results, 1))
    return f"""You are an internal Databricks documentation assistant. Answer only from the official-document excerpts below. Every factual claim must cite one or more source labels such as [S1]. Do not use background knowledge. If the excerpts do not support an answer, reply exactly: I could not verify this from the indexed official documentation.

Question: {question}

Official documentation excerpts:
{evidence}"""


def answer_groundedly(question: str, results: list[RetrievalResult], provider: AnswerProvider, *, threshold: float) -> Answer:
    snapshot_id = results[0].snapshot_id if results else "none"
    citations = tuple(Citation(f"S{i}", result.chunk.source_title, result.chunk.source_url, result.chunk.text[:500], result.chunk.chunk_id) for i, result in enumerate(results, 1))
    if not results or results[0].score < threshold:
        return Answer("I could not verify this from the indexed official documentation.", citations, False, provider.name, snapshot_id)
    evidence = select_evidence(question, results, provider)
    citations = tuple(Citation(f"S{i}", item.chunk.source_title, item.chunk.source_url, item.chunk.text[:500], item.chunk.chunk_id) for i, item in enumerate(evidence, 1))
    text = provider.complete(build_prompt(question, evidence))
    labels = set(LABEL.findall(text))
    if not labels:
        return Answer("I could not verify this from the indexed official documentation.", citations, False, provider.name, snapshot_id)
    return Answer(text, tuple(citation for citation in citations if citation.label in labels), True, provider.name, snapshot_id)
