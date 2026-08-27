"""Evidence-only prompting and conservative unsupported-answer behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from rag.llm.providers import AnswerProvider
from rag.models import Answer, Citation, RetrievalResult

CITATION = re.compile(r"\[((?:S\d+)(?:\s*,\s*S\d+)*)\]")
LABEL = re.compile(r"S\d+")


@dataclass(frozen=True)
class GroundingTrace:
    raw_model_output: str | None
    parsed_citation_labels: tuple[str, ...]
    fallback_reason: str | None


def build_prompt(
    question: str,
    results: list[RetrievalResult],
    *,
    evidence_support: tuple[dict, ...] = (),
    unverified_points: tuple[str, ...] = (),
) -> str:
    evidence = "\n\n".join(
        f"[S{i}] {result.chunk.source_title}\nURL: {result.chunk.source_url}\nExcerpt: {result.chunk.text}"
        for i, result in enumerate(results, 1)
    )
    support_by_chunk = {item["chunk_id"]: item["supports"] for item in evidence_support}
    selection = "\n".join(
        f"[S{i}] may be used only for: {'; '.join(support_by_chunk.get(result.chunk.chunk_id, ()))}"
        for i, result in enumerate(results, 1)
        if result.chunk.chunk_id in support_by_chunk
    )
    unresolved = "\n".join(f"- {point}" for point in unverified_points)
    selection_rules = (
        (
            f"\n\nRetrieval selection scope:\n{selection}\n"
            "Do not use a cited excerpt to establish a point outside its listed scope."
        )
        if selection
        else ""
    )
    unresolved_rules = (
        (
            f"\n\nThe retrieval investigation could not verify these requested points:\n{unresolved}\n"
            "State that they could not be verified; do not infer an answer for them."
        )
        if unresolved
        else ""
    )
    return f"""You are an internal Databricks documentation assistant. Answer only from the official Databricks documentation excerpts below.
Every factual claim must cite one or more source labels such as [S1]. Do not use background knowledge and do not infer a general product behavior
from an excerpt that discusses a narrower context such as benchmarks, APIs, or trusted assets. For comparison questions, state only differences that are directly supported;
explicitly say which requested differences the excerpts do not establish. If the excerpts support only part of a multi-part question,
answer that supported part with citations and explicitly say which remaining part could not be verified. Reply exactly "I could not verify this from the indexed documentation."
only when no part of the question is supported.{selection_rules}{unresolved_rules}

Question: {question}

Indexed documentation excerpts:
{evidence}"""


def answer_groundedly(
    question: str,
    results: list[RetrievalResult],
    provider: AnswerProvider,
    *,
    threshold: float,
    evidence_support: tuple[dict, ...] = (),
    unverified_points: tuple[str, ...] = (),
) -> Answer:
    return answer_groundedly_with_trace(
        question,
        results,
        provider,
        threshold=threshold,
        evidence_support=evidence_support,
        unverified_points=unverified_points,
    )[0]


def answer_groundedly_with_trace(
    question: str,
    results: list[RetrievalResult],
    provider: AnswerProvider,
    *,
    threshold: float,
    evidence_support: tuple[dict, ...] = (),
    unverified_points: tuple[str, ...] = (),
) -> tuple[Answer, GroundingTrace]:
    """Answer from the retrieval agent's evidence without re-selecting it.

    Evidence selection belongs to retrieval. A second LLM selector here can
    discard chunks chosen to cover another aspect of a multi-part question.
    """
    snapshot_id = results[0].snapshot_id if results else "none"
    citations = tuple(
        Citation(
            f"S{i}",
            result.chunk.source_title,
            result.chunk.source_url,
            result.chunk.text,
            result.chunk.chunk_id,
        )
        for i, result in enumerate(results, 1)
    )
    if not results:
        return Answer(
            "I could not verify this from the indexed documentation.",
            citations,
            False,
            provider.name,
            snapshot_id,
        ), GroundingTrace(None, (), "no_results")
    if results[0].score < threshold:
        return Answer(
            "I could not verify this from the indexed documentation.",
            citations,
            False,
            provider.name,
            snapshot_id,
        ), GroundingTrace(None, (), "below_relevance_threshold")
    raw_model_output = provider.complete(
        build_prompt(
            question,
            results,
            evidence_support=evidence_support,
            unverified_points=unverified_points,
        )
    )
    labels = tuple(
        dict.fromkeys(
            label
            for citation in CITATION.findall(raw_model_output)
            for label in LABEL.findall(citation)
        )
    )
    valid_labels = set(labels).intersection(citation.label for citation in citations)
    if not labels:
        return Answer(
            "I could not verify this from the indexed documentation.",
            citations,
            False,
            provider.name,
            snapshot_id,
        ), GroundingTrace(raw_model_output, labels, "no_citations_in_model_output")
    if not valid_labels:
        return Answer(
            "I could not verify this from the indexed documentation.",
            citations,
            False,
            provider.name,
            snapshot_id,
        ), GroundingTrace(raw_model_output, labels, "invalid_citation_labels")
    cited = tuple(citation for citation in citations if citation.label in valid_labels)
    renumbered = {citation.label: f"S{index}" for index, citation in enumerate(cited, 1)}
    text = CITATION.sub(
        lambda match: "["
        + ", ".join(renumbered.get(label, label) for label in LABEL.findall(match.group(1)))
        + "]",
        raw_model_output,
    )
    citations = tuple(replace(citation, label=renumbered[citation.label]) for citation in cited)
    return Answer(text, citations, True, provider.name, snapshot_id), GroundingTrace(
        raw_model_output, labels, None
    )
