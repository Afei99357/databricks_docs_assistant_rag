"""Evidence-only prompting and conservative unsupported-answer behavior."""
from __future__ import annotations

import re

from rag.llm.providers import AnswerProvider
from rag.models import Answer, Citation, RetrievalResult

LABEL = re.compile(r"\[(S\d+)\]")
def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    evidence = "\n\n".join(f"[S{i}] {result.chunk.source_title}\nURL: {result.chunk.source_url}\nExcerpt: {result.chunk.text}" for i, result in enumerate(results, 1))
    return f"""You are an internal Databricks documentation assistant. Answer only from the official-document excerpts below. Every factual claim must cite one or more source labels such as [S1]. Do not use background knowledge. If the excerpts do not support an answer, reply exactly: I could not verify this from the indexed official documentation.

Question: {question}

Official documentation excerpts:
{evidence}"""


def answer_groundedly(question: str, results: list[RetrievalResult], provider: AnswerProvider, *, threshold: float) -> Answer:
    """Answer from the retrieval agent's evidence without re-selecting it.

    Evidence selection belongs to retrieval. A second LLM selector here can
    discard chunks chosen to cover another aspect of a multi-part question.
    """
    snapshot_id = results[0].snapshot_id if results else "none"
    citations = tuple(Citation(f"S{i}", result.chunk.source_title, result.chunk.source_url, result.chunk.text, result.chunk.chunk_id) for i, result in enumerate(results, 1))
    if not results or results[0].score < threshold:
        return Answer("I could not verify this from the indexed official documentation.", citations, False, provider.name, snapshot_id)
    text = provider.complete(build_prompt(question, results))
    labels = set(LABEL.findall(text))
    if not labels:
        return Answer("I could not verify this from the indexed official documentation.", citations, False, provider.name, snapshot_id)
    return Answer(text, tuple(citation for citation in citations if citation.label in labels), True, provider.name, snapshot_id)
