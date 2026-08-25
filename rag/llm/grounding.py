"""Evidence-only prompting and conservative unsupported-answer behavior."""
from __future__ import annotations

import re
from dataclasses import replace

from rag.llm.providers import AnswerProvider
from rag.models import Answer, Citation, RetrievalResult

LABEL = re.compile(r"\[(S\d+)\]")
def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    evidence = "\n\n".join(f"[S{i}] {result.chunk.source_title}\nURL: {result.chunk.source_url}\nExcerpt: {result.chunk.text}" for i, result in enumerate(results, 1))
    return f"""You are an internal Databricks documentation assistant. Answer only from the indexed documentation excerpts below. The source set includes official Databricks documentation and approved supplemental guidance; identify a source as official only when its URL is on docs.databricks.com. Every factual claim must cite one or more source labels such as [S1]. Do not use background knowledge and do not infer a general product behavior from an excerpt that discusses a narrower context such as benchmarks, APIs, or trusted assets. For comparison questions, state only differences that are directly supported; explicitly say which requested differences the excerpts do not establish. If the excerpts do not support an answer, reply exactly: I could not verify this from the indexed documentation.

Question: {question}

Indexed documentation excerpts:
{evidence}"""


def answer_groundedly(question: str, results: list[RetrievalResult], provider: AnswerProvider, *, threshold: float) -> Answer:
    """Answer from the retrieval agent's evidence without re-selecting it.

    Evidence selection belongs to retrieval. A second LLM selector here can
    discard chunks chosen to cover another aspect of a multi-part question.
    """
    snapshot_id = results[0].snapshot_id if results else "none"
    citations = tuple(Citation(f"S{i}", result.chunk.source_title, result.chunk.source_url, result.chunk.text, result.chunk.chunk_id) for i, result in enumerate(results, 1))
    if not results or results[0].score < threshold:
        return Answer("I could not verify this from the indexed documentation.", citations, False, provider.name, snapshot_id)
    text = provider.complete(build_prompt(question, results))
    labels = set(LABEL.findall(text))
    if not labels:
        return Answer("I could not verify this from the indexed documentation.", citations, False, provider.name, snapshot_id)
    cited = tuple(citation for citation in citations if citation.label in labels)
    renumbered = {citation.label: f"S{index}" for index, citation in enumerate(cited, 1)}
    text = LABEL.sub(lambda match: f"[{renumbered.get(match.group(1), match.group(1))}]", text)
    citations = tuple(replace(citation, label=renumbered[citation.label]) for citation in cited)
    return Answer(text, citations, True, provider.name, snapshot_id)
