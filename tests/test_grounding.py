from rag.llm.grounding import answer_groundedly, build_prompt, select_evidence
from rag.models import Chunk, RetrievalResult


class FakeProvider:
    name = "fake"
    model = "fake-model"
    def __init__(self, response): self.response = response
    def complete(self, prompt): return self.response.pop(0) if isinstance(self.response, list) else self.response


def _result(score=0.9):
    chunk = Chunk("c", "d", "v", 0, "Genie uses official docs.", ("Genie",), "https://docs.databricks.com/x", "Genie",)
    return RetrievalResult(chunk, score, "snapshot")


def test_grounded_answer_keeps_only_cited_sources():
    answer = answer_groundedly("What is Genie?", [_result()], FakeProvider(["S1,S1,S1,S1", "Genie is documented [S1]."]), threshold=0.3)
    assert answer.supported
    assert answer.citations[0].url.startswith("https://docs.databricks.com")
    assert "[S1]" in build_prompt("What?", [_result()])


def test_low_relevance_refuses_without_calling_provider():
    answer = answer_groundedly("Unknown", [_result(0.1)], FakeProvider("unsafe"), threshold=0.3)
    assert not answer.supported
    assert "could not verify" in answer.text


def test_invalid_evidence_selection_falls_back_to_top_candidates():
    results = [_result() for _ in range(6)]
    assert len(select_evidence("question", results, FakeProvider("not labels"))) == 6
