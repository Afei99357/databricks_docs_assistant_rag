from rag.llm.grounding import answer_groundedly, answer_groundedly_with_trace, build_prompt
from rag.models import Chunk, RetrievalResult


class FakeProvider:
    name = "fake"
    model = "fake-model"
    def __init__(self, response): self.response = response
    def complete(self, prompt): return self.response.pop(0) if isinstance(self.response, list) else self.response


def _result(score=0.9, *, chunk_id="c"):
    chunk = Chunk(chunk_id, "d", "v", 0, "Genie uses official docs.", ("Genie",), "https://docs.databricks.com/x", "Genie")
    return RetrievalResult(chunk, score, "snapshot")


def test_grounded_answer_keeps_only_cited_sources():
    answer = answer_groundedly("What is Genie?", [_result()], FakeProvider("Genie is documented [S1]."), threshold=0.3)
    assert answer.supported
    assert answer.citations[0].url.startswith("https://docs.databricks.com")
    assert "[S1]" in build_prompt("What?", [_result()])


def test_grounding_prompt_respects_the_retrieval_selection_scope():
    prompt = build_prompt(
        "What is Genie?", [_result()],
        evidence_support=({"chunk_id": "c", "supports": ("the documented definition",)},),
        unverified_points=("pricing",),
    )
    assert "[S1] may be used only for: the documented definition" in prompt
    assert "pricing" in prompt


def test_grounding_prompt_treats_the_index_as_official_documentation():
    prompt = build_prompt("What is Genie?", [_result()])
    assert "official Databricks documentation excerpts" in prompt
    assert "identify a source as official" not in prompt


def test_low_relevance_refuses_without_calling_provider():
    answer = answer_groundedly("Unknown", [_result(0.1)], FakeProvider("unsafe"), threshold=0.3)
    assert not answer.supported
    assert "could not verify" in answer.text
    assert answer.citations == ()


def test_no_results_refuses_without_citations():
    answer = answer_groundedly("Unknown", [], FakeProvider("unsafe"), threshold=0.3)

    assert not answer.supported
    assert answer.citations == ()


def test_grounded_answer_renumbers_cited_sources_without_gaps():
    answer = answer_groundedly(
        "What is Genie?",
        [_result(chunk_id="first"), _result(chunk_id="second"), _result(chunk_id="third")],
        FakeProvider("First [S1]. Third [S3]."),
        threshold=0.3,
    )
    assert answer.text == "First [S1]. Third [S2]."
    assert [citation.label for citation in answer.citations] == ["S1", "S2"]
    assert [citation.chunk_id for citation in answer.citations] == ["first", "third"]


def test_grounding_trace_preserves_model_labels_before_display_renumbering():
    answer, trace = answer_groundedly_with_trace(
        "What is Genie?",
        [_result(chunk_id="first"), _result(chunk_id="second"), _result(chunk_id="third")],
        FakeProvider("First [S1]. Third [S3]."),
        threshold=0.3,
    )
    assert answer.text == "First [S1]. Third [S2]."
    assert trace.raw_model_output == "First [S1]. Third [S3]."
    assert trace.parsed_citation_labels == ("S1", "S3")


def test_grounded_answer_accepts_comma_separated_citations():
    answer, trace = answer_groundedly_with_trace(
        "What is Genie?",
        [_result(chunk_id="first"), _result(chunk_id="second"), _result(chunk_id="third")],
        FakeProvider("First [S1, S3]."),
        threshold=0.3,
    )
    assert answer.supported
    assert answer.text == "First [S1, S2]."
    assert trace.parsed_citation_labels == ("S1", "S3")


def test_grounding_trace_records_uncited_model_fallback_reason():
    answer, trace = answer_groundedly_with_trace(
        "What is Genie?", [_result()], FakeProvider("I cannot verify this."), threshold=0.3
    )
    assert not answer.supported
    assert answer.citations == ()
    assert trace.raw_model_output == "I cannot verify this."
    assert trace.fallback_reason == "no_citations_in_model_output"


def test_invalid_model_citation_refuses_without_citations():
    answer, trace = answer_groundedly_with_trace(
        "What is Genie?", [_result()], FakeProvider("Unsupported [S99]."), threshold=0.3
    )

    assert not answer.supported
    assert answer.citations == ()
    assert trace.fallback_reason == "invalid_citation_labels"
