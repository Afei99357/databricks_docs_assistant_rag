import pytest

from rag.agent.retrieval import RetrievalAgent
from rag.llm.providers import ToolCall
from rag.models import Chunk, RetrievalResult

# Realistic opaque chunk IDs. The agent protocol never asks the model to retype
# one of these; it addresses evidence by short label instead.
IDS = {
    "generic": "d2b162a114fec21dbe6e23c3",
    "permission": "e08a9338e087e3c41ccae492",
    "neighbor": "bb96a17fe5d4095b5a300647",
    "within": "31bc6e7a5133517fb7788585",
}


class Provider:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.prompts = []
        self.declared_tools = []

    def complete(self, prompt):
        raise AssertionError("the retrieval agent must not use free-text completion")

    def call_tool(self, prompt, tools):
        self.prompts.append(prompt)
        self.declared_tools.append(tools)
        return next(self.calls)


def _result(name, *, text="evidence", url="https://docs.databricks.com/x", position=0, score=.9):
    return RetrievalResult(Chunk(IDS[name], "doc", "v", position, text, (), url, "Docs"), score, "snap")


class Tools:
    def __init__(self):
        self.calls = []
        self.known = {}
        self.results = {
            "broad question": [_result("generic", text="generic evidence", score=.95)],
            "specific permission": [_result("permission", text="SELECT is required", score=.88)],
        }

    def _emit(self, results):
        # The real snapshot can read back any chunk it has ever returned.
        self.known.update({item.chunk.chunk_id: item for item in results})
        return results

    def retrieve(self, query, top_k=None):
        self.calls.append(("search_docs", query))
        return self._emit(self.results.get(query, []))

    def read_chunks(self, ids):
        self.calls.append(("read_chunks", tuple(ids)))
        return [self.known[chunk_id] for chunk_id in ids if chunk_id in self.known]

    def related_chunks(self, chunk_id, *, radius=1):
        # Mirrors ActiveSnapshotRetriever: the anchor chunk itself is included,
        # and neighbours carry a placeholder score rather than a ranked one.
        self.calls.append(("get_related_chunks", chunk_id))
        return self._emit([_result("generic", text="generic evidence", score=1.0),
                           _result("neighbor", text="continued section", position=1, score=1.0)])

    def search_within_document(self, source_url, query, top_k):
        self.calls.append(("search_within_document", source_url, query))
        return self._emit([_result("within", text="specific section", url=source_url)])


def test_agent_declares_every_tool_it_dispatches():
    tools = Tools()
    provider = Provider([ToolCall("search_docs", {"query": "broad question"}),
                         ToolCall("read_chunks", {"labels": ["S1"]}),
                         ToolCall("final", {"selected": ["S1"]})])
    RetrievalAgent(tools, provider).retrieve("question")
    declared = {tool["function"]["name"] for tool in provider.declared_tools[0]}
    assert declared == {"search_docs", "read_chunks", "get_related_chunks",
                        "search_within_document", "final"}
    assert all(tool["type"] == "function" for tool in provider.declared_tools[0])


def test_agent_searches_reads_then_selects_opened_evidence():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("search_docs", {"query": "specific permission"}),
        ToolCall("read_chunks", {"labels": ["S2"]}),
        ToolCall("final", {"selected": ["S1", "S2"]}),
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"], IDS["permission"]]
    assert tools.calls == [
        ("search_docs", "broad question"), ("read_chunks", (IDS["generic"],)),
        ("search_docs", "specific permission"), ("read_chunks", (IDS["permission"],)),
    ]
    assert agent.last_trace.status == "answered"
    assert agent.last_trace.stop_reason == "agent_satisfied"


def test_observations_address_evidence_by_label_and_never_expose_chunk_ids():
    tools = Tools()
    provider = Provider([ToolCall("search_docs", {"query": "broad question"}),
                         ToolCall("read_chunks", {"labels": ["S1"]}),
                         ToolCall("final", {"selected": ["S1"]})])
    RetrievalAgent(tools, provider).retrieve("question")
    assert '"label": "S1"' in provider.prompts[1]
    assert IDS["generic"] not in provider.prompts[1]
    assert IDS["generic"] not in provider.prompts[2]


def test_agent_rejects_final_evidence_that_was_not_opened():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("final", {"selected": ["S1"]}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"]}),
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"]]
    assert agent.last_trace.steps[1].status == "rejected"


def test_agent_rejects_duplicate_searches_without_calling_retriever_twice():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "Broad question!"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"], "unverified_points": ["limits"]}),
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"]]
    assert [call for call in tools.calls if call[0] == "search_docs"] == [("search_docs", "broad question")]
    assert agent.last_trace.status == "partial"
    assert agent.last_trace.steps[1].status == "rejected_duplicate"


def test_agent_can_open_related_chunks_by_label():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("get_related_chunks", {"label": "S1"}),
        ToolCall("final", {"selected": ["S2"]}),
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["neighbor"]]
    assert ("get_related_chunks", IDS["generic"]) in tools.calls


def test_search_within_document_resolves_a_label_to_its_source_url():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_within_document", {"source": "S1", "query": "specific section"}),
        ToolCall("read_chunks", {"labels": ["S2"]}),
        ToolCall("final", {"selected": ["S2"]}),
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["within"]]
    assert ("search_within_document", "https://docs.databricks.com/x", "specific section") in tools.calls


def test_agent_rejects_a_tool_name_it_does_not_implement():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("summarise_everything", {}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"]}),
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"]]
    assert agent.last_trace.steps[1].status == "rejected"


def test_a_provider_failure_is_not_swallowed():
    class Failing(Provider):
        def call_tool(self, prompt, tools):
            raise RuntimeError("qwen returned prose instead of a tool call")

    with pytest.raises(RuntimeError, match="prose instead of a tool call"):
        RetrievalAgent(Tools(), Failing([])).retrieve("question")


def test_opening_related_chunks_preserves_the_anchor_retrieval_score():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("get_related_chunks", {"label": "S1"}),
        ToolCall("final", {"selected": ["S1", "S2"]}),
    ]))
    scores = {item.chunk.chunk_id: item.score for item in agent.retrieve("question")}
    # The anchor keeps its ranked score, and its neighbour inherits that score
    # instead of the retriever's placeholder 1.0.
    assert scores[IDS["generic"]] == pytest.approx(.95)
    assert scores[IDS["neighbor"]] == pytest.approx(.95)


def test_final_evidence_is_returned_in_descending_score_order():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "specific permission"}),
        ToolCall("read_chunks", {"labels": ["S1", "S2"]}),
        ToolCall("final", {"selected": ["S2", "S1"]}),
    ]))
    # The grounding layer gates on results[0].score, so the strongest evidence
    # must lead regardless of the order the model listed it in.
    assert [item.score for item in agent.retrieve("question")] == [pytest.approx(.95), pytest.approx(.88)]


def test_final_evidence_is_capped_at_the_configured_maximum():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "specific permission"}),
        ToolCall("read_chunks", {"labels": ["S1", "S2"]}),
        ToolCall("final", {"selected": ["S1", "S2"]}),
    ]), max_evidence=1)
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"]]


def test_evidence_is_ranked_and_capped_when_the_agent_never_finalizes():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "specific permission"}),
        ToolCall("read_chunks", {"labels": ["S1", "S2"]}),
    ]), max_steps=3, max_evidence=1)
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"]]
    assert agent.last_trace.status == "partial"


def test_agent_stops_after_the_step_budget():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("get_related_chunks", {"label": "S1"}),
    ]), max_steps=3)
    agent.retrieve("question")
    assert agent.last_trace.stop_reason == "step_budget_exhausted"
    assert len(agent.last_trace.steps) == 3


def test_reordered_search_terms_are_treated_as_a_duplicate_search():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "question broad"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"]}),
    ]))
    agent.retrieve("question")
    assert [call for call in tools.calls if call[0] == "search_docs"] == [("search_docs", "broad question")]
    assert agent.last_trace.steps[1].status == "rejected_duplicate"


def test_a_search_that_mostly_repeats_earlier_evidence_is_reported_as_low_novelty():
    tools = Tools()
    tools.results["restated question"] = [
        _result("generic", text="generic evidence", score=.95),
        _result("permission", text="SELECT is required", score=.88),
    ]
    agent = RetrievalAgent(tools, Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "restated question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"]}),
    ]), min_new_chunks=2)
    agent.retrieve("question")
    assert agent.last_trace.steps[1].status == "low_novelty"


def test_superseded_search_results_are_compacted_out_of_the_prompt():
    tools = Tools()
    provider = Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("search_docs", {"query": "specific permission"}),
        ToolCall("read_chunks", {"labels": ["S2"]}),
        ToolCall("final", {"selected": ["S2"]}),
    ])
    RetrievalAgent(tools, provider).retrieve("question")
    # The first search's excerpts survive only while that search is the latest
    # one; afterwards the label and title remain but the body text does not.
    assert "generic evidence" in provider.prompts[1]
    assert "generic evidence" not in provider.prompts[2]
    assert '"S1"' in provider.prompts[2]


def test_opened_chunk_text_is_never_compacted_away():
    tools = Tools()
    provider = Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("search_docs", {"query": "specific permission"}),
        ToolCall("final", {"selected": ["S1"]}),
    ])
    RetrievalAgent(tools, provider).retrieve("question")
    assert "generic evidence" in provider.prompts[3]


def test_agent_is_told_to_finalize_on_its_last_step():
    tools = Tools()
    provider = Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"]}),
    ])
    agent = RetrievalAgent(tools, provider, max_steps=3)
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == [IDS["generic"]]
    # Without this the model keeps reformulating searches until the budget runs
    # out and the harness has to guess the evidence for it.
    assert "final_step" in provider.prompts[2]
    assert agent.last_trace.stop_reason == "agent_satisfied"


def test_the_finalize_notice_is_not_issued_while_steps_remain():
    tools = Tools()
    provider = Provider([
        ToolCall("search_docs", {"query": "broad question"}),
        ToolCall("read_chunks", {"labels": ["S1"]}),
        ToolCall("final", {"selected": ["S1"]}),
    ])
    RetrievalAgent(tools, provider, max_steps=8).retrieve("question")
    assert "final_step" not in provider.prompts[2]
