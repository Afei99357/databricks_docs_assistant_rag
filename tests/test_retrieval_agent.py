from rag.agent.retrieval import RetrievalAgent
from rag.models import Chunk, RetrievalResult


class Provider:
    def __init__(self, replies):
        self.replies = iter(replies)

    def complete(self, prompt):
        return next(self.replies)


def _result(identifier, *, text="evidence", url="https://docs.databricks.com/x", position=0, score=.9):
    return RetrievalResult(Chunk(identifier, "doc", "v", position, text, (), url, "Docs"), score, "snap")


class Tools:
    def __init__(self):
        self.calls = []
        self.results = {
            "broad question": [_result("generic", text="generic evidence", score=.95)],
            "specific permission": [_result("permission", text="SELECT is required", score=.88)],
        }

    def retrieve(self, query, top_k=None):
        self.calls.append(("search_docs", query))
        return self.results.get(query, [])

    def read_chunks(self, ids):
        self.calls.append(("read_chunks", tuple(ids)))
        return [item for values in self.results.values() for item in values if item.chunk.chunk_id in ids]

    def related_chunks(self, chunk_id, *, radius=1):
        self.calls.append(("get_related_chunks", chunk_id))
        return [_result("neighbor", text="continued section", position=1)]

    def search_within_document(self, source_url, query, top_k):
        self.calls.append(("search_within_document", source_url, query))
        return [_result("within", text="specific section", url=source_url)]


def test_agent_searches_reads_then_selects_opened_evidence():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        '{"action":"search_docs","query":"broad question"}',
        '{"action":"read_chunks","chunk_ids":["generic"]}',
        '{"action":"search_docs","query":"specific permission"}',
        '{"action":"read_chunks","chunk_ids":["permission"]}',
        '{"action":"final","selected_chunk_ids":["generic","permission"]}',
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == ["generic", "permission"]
    assert tools.calls == [
        ("search_docs", "broad question"), ("read_chunks", ("generic",)),
        ("search_docs", "specific permission"), ("read_chunks", ("permission",)),
    ]
    assert agent.last_trace.status == "answered"
    assert agent.last_trace.stop_reason == "agent_satisfied"


def test_agent_rejects_final_evidence_that_was_not_opened():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        '{"action":"search_docs","query":"broad question"}',
        '{"action":"final","selected_chunk_ids":["generic"]}',
        '{"action":"read_chunks","chunk_ids":["generic"]}',
        '{"action":"final","selected_chunk_ids":["generic"]}',
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == ["generic"]
    assert agent.last_trace.steps[1].status == "rejected"


def test_agent_rejects_duplicate_searches_without_calling_retriever_twice():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        '{"action":"search_docs","query":"broad question"}',
        '{"action":"search_docs","query":"Broad question!"}',
        '{"action":"read_chunks","chunk_ids":["generic"]}',
        '{"action":"final","selected_chunk_ids":["generic"],"unverified_points":["limits"]}',
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == ["generic"]
    assert [call for call in tools.calls if call[0] == "search_docs"] == [("search_docs", "broad question")]
    assert agent.last_trace.status == "partial"
    assert agent.last_trace.steps[1].status == "rejected_duplicate"


def test_agent_can_open_related_chunks():
    tools = Tools()
    agent = RetrievalAgent(tools, Provider([
        '{"action":"search_docs","query":"broad question"}',
        '{"action":"read_chunks","chunk_ids":["generic"]}',
        '{"action":"get_related_chunks","chunk_id":"generic"}',
        '{"action":"final","selected_chunk_ids":["neighbor"]}',
    ]))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == ["neighbor"]
    assert ("get_related_chunks", "generic") in tools.calls
