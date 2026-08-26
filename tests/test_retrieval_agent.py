from rag.agent.retrieval import RetrievalAgent
from rag.models import Chunk, RetrievalResult


class Provider:
    def __init__(self, replies): self.replies = iter(replies)
    def complete(self, prompt): return next(self.replies)


def _result(identifier):
    return RetrievalResult(Chunk(identifier, "doc", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs"), .9, "snap")


def test_agent_can_search_again_then_select_evidence():
    calls = []
    def search(query, count): calls.append(query); return [_result("a")] if len(calls) == 1 else [_result("b")]
    agent = RetrievalAgent(search, Provider(['{"queries":[]}', '{"action":"search","query":"specific permission"}', '{"action":"answer","selected_chunk_ids":["a","b"]}']))
    assert [item.chunk.chunk_id for item in agent.retrieve("question")] == ["a", "b"]
    assert calls == ["question", "specific permission"]


def test_agent_searches_each_planned_aspect_before_selecting_evidence():
    calls = []
    def search(query, count):
        calls.append(query)
        if query == "What is the feature and its limitations?":
            return []
        return [_result("definition")] if "definition" in query else [_result("limits")]
    provider = Provider([
        '{"queries":["feature definition","feature limitations"]}',
        '{"action":"answer","selected_chunk_ids":["definition","limits"]}',
    ])
    agent = RetrievalAgent(search, provider)
    assert [item.chunk.chunk_id for item in agent.retrieve("What is the feature and its limitations?")] == ["definition", "limits"]
    assert calls == ["What is the feature and its limitations?", "feature definition", "feature limitations"]
