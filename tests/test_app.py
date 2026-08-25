from rag.app.web import create_app
from rag.models import Chunk, RetrievalResult


class Provider:
    name = "fake"; model = "fake"
    def complete(self, prompt):
        if "Select the" in prompt: return "S1"
        if "standalone_query" in prompt: return '{"standalone_query":"resolved follow-up"}'
        return "Supported answer [S1]."


class Identity:
    def current_user_id(self, request): return "eric@example.com"


class History:
    def __init__(self): self.data = {}; self.resolved = []
    def create(self, owner, title): self.data["c1"] = []; return "c1"
    def list(self, owner): return [("c1", "first question", "today")]
    def turns_for(self, owner, conversation_id): return self.data.get(conversation_id, [])
    def append_turn(self, owner, conversation_id, **kwargs):
        self.resolved.append(kwargs["resolved_query"])
        self.data.setdefault(conversation_id, []).append(("t", len(self.data.get(conversation_id, [])) + 1, kwargs["question"], kwargs["answer"].text, kwargs["answer"].supported, kwargs["answer"].snapshot_id, kwargs["citation_ids"], "today"))


def test_answer_and_feedback_routes():
    received = []
    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")
    app = create_app(retrieve=lambda _: [RetrievalResult(chunk, .9, "s")], provider=Provider(), threshold=.3, feedback_sink=received.append)
    client = app.test_client()
    response = client.post("/api/answer", json={"question": "test"})
    assert response.status_code == 200 and response.json["supported"]
    assert client.post("/api/feedback", json={"rating": "up"}).status_code == 204
    assert received[0]["rating"] == "up"


def test_conversation_reuses_id_and_rewrites_follow_up():
    history = History(); queries = []
    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")
    app = create_app(retrieve=lambda query: queries.append(query) or [RetrievalResult(chunk, .9, "s")], provider=Provider(), threshold=.3, history=history, identity=Identity())
    client = app.test_client()
    first = client.post("/api/answer", json={"question": "first question"})
    assert first.status_code == 200 and first.json["conversation_id"] == "c1"
    second = client.post("/api/answer", json={"question": "and what about requirements?", "conversation_id": "c1"})
    assert second.status_code == 200
    assert queries == ["first question", "resolved follow-up"]
    assert history.resolved == ["first question", "resolved follow-up"]
    assert client.get("/api/conversations").json["conversations"][0]["conversation_id"] == "c1"
    assert len(client.get("/api/conversations/c1/turns").json["turns"]) == 2
