from rag.app.web import create_app
from rag.llm.providers import ToolCall
from rag.models import Chunk, RetrievalResult


class Provider:
    name = "fake"; model = "fake"
    def complete(self, prompt):
        if "Select the" in prompt: return "S1"
        return "Supported answer [S1]."
    def call_tool(self, messages, tools):
        return ToolCall("standalone_query", {"query": "resolved follow-up"})


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

    def delete(self, owner, conversation_id):
        if conversation_id not in self.data: return False
        del self.data[conversation_id]
        return True


def test_answer_and_feedback_routes():
    received = []
    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")
    app = create_app(retrieve=lambda _: [RetrievalResult(chunk, .9, "s")], provider=Provider(), threshold=.3, feedback_sink=received.append)
    client = app.test_client()
    response = client.post("/api/answer", json={"question": "test"})
    assert response.status_code == 200 and response.json["supported"]
    assert client.post("/api/feedback", json={"rating": "up"}).status_code == 204
    assert received[0]["rating"] == "up"


def test_answer_records_retrieval_and_grounding_trace_when_configured():
    received = []
    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")
    sink = type("TraceSink", (), {"record": lambda self, **kwargs: received.append(kwargs)})()
    app = create_app(
        retrieve=lambda _: [RetrievalResult(chunk, .9, "s")], provider=Provider(), threshold=.3,
        trace_getter=lambda: None, trace_sink=sink,
    )
    assert app.test_client().post("/api/answer", json={"question": "test"}).status_code == 200
    assert received[0]["grounding_trace"].fallback_reason is None
    assert received[0]["results"][0].chunk.chunk_id == "c"


def test_streamed_answer_emits_retrieval_progress_and_a_final_answer():
    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")

    def live_retrieve(_, *, on_progress):
        on_progress({"kind": "step", "turn": 1, "action": "search_docs", "status": "ok", "count": 1,
                     "message": "Searched documentation"})
        return [RetrievalResult(chunk, .9, "s")]

    app = create_app(retrieve=lambda _: [RetrievalResult(chunk, .9, "s")], progress_retrieve=live_retrieve,
                     provider=Provider(), threshold=.3)
    response = app.test_client().post("/api/answer/stream", json={"question": "test"})

    assert response.status_code == 200
    assert b"event: progress" in response.data
    assert b"Searched documentation" in response.data
    assert b"event: answer" in response.data


def test_home_serves_the_preact_application_shell():
    app = create_app(retrieve=lambda _: [], provider=Provider(), threshold=.3)
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b'static/ui/app.js' in response.data
    assert b'static/ui/assets/index.css' in response.data


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


def test_owner_can_remove_a_conversation_from_history():
    history = History(); history.data["c1"] = []
    app = create_app(retrieve=lambda _: [], provider=Provider(), threshold=.3, history=history, identity=Identity())
    client = app.test_client()
    assert client.delete("/api/conversations/c1").status_code == 204
    assert client.delete("/api/conversations/c1").status_code == 404


def test_unexpected_answer_error_is_json_not_an_html_error_page():
    class BrokenProvider:
        name = "broken"; model = "broken"
        def complete(self, prompt): raise AttributeError("provider failure")

    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")
    app = create_app(
        retrieve=lambda _: [RetrievalResult(chunk, .9, "s")], provider=BrokenProvider(), threshold=.3,
    )
    response = app.test_client().post("/api/answer", json={"question": "test"})
    assert response.status_code == 500
    assert response.json == {"error": "The service could not process this request."}
