from rag.app.web import create_app
from rag.models import Chunk, RetrievalResult


class Provider:
    name = "fake"; model = "fake"
    def complete(self, prompt): return "Supported answer [S1]."


def test_answer_and_feedback_routes():
    received = []
    chunk = Chunk("c", "d", "v", 0, "evidence", (), "https://docs.databricks.com/x", "Docs")
    app = create_app(retrieve=lambda _: [RetrievalResult(chunk, .9, "s")], provider=Provider(), threshold=.3, feedback_sink=received.append)
    client = app.test_client()
    response = client.post("/api/answer", json={"question": "test"})
    assert response.status_code == 200 and response.json["supported"]
    assert client.post("/api/feedback", json={"rating": "up"}).status_code == 204
    assert received[0]["rating"] == "up"

