from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Callable

from flask import Flask, jsonify, render_template, request

from rag.llm.grounding import answer_groundedly
from rag.models import RetrievalResult

STARTER_QUESTIONS = [
    "What is Volume Content Search and what are its limitations?",
    "How do Unity Catalog privileges affect Genie?",
    "How can I embed Genie in an application?",
    "How do Genie Agents use unstructured data in Volumes?",
]


def create_app(*, retrieve: Callable[[str], list[RetrievalResult]], provider, threshold: float,
               feedback_sink: Callable[[dict], None] | None = None, history=None, identity=None) -> Flask:
    app = Flask(__name__)
    app.config["STARTER_QUESTIONS"] = STARTER_QUESTIONS

    @app.get("/")
    def home():
        return render_template("index.html", starter_questions=STARTER_QUESTIONS)

    @app.get("/api/conversations")
    def conversations():
        if not history or not identity: return jsonify({"conversations": []})
        owner = identity.current_user_id(request)
        return jsonify({"conversations": [{"conversation_id": row[0], "title": row[1], "updated_at": row[2]} for row in history.list(owner)]})

    @app.get("/api/conversations/<conversation_id>/turns")
    def turns(conversation_id):
        if not history or not identity: return jsonify({"turns": []})
        owner = identity.current_user_id(request)
        rows = history.turns_for(owner, conversation_id)
        return jsonify({"turns": [{"turn_id": row[0], "turn_number": row[1], "question": row[2], "answer": row[3], "supported": row[4], "snapshot_id": row[5], "citation_chunk_ids": row[6], "created_at": row[7]} for row in rows]})

    @app.post("/api/answer")
    def answer():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "").strip()
        if not question:
            return jsonify({"error": "Enter a question."}), 400
        started = perf_counter()
        retrieved = retrieve(question)
        result = answer_groundedly(question, retrieved, provider, threshold=threshold)
        conversation_id = payload.get("conversation_id") or None
        if history and identity:
            owner = identity.current_user_id(request)
            conversation_id = conversation_id or history.create(owner, question)
            history.append_turn(owner, conversation_id, question=question, resolved_query=question, answer=result,
                                citation_ids=[item.chunk.chunk_id for item in result.citations], latency_ms=round((perf_counter() - started) * 1000))
        return jsonify({"question": question, "answer": result.text, "supported": result.supported,
                        "provider": result.provider, "snapshot_id": result.snapshot_id,
                        "latency_ms": round((perf_counter() - started) * 1000),
                        "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved],
                        "citations": [asdict(citation) for citation in result.citations], "conversation_id": conversation_id})

    @app.post("/api/feedback")
    def feedback():
        payload = request.get_json(silent=True) or {}
        if payload.get("rating") not in {"up", "down"}:
            return jsonify({"error": "rating must be up or down"}), 400
        if feedback_sink:
            feedback_sink(payload)
        return ("", 204)

    return app
