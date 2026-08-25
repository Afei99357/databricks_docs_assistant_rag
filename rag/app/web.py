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
               feedback_sink: Callable[[dict], None] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["STARTER_QUESTIONS"] = STARTER_QUESTIONS

    @app.get("/")
    def home():
        return render_template("index.html", starter_questions=STARTER_QUESTIONS)

    @app.post("/api/answer")
    def answer():
        question = (request.get_json(silent=True) or {}).get("question", "").strip()
        if not question:
            return jsonify({"error": "Enter a question."}), 400
        started = perf_counter()
        retrieved = retrieve(question)
        result = answer_groundedly(question, retrieved, provider, threshold=threshold)
        return jsonify({"question": question, "answer": result.text, "supported": result.supported,
                        "provider": result.provider, "snapshot_id": result.snapshot_id,
                        "latency_ms": round((perf_counter() - started) * 1000),
                        "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved],
                        "citations": [asdict(citation) for citation in result.citations]})

    @app.post("/api/feedback")
    def feedback():
        payload = request.get_json(silent=True) or {}
        if payload.get("rating") not in {"up", "down"}:
            return jsonify({"error": "rating must be up or down"}), 400
        if feedback_sink:
            feedback_sink(payload)
        return ("", 204)

    return app
