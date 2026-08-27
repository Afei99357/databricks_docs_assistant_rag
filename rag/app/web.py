from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from queue import Queue
from threading import Thread
from time import perf_counter

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from rag.conversation import resolve_follow_up
from rag.llm.grounding import answer_groundedly_with_trace
from rag.llm.providers import capture_llm_usage
from rag.models import RetrievalResult

STARTER_QUESTIONS = [
    "What is Volume Content Search and what are its limitations?",
    "How do Unity Catalog privileges affect Genie?",
    "How can I embed Genie in an application?",
    "How do Genie Agents use unstructured data in Volumes?",
]


def create_app(*, retrieve: Callable[[str], list[RetrievalResult]], provider, threshold: float,
               feedback_sink: Callable[[dict], None] | None = None, history=None, identity=None,
               trace_getter: Callable[[], object | None] | None = None, trace_sink=None,
               progress_retrieve: Callable | None = None) -> Flask:
    app = Flask(__name__)
    app.config["STARTER_QUESTIONS"] = STARTER_QUESTIONS

    def complete_answer(question, conversation_id, owner, prior_turns, on_progress=None):
        started = perf_counter()
        with capture_llm_usage() as llm_usage:
            resolved_query = resolve_follow_up(question, prior_turns, provider) if prior_turns else question
            if on_progress:
                on_progress({"kind": "starting", "message": "Searching indexed documentation."})
            if on_progress and progress_retrieve:
                retrieved = progress_retrieve(resolved_query, on_progress=on_progress)
            else:
                retrieved = retrieve(resolved_query)
            retrieval_trace = trace_getter() if trace_getter else None
            if on_progress:
                on_progress({"kind": "answering", "message": "Writing a grounded answer from the selected evidence."})
            result, grounding_trace = answer_groundedly_with_trace(
                question, retrieved, provider, threshold=threshold,
                evidence_support=tuple(getattr(retrieval_trace, "evidence_support", ()) or ()),
                unverified_points=tuple(getattr(retrieval_trace, "unverified_points", ()) or ()),
            )
        turn_id = None
        if history and identity:
            conversation_id = conversation_id or history.create(owner, question)
            turn_id = history.append_turn(owner, conversation_id, question=question, resolved_query=resolved_query, answer=result,
                                          citation_ids=[item.chunk_id for item in result.citations], latency_ms=round((perf_counter() - started) * 1000))
        if trace_sink:
            try:
                trace_sink.record(turn_id=turn_id, conversation_id=conversation_id, owner=owner, question=question,
                                  resolved_query=resolved_query, retrieval_trace=retrieval_trace,
                                  results=retrieved, grounding_trace=grounding_trace, answer=result,
                                  latency_ms=round((perf_counter() - started) * 1000), llm_usage=llm_usage)
            except Exception:
                app.logger.exception("failed to persist request trace")
        return {"question": question, "answer": result.text, "supported": result.supported,
                "provider": result.provider, "snapshot_id": result.snapshot_id,
                "latency_ms": round((perf_counter() - started) * 1000),
                "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved],
                "citations": [asdict(citation) for citation in result.citations], "conversation_id": conversation_id}

    def request_context():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "").strip()
        if not question:
            raise ValueError("Enter a question.")
        conversation_id = payload.get("conversation_id") or None
        owner = identity.current_user_id(request) if history and identity else None
        prior_turns = history.turns_for(owner, conversation_id) if conversation_id else []
        if conversation_id and history and not prior_turns:
            raise LookupError("Conversation not found.")
        return question, conversation_id, owner, prior_turns

    @app.errorhandler(PermissionError)
    def forbidden(error):
        return jsonify({"error": str(error) or "You are not authorized to use this application."}), 403

    @app.errorhandler(RuntimeError)
    def runtime_error(error):
        # OBO failures are user-facing authorization errors. Other runtime
        # failures still return a concise API response instead of Flask HTML.
        from rag.app.auth import MissingForwardedTokenError
        status = 401 if isinstance(error, MissingForwardedTokenError) else 500
        return jsonify({"error": str(error) or "The service could not process this request."}), status

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("unexpected API error", exc_info=error)
        return jsonify({"error": "The service could not process this request."}), 500

    @app.get("/")
    def home():
        return render_template("index.html", starter_questions=STARTER_QUESTIONS)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

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

    @app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id):
        if not history or not identity:
            return jsonify({"error": "Conversation history is not enabled."}), 404
        owner = identity.current_user_id(request)
        if not history.delete(owner, conversation_id):
            return jsonify({"error": "Conversation not found."}), 404
        return ("", 204)

    @app.post("/api/answer")
    def answer():
        try:
            question, conversation_id, owner, prior_turns = request_context()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        return jsonify(complete_answer(question, conversation_id, owner, prior_turns))

    @app.post("/api/answer/stream")
    def answer_stream():
        try:
            question, conversation_id, owner, prior_turns = request_context()
        except ValueError:
            return jsonify({"error": "Enter a question."}), 400
        except LookupError as error:
            return jsonify({"error": str(error)}), 404
        events: Queue[tuple[str, dict]] = Queue()

        def run():
            try:
                events.put(("answer", complete_answer(question, conversation_id, owner, prior_turns,
                                                       on_progress=lambda event: events.put(("progress", event)))))
            except Exception as error:
                app.logger.exception("streamed answer failed")
                events.put(("error", {"error": str(error) or "The request could not be completed."}))
            finally:
                events.put(("done", {}))

        Thread(target=run, daemon=True).start()

        @stream_with_context
        def stream():
            while True:
                event, payload = events.get()
                yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                if event == "done":
                    break

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/feedback")
    def feedback():
        payload = request.get_json(silent=True) or {}
        if payload.get("rating") not in {"up", "down"}:
            return jsonify({"error": "rating must be up or down"}), 400
        if feedback_sink:
            feedback_sink(payload)
        return ("", 204)

    return app
