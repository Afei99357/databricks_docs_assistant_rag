"""Small, bounded follow-up query resolution for documentation retrieval."""
from __future__ import annotations

from rag.llm.providers import AnswerProvider


def history_title(question: str) -> str:
    """Keep sidebar labels useful without turning them into horizontal documents."""
    normalized = " ".join(question.split())
    if len(normalized) <= 88:
        return normalized or "New conversation"
    shortened = normalized[:85].rsplit(" ", 1)[0] or normalized[:85]
    return shortened + "…"


REWRITE_TOOL = [{
    "type": "function",
    "function": {
        "name": "standalone_query",
        "description": "Record the user's latest question, rewritten so it can be searched alone.",
        "parameters": {
            "type": "object",
            "properties": {"query": {
                "type": "string",
                "description": "The question with any references to the conversation resolved. "
                               "Preserve the user's intent; do not answer it or add facts.",
            }},
            "required": ["query"],
        },
    },
}]

INSTRUCTION = ("Rewrite the user's latest documentation question so it can be searched without "
               "the conversation. Report the result by calling standalone_query.")


def resolve_follow_up(question: str, turns: list[tuple], provider: AnswerProvider, *, limit: int = 3) -> str:
    """Make a follow-up independently searchable without treating history as evidence.

    The rewrite comes back as a tool call, so its shape is guaranteed by the
    runtime's decoder rather than scraped out of prose. The only thing left to
    guard is an empty rewrite, which is not a rewrite.
    """
    if not turns:
        return question
    context = "\n\n".join(
        f"User: {str(turn[2])[:900]}\nAssistant: {str(turn[3])[:1200]}"
        for turn in turns[-limit:]
    )
    call = provider.call_tool([
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"Recent conversation:\n{context}\n\nLatest question: {question}"},
    ], REWRITE_TOOL)
    return (call.arguments.get("query") or "").strip() or question
