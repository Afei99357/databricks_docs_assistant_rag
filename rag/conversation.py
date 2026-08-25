"""Small, bounded follow-up query resolution for documentation retrieval."""
from __future__ import annotations

import json

from rag.llm.providers import AnswerProvider


def resolve_follow_up(question: str, turns: list[tuple], provider: AnswerProvider, *, limit: int = 3) -> str:
    """Make a follow-up independently searchable without treating history as evidence."""
    if not turns:
        return question
    context = "\n\n".join(
        f"User: {str(turn[2])[:900]}\nAssistant: {str(turn[3])[:1200]}"
        for turn in turns[-limit:]
    )
    prompt = f'''Rewrite the user's latest documentation question so it can be searched without the conversation.
Return JSON only: {{"standalone_query":"..."}}. Preserve the user's intent. Do not answer the question and do not add facts.

Recent conversation:
{context}

Latest question: {question}'''
    try:
        value = json.loads(provider.complete(prompt)).get("standalone_query", "")
        return value.strip() if isinstance(value, str) and value.strip() else question
    except (json.JSONDecodeError, AttributeError, TypeError):
        return question
