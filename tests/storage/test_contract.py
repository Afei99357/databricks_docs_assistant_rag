"""Behavioural contract suite for the storage protocols.

Written against `rag.storage.protocol.ConversationStore` and obtained through
`rag.storage.create_store`, not against `DatabricksStore` internals -- so this
file runs unchanged once a second backend (e.g. SQLite) exists.

*** WRITES TO A REAL WAREHOUSE ***

This suite exercises a live store: it creates and (soft-)deletes conversation
rows in the shared Delta tables. `delete_conversation` is a soft delete (it
sets status='deleted'; it does not remove the row), so every run of this
suite leaves rows behind in `rag_conversations` / `rag_conversation_turns`.
All rows are written under the clearly-marked owner id
`storage-contract-test@example.invalid` so they are obviously test data and
easy to find/clean up later.

Skipped by default. Opt in with `RAG_STORAGE_CONTRACT=1`, e.g. via
`just test-storage-databricks`. Do NOT gate this on RAG_WAREHOUSE_ID --
.env sets that for every `just check` run (justfile has `dotenv-load :=
true`), so that guard would never skip and this suite would write to
production on every routine test run.
"""

from __future__ import annotations

import os

import pytest

from rag.config import Settings
from rag.models import Answer
from rag.storage import create_store

pytestmark = pytest.mark.skipif(
    not os.getenv("RAG_STORAGE_CONTRACT"),
    reason="opt-in: writes to a real warehouse; run via just test-storage-databricks",
)

OWNER = "storage-contract-test@example.invalid"


@pytest.fixture(scope="module")
def store():
    return create_store(Settings.from_env())


def _answer(text: str = "a test answer") -> Answer:
    return Answer(
        text=text,
        citations=(),
        supported=True,
        provider="contract-test",
        snapshot_id="contract-test-snapshot",
    )


def test_conversation_title_with_apostrophe_survives_round_trip(store):
    # This is the regression this whole project exists to prevent: a title
    # containing an apostrophe must come back byte-for-byte, not silently
    # stripped of it by a broken escaping path.
    title = "Delta Lake's ACID guarantees"

    conversation_id = store.create_conversation(OWNER, title)
    try:
        rows = store.list_conversations(OWNER)
        matching = [row for row in rows if row[0] == conversation_id]
        assert len(matching) == 1
        assert matching[0][1] == title
    finally:
        store.delete_conversation(OWNER, conversation_id)


def test_owner_cannot_read_another_owners_conversation(store):
    other_owner = "storage-contract-test-other@example.invalid"
    conversation_id = store.create_conversation(OWNER, "owned by the first owner")
    try:
        rows = store.list_conversations(other_owner)
        assert all(row[0] != conversation_id for row in rows)

        turns = store.turns_for(other_owner, conversation_id)
        assert turns == []

        with pytest.raises(PermissionError):
            store.append_turn(
                other_owner,
                conversation_id,
                question="can I see this?",
                resolved_query="can I see this?",
                answer=_answer(),
                citation_ids=[],
                latency_ms=1,
            )

        assert store.delete_conversation(other_owner, conversation_id) is False
    finally:
        store.delete_conversation(OWNER, conversation_id)


def test_delete_conversation_is_true_then_false(store):
    conversation_id = store.create_conversation(OWNER, "delete-twice contract")

    assert store.delete_conversation(OWNER, conversation_id) is True
    assert store.delete_conversation(OWNER, conversation_id) is False


def test_append_turn_round_trips_through_turns_for(store):
    conversation_id = store.create_conversation(OWNER, "turn round-trip")
    try:
        turn_id = store.append_turn(
            OWNER,
            conversation_id,
            question="what is Unity Catalog?",
            resolved_query="what is Unity Catalog?",
            answer=_answer("Unity Catalog is Databricks' governance layer."),
            citation_ids=["chunk-1", "chunk-2"],
            latency_ms=42,
        )

        turns = store.turns_for(OWNER, conversation_id)
        assert len(turns) == 1
        turn_id_col, _turn_number, question_col, answer_text_col = turns[0][:4]
        assert turn_id_col == turn_id
        assert question_col == "what is Unity Catalog?"
        assert answer_text_col == "Unity Catalog is Databricks' governance layer."
    finally:
        store.delete_conversation(OWNER, conversation_id)
