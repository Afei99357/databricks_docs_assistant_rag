from rag.ingest.lifecycle import (
    REMOVAL_THRESHOLD,
    DocumentState,
    apply_fetch_outcome,
    needs_reindex,
)


def test_transient_failure_preserves_active_document():
    state = DocumentState(status="ok", content_hash="v1", indexed_content_hash="v1")
    assert apply_fetch_outcome(state, "network_error", now="now") == state


def test_three_confirmed_404s_are_required_for_removal():
    state = DocumentState(status="ok")
    for _ in range(REMOVAL_THRESHOLD - 1):
        state = apply_fetch_outcome(state, "not_found", now="now")
        assert state.status == "ok"
    state = apply_fetch_outcome(state, "not_found", now="later")
    assert state.status == "removed"
    assert state.removed_at == "later"


def test_successful_fetch_resets_missing_counter():
    state = DocumentState(status="ok", consecutive_404_count=2)
    assert apply_fetch_outcome(state, "ok", now="now").consecutive_404_count == 0


def test_content_hash_drives_reindexing():
    assert needs_reindex(DocumentState(status="ok", content_hash="new", indexed_content_hash="old"))
    assert not needs_reindex(
        DocumentState(status="ok", content_hash="same", indexed_content_hash="same")
    )


def test_source_last_updated_can_conservatively_request_reindexing():
    state = DocumentState(status="ok", content_hash="same", indexed_content_hash="same")
    assert needs_reindex(state, source_last_updated_changed=True)
