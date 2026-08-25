"""Pure lifecycle rules for source refreshes.

No failed or partial refresh can make a previously active document disappear.
In particular, an official page is only marked removed after three consecutive
successful runs report a 404.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

REMOVAL_THRESHOLD = 3


@dataclass(frozen=True)
class DocumentState:
    status: str  # ok | fetch_failed | removed
    consecutive_404_count: int = 0
    removed_at: str | None = None
    content_hash: str | None = None
    indexed_content_hash: str | None = None


def apply_fetch_outcome(
    previous: DocumentState | None,
    outcome: str,
    *,
    now: str,
) -> DocumentState:
    """Apply a fetch outcome without losing a usable prior document on errors."""
    if previous is None:
        previous = DocumentState(status="fetch_failed")
    if outcome == "ok":
        return replace(previous, status="ok", consecutive_404_count=0)
    if outcome == "not_found":
        count = previous.consecutive_404_count + 1
        return replace(previous, status="removed" if count >= REMOVAL_THRESHOLD else previous.status,
                       consecutive_404_count=count,
                       removed_at=now if count >= REMOVAL_THRESHOLD else previous.removed_at)
    if outcome not in {"http_error", "network_error"}:
        raise ValueError(f"unknown fetch outcome: {outcome}")
    return previous


def needs_reindex(state: DocumentState, *, source_last_updated_changed: bool = False) -> bool:
    """A content change (or distinct source update signal) requests a new index build."""
    return state.content_hash != state.indexed_content_hash or source_last_updated_changed

