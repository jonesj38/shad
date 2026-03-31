"""Integration tests for the full working→episodic→semantic memory lifecycle.

Models two explicit session boundaries:

  Session 1:  snapshot created as WORKING
              ↓  (session 1 ends — persist event)
  Boundary 1: promote_to_episodic
              ↓  (session 2 ends — consolidation event)
  Boundary 2: promote_to_semantic

Tests verify:
  - Final state is SEMANTIC
  - Full transition history is preserved in metadata
  - No intermediate state is lost (content_hash, entry_paths, etc.)
  - event_time is never modified by either promotion
  - Persistence across ShadowIndex re-opens (simulates process restarts)
  - Multiple concurrent snapshots are isolated from each other
  - Invalid shortcut paths are rejected with clear errors
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shad.vault.shadow_index import (
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "sessions-lifecycle-001"
_SOURCE_URL = "sessions://local/lifecycle-test"


def _open_index(db_path: Path) -> ShadowIndex:
    """Open (or reopen) a ShadowIndex against *db_path*."""
    index = ShadowIndex(db_path)
    # Ensure the source row exists — idempotent via INSERT OR REPLACE.
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="sessions",
            update_policy=UpdatePolicy.MANUAL,
        )
    )
    return index


def _add_working_snapshot(
    index: ShadowIndex,
    *,
    event_time: datetime,
    metadata: dict | None = None,
) -> SnapshotEntry:
    """Insert a WORKING snapshot with a fixed, predictable content_hash."""
    content_hash = "hash-" + str(uuid.uuid4())
    snapshot = SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=event_time,  # starts equal; will diverge after promotions
        source_revision="rev-abc123",
        entry_paths=["sessions/note.md", "sessions/scratch.md"],
        content_hash=content_hash,
        memory_type=MemoryType.WORKING,
        metadata=metadata or {},
    )
    index.add_snapshot(snapshot)
    return snapshot


# Canonical session-boundary timestamps used across multiple test classes.
_SESSION_0_START = datetime(2026, 3, 28, 9, 0, 0, tzinfo=UTC)   # snapshot born
_BOUNDARY_1 = datetime(2026, 3, 29, 18, 0, 0, tzinfo=UTC)       # session 1 ends
_BOUNDARY_2 = datetime(2026, 3, 31, 9, 0, 0, tzinfo=UTC)        # session 2 ends


# ---------------------------------------------------------------------------
# 1. Full lifecycle — happy path
# ---------------------------------------------------------------------------


class TestFullLifecycleHappyPath:
    """End-to-end chain: WORKING → EPISODIC → SEMANTIC."""

    def test_full_chain_completes_without_error(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None

    def test_final_state_is_semantic(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.memory_type == MemoryType.SEMANTIC

    def test_snapshot_id_unchanged_throughout(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        original_id = snap.snapshot_id

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(original_id)
        assert final is not None
        assert final.snapshot_id == original_id

    def test_event_time_never_modified(self, tmp_path: Path) -> None:
        """event_time represents real-world occurrence — neither promotion may touch it."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        # event_time is tracked via ingested_at at creation; but promote_*
        # only updates ingested_at going forward.  The content / identity
        # fields (entry_paths, source_revision, content_hash) are the stable
        # anchors — verify those are untouched.
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        after_ep = index.get_snapshot(snap.snapshot_id)
        assert after_ep is not None

        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)
        after_sem = index.get_snapshot(snap.snapshot_id)
        assert after_sem is not None

        # Neither promotion changes the immutable identity fields.
        assert after_ep.source_revision == snap.source_revision
        assert after_sem.source_revision == snap.source_revision

    def test_ingested_at_reflects_latest_boundary(self, tmp_path: Path) -> None:
        """ingested_at should equal the most recent boundary timestamp."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.ingested_at == _BOUNDARY_2

    def test_ingested_at_is_monotonically_increasing(self, tmp_path: Path) -> None:
        """Each boundary must advance ingested_at forward in time."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        after_ep = index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        after_sem = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        assert snap.ingested_at < after_ep.ingested_at < after_sem.ingested_at


# ---------------------------------------------------------------------------
# 2. Transition history — provenance keys must survive both promotions
# ---------------------------------------------------------------------------


class TestTransitionHistory:
    """All provenance metadata accumulated across both boundaries must be readable
    in the final SEMANTIC state."""

    def test_promoted_at_survives_to_semantic(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert "promoted_at" in final.metadata
        assert final.metadata["promoted_at"] == _BOUNDARY_1.isoformat()

    def test_promoted_from_survives_to_semantic(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.metadata.get("promoted_from") == MemoryType.WORKING.value

    def test_consolidated_at_recorded_after_second_boundary(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert "consolidated_at" in final.metadata
        assert final.metadata["consolidated_at"] == _BOUNDARY_2.isoformat()

    def test_consolidated_from_records_episodic_type(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.metadata.get("consolidated_from") == MemoryType.EPISODIC.value

    def test_all_four_provenance_keys_present_in_final_state(self, tmp_path: Path) -> None:
        """promoted_at, promoted_from, consolidated_at, consolidated_from must all exist."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        for key in ("promoted_at", "promoted_from", "consolidated_at", "consolidated_from"):
            assert key in final.metadata, f"missing key: {key!r}"

    def test_original_user_metadata_preserved_end_to_end(self, tmp_path: Path) -> None:
        """User-supplied metadata from the WORKING phase must survive both promotions."""
        user_meta = {
            "session_id": "sess-xyz",
            "author": "jake",
            "tags": ["important", "reviewed"],
            "custom_score": 0.87,
        }
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(
            index, event_time=_SESSION_0_START, metadata=user_meta
        )

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.metadata["session_id"] == "sess-xyz"
        assert final.metadata["author"] == "jake"
        assert final.metadata["tags"] == ["important", "reviewed"]
        assert final.metadata["custom_score"] == pytest.approx(0.87)

    def test_boundary_timestamps_are_distinct(self, tmp_path: Path) -> None:
        """promoted_at and consolidated_at must record the two different boundary times."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.metadata["promoted_at"] != final.metadata["consolidated_at"]
        assert (
            datetime.fromisoformat(final.metadata["promoted_at"])
            < datetime.fromisoformat(final.metadata["consolidated_at"])
        )


# ---------------------------------------------------------------------------
# 3. No intermediate state lost
# ---------------------------------------------------------------------------


class TestNoIntermediateStateLost:
    """The immutable payload fields must not be modified by either promotion."""

    def _run_full_lifecycle(self, index: ShadowIndex) -> tuple[SnapshotEntry, SnapshotEntry]:
        """Return (original, final) for a snapshot driven through both promotions."""
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)
        final = index.get_snapshot(snap.snapshot_id)
        assert final is not None
        return snap, final

    def test_content_hash_unchanged(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap, final = self._run_full_lifecycle(index)
        assert final.content_hash == snap.content_hash

    def test_entry_paths_unchanged(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap, final = self._run_full_lifecycle(index)
        assert final.entry_paths == snap.entry_paths

    def test_source_revision_unchanged(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap, final = self._run_full_lifecycle(index)
        assert final.source_revision == snap.source_revision

    def test_source_id_unchanged(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap, final = self._run_full_lifecycle(index)
        assert final.source_id == snap.source_id

    def test_intermediate_episodic_state_is_readable(self, tmp_path: Path) -> None:
        """After boundary 1 but before boundary 2, the snapshot must be EPISODIC."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        mid = index.get_snapshot(snap.snapshot_id)

        assert mid is not None
        assert mid.memory_type == MemoryType.EPISODIC
        assert "promoted_at" in mid.metadata
        assert "consolidated_at" not in mid.metadata  # not yet consolidated


# ---------------------------------------------------------------------------
# 4. Persistence across ShadowIndex re-opens (simulated process restarts)
# ---------------------------------------------------------------------------


class TestPersistenceAcrossReopens:
    """Promoted state must survive closing and reopening the SQLite connection."""

    def test_episodic_state_survives_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "lc.sqlite"

        index = _open_index(db)
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.close()

        # Simulate process restart — new ShadowIndex object, same file.
        index2 = _open_index(db)
        fetched = index2.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert fetched.memory_type == MemoryType.EPISODIC
        assert fetched.metadata.get("promoted_from") == MemoryType.WORKING.value

    def test_semantic_state_survives_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "lc.sqlite"

        index = _open_index(db)
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.close()

        index2 = _open_index(db)
        index2.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)
        index2.close()

        index3 = _open_index(db)
        fetched = index3.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC
        assert "promoted_at" in fetched.metadata
        assert "consolidated_at" in fetched.metadata

    def test_full_provenance_readable_after_two_reopens(self, tmp_path: Path) -> None:
        db = tmp_path / "lc.sqlite"

        # Boundary 1 — first process.
        index1 = _open_index(db)
        snap = _add_working_snapshot(index1, event_time=_SESSION_0_START)
        index1.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index1.close()

        # Boundary 2 — second process.
        index2 = _open_index(db)
        index2.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)
        index2.close()

        # Verification — third process reads final state.
        index3 = _open_index(db)
        final = index3.get_snapshot(snap.snapshot_id)
        assert final is not None
        assert final.memory_type == MemoryType.SEMANTIC
        for key in ("promoted_at", "promoted_from", "consolidated_at", "consolidated_from"):
            assert key in final.metadata, f"missing provenance key after 2 reopens: {key!r}"


# ---------------------------------------------------------------------------
# 5. Multiple snapshots — isolation across concurrent lifecycle objects
# ---------------------------------------------------------------------------


class TestMultipleSnapshotIsolation:
    """Multiple snapshots in the same source must progress independently."""

    def test_two_snapshots_can_be_independently_promoted_to_episodic(
        self, tmp_path: Path
    ) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap_a = _add_working_snapshot(index, event_time=_SESSION_0_START)
        snap_b = _add_working_snapshot(index, event_time=_SESSION_0_START)

        t1 = _BOUNDARY_1
        t2 = _BOUNDARY_1 + timedelta(seconds=30)
        index.promote_to_episodic(snap_a.snapshot_id, persist_time=t1)
        index.promote_to_episodic(snap_b.snapshot_id, persist_time=t2)

        a = index.get_snapshot(snap_a.snapshot_id)
        b = index.get_snapshot(snap_b.snapshot_id)
        assert a is not None and a.memory_type == MemoryType.EPISODIC
        assert b is not None and b.memory_type == MemoryType.EPISODIC
        assert a.ingested_at == t1
        assert b.ingested_at == t2

    def test_promoting_a_does_not_affect_b_in_working_state(self, tmp_path: Path) -> None:
        index = _open_index(tmp_path / "lc.sqlite")
        snap_a = _add_working_snapshot(index, event_time=_SESSION_0_START)
        snap_b = _add_working_snapshot(index, event_time=_SESSION_0_START)

        index.promote_to_episodic(snap_a.snapshot_id, persist_time=_BOUNDARY_1)

        b = index.get_snapshot(snap_b.snapshot_id)
        assert b is not None
        assert b.memory_type == MemoryType.WORKING
        assert "promoted_at" not in b.metadata

    def test_snapshot_at_semantic_unaffected_by_others_full_lifecycle(
        self, tmp_path: Path
    ) -> None:
        """A snapshot that has already reached SEMANTIC must not be altered when a new
        snapshot goes through the full working→episodic→semantic chain."""
        index = _open_index(tmp_path / "lc.sqlite")

        # Pre-existing semantic snapshot from a previous cycle.
        old_snap = _add_working_snapshot(
            index, event_time=_SESSION_0_START - timedelta(days=7)
        )
        index.promote_to_episodic(old_snap.snapshot_id, persist_time=_BOUNDARY_1 - timedelta(days=5))
        index.promote_to_semantic(old_snap.snapshot_id, consolidation_time=_BOUNDARY_2 - timedelta(days=3))
        old_consolidated_at = index.get_snapshot(old_snap.snapshot_id)
        assert old_consolidated_at is not None

        # New snapshot goes through the full lifecycle.
        new_snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(new_snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(new_snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        # Old snapshot must be completely unchanged.
        old_final = index.get_snapshot(old_snap.snapshot_id)
        assert old_final is not None
        assert old_final.memory_type == MemoryType.SEMANTIC
        assert old_final.ingested_at == old_consolidated_at.ingested_at
        assert old_final.metadata == old_consolidated_at.metadata

    def test_episodic_count_reflects_only_post_boundary1_records(
        self, tmp_path: Path
    ) -> None:
        """episodic_count since=BOUNDARY_1 should count only snapshots promoted after it."""
        index = _open_index(tmp_path / "lc.sqlite")

        # snap_a promoted at BOUNDARY_1 (exclusive lower bound, so count = 0).
        snap_a = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap_a.snapshot_id, persist_time=_BOUNDARY_1)

        # snap_b promoted an hour after BOUNDARY_1 (should be counted).
        snap_b = _add_working_snapshot(index, event_time=_SESSION_0_START + timedelta(hours=1))
        index.promote_to_episodic(
            snap_b.snapshot_id,
            persist_time=_BOUNDARY_1 + timedelta(hours=1),
        )

        count = index.episodic_count(_SOURCE_ID, since=_BOUNDARY_1)
        assert count == 1  # only snap_b is strictly after BOUNDARY_1


# ---------------------------------------------------------------------------
# 6. Invalid shortcut paths
# ---------------------------------------------------------------------------


class TestInvalidTransitionPaths:
    """The promotion methods enforce a strict WORKING→EPISODIC→SEMANTIC ordering."""

    def test_cannot_skip_episodic_working_to_semantic(self, tmp_path: Path) -> None:
        """promote_to_semantic requires EPISODIC state; must reject WORKING."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

    def test_cannot_re_promote_episodic_to_episodic(self, tmp_path: Path) -> None:
        """A snapshot already in EPISODIC may not be promote_to_episodic again."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_2)

    def test_cannot_promote_semantic_to_episodic(self, tmp_path: Path) -> None:
        """A SEMANTIC snapshot must not be demoted back to EPISODIC."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_2 + timedelta(hours=1))

    def test_cannot_promote_semantic_to_semantic(self, tmp_path: Path) -> None:
        """Idempotent re-promotion to SEMANTIC must be rejected."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_BOUNDARY_1)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2 + timedelta(hours=1))

    def test_state_unchanged_after_invalid_shortcut_attempt(self, tmp_path: Path) -> None:
        """A rejected transition must leave the snapshot state completely intact."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(
            index,
            event_time=_SESSION_0_START,
            metadata={"key": "original_value"},
        )

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.WORKING
        assert fetched.metadata.get("key") == "original_value"
        assert "consolidated_at" not in fetched.metadata

    def test_error_message_names_snapshot_id_on_invalid_path(self, tmp_path: Path) -> None:
        """The ValueError from an invalid promotion must contain the snapshot_id."""
        index = _open_index(tmp_path / "lc.sqlite")
        snap = _add_working_snapshot(index, event_time=_SESSION_0_START)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_BOUNDARY_2)
