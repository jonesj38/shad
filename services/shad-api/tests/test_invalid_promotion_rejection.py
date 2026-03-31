"""Tests for invalid promotion rejection in the memory hierarchy.

The valid promotion path is strictly one-directional:
  WORKING → EPISODIC → SEMANTIC

Any attempt to skip a step, demote, or loop back must be rejected with a
descriptive ValueError that identifies the snapshot and the violated constraint.

Three invalid-transition patterns are covered:

  1. working→semantic (skip)      — promote_to_semantic on a WORKING snapshot
  2. semantic→episodic (demotion) — promote_to_episodic on a SEMANTIC snapshot
  3. episodic→working (demotion)  — promote_to_episodic on an EPISODIC snapshot
                                    (already past WORKING; cannot be re-promoted
                                    or rolled back)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "invalid-promo-src-001"
_SOURCE_URL = "sessions://local/invalid-promotion-test"


def _make_index(tmp_path: Path) -> ShadowIndex:
    index = ShadowIndex(tmp_path / "test.sqlite")
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="sessions",
            update_policy=UpdatePolicy.MANUAL,
        )
    )
    return index


def _add_snapshot(
    index: ShadowIndex,
    *,
    memory_type: MemoryType | None,
) -> SnapshotEntry:
    snapshot = SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["sessions/note.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
        metadata={},
    )
    index.add_snapshot(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# 1. working → semantic skip (missing EPISODIC step)
# ---------------------------------------------------------------------------


class TestWorkingToSemanticSkip:
    """promote_to_semantic must reject WORKING snapshots.

    The caller must first promote WORKING → EPISODIC; skipping the episodic
    phase is not allowed regardless of the caller's intent.
    """

    def test_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

    def test_error_message_cites_expected_episodic_type(self, tmp_path: Path) -> None:
        """Error must name the required source type (episodic_memory)."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError, match=MemoryType.EPISODIC.value):
            index.promote_to_semantic(snap.snapshot_id)

    def test_error_message_contains_snapshot_id(self, tmp_path: Path) -> None:
        """Error must identify the offending snapshot."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_semantic(snap.snapshot_id)

    def test_snapshot_remains_working_after_rejection(self, tmp_path: Path) -> None:
        """Rejected call must leave memory_type unchanged."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.WORKING

    def test_no_consolidation_provenance_added_on_rejection(self, tmp_path: Path) -> None:
        """consolidated_at / consolidated_from must not be written on rejection."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert "consolidated_at" not in fetched.metadata
        assert "consolidated_from" not in fetched.metadata


# ---------------------------------------------------------------------------
# 2. semantic → episodic demotion
# ---------------------------------------------------------------------------


class TestSemanticToEpisodicDemotion:
    """promote_to_episodic must reject SEMANTIC snapshots.

    SEMANTIC is the terminal state in the hierarchy.  There is no valid
    demotion path; once consolidated, a snapshot cannot be moved back.
    """

    def test_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_cites_expected_working_type(self, tmp_path: Path) -> None:
        """Error must name the required source type (working_memory)."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError, match=MemoryType.WORKING.value):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_contains_snapshot_id(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_episodic(snap.snapshot_id)

    def test_snapshot_remains_semantic_after_rejection(self, tmp_path: Path) -> None:
        """Rejected demotion must leave memory_type unchanged."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC

    def test_no_promotion_provenance_added_on_rejection(self, tmp_path: Path) -> None:
        """promoted_at / promoted_from must not be written on a rejected demotion."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert "promoted_at" not in fetched.metadata
        assert "promoted_from" not in fetched.metadata

    def test_fully_promoted_snapshot_cannot_be_demoted(self, tmp_path: Path) -> None:
        """A snapshot that completed WORKING → EPISODIC → SEMANTIC must still
        reject a subsequent promote_to_episodic call."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        episodic = index.promote_to_episodic(snap.snapshot_id)
        semantic = index.promote_to_semantic(episodic.snapshot_id)

        with pytest.raises(ValueError):
            index.promote_to_episodic(semantic.snapshot_id)

        fetched = index.get_snapshot(semantic.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# 3. episodic → working demotion (promote_to_episodic on EPISODIC snapshot)
# ---------------------------------------------------------------------------


class TestEpisodicToWorkingDemotion:
    """promote_to_episodic must reject EPISODIC snapshots.

    An EPISODIC snapshot has already been persisted.  Calling promote_to_episodic
    again would constitute a silent re-promotion / rollback attempt.  The
    operation must raise ValueError; the caller should use promote_to_semantic
    to advance it further.
    """

    def test_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_cites_expected_working_type(self, tmp_path: Path) -> None:
        """Error must name the required source type (working_memory)."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        with pytest.raises(ValueError, match=MemoryType.WORKING.value):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_contains_snapshot_id(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_episodic(snap.snapshot_id)

    def test_snapshot_remains_episodic_after_rejection(self, tmp_path: Path) -> None:
        """Rejected call must leave memory_type unchanged."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.EPISODIC

    def test_existing_provenance_not_overwritten_on_rejection(
        self, tmp_path: Path
    ) -> None:
        """Provenance written during the valid WORKING → EPISODIC step must
        survive a subsequent (invalid) promote_to_episodic call."""
        persist_time = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        index = _make_index(tmp_path)
        working_snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        episodic = index.promote_to_episodic(
            working_snap.snapshot_id, persist_time=persist_time
        )
        original_promoted_at = episodic.metadata["promoted_at"]

        with pytest.raises(ValueError):
            index.promote_to_episodic(episodic.snapshot_id)

        fetched = index.get_snapshot(episodic.snapshot_id)
        assert fetched is not None
        assert fetched.metadata["promoted_at"] == original_promoted_at
        assert fetched.metadata["promoted_from"] == MemoryType.WORKING.value

    def test_idempotent_call_rejected_not_silently_ignored(
        self, tmp_path: Path
    ) -> None:
        """A second promote_to_episodic on the same snapshot must raise ValueError,
        not silently succeed or act as a no-op."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)


# ---------------------------------------------------------------------------
# Cross-cutting: atomicity across all three invalid patterns
# ---------------------------------------------------------------------------


class TestInvalidPromotionAtomicity:
    """Every invalid promotion must be atomic: raise ValueError with no state change."""

    @pytest.mark.parametrize(
        "from_type, promote_method",
        [
            (MemoryType.WORKING, "promote_to_semantic"),   # skip
            (MemoryType.SEMANTIC, "promote_to_episodic"),  # demotion
            (MemoryType.EPISODIC, "promote_to_episodic"),  # demotion / re-promote
        ],
    )
    def test_always_raises_value_error(
        self,
        tmp_path: Path,
        from_type: MemoryType,
        promote_method: str,
    ) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=from_type)

        with pytest.raises(ValueError):
            getattr(index, promote_method)(snap.snapshot_id)

    @pytest.mark.parametrize(
        "from_type, promote_method",
        [
            (MemoryType.WORKING, "promote_to_semantic"),
            (MemoryType.SEMANTIC, "promote_to_episodic"),
            (MemoryType.EPISODIC, "promote_to_episodic"),
        ],
    )
    def test_memory_type_unchanged_after_rejection(
        self,
        tmp_path: Path,
        from_type: MemoryType,
        promote_method: str,
    ) -> None:
        """memory_type must be identical before and after a rejected call."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=from_type)

        with pytest.raises(ValueError):
            getattr(index, promote_method)(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == from_type

    @pytest.mark.parametrize(
        "from_type, promote_method",
        [
            (MemoryType.WORKING, "promote_to_semantic"),
            (MemoryType.SEMANTIC, "promote_to_episodic"),
            (MemoryType.EPISODIC, "promote_to_episodic"),
        ],
    )
    def test_error_message_contains_snapshot_id(
        self,
        tmp_path: Path,
        from_type: MemoryType,
        promote_method: str,
    ) -> None:
        """Every rejection must name the offending snapshot in the error message."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=from_type)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            getattr(index, promote_method)(snap.snapshot_id)
