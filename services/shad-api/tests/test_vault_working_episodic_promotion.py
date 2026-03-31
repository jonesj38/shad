"""Tests for working→episodic memory promotion via ShadowIndex.promote_to_episodic.

Verifies three behavioural contracts:
  1. State transition — memory_type changes from WORKING to EPISODIC.
  2. Timestamp update — ingested_at is set to the persist event time (or now).
  3. Provenance — metadata records promoted_at and promoted_from without
     clobbering pre-existing metadata keys.

Also covers error cases (wrong state, missing snapshot) and isolation
(unrelated snapshots in the same source are untouched).
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
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "sessions-src-001"
_SOURCE_URL = "sessions://local/test-session"


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
    metadata: dict | None = None,
    ingested_at: datetime | None = None,
) -> SnapshotEntry:
    snapshot = SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=ingested_at or datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC),
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["sessions/note.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
        metadata=metadata or {},
    )
    index.add_snapshot(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# Happy-path: state transition, timestamps, provenance
# ---------------------------------------------------------------------------


class TestWorkingToEpisodicTransition:
    """Core promotion semantics: WORKING → EPISODIC."""

    def test_memory_type_transitions_to_episodic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert promoted.memory_type == MemoryType.EPISODIC

    def test_ingested_at_updated_to_persist_time(self, tmp_path: Path) -> None:
        """ingested_at must be replaced with the exact persist event timestamp."""
        index = _make_index(tmp_path)
        original_time = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING, ingested_at=original_time)
        persist_time = datetime(2026, 3, 31, 15, 30, 0, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert promoted.ingested_at == persist_time
        assert promoted.ingested_at != original_time

    def test_default_persist_time_is_close_to_now(self, tmp_path: Path) -> None:
        """When persist_time is omitted, ingested_at is stamped to approximately now."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        before = datetime.now(UTC)
        promoted = index.promote_to_episodic(snap.snapshot_id)
        after = datetime.now(UTC)

        assert before <= promoted.ingested_at <= after + timedelta(milliseconds=1)

    def test_default_persist_time_advances_ingested_at(self, tmp_path: Path) -> None:
        """Default persist_time must be after the original ingested_at."""
        original_time = datetime(2026, 1, 1, tzinfo=UTC)
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING, ingested_at=original_time)

        promoted = index.promote_to_episodic(snap.snapshot_id)

        assert promoted.ingested_at > original_time

    def test_persist_time_microseconds_preserved(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 9, 15, 45, 123456, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert promoted.ingested_at == persist_time


# ---------------------------------------------------------------------------
# Provenance: metadata updates
# ---------------------------------------------------------------------------


class TestPromotionProvenance:
    """Metadata must record the promotion event without clobbering prior keys."""

    def test_promoted_at_recorded_in_metadata(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert "promoted_at" in promoted.metadata
        assert promoted.metadata["promoted_at"] == persist_time.isoformat()

    def test_promoted_from_records_working_type(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert promoted.metadata.get("promoted_from") == MemoryType.WORKING.value

    def test_existing_metadata_preserved_on_promotion(self, tmp_path: Path) -> None:
        """Pre-existing metadata keys must survive the promotion update."""
        prior_metadata = {"session_id": "abc123", "user": "jake", "tags": ["important"]}
        index = _make_index(tmp_path)
        snap = _add_snapshot(
            index, memory_type=MemoryType.WORKING, metadata=prior_metadata
        )
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert promoted.metadata["session_id"] == "abc123"
        assert promoted.metadata["user"] == "jake"
        assert promoted.metadata["tags"] == ["important"]

    def test_promoted_at_uses_persist_time_not_original_ingested_at(
        self, tmp_path: Path
    ) -> None:
        index = _make_index(tmp_path)
        original_time = datetime(2026, 1, 15, tzinfo=UTC)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING, ingested_at=original_time)
        persist_time = datetime(2026, 3, 31, 9, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        assert promoted.metadata["promoted_at"] == persist_time.isoformat()
        assert promoted.metadata["promoted_at"] != original_time.isoformat()

    def test_default_persist_time_promoted_at_close_to_now(self, tmp_path: Path) -> None:
        """When no persist_time given, promoted_at should be near now."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        before = datetime.now(UTC)
        promoted = index.promote_to_episodic(snap.snapshot_id)
        after = datetime.now(UTC)

        promoted_at = datetime.fromisoformat(promoted.metadata["promoted_at"])
        assert before <= promoted_at <= after + timedelta(milliseconds=1)


# ---------------------------------------------------------------------------
# Round-trip: promoted state persists in the database
# ---------------------------------------------------------------------------


class TestPromotionRoundTrip:
    """Promoted state must survive a fresh read from the database."""

    def test_promoted_snapshot_readable_with_episodic_type(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        # Re-read from DB — must reflect the promotion.
        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert fetched.memory_type == MemoryType.EPISODIC
        assert fetched.ingested_at == persist_time

    def test_promoted_snapshot_appears_in_episodic_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        episodic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.EPISODIC)
        ids = [s.snapshot_id for s in episodic]

        assert snap.snapshot_id in ids

    def test_promoted_snapshot_absent_from_working_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        working = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.WORKING)
        ids = [s.snapshot_id for s in working]

        assert snap.snapshot_id not in ids

    def test_promoted_snapshot_counts_in_episodic_count(self, tmp_path: Path) -> None:
        since = datetime(2026, 3, 1, tzinfo=UTC)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=persist_time)

        count = index.episodic_count(_SOURCE_ID, since=since)
        assert count == 1


# ---------------------------------------------------------------------------
# Isolation: unrelated snapshots are untouched
# ---------------------------------------------------------------------------


class TestPromotionIsolation:
    """Promotion of one snapshot must not affect others."""

    def test_other_working_snapshots_unaffected(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap1 = _add_snapshot(index, memory_type=MemoryType.WORKING)
        snap2 = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_episodic(snap1.snapshot_id, persist_time=persist_time)

        # snap2 must remain WORKING
        fetched2 = index.get_snapshot(snap2.snapshot_id)
        assert fetched2 is not None
        assert fetched2.memory_type == MemoryType.WORKING

    def test_semantic_snapshots_unaffected(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        working_snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        semantic_snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_episodic(working_snap.snapshot_id, persist_time=persist_time)

        fetched = index.get_snapshot(semantic_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC

    def test_existing_episodic_snapshots_unaffected(self, tmp_path: Path) -> None:
        existing_time = datetime(2026, 2, 1, tzinfo=UTC)
        index = _make_index(tmp_path)
        working_snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        episodic_snap = _add_snapshot(
            index, memory_type=MemoryType.EPISODIC, ingested_at=existing_time
        )
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_episodic(working_snap.snapshot_id, persist_time=persist_time)

        fetched = index.get_snapshot(episodic_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.EPISODIC
        # ingested_at must be the original value, not the new persist_time
        assert fetched.ingested_at == existing_time

    def test_only_promoted_snapshot_has_provenance_keys(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap1 = _add_snapshot(index, memory_type=MemoryType.WORKING)
        snap2 = _add_snapshot(index, memory_type=MemoryType.WORKING)
        persist_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_episodic(snap1.snapshot_id, persist_time=persist_time)

        fetched2 = index.get_snapshot(snap2.snapshot_id)
        assert fetched2 is not None
        assert "promoted_at" not in fetched2.metadata
        assert "promoted_from" not in fetched2.metadata


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestPromotionErrors:
    """promote_to_episodic must reject invalid inputs with clear exceptions."""

    def test_missing_snapshot_raises_key_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)

        with pytest.raises(KeyError, match="not found"):
            index.promote_to_episodic("nonexistent-id")

    def test_semantic_snapshot_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError, match="working_memory"):
            index.promote_to_episodic(snap.snapshot_id)

    def test_episodic_snapshot_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

    def test_none_type_snapshot_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=None)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_contains_snapshot_id(self, tmp_path: Path) -> None:
        """The error message must identify which snapshot was rejected."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_episodic(snap.snapshot_id)

    def test_state_unchanged_after_failed_promotion(self, tmp_path: Path) -> None:
        """A failed promotion must leave the snapshot state untouched."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC
