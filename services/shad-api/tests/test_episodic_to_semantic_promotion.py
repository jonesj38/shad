"""Tests for episodic→semantic memory promotion via ShadowIndex.promote_to_semantic.

Session boundary semantics
--------------------------
The EPISODIC → SEMANTIC transition is NOT free-running; it is gated by a
*session boundary* — the moment ConsolidationConfig.should_consolidate returns
True (i.e. episodic_count >= threshold).  These tests verify three things:

  1. Session boundary gate — promote_to_semantic is only called after
     should_consolidate fires; episodic snapshots below the threshold are
     left untouched.

  2. State transition — memory_type changes from EPISODIC to SEMANTIC and
     ingested_at is replaced with the consolidation timestamp.

  3. Context carry-forward — all metadata accumulated during the episodic
     phase (session_id, tags, promoted_at, etc.) is preserved verbatim;
     the consolidation provenance keys are *added*, never substituted.

Also covers error cases (wrong source state, missing snapshot) and isolation
(unrelated snapshots in the same source are untouched).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from shad.vault.consolidation import ConsolidationConfig
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

_SOURCE_ID = "feed-src-semantic-001"
_SOURCE_URL = "https://feeds.example.com/semantic-test"


def _make_index(tmp_path: Path) -> ShadowIndex:
    index = ShadowIndex(tmp_path / "test.sqlite")
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
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
        entry_paths=["feeds/item.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
        metadata=metadata or {},
    )
    index.add_snapshot(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# Session boundary gate
# ---------------------------------------------------------------------------


class TestSessionBoundaryGate:
    """promote_to_semantic must only be called when the consolidation trigger fires.

    These tests verify the gate logic by wiring ConsolidationConfig.should_consolidate
    to episodic_count and confirming no promotion happens below the threshold.
    """

    def test_no_promotion_below_threshold(self, tmp_path: Path) -> None:
        """threshold-1 episodic snapshots → should_consolidate is False → no state change."""
        threshold = 3
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)
        base = datetime(2026, 3, 1, tzinfo=UTC)

        snaps = [
            _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=base + timedelta(hours=i + 1))
            for i in range(threshold - 1)
        ]

        count = index.episodic_count(_SOURCE_ID, since=base)
        assert cfg.should_consolidate(count) is False

        # All snapshots must remain EPISODIC — no promote_to_semantic call was warranted.
        for snap in snaps:
            fetched = index.get_snapshot(snap.snapshot_id)
            assert fetched is not None
            assert fetched.memory_type == MemoryType.EPISODIC

    def test_promotion_fires_at_exactly_threshold(self, tmp_path: Path) -> None:
        """Exactly threshold episodic snapshots → should_consolidate is True."""
        threshold = 3
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)
        base = datetime(2026, 3, 1, tzinfo=UTC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        snaps = [
            _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=base + timedelta(hours=i + 1))
            for i in range(threshold)
        ]

        count = index.episodic_count(_SOURCE_ID, since=base)
        assert cfg.should_consolidate(count) is True

        # Simulate the consolidation: promote all episodic entries to semantic.
        for snap in snaps:
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        for snap in snaps:
            fetched = index.get_snapshot(snap.snapshot_id)
            assert fetched is not None
            assert fetched.memory_type == MemoryType.SEMANTIC

    def test_boundary_does_not_fire_one_short(self, tmp_path: Path) -> None:
        """threshold-1 count: should_consolidate is False; threshold count: True."""
        threshold = 5
        cfg = ConsolidationConfig(threshold=threshold)

        assert cfg.should_consolidate(threshold - 1) is False
        assert cfg.should_consolidate(threshold) is True

    def test_promotion_fires_above_threshold(self, tmp_path: Path) -> None:
        """threshold+2 episodic entries: should_consolidate still True."""
        threshold = 3
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)
        base = datetime(2026, 3, 1, tzinfo=UTC)

        for i in range(threshold + 2):
            _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=base + timedelta(hours=i + 1))

        count = index.episodic_count(_SOURCE_ID, since=base)
        assert cfg.should_consolidate(count) is True

    def test_disabled_config_never_fires(self, tmp_path: Path) -> None:
        """Master switch off → should_consolidate never returns True."""
        cfg = ConsolidationConfig(threshold=1, enabled=False)
        index = _make_index(tmp_path)
        base = datetime(2026, 3, 1, tzinfo=UTC)

        for i in range(10):
            _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=base + timedelta(hours=i + 1))

        count = index.episodic_count(_SOURCE_ID, since=base)
        assert cfg.should_consolidate(count) is False

    def test_advancing_baseline_resets_effective_count(self, tmp_path: Path) -> None:
        """After a consolidation, advancing the since-baseline resets the window.

        Snapshots from the previous window must not re-trigger consolidation.
        """
        threshold = 3
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)

        first_baseline = datetime(2026, 3, 1, tzinfo=UTC)
        second_baseline = datetime(2026, 3, 15, tzinfo=UTC)

        # First window: threshold entries → fires.
        for i in range(threshold):
            _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=first_baseline + timedelta(hours=i + 1))
        assert cfg.should_consolidate(index.episodic_count(_SOURCE_ID, since=first_baseline)) is True

        # After advancing the baseline only threshold-1 new entries → does not fire.
        for i in range(threshold - 1):
            _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=second_baseline + timedelta(hours=i + 1))
        assert cfg.should_consolidate(index.episodic_count(_SOURCE_ID, since=second_baseline)) is False

    def test_callback_invoked_only_at_boundary(self) -> None:
        """Simulate a caller that invokes a callback only when should_consolidate fires."""
        threshold = 4
        cfg = ConsolidationConfig(threshold=threshold)
        consolidate = MagicMock()

        for count in range(threshold + 2):
            if cfg.should_consolidate(count):
                consolidate(count)

        # callback should have been invoked for count == threshold and threshold+1 only
        assert consolidate.call_count == 2
        consolidate.assert_has_calls([call(threshold), call(threshold + 1)])


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------


class TestEpisodicToSemanticTransition:
    """Core promotion semantics: EPISODIC → SEMANTIC."""

    def test_memory_type_transitions_to_semantic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert promoted.memory_type == MemoryType.SEMANTIC

    def test_ingested_at_updated_to_consolidation_time(self, tmp_path: Path) -> None:
        """ingested_at must be replaced with the exact consolidation timestamp."""
        original_time = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=original_time)
        consolidation_time = datetime(2026, 3, 31, 15, 30, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert promoted.ingested_at == consolidation_time
        assert promoted.ingested_at != original_time

    def test_default_consolidation_time_is_close_to_now(self, tmp_path: Path) -> None:
        """When consolidation_time is omitted, ingested_at is stamped to approximately now."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        before = datetime.now(UTC)
        promoted = index.promote_to_semantic(snap.snapshot_id)
        after = datetime.now(UTC)

        assert before <= promoted.ingested_at <= after + timedelta(milliseconds=1)

    def test_default_consolidation_time_advances_ingested_at(self, tmp_path: Path) -> None:
        """Default consolidation_time must be after the original ingested_at."""
        original_time = datetime(2026, 1, 1, tzinfo=UTC)
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=original_time)

        promoted = index.promote_to_semantic(snap.snapshot_id)

        assert promoted.ingested_at > original_time

    def test_consolidation_time_microseconds_preserved(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 9, 15, 45, 123456, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert promoted.ingested_at == consolidation_time


# ---------------------------------------------------------------------------
# Context carry-forward: accumulated episodic metadata must survive
# ---------------------------------------------------------------------------


class TestEpisodicContextCarryForward:
    """All metadata accumulated during the episodic phase must be preserved verbatim.

    This is the "carries forward accumulated context" contract: the semantic
    record inherits every key the episodic record built up, and the promotion
    only *appends* provenance keys without overwriting anything.
    """

    def test_episodic_metadata_preserved_after_semantic_promotion(self, tmp_path: Path) -> None:
        """Plain metadata keys from the episodic phase survive the transition."""
        episodic_metadata = {"session_id": "sess-abc", "user": "jake", "tags": ["important"]}
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC, metadata=episodic_metadata)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert promoted.metadata["session_id"] == "sess-abc"
        assert promoted.metadata["user"] == "jake"
        assert promoted.metadata["tags"] == ["important"]

    def test_promoted_at_from_episodic_phase_preserved(self, tmp_path: Path) -> None:
        """If the episodic record already carried a promoted_at key from its own
        WORKING→EPISODIC promotion, that key must not be overwritten by the
        EPISODIC→SEMANTIC consolidation.
        """
        working_promoted_at = datetime(2026, 3, 10, 8, 0, 0, tzinfo=UTC).isoformat()
        episodic_metadata = {
            "promoted_at": working_promoted_at,
            "promoted_from": MemoryType.WORKING.value,
            "session_id": "sess-xyz",
        }
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC, metadata=episodic_metadata)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        # Prior episodic provenance intact.
        assert promoted.metadata["promoted_at"] == working_promoted_at
        assert promoted.metadata["promoted_from"] == MemoryType.WORKING.value
        assert promoted.metadata["session_id"] == "sess-xyz"

    def test_consolidated_at_added_to_metadata(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert "consolidated_at" in promoted.metadata
        assert promoted.metadata["consolidated_at"] == consolidation_time.isoformat()

    def test_consolidated_from_records_episodic_type(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert promoted.metadata.get("consolidated_from") == MemoryType.EPISODIC.value

    def test_consolidated_at_uses_consolidation_time_not_original_ingested_at(
        self, tmp_path: Path
    ) -> None:
        original_time = datetime(2026, 1, 15, tzinfo=UTC)
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC, ingested_at=original_time)
        consolidation_time = datetime(2026, 3, 31, 9, 0, 0, tzinfo=UTC)

        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        assert promoted.metadata["consolidated_at"] == consolidation_time.isoformat()
        assert promoted.metadata["consolidated_at"] != original_time.isoformat()

    def test_default_consolidation_time_consolidated_at_close_to_now(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)

        before = datetime.now(UTC)
        promoted = index.promote_to_semantic(snap.snapshot_id)
        after = datetime.now(UTC)

        consolidated_at = datetime.fromisoformat(promoted.metadata["consolidated_at"])
        assert before <= consolidated_at <= after + timedelta(milliseconds=1)

    def test_accumulated_metadata_unchanged_after_full_working_episodic_semantic_lifecycle(
        self, tmp_path: Path
    ) -> None:
        """Full WORKING → EPISODIC → SEMANTIC lifecycle: all accumulated context survives."""
        initial_metadata = {"task_id": "task-99", "priority": "high"}
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING, metadata=initial_metadata)

        # Step 1: WORKING → EPISODIC
        episodic_time = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)
        episodic = index.promote_to_episodic(snap.snapshot_id, persist_time=episodic_time)
        assert episodic.memory_type == MemoryType.EPISODIC

        # Step 2: EPISODIC → SEMANTIC
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        semantic = index.promote_to_semantic(episodic.snapshot_id, consolidation_time=consolidation_time)

        # Original metadata intact.
        assert semantic.metadata["task_id"] == "task-99"
        assert semantic.metadata["priority"] == "high"
        # Episodic provenance intact.
        assert semantic.metadata["promoted_at"] == episodic_time.isoformat()
        assert semantic.metadata["promoted_from"] == MemoryType.WORKING.value
        # Semantic provenance added.
        assert semantic.metadata["consolidated_at"] == consolidation_time.isoformat()
        assert semantic.metadata["consolidated_from"] == MemoryType.EPISODIC.value
        # Final type.
        assert semantic.memory_type == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# Round-trip: promoted state persists in the database
# ---------------------------------------------------------------------------


class TestSemanticPromotionRoundTrip:
    """Promoted state must survive a fresh read from the database."""

    def test_promoted_snapshot_readable_with_semantic_type(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC
        assert fetched.ingested_at == consolidation_time

    def test_promoted_snapshot_appears_in_semantic_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        semantic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.SEMANTIC)
        ids = [s.snapshot_id for s in semantic]

        assert snap.snapshot_id in ids

    def test_promoted_snapshot_absent_from_episodic_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        episodic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.EPISODIC)
        ids = [s.snapshot_id for s in episodic]

        assert snap.snapshot_id not in ids

    def test_metadata_persisted_with_provenance_keys(self, tmp_path: Path) -> None:
        prior_metadata = {"origin": "morning-standup"}
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC, metadata=prior_metadata)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.metadata["origin"] == "morning-standup"
        assert fetched.metadata["consolidated_at"] == consolidation_time.isoformat()
        assert fetched.metadata["consolidated_from"] == MemoryType.EPISODIC.value


# ---------------------------------------------------------------------------
# Isolation: unrelated snapshots are untouched
# ---------------------------------------------------------------------------


class TestSemanticPromotionIsolation:
    """Promotion of one snapshot must not affect others."""

    def test_other_episodic_snapshots_unaffected(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap1 = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        snap2 = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_semantic(snap1.snapshot_id, consolidation_time=consolidation_time)

        fetched2 = index.get_snapshot(snap2.snapshot_id)
        assert fetched2 is not None
        assert fetched2.memory_type == MemoryType.EPISODIC

    def test_working_snapshots_unaffected(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        episodic_snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        working_snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_semantic(episodic_snap.snapshot_id, consolidation_time=consolidation_time)

        fetched = index.get_snapshot(working_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.WORKING

    def test_existing_semantic_snapshots_unaffected(self, tmp_path: Path) -> None:
        existing_time = datetime(2026, 2, 1, tzinfo=UTC)
        index = _make_index(tmp_path)
        episodic_snap = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        semantic_snap = _add_snapshot(
            index, memory_type=MemoryType.SEMANTIC, ingested_at=existing_time
        )
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_semantic(episodic_snap.snapshot_id, consolidation_time=consolidation_time)

        fetched = index.get_snapshot(semantic_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC
        assert fetched.ingested_at == existing_time

    def test_only_promoted_snapshot_has_provenance_keys(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap1 = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        snap2 = _add_snapshot(index, memory_type=MemoryType.EPISODIC)
        consolidation_time = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

        index.promote_to_semantic(snap1.snapshot_id, consolidation_time=consolidation_time)

        fetched2 = index.get_snapshot(snap2.snapshot_id)
        assert fetched2 is not None
        assert "consolidated_at" not in fetched2.metadata
        assert "consolidated_from" not in fetched2.metadata


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestSemanticPromotionErrors:
    """promote_to_semantic must reject invalid inputs with clear exceptions."""

    def test_missing_snapshot_raises_key_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)

        with pytest.raises(KeyError, match="not found"):
            index.promote_to_semantic("nonexistent-id")

    def test_working_snapshot_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError, match="episodic_memory"):
            index.promote_to_semantic(snap.snapshot_id)

    def test_semantic_snapshot_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.SEMANTIC)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

    def test_none_type_snapshot_raises_value_error(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=None)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

    def test_error_message_contains_snapshot_id(self, tmp_path: Path) -> None:
        """The error message must identify which snapshot was rejected."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_semantic(snap.snapshot_id)

    def test_state_unchanged_after_failed_promotion(self, tmp_path: Path) -> None:
        """A failed promotion must leave the snapshot state untouched."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.WORKING
