"""Tests for episodic→semantic memory promotion via ShadowIndex.promote_to_semantic.

Mirrors the structure of test_vault_working_episodic_promotion.py for the second
transition in the WORKING → EPISODIC → SEMANTIC lifecycle.

Verifies:
  1. State transition   — memory_type changes from EPISODIC to SEMANTIC.
  2. Timestamp update   — ingested_at is set to the consolidation event time (or now).
  3. Provenance         — metadata records consolidated_at and consolidated_from;
                          prior keys (including promoted_at from the first promotion)
                          are preserved.
  4. Round-trip         — consolidated state survives a fresh read from the database.
  5. Isolation          — unrelated snapshots in the same source are untouched.
  6. Error cases        — wrong state and missing snapshot raise the expected exceptions.
  7. promotion_history  — both promote_to_episodic and promote_to_semantic append
                          correct PromotionHistoryEntry records; the full chain is
                          preserved in order across both transitions.
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
# Shared helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "sessions-episodic-semantic-001"
_SOURCE_URL = "sessions://local/episodic-semantic-test"

_PERSIST_TIME = datetime(2026, 3, 29, 18, 0, 0, tzinfo=UTC)
_CONSOLIDATION_TIME = datetime(2026, 3, 31, 9, 0, 0, tzinfo=UTC)


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
        ingested_at=ingested_at or datetime(2026, 3, 28, 9, 0, 0, tzinfo=UTC),
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["sessions/note.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
        metadata=metadata or {},
    )
    index.add_snapshot(snapshot)
    return snapshot


def _add_promoted_snapshot(
    index: ShadowIndex,
    *,
    metadata: dict | None = None,
) -> SnapshotEntry:
    """Add a WORKING snapshot and advance it to EPISODIC via promote_to_episodic."""
    snap = _add_snapshot(index, memory_type=MemoryType.WORKING, metadata=metadata)
    return index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)


# ---------------------------------------------------------------------------
# 1. State transition: EPISODIC → SEMANTIC
# ---------------------------------------------------------------------------


class TestEpisodicToSemanticTransition:
    """Core promotion semantics: EPISODIC → SEMANTIC."""

    def test_memory_type_transitions_to_semantic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.memory_type == MemoryType.SEMANTIC

    def test_ingested_at_updated_to_consolidation_time(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.ingested_at == _CONSOLIDATION_TIME

    def test_ingested_at_differs_from_episodic_persist_time(self, tmp_path: Path) -> None:
        """ingested_at after semantic promotion must be the consolidation time, not
        the earlier persist_time that was written during episodic promotion."""
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.ingested_at == _CONSOLIDATION_TIME
        assert consolidated.ingested_at != _PERSIST_TIME

    def test_default_consolidation_time_is_close_to_now(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        before = datetime.now(UTC)
        consolidated = index.promote_to_semantic(snap.snapshot_id)
        after = datetime.now(UTC)

        assert before <= consolidated.ingested_at <= after + timedelta(milliseconds=1)

    def test_default_consolidation_time_advances_ingested_at(self, tmp_path: Path) -> None:
        """Default consolidation_time must be after the episodic ingested_at."""
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        episodic_ingested_at = snap.ingested_at

        consolidated = index.promote_to_semantic(snap.snapshot_id)

        assert consolidated.ingested_at > episodic_ingested_at

    def test_consolidation_time_microseconds_preserved(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        consolidation_time = datetime(2026, 3, 31, 9, 15, 45, 123456, tzinfo=UTC)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=consolidation_time
        )

        assert consolidated.ingested_at == consolidation_time


# ---------------------------------------------------------------------------
# 2. Provenance: metadata updates
# ---------------------------------------------------------------------------


class TestSemanticPromotionProvenance:
    """metadata must record the consolidation event without clobbering prior keys."""

    def test_consolidated_at_recorded_in_metadata(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert "consolidated_at" in consolidated.metadata
        assert consolidated.metadata["consolidated_at"] == _CONSOLIDATION_TIME.isoformat()

    def test_consolidated_from_records_episodic_type(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.metadata.get("consolidated_from") == MemoryType.EPISODIC.value

    def test_prior_promoted_at_key_preserved_after_semantic_promotion(
        self, tmp_path: Path
    ) -> None:
        """promoted_at written during the episodic promotion must survive semantic promotion."""
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert "promoted_at" in consolidated.metadata
        assert consolidated.metadata["promoted_at"] == _PERSIST_TIME.isoformat()

    def test_prior_promoted_from_key_preserved_after_semantic_promotion(
        self, tmp_path: Path
    ) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.metadata.get("promoted_from") == MemoryType.WORKING.value

    def test_user_metadata_preserved_through_both_promotions(self, tmp_path: Path) -> None:
        """Arbitrary user metadata set before any promotion must survive both transitions."""
        prior_metadata = {"session_id": "xyz999", "user": "jake", "tags": ["important"]}
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING, metadata=prior_metadata)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.metadata["session_id"] == "xyz999"
        assert consolidated.metadata["user"] == "jake"
        assert consolidated.metadata["tags"] == ["important"]

    def test_consolidated_at_uses_consolidation_time_not_persist_time(
        self, tmp_path: Path
    ) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.metadata["consolidated_at"] == _CONSOLIDATION_TIME.isoformat()
        assert consolidated.metadata["consolidated_at"] != _PERSIST_TIME.isoformat()


# ---------------------------------------------------------------------------
# 3. Round-trip: consolidated state persists in the database
# ---------------------------------------------------------------------------


class TestSemanticPromotionRoundTrip:
    """Consolidated state must survive a fresh read from the database."""

    def test_consolidated_snapshot_readable_with_semantic_type(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC
        assert fetched.ingested_at == _CONSOLIDATION_TIME

    def test_consolidated_snapshot_appears_in_semantic_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        semantic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.SEMANTIC)

        assert snap.snapshot_id in [s.snapshot_id for s in semantic]

    def test_consolidated_snapshot_absent_from_episodic_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        episodic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.EPISODIC)

        assert snap.snapshot_id not in [s.snapshot_id for s in episodic]

    def test_consolidated_snapshot_absent_from_working_list(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        working = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.WORKING)

        assert snap.snapshot_id not in [s.snapshot_id for s in working]

    def test_consolidated_at_metadata_persists_on_reread(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_promoted_snapshot(index)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert fetched.metadata.get("consolidated_at") == _CONSOLIDATION_TIME.isoformat()
        assert fetched.metadata.get("consolidated_from") == MemoryType.EPISODIC.value


# ---------------------------------------------------------------------------
# 4. Isolation: unrelated snapshots are untouched
# ---------------------------------------------------------------------------


class TestSemanticPromotionIsolation:
    """Promotion of one snapshot must not affect others."""

    def test_other_episodic_snapshots_unaffected(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap1 = _add_promoted_snapshot(index)
        snap2 = _add_promoted_snapshot(index)

        index.promote_to_semantic(snap1.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched2 = index.get_snapshot(snap2.snapshot_id)
        assert fetched2 is not None
        assert fetched2.memory_type == MemoryType.EPISODIC

    def test_working_snapshots_unaffected(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        episodic_snap = _add_promoted_snapshot(index)
        working_snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_semantic(
            episodic_snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        fetched = index.get_snapshot(working_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.WORKING

    def test_existing_semantic_snapshots_unaffected(self, tmp_path: Path) -> None:
        existing_time = datetime(2026, 2, 1, tzinfo=UTC)
        index = _make_index(tmp_path)
        episodic_snap = _add_promoted_snapshot(index)
        semantic_snap = _add_snapshot(
            index, memory_type=MemoryType.SEMANTIC, ingested_at=existing_time
        )

        index.promote_to_semantic(
            episodic_snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        fetched = index.get_snapshot(semantic_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.SEMANTIC
        assert fetched.ingested_at == existing_time

    def test_only_consolidated_snapshot_has_consolidated_at_key(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap1 = _add_promoted_snapshot(index)
        snap2 = _add_promoted_snapshot(index)

        index.promote_to_semantic(snap1.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched2 = index.get_snapshot(snap2.snapshot_id)
        assert fetched2 is not None
        assert "consolidated_at" not in fetched2.metadata
        assert "consolidated_from" not in fetched2.metadata


# ---------------------------------------------------------------------------
# 5. Error cases
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
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_semantic(snap.snapshot_id)

    def test_state_unchanged_after_failed_promotion(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.WORKING


# ---------------------------------------------------------------------------
# 6. promotion_history entries — both transitions in one place
# ---------------------------------------------------------------------------


class TestPromotionHistoryWorkingToEpisodic:
    """promote_to_episodic appends exactly one correctly-populated history entry."""

    def test_one_entry_appended_to_empty_history(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        assert len(promoted.promotion_history) == 1

    def test_entry_from_type_is_working(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        assert promoted.promotion_history[0].from_type == MemoryType.WORKING

    def test_entry_to_type_is_episodic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        assert promoted.promotion_history[0].to_type == MemoryType.EPISODIC

    def test_entry_timestamp_matches_persist_time(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        assert promoted.promotion_history[0].timestamp == _PERSIST_TIME

    def test_entry_reason_is_persist_event(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        promoted = index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        assert promoted.promotion_history[0].reason == "persist event"

    def test_history_entry_persists_on_reread(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert len(fetched.promotion_history) == 1
        assert fetched.promotion_history[0].from_type == MemoryType.WORKING
        assert fetched.promotion_history[0].to_type == MemoryType.EPISODIC
        assert fetched.promotion_history[0].timestamp == _PERSIST_TIME


class TestPromotionHistoryEpisodicToSemantic:
    """promote_to_semantic appends a second correctly-populated history entry."""

    def test_second_entry_appended_after_semantic_promotion(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert len(consolidated.promotion_history) == 2

    def test_second_entry_from_type_is_episodic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.promotion_history[1].from_type == MemoryType.EPISODIC

    def test_second_entry_to_type_is_semantic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.promotion_history[1].to_type == MemoryType.SEMANTIC

    def test_second_entry_timestamp_matches_consolidation_time(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.promotion_history[1].timestamp == _CONSOLIDATION_TIME

    def test_second_entry_reason_is_session_consolidation(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert consolidated.promotion_history[1].reason == "session consolidation"

    def test_first_history_entry_preserved_verbatim(self, tmp_path: Path) -> None:
        """The episodic promotion entry must be unchanged after semantic promotion."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        first = consolidated.promotion_history[0]
        assert first.from_type == MemoryType.WORKING
        assert first.to_type == MemoryType.EPISODIC
        assert first.timestamp == _PERSIST_TIME
        assert first.reason == "persist event"

    def test_history_entries_are_chronologically_ordered(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        assert (
            consolidated.promotion_history[0].timestamp
            < consolidated.promotion_history[1].timestamp
        )

    def test_full_history_persists_on_reread(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert len(fetched.promotion_history) == 2
        assert fetched.promotion_history[0].from_type == MemoryType.WORKING
        assert fetched.promotion_history[0].to_type == MemoryType.EPISODIC
        assert fetched.promotion_history[1].from_type == MemoryType.EPISODIC
        assert fetched.promotion_history[1].to_type == MemoryType.SEMANTIC

    def test_full_history_survives_close_and_reopen(self, tmp_path: Path) -> None:
        """Both history entries must be readable from a freshly opened ShadowIndex."""
        db_path = tmp_path / "test.sqlite"
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)
        index.close()

        index2 = ShadowIndex(db_path)
        fetched = index2.get_snapshot(snap.snapshot_id)
        index2.close()

        assert fetched is not None
        assert len(fetched.promotion_history) == 2
        assert fetched.promotion_history[0].timestamp == _PERSIST_TIME
        assert fetched.promotion_history[1].timestamp == _CONSOLIDATION_TIME

    def test_to_type_sequence_traces_valid_path(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        consolidated = index.promote_to_semantic(
            snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME
        )

        to_types = [e.to_type for e in consolidated.promotion_history]
        assert to_types == [MemoryType.EPISODIC, MemoryType.SEMANTIC]

    def test_failed_semantic_promotion_does_not_append_history(self, tmp_path: Path) -> None:
        """A failed promote_to_semantic must not write any history entry."""
        index = _make_index(tmp_path)
        # WORKING — not eligible for semantic promotion
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.promotion_history == []
