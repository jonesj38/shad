"""Tests for the PromotionHistoryEntry chain and serialize/deserialize round-trips.

The `promotion_history` field on SnapshotEntry records every memory-type
transition as a structured list of PromotionHistoryEntry objects.  These tests
verify:

  1. Serialisation — serialize_promotion_history produces the expected JSON for
     empty lists, single entries with None from_type, and multi-entry chains.

  2. Deserialisation — deserialize_promotion_history round-trips all field types
     correctly (MemoryType enums, UTC datetimes, reason strings, None from_type).

  3. Chain integrity after promote_to_episodic — exactly one entry appended with
     the correct from_type / to_type / timestamp / reason.

  4. Chain integrity after promote_to_semantic — a second entry is appended;
     the first entry is preserved verbatim.

  5. Full WORKING→EPISODIC→SEMANTIC chain — promotion_history carries both
     transitions in order with correct timestamps.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shad.vault.shadow_index import (
    MemoryType,
    PromotionHistoryEntry,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
    deserialize_promotion_history,
    serialize_promotion_history,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "promo-history-src-001"
_SOURCE_URL = "sessions://local/promotion-history-test"

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
) -> SnapshotEntry:
    snapshot = SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=datetime(2026, 3, 28, 9, 0, 0, tzinfo=UTC),
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["sessions/note.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
        metadata=metadata or {},
    )
    index.add_snapshot(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# 1. Serialisation
# ---------------------------------------------------------------------------


class TestSerializePromotionHistory:
    """serialize_promotion_history must produce valid, stable JSON."""

    def test_empty_list_serialises_to_empty_json_array(self) -> None:
        result = serialize_promotion_history([])
        assert json.loads(result) == []

    def test_single_entry_with_none_from_type(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=None,
            to_type=MemoryType.EPISODIC,
            timestamp=_PERSIST_TIME,
            reason="initial ingest",
        )
        raw = serialize_promotion_history([entry])
        data = json.loads(raw)

        assert len(data) == 1
        assert data[0]["from_type"] is None
        assert data[0]["to_type"] == MemoryType.EPISODIC.value
        assert data[0]["reason"] == "initial ingest"

    def test_single_entry_with_known_from_type(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=MemoryType.WORKING,
            to_type=MemoryType.EPISODIC,
            timestamp=_PERSIST_TIME,
            reason="persist event",
        )
        raw = serialize_promotion_history([entry])
        data = json.loads(raw)

        assert data[0]["from_type"] == MemoryType.WORKING.value
        assert data[0]["to_type"] == MemoryType.EPISODIC.value

    def test_two_entry_chain_preserves_order(self) -> None:
        entries = [
            PromotionHistoryEntry(
                from_type=MemoryType.WORKING,
                to_type=MemoryType.EPISODIC,
                timestamp=_PERSIST_TIME,
                reason="persist event",
            ),
            PromotionHistoryEntry(
                from_type=MemoryType.EPISODIC,
                to_type=MemoryType.SEMANTIC,
                timestamp=_CONSOLIDATION_TIME,
                reason="session consolidation",
            ),
        ]
        raw = serialize_promotion_history(entries)
        data = json.loads(raw)

        assert len(data) == 2
        assert data[0]["to_type"] == MemoryType.EPISODIC.value
        assert data[1]["to_type"] == MemoryType.SEMANTIC.value

    def test_timestamp_is_isoformat_string(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=MemoryType.WORKING,
            to_type=MemoryType.EPISODIC,
            timestamp=_PERSIST_TIME,
            reason="persist event",
        )
        raw = serialize_promotion_history([entry])
        data = json.loads(raw)

        ts = data[0]["timestamp"]
        # Must be parseable back to the original datetime.
        assert datetime.fromisoformat(ts) == _PERSIST_TIME


# ---------------------------------------------------------------------------
# 2. Deserialisation
# ---------------------------------------------------------------------------


class TestDeserializePromotionHistory:
    """deserialize_promotion_history must faithfully reconstruct all field types."""

    def test_none_or_empty_string_returns_empty_list(self) -> None:
        assert deserialize_promotion_history(None) == []
        assert deserialize_promotion_history("") == []

    def test_empty_json_array_returns_empty_list(self) -> None:
        assert deserialize_promotion_history("[]") == []

    def test_none_from_type_round_trips(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=None,
            to_type=MemoryType.EPISODIC,
            timestamp=_PERSIST_TIME,
            reason="initial ingest",
        )
        raw = serialize_promotion_history([entry])
        restored = deserialize_promotion_history(raw)

        assert len(restored) == 1
        assert restored[0].from_type is None
        assert restored[0].to_type == MemoryType.EPISODIC

    def test_known_from_type_round_trips(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=MemoryType.WORKING,
            to_type=MemoryType.EPISODIC,
            timestamp=_PERSIST_TIME,
            reason="persist event",
        )
        raw = serialize_promotion_history([entry])
        restored = deserialize_promotion_history(raw)

        assert restored[0].from_type == MemoryType.WORKING
        assert restored[0].to_type == MemoryType.EPISODIC

    def test_timestamp_round_trips_with_utc(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=MemoryType.WORKING,
            to_type=MemoryType.EPISODIC,
            timestamp=_PERSIST_TIME,
            reason="persist event",
        )
        raw = serialize_promotion_history([entry])
        restored = deserialize_promotion_history(raw)

        assert restored[0].timestamp == _PERSIST_TIME

    def test_reason_string_round_trips(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=MemoryType.EPISODIC,
            to_type=MemoryType.SEMANTIC,
            timestamp=_CONSOLIDATION_TIME,
            reason="session consolidation",
        )
        raw = serialize_promotion_history([entry])
        restored = deserialize_promotion_history(raw)

        assert restored[0].reason == "session consolidation"

    def test_two_entry_chain_round_trips_in_order(self) -> None:
        entries = [
            PromotionHistoryEntry(
                from_type=MemoryType.WORKING,
                to_type=MemoryType.EPISODIC,
                timestamp=_PERSIST_TIME,
                reason="persist event",
            ),
            PromotionHistoryEntry(
                from_type=MemoryType.EPISODIC,
                to_type=MemoryType.SEMANTIC,
                timestamp=_CONSOLIDATION_TIME,
                reason="session consolidation",
            ),
        ]
        raw = serialize_promotion_history(entries)
        restored = deserialize_promotion_history(raw)

        assert len(restored) == 2
        assert restored[0].from_type == MemoryType.WORKING
        assert restored[0].to_type == MemoryType.EPISODIC
        assert restored[1].from_type == MemoryType.EPISODIC
        assert restored[1].to_type == MemoryType.SEMANTIC
        assert restored[0].timestamp == _PERSIST_TIME
        assert restored[1].timestamp == _CONSOLIDATION_TIME


# ---------------------------------------------------------------------------
# 3. Chain integrity after promote_to_episodic
# ---------------------------------------------------------------------------


class TestPromoteToEpisodicHistory:
    """promote_to_episodic must append exactly one PromotionHistoryEntry."""

    def test_one_entry_appended(self, tmp_path: Path) -> None:
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

    def test_history_persists_on_reread(self, tmp_path: Path) -> None:
        """promotion_history must survive a fresh get_snapshot read."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)
        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)

        assert fetched is not None
        assert len(fetched.promotion_history) == 1
        assert fetched.promotion_history[0].to_type == MemoryType.EPISODIC

    def test_snapshot_with_no_prior_history_starts_clean(self, tmp_path: Path) -> None:
        """A fresh WORKING snapshot has an empty promotion_history before any promotion."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.promotion_history == []


# ---------------------------------------------------------------------------
# 4. Chain integrity after promote_to_semantic
# ---------------------------------------------------------------------------


class TestPromoteToSemanticHistory:
    """promote_to_semantic must append a second entry without disturbing the first."""

    def test_two_entries_after_both_promotions(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        assert len(final.promotion_history) == 2

    def test_first_entry_preserved_after_semantic_promotion(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        first = final.promotion_history[0]
        assert first.from_type == MemoryType.WORKING
        assert first.to_type == MemoryType.EPISODIC
        assert first.timestamp == _PERSIST_TIME
        assert first.reason == "persist event"

    def test_second_entry_is_episodic_to_semantic(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        second = final.promotion_history[1]
        assert second.from_type == MemoryType.EPISODIC
        assert second.to_type == MemoryType.SEMANTIC

    def test_second_entry_timestamp_matches_consolidation_time(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        assert final.promotion_history[1].timestamp == _CONSOLIDATION_TIME

    def test_second_entry_reason_is_session_consolidation(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        assert final.promotion_history[1].reason == "session consolidation"

    def test_history_persists_after_semantic_promotion_reread(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert len(fetched.promotion_history) == 2


# ---------------------------------------------------------------------------
# 5. Full WORKING → EPISODIC → SEMANTIC chain — ordered timestamps
# ---------------------------------------------------------------------------


class TestFullPromotionHistoryChain:
    """The full two-step chain must produce a correctly ordered history."""

    def test_history_entries_are_chronologically_ordered(self, tmp_path: Path) -> None:
        """Entries must be ordered oldest-first (WORKING→EPISODIC before EPISODIC→SEMANTIC)."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        assert final.promotion_history[0].timestamp < final.promotion_history[1].timestamp

    def test_history_to_types_trace_the_valid_path(self, tmp_path: Path) -> None:
        """to_type values must trace EPISODIC then SEMANTIC in that order."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        index.promote_to_episodic(snap.snapshot_id, persist_time=_PERSIST_TIME)
        final = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        to_types = [e.to_type for e in final.promotion_history]
        assert to_types == [MemoryType.EPISODIC, MemoryType.SEMANTIC]

    def test_rejected_promotion_does_not_append_to_history(self, tmp_path: Path) -> None:
        """A failed promote_to_semantic call must not lengthen promotion_history."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, memory_type=MemoryType.WORKING)

        # Attempt invalid skip: WORKING → SEMANTIC (must raise, must not write history)
        with pytest.raises(ValueError):
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_CONSOLIDATION_TIME)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.promotion_history == []

    def test_history_survives_close_and_reopen(self, tmp_path: Path) -> None:
        """Full history must be readable from a freshly opened ShadowIndex."""
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
        assert fetched.promotion_history[0].from_type == MemoryType.WORKING
        assert fetched.promotion_history[1].from_type == MemoryType.EPISODIC
