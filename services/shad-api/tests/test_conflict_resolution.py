"""Unit tests for resolve_latest_ingestion_wins.

Contract under test
-------------------
``resolve_latest_ingestion_wins(group)`` must:

1. **Basic win** — the SnapshotEntry with the latest ``ingested_at`` is set as
   ``resolved_record``.
2. **Two candidates** — when two records differ only in ``ingested_at``, the
   later one wins.
3. **Three-way** — correct winner selected from three records with distinct
   ``ingested_at`` values regardless of insertion order.
4. **Tie** — when two records share the same maximum ``ingested_at``, the first
   encountered (stable, deterministic) is chosen.
5. **In-place mutation** — ``group.resolved_record`` is mutated *and* the same
   object is returned (convenient chaining).
6. **Single record** — a lone record always wins its own conflict group.

Sources / prior art:
  - teranode/stores/utxo/process-conflicting.go — "last-writer-wins by system
    time" pattern for UTXO conflict resolution.
  - openclaw/src/memory/temporal-decay.test.ts (openclaw@2026-02-22) — parallel
    TypeScript implementation of the same bi-temporal conflict model.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.shadow_index import ConflictGroup, SnapshotEntry, resolve_latest_ingestion_wins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_TIME = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _snapshot(ingested_at: datetime, *, revision: str = "rev-x") -> SnapshotEntry:
    """Minimal SnapshotEntry with only ingested_at varied."""
    return SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id="src-001",
        ingested_at=ingested_at,
        source_revision=revision,
        entry_paths=["notes/item.md"],
        content_hash=str(uuid.uuid4()),
    )


def _group(*snapshots: SnapshotEntry) -> ConflictGroup:
    return ConflictGroup(
        event_time=_EVENT_TIME,
        conflicting_records=list(snapshots),
    )


# ---------------------------------------------------------------------------
# Basic win
# ---------------------------------------------------------------------------

class TestBasicWin:
    def test_later_ingestion_wins_over_earlier(self) -> None:
        early = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        late = _snapshot(datetime(2026, 1, 2, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(early, late))
        assert result.resolved_record is late

    def test_winner_identity_is_exact_object_not_copy(self) -> None:
        early = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        late = _snapshot(datetime(2026, 1, 2, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(early, late))
        assert result.resolved_record is late  # identity, not equality

    def test_winner_ingested_at_is_max_of_group(self) -> None:
        t1 = datetime(2026, 1, 10, tzinfo=UTC)
        t2 = datetime(2026, 1, 20, tzinfo=UTC)
        s1, s2 = _snapshot(t1), _snapshot(t2)
        result = resolve_latest_ingestion_wins(_group(s1, s2))
        assert result.resolved_record is not None
        assert result.resolved_record.ingested_at == t2


# ---------------------------------------------------------------------------
# Insertion order must not affect the result
# ---------------------------------------------------------------------------

class TestInsertionOrderIndependence:
    def test_latest_wins_when_inserted_first(self) -> None:
        late = _snapshot(datetime(2026, 2, 28, tzinfo=UTC))
        early = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(late, early))  # late first
        assert result.resolved_record is late

    def test_latest_wins_when_inserted_last(self) -> None:
        early = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        late = _snapshot(datetime(2026, 2, 28, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(early, late))  # late last
        assert result.resolved_record is late


# ---------------------------------------------------------------------------
# Three-way conflict
# ---------------------------------------------------------------------------

class TestThreeWayConflict:
    def test_middle_record_not_chosen(self) -> None:
        t_old = datetime(2026, 1, 1, tzinfo=UTC)
        t_mid = datetime(2026, 1, 15, tzinfo=UTC)
        t_new = datetime(2026, 1, 31, tzinfo=UTC)
        s_old, s_mid, s_new = _snapshot(t_old), _snapshot(t_mid), _snapshot(t_new)
        result = resolve_latest_ingestion_wins(_group(s_old, s_mid, s_new))
        assert result.resolved_record is s_new

    def test_winner_in_shuffled_order(self) -> None:
        t_old = datetime(2026, 1, 1, tzinfo=UTC)
        t_mid = datetime(2026, 1, 15, tzinfo=UTC)
        t_new = datetime(2026, 1, 31, tzinfo=UTC)
        s_old, s_mid, s_new = _snapshot(t_old), _snapshot(t_mid), _snapshot(t_new)
        # Insert newest in the middle
        result = resolve_latest_ingestion_wins(_group(s_old, s_new, s_mid))
        assert result.resolved_record is s_new

    def test_correct_ingested_at_on_resolved_record(self) -> None:
        t_new = datetime(2026, 3, 31, tzinfo=UTC)
        records = [
            _snapshot(datetime(2026, 1, 1, tzinfo=UTC)),
            _snapshot(datetime(2026, 2, 1, tzinfo=UTC)),
            _snapshot(t_new),
        ]
        result = resolve_latest_ingestion_wins(_group(*records))
        assert result.resolved_record is not None
        assert result.resolved_record.ingested_at == t_new


# ---------------------------------------------------------------------------
# Tie-breaking (equal maximum ingested_at)
# ---------------------------------------------------------------------------

class TestEqualIngestedAt:
    def test_first_encountered_wins_on_tie(self) -> None:
        """When two records share the same max ingested_at, the first one wins."""
        t_same = datetime(2026, 1, 15, tzinfo=UTC)
        s1 = _snapshot(t_same, revision="rev-first")
        s2 = _snapshot(t_same, revision="rev-second")
        result = resolve_latest_ingestion_wins(_group(s1, s2))
        assert result.resolved_record is s1

    def test_tie_among_three_first_max_wins(self) -> None:
        t_old = datetime(2026, 1, 1, tzinfo=UTC)
        t_tie = datetime(2026, 3, 1, tzinfo=UTC)
        s_old = _snapshot(t_old, revision="rev-old")
        s_tie_a = _snapshot(t_tie, revision="rev-tie-a")
        s_tie_b = _snapshot(t_tie, revision="rev-tie-b")
        result = resolve_latest_ingestion_wins(_group(s_old, s_tie_a, s_tie_b))
        # s_tie_a is the first of the two equal-max entries
        assert result.resolved_record is s_tie_a


# ---------------------------------------------------------------------------
# In-place mutation and return value
# ---------------------------------------------------------------------------

class TestInPlaceMutation:
    def test_returns_same_group_object(self) -> None:
        s1 = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        s2 = _snapshot(datetime(2026, 1, 2, tzinfo=UTC))
        group = _group(s1, s2)
        returned = resolve_latest_ingestion_wins(group)
        assert returned is group

    def test_resolved_record_written_to_group(self) -> None:
        s1 = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        s2 = _snapshot(datetime(2026, 3, 31, tzinfo=UTC))
        group = _group(s1, s2)
        assert group.resolved_record is None  # pre-condition
        resolve_latest_ingestion_wins(group)
        assert group.resolved_record is s2  # mutated in place

    def test_resolved_record_is_none_before_resolution(self) -> None:
        group = _group(_snapshot(datetime(2026, 1, 1, tzinfo=UTC)))
        assert group.resolved_record is None


# ---------------------------------------------------------------------------
# Single-record group (degenerate case)
# ---------------------------------------------------------------------------

class TestSingleRecord:
    def test_single_record_always_wins(self) -> None:
        sole = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(sole))
        assert result.resolved_record is sole

    def test_single_record_resolved_record_identity(self) -> None:
        sole = _snapshot(datetime(2026, 6, 1, tzinfo=UTC))
        group = _group(sole)
        resolve_latest_ingestion_wins(group)
        assert group.resolved_record is sole


# ---------------------------------------------------------------------------
# Resolved record preserves full SnapshotEntry fields
# ---------------------------------------------------------------------------

class TestResolvedRecordFields:
    def test_snapshot_id_preserved_on_winner(self) -> None:
        early = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        late = _snapshot(datetime(2026, 3, 31, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(early, late))
        assert result.resolved_record is not None
        assert result.resolved_record.snapshot_id == late.snapshot_id

    def test_source_id_preserved_on_winner(self) -> None:
        early = _snapshot(datetime(2026, 1, 1, tzinfo=UTC))
        late = _snapshot(datetime(2026, 3, 31, tzinfo=UTC))
        result = resolve_latest_ingestion_wins(_group(early, late))
        assert result.resolved_record is not None
        assert result.resolved_record.source_id == "src-001"
