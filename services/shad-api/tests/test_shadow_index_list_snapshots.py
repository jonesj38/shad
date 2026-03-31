"""Unit tests for ShadowIndex.list_snapshots memory_type filtering.

Contract under test
-------------------
``ShadowIndex.list_snapshots(source_id, memory_type=None, memory_types=None)`` must:

1. **No filter** — return all snapshots for the source regardless of memory_type.
2. **memory_type (single)** — return only snapshots whose memory_type matches.
3. **memory_types (list, single element)** — equivalent to memory_type shorthand.
4. **memory_types (list, multiple)** — return snapshots matching any of the types.
5. **memory_type mismatch** — return empty list when no snapshots match the type.
6. **None-typed snapshots excluded** — unclassified (memory_type=None) snapshots
   are excluded from typed queries but included in unfiltered queries.
7. **Cross-source isolation** — filters never leak snapshots from other sources.
8. **memory_type wins when both params absent** — no filter = all rows returned.
9. **memory_types takes precedence over memory_type when both supplied** —
   documented behaviour: memory_types is checked first.
"""

from __future__ import annotations

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
# Shared timestamps
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source(source_id: str) -> SourceEntry:
    return SourceEntry(
        source_url=f"https://example.com/{source_id}",
        source_id=source_id,
        source_type="github",
        update_policy=UpdatePolicy.MANUAL,
    )


def _snapshot(
    snapshot_id: str,
    source_id: str,
    memory_type: MemoryType | None,
    *,
    offset_days: int = 0,
) -> SnapshotEntry:
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id=source_id,
        ingested_at=_BASE + timedelta(days=offset_days),
        source_revision="rev-" + snapshot_id,
        entry_paths=[f"docs/{snapshot_id}.md"],
        content_hash="hash-" + snapshot_id,
        memory_type=memory_type,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def index(tmp_path: Path) -> ShadowIndex:
    db_path = tmp_path / "shadow.sqlite"
    idx = ShadowIndex(db_path=db_path)

    # Primary source: one snapshot per taxonomy type + one unclassified
    idx.add_source(_source("primary"))
    idx.add_snapshot(_snapshot("p-working",   "primary", MemoryType.WORKING,   offset_days=0))
    idx.add_snapshot(_snapshot("p-episodic",  "primary", MemoryType.EPISODIC,  offset_days=1))
    idx.add_snapshot(_snapshot("p-semantic",  "primary", MemoryType.SEMANTIC,  offset_days=2))
    idx.add_snapshot(_snapshot("p-procedural","primary", MemoryType.PROCEDURAL,offset_days=3))
    idx.add_snapshot(_snapshot("p-untyped",   "primary", None,                 offset_days=4))

    # Secondary source: only EPISODIC — used for isolation tests
    idx.add_source(_source("secondary"))
    idx.add_snapshot(_snapshot("s-episodic-1", "secondary", MemoryType.EPISODIC, offset_days=0))
    idx.add_snapshot(_snapshot("s-episodic-2", "secondary", MemoryType.EPISODIC, offset_days=1))

    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListSnapshotsNoFilter:
    """No memory_type filter — all rows returned."""

    def test_returns_all_snapshots_including_untyped(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary")
        ids = {s.snapshot_id for s in results}
        assert ids == {"p-working", "p-episodic", "p-semantic", "p-procedural", "p-untyped"}

    def test_ordered_by_ingested_at_descending(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary")
        timestamps = [s.ingested_at for s in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_empty_source_returns_empty_list(self, index: ShadowIndex, tmp_path: Path) -> None:
        index.add_source(_source("empty-src"))
        assert index.list_snapshots("empty-src") == []


class TestListSnapshotsSingleMemoryType:
    """memory_type= (single-value shorthand) filtering."""

    def test_filters_to_working(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary", memory_type=MemoryType.WORKING)
        assert len(results) == 1
        assert results[0].snapshot_id == "p-working"
        assert results[0].memory_type == MemoryType.WORKING

    def test_filters_to_episodic(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary", memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].snapshot_id == "p-episodic"

    def test_filters_to_semantic(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary", memory_type=MemoryType.SEMANTIC)
        assert len(results) == 1
        assert results[0].snapshot_id == "p-semantic"

    def test_filters_to_procedural(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary", memory_type=MemoryType.PROCEDURAL)
        assert len(results) == 1
        assert results[0].snapshot_id == "p-procedural"

    def test_no_match_returns_empty_list(self, index: ShadowIndex) -> None:
        # secondary only has EPISODIC; querying SEMANTIC should return nothing
        results = index.list_snapshots("secondary", memory_type=MemoryType.SEMANTIC)
        assert results == []

    def test_untyped_snapshots_excluded_from_typed_query(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary", memory_type=MemoryType.WORKING)
        assert all(s.memory_type is not None for s in results)
        assert not any(s.snapshot_id == "p-untyped" for s in results)


class TestListSnapshotsMemoryTypesList:
    """memory_types= (multi-value list) filtering."""

    def test_single_element_list_equivalent_to_scalar(self, index: ShadowIndex) -> None:
        scalar = index.list_snapshots("primary", memory_type=MemoryType.EPISODIC)
        list_form = index.list_snapshots("primary", memory_types=[MemoryType.EPISODIC])
        assert {s.snapshot_id for s in scalar} == {s.snapshot_id for s in list_form}

    def test_two_types_returns_union(self, index: ShadowIndex) -> None:
        results = index.list_snapshots(
            "primary",
            memory_types=[MemoryType.WORKING, MemoryType.EPISODIC],
        )
        ids = {s.snapshot_id for s in results}
        assert ids == {"p-working", "p-episodic"}

    def test_all_four_types_excludes_only_untyped(self, index: ShadowIndex) -> None:
        results = index.list_snapshots(
            "primary",
            memory_types=[
                MemoryType.WORKING,
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
                MemoryType.PROCEDURAL,
            ],
        )
        ids = {s.snapshot_id for s in results}
        assert "p-untyped" not in ids
        assert len(ids) == 4

    def test_empty_list_returns_no_rows(self, index: ShadowIndex) -> None:
        # An empty list means "match nothing in this set" — no rows should come back.
        results = index.list_snapshots("primary", memory_types=[])
        assert results == []

    def test_unmatched_types_return_empty(self, index: ShadowIndex) -> None:
        # secondary only has EPISODIC; requesting SEMANTIC + PROCEDURAL → empty
        results = index.list_snapshots(
            "secondary",
            memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
        )
        assert results == []

    def test_results_are_ordered_descending(self, index: ShadowIndex) -> None:
        results = index.list_snapshots(
            "secondary",
            memory_types=[MemoryType.EPISODIC],
        )
        assert len(results) == 2
        timestamps = [s.ingested_at for s in results]
        assert timestamps == sorted(timestamps, reverse=True)


class TestListSnapshotsCrossSourceIsolation:
    """Filters must never include snapshots from a different source."""

    def test_primary_query_excludes_secondary_snapshots(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary", memory_type=MemoryType.EPISODIC)
        assert all(s.source_id == "primary" for s in results)

    def test_secondary_query_excludes_primary_snapshots(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("secondary", memory_type=MemoryType.EPISODIC)
        secondary_ids = {"s-episodic-1", "s-episodic-2"}
        assert {s.snapshot_id for s in results} == secondary_ids

    def test_unknown_source_returns_empty(self, index: ShadowIndex) -> None:
        assert index.list_snapshots("does-not-exist") == []
        assert index.list_snapshots("does-not-exist", memory_type=MemoryType.EPISODIC) == []


class TestListSnapshotsUntypedBehaviour:
    """Snapshots stored with memory_type=None are unclassified."""

    def test_untyped_returned_in_unfiltered_query(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary")
        assert any(s.snapshot_id == "p-untyped" for s in results)

    def test_untyped_memory_type_field_is_none(self, index: ShadowIndex) -> None:
        results = index.list_snapshots("primary")
        untyped = next(s for s in results if s.snapshot_id == "p-untyped")
        assert untyped.memory_type is None

    def test_untyped_not_returned_for_any_typed_query(self, index: ShadowIndex) -> None:
        for mt in MemoryType:
            results = index.list_snapshots("primary", memory_type=mt)
            assert not any(s.snapshot_id == "p-untyped" for s in results), (
                f"p-untyped should not appear when filtering by {mt}"
            )
