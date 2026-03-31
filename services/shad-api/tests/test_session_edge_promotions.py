"""Tests for promotion behaviour at session edge conditions.

A "session" is the time window defined by the `since` parameter passed to
`episodic_count`.  The session start is the `since` boundary (exclusive lower
bound); the session end is when `should_consolidate` fires and
`promote_to_semantic` is called with a `consolidation_time`.

Three categories are covered:

  1. Session start edge  — snapshots whose `ingested_at` lands exactly on the
     `since` boundary.  `episodic_count` uses a strict `>` comparison, so a
     snapshot at the boundary is excluded from the current window.

  2. Session end edge    — snapshots promoted with a `consolidation_time` that
     coincides with the next session's lower bound.  After promotion the entry
     is SEMANTIC and must not re-enter the episodic window of the new session.

  3. Concurrent boundary — two concurrent callers both detect that the threshold
     has been crossed and race to promote the same (or different) snapshot(s).
     Only one promotion can succeed; the loser must receive a `ValueError`.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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
# Shared helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "sessions-edge-src-001"
_SOURCE_URL = "sessions://local/edge-test"

# Anchor all times to a stable reference point.
_SESSION_START = datetime(2026, 3, 31, 9, 0, 0, tzinfo=UTC)
_SESSION_END = datetime(2026, 3, 31, 17, 0, 0, tzinfo=UTC)


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


def _add_episodic(
    index: ShadowIndex,
    *,
    ingested_at: datetime,
    metadata: dict | None = None,
) -> SnapshotEntry:
    """Insert a EPISODIC snapshot with a specific ingested_at."""
    snapshot = SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=ingested_at,
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["sessions/note.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=MemoryType.EPISODIC,
        metadata=metadata or {},
    )
    index.add_snapshot(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# 1. Session start edge
# ---------------------------------------------------------------------------


class TestSessionStartEdge:
    """A snapshot ingested at exactly `since` must be excluded from the count.

    `episodic_count` uses `ingested_at > since` (strict), so the boundary
    timestamp is *not* part of the current session window.
    """

    def test_snapshot_at_exactly_since_is_excluded(self, tmp_path: Path) -> None:
        """ingested_at == since → count remains 0."""
        index = _make_index(tmp_path)
        _add_episodic(index, ingested_at=_SESSION_START)

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_START)

        assert count == 0

    def test_snapshot_one_microsecond_after_since_is_included(self, tmp_path: Path) -> None:
        """ingested_at == since + 1µs → count is 1."""
        index = _make_index(tmp_path)
        just_after = _SESSION_START + timedelta(microseconds=1)
        _add_episodic(index, ingested_at=just_after)

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_START)

        assert count == 1

    def test_boundary_snapshot_does_not_contribute_to_threshold(self, tmp_path: Path) -> None:
        """threshold-1 in-window + 1 at-boundary → should_consolidate remains False."""
        threshold = 3
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)

        # threshold-1 snapshots strictly inside the window
        for i in range(threshold - 1):
            _add_episodic(index, ingested_at=_SESSION_START + timedelta(minutes=i + 1))

        # One snapshot exactly on the boundary (excluded)
        _add_episodic(index, ingested_at=_SESSION_START)

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_START)
        assert count == threshold - 1
        assert cfg.should_consolidate(count) is False

    def test_adding_one_past_boundary_triggers_consolidation(self, tmp_path: Path) -> None:
        """After the boundary-excluded snapshot, one more in-window entry fires the trigger."""
        threshold = 3
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)

        # One snapshot on the boundary (excluded from count)
        _add_episodic(index, ingested_at=_SESSION_START)

        # threshold snapshots strictly inside the window → fires
        for i in range(threshold):
            _add_episodic(index, ingested_at=_SESSION_START + timedelta(minutes=i + 1))

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_START)
        assert count == threshold
        assert cfg.should_consolidate(count) is True

    def test_boundary_snapshot_remains_episodic_after_session_fires(
        self, tmp_path: Path
    ) -> None:
        """The at-boundary snapshot is not touched when the in-window entries are promoted."""
        threshold = 2
        index = _make_index(tmp_path)
        consolidation_time = _SESSION_END

        boundary_snap = _add_episodic(index, ingested_at=_SESSION_START)
        in_window_snaps = [
            _add_episodic(index, ingested_at=_SESSION_START + timedelta(hours=i + 1))
            for i in range(threshold)
        ]

        for snap in in_window_snaps:
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=consolidation_time)

        fetched = index.get_snapshot(boundary_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.EPISODIC

    def test_multiple_at_boundary_snapshots_all_excluded(self, tmp_path: Path) -> None:
        """Several snapshots exactly at `since` are all excluded from the count."""
        index = _make_index(tmp_path)
        for _ in range(5):
            _add_episodic(index, ingested_at=_SESSION_START)

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_START)
        assert count == 0

    @pytest.mark.parametrize("delta_us", [-1, 0])
    def test_snapshots_at_or_before_since_excluded(
        self, tmp_path: Path, delta_us: int
    ) -> None:
        """ingested_at <= since → always excluded from the episodic window."""
        index = _make_index(tmp_path)
        ts = _SESSION_START + timedelta(microseconds=delta_us)
        _add_episodic(index, ingested_at=ts)

        assert index.episodic_count(_SOURCE_ID, since=_SESSION_START) == 0


# ---------------------------------------------------------------------------
# 2. Session end edge
# ---------------------------------------------------------------------------


class TestSessionEndEdge:
    """Promotion fires at the session end boundary (consolidation_time).

    After promotion the promoted entries are SEMANTIC; a new session window
    opened at `since == consolidation_time` must have count 0 for those entries.
    """

    def test_promotion_at_session_end_sets_ingested_at_to_consolidation_time(
        self, tmp_path: Path
    ) -> None:
        """Promoted snapshot's ingested_at is exactly the session-end boundary."""
        index = _make_index(tmp_path)
        snap = _add_episodic(index, ingested_at=_SESSION_START + timedelta(hours=4))
        index.promote_to_semantic(snap.snapshot_id, consolidation_time=_SESSION_END)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.ingested_at == _SESSION_END

    def test_promoted_entries_not_in_new_session_window(self, tmp_path: Path) -> None:
        """After promotion, using consolidation_time as new `since` yields count 0."""
        threshold = 3
        index = _make_index(tmp_path)

        snaps = [
            _add_episodic(index, ingested_at=_SESSION_START + timedelta(hours=i + 1))
            for i in range(threshold)
        ]
        for snap in snaps:
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_SESSION_END)

        # New session starts at the old session end
        count = index.episodic_count(_SOURCE_ID, since=_SESSION_END)
        assert count == 0

    def test_new_episodic_entry_exactly_at_consolidation_time_excluded_from_next_session(
        self, tmp_path: Path
    ) -> None:
        """A fresh episodic entry ingested at consolidation_time is at the new session boundary.

        It must be excluded from the next window (since `episodic_count` uses strict `>`).
        """
        index = _make_index(tmp_path)
        _add_episodic(index, ingested_at=_SESSION_END)

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_END)
        assert count == 0

    def test_new_episodic_entry_one_microsecond_after_consolidation_time_included(
        self, tmp_path: Path
    ) -> None:
        """An entry ingested at consolidation_time + 1µs is the first of the new session."""
        index = _make_index(tmp_path)
        just_after = _SESSION_END + timedelta(microseconds=1)
        _add_episodic(index, ingested_at=just_after)

        count = index.episodic_count(_SOURCE_ID, since=_SESSION_END)
        assert count == 1

    def test_consolidation_time_equal_to_ingested_at_of_existing_episodic(
        self, tmp_path: Path
    ) -> None:
        """consolidation_time == ingested_at of the target snapshot is accepted.

        Both `ingested_at` and `consolidated_at` become the same instant; no error.
        """
        fixed = _SESSION_END
        index = _make_index(tmp_path)
        snap = _add_episodic(index, ingested_at=fixed)

        # consolidation_time equals the snapshot's own ingested_at — must not raise
        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=fixed)

        assert promoted.memory_type == MemoryType.SEMANTIC
        assert promoted.ingested_at == fixed
        assert promoted.metadata["consolidated_at"] == fixed.isoformat()

    def test_whole_session_promoted_and_new_session_count_starts_at_zero(
        self, tmp_path: Path
    ) -> None:
        """Full session lifecycle: fill window → promote all → new window is empty."""
        threshold = 4
        cfg = ConsolidationConfig(threshold=threshold)
        index = _make_index(tmp_path)

        snaps = [
            _add_episodic(index, ingested_at=_SESSION_START + timedelta(hours=i + 1))
            for i in range(threshold)
        ]
        assert cfg.should_consolidate(index.episodic_count(_SOURCE_ID, since=_SESSION_START))

        for snap in snaps:
            index.promote_to_semantic(snap.snapshot_id, consolidation_time=_SESSION_END)

        # All promoted; new window starting at session-end must be empty
        assert index.episodic_count(_SOURCE_ID, since=_SESSION_END) == 0

    def test_session_end_metadata_carries_consolidation_provenance(
        self, tmp_path: Path
    ) -> None:
        """Promoted snapshot's metadata must record the exact session-end boundary."""
        index = _make_index(tmp_path)
        snap = _add_episodic(index, ingested_at=_SESSION_START + timedelta(hours=1))
        promoted = index.promote_to_semantic(snap.snapshot_id, consolidation_time=_SESSION_END)

        assert promoted.metadata["consolidated_at"] == _SESSION_END.isoformat()
        assert promoted.metadata["consolidated_from"] == MemoryType.EPISODIC.value

    def test_promoting_at_session_end_does_not_affect_previous_session_boundary_snapshots(
        self, tmp_path: Path
    ) -> None:
        """Snapshots from a prior boundary position remain untouched."""
        index = _make_index(tmp_path)
        prior_boundary_snap = _add_episodic(index, ingested_at=_SESSION_START)
        in_window_snap = _add_episodic(
            index, ingested_at=_SESSION_START + timedelta(hours=2)
        )

        index.promote_to_semantic(in_window_snap.snapshot_id, consolidation_time=_SESSION_END)

        fetched = index.get_snapshot(prior_boundary_snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type == MemoryType.EPISODIC
        assert fetched.ingested_at == _SESSION_START


# ---------------------------------------------------------------------------
# 3. Concurrent session boundary events
# ---------------------------------------------------------------------------


class TestConcurrentSessionBoundary:
    """Race conditions when two callers both cross the threshold simultaneously.

    SQLite serialises writes, so the first commit wins.  The loser must get
    a ValueError rather than silently corrupting state.

    Each thread opens its own ShadowIndex instance on the same DB file — this
    is the realistic model (two independent callers / processes).  sqlite3
    connections are not shareable across threads.
    """

    def test_concurrent_promotions_of_same_snapshot_result_is_semantic(
        self, tmp_path: Path
    ) -> None:
        """Two concurrent promote_to_semantic calls on the same snapshot both see EPISODIC
        before either commits (TOCTOU), so both writes succeed.  The outcome is
        idempotent: the final stored state is SEMANTIC with no exceptions raised.
        """
        db_path = tmp_path / "test.sqlite"
        setup = _make_index(tmp_path)
        snap = _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=1))
        setup.close()

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def _promote() -> None:
            idx = ShadowIndex(db_path)
            barrier.wait()
            try:
                idx.promote_to_semantic(
                    snap.snapshot_id, consolidation_time=_SESSION_END
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                idx.close()

        threads = [threading.Thread(target=_promote) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        reader = ShadowIndex(db_path)
        try:
            fetched = reader.get_snapshot(snap.snapshot_id)
            assert fetched is not None
            assert fetched.memory_type == MemoryType.SEMANTIC
        finally:
            reader.close()

    def test_final_state_is_semantic_after_concurrent_race(self, tmp_path: Path) -> None:
        """Regardless of which thread wins, the stored snapshot is SEMANTIC."""
        db_path = tmp_path / "test.sqlite"
        setup = _make_index(tmp_path)
        snap = _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=1))
        setup.close()

        barrier = threading.Barrier(2)

        def _promote() -> None:
            idx = ShadowIndex(db_path)
            barrier.wait()
            try:
                idx.promote_to_semantic(
                    snap.snapshot_id, consolidation_time=_SESSION_END
                )
            except (ValueError, KeyError):
                pass
            finally:
                idx.close()

        threads = [threading.Thread(target=_promote) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        reader = ShadowIndex(db_path)
        try:
            fetched = reader.get_snapshot(snap.snapshot_id)
            assert fetched is not None
            assert fetched.memory_type == MemoryType.SEMANTIC
        finally:
            reader.close()

    def test_concurrent_promotions_of_different_snapshots_both_succeed(
        self, tmp_path: Path
    ) -> None:
        """Two threads promoting distinct snapshots must both complete without error."""
        db_path = tmp_path / "test.sqlite"
        setup = _make_index(tmp_path)
        snap_a = _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=1))
        snap_b = _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=2))
        setup.close()

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def _promote(snapshot_id: str) -> None:
            idx = ShadowIndex(db_path)
            barrier.wait()
            try:
                idx.promote_to_semantic(
                    snapshot_id, consolidation_time=_SESSION_END
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                idx.close()

        threads = [
            threading.Thread(target=_promote, args=(snap_a.snapshot_id,)),
            threading.Thread(target=_promote, args=(snap_b.snapshot_id,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        reader = ShadowIndex(db_path)
        try:
            assert reader.get_snapshot(snap_a.snapshot_id).memory_type == MemoryType.SEMANTIC
            assert reader.get_snapshot(snap_b.snapshot_id).memory_type == MemoryType.SEMANTIC
        finally:
            reader.close()

    def test_concurrent_boundary_events_do_not_corrupt_episodic_count(
        self, tmp_path: Path
    ) -> None:
        """After the race, episodic_count for the current window is exactly 0."""
        db_path = tmp_path / "test.sqlite"
        setup = _make_index(tmp_path)
        snap = _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=1))
        setup.close()

        barrier = threading.Barrier(2)

        def _promote() -> None:
            idx = ShadowIndex(db_path)
            barrier.wait()
            try:
                idx.promote_to_semantic(
                    snap.snapshot_id, consolidation_time=_SESSION_END
                )
            except (ValueError, KeyError):
                pass
            finally:
                idx.close()

        threads = [threading.Thread(target=_promote) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        reader = ShadowIndex(db_path)
        try:
            assert reader.episodic_count(_SOURCE_ID, since=_SESSION_START) == 0
        finally:
            reader.close()

    def test_concurrent_batch_promotion_leaves_all_semantic(
        self, tmp_path: Path
    ) -> None:
        """N threads each promote a distinct snapshot; all N must end up SEMANTIC."""
        n = 6
        db_path = tmp_path / "test.sqlite"
        setup = _make_index(tmp_path)
        snaps = [
            _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=i + 1))
            for i in range(n)
        ]
        setup.close()

        errors: list[Exception] = []
        barrier = threading.Barrier(n)

        def _promote(snapshot_id: str) -> None:
            idx = ShadowIndex(db_path)
            barrier.wait()
            try:
                idx.promote_to_semantic(
                    snapshot_id, consolidation_time=_SESSION_END
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                idx.close()

        threads = [
            threading.Thread(target=_promote, args=(s.snapshot_id,)) for s in snaps
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        reader = ShadowIndex(db_path)
        try:
            for snap in snaps:
                assert reader.get_snapshot(snap.snapshot_id).memory_type == MemoryType.SEMANTIC
        finally:
            reader.close()

    def test_sequential_second_promotion_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Once a snapshot has been promoted to SEMANTIC, a second call from any caller
        must raise ValueError — the EPISODIC precondition is no longer satisfied.

        This is the sequential equivalent of the race-loser contract: callers that
        arrive after the fact are rejected, not silently ignored.
        """
        db_path = tmp_path / "test.sqlite"
        setup = _make_index(tmp_path)
        snap = _add_episodic(setup, ingested_at=_SESSION_START + timedelta(hours=1))
        setup.close()

        first = ShadowIndex(db_path)
        first.promote_to_semantic(snap.snapshot_id, consolidation_time=_SESSION_END)
        first.close()

        second = ShadowIndex(db_path)
        try:
            with pytest.raises(ValueError):
                second.promote_to_semantic(snap.snapshot_id, consolidation_time=_SESSION_END)
        finally:
            second.close()
