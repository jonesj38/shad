"""Tests for cooldown period after consolidation completes.

After consolidation fires, ``SourceManager._check_consolidation`` stamps
``source.metadata["last_consolidation"]`` with the current UTC time.  This
advances the episodic-count baseline so that subsequent trigger events are
suppressed until enough *new* entries re-accumulate to the threshold.

Three invariants under test:

1. Immediate re-trigger is suppressed — the post-fire count drops to zero
   because no new entries have arrived after the new baseline.
2. Partial re-accumulation is suppressed — entries that arrive after
   consolidation but below the threshold do not fire.
3. Consolidation resumes once the post-cooldown entry count reaches the
   threshold, and subsequent cycles repeat the same pattern.

Helper convention: pre-cooldown snapshots use a fixed past date (2026-01-01);
post-cooldown snapshots use a timestamp derived from ``last_consolidation +
timedelta(seconds=1)`` so they are guaranteed to be strictly after the
baseline regardless of wall-clock drift.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from shad.sources.config import Source, SourceType
from shad.sources.manager import SourceManager
from shad.vault.consolidation import ConsolidationConfig
from shad.vault.shadow_index import MemoryType, ShadowIndex, SnapshotEntry, SourceEntry, UpdatePolicy

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_THRESHOLD = 3
_PRE_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _make_index_and_source(tmp_path: Path) -> tuple[ShadowIndex, Source]:
    index = ShadowIndex(tmp_path / "index.sqlite")
    source = Source(
        type=SourceType.FEED,
        url="https://feeds.example.com/cooldown-test",
        collection_path="/tmp/shad-cooldown",
    )
    index.add_source(
        SourceEntry(
            source_url=source.url,
            source_id=source.id,
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
        )
    )
    return index, source


def _insert_snapshots(
    index: ShadowIndex,
    source_id: str,
    count: int,
    base: datetime,
) -> None:
    for i in range(count):
        index.add_snapshot(
            SnapshotEntry(
                snapshot_id=str(uuid.uuid4()),
                source_id=source_id,
                ingested_at=base + timedelta(hours=i + 1),
                source_revision=f"rev-{i}",
                entry_paths=[f"feeds/item-{i}.md"],
                content_hash=str(uuid.uuid4()),
                memory_type=MemoryType.EPISODIC,
            )
        )


def _make_manager(
    index: ShadowIndex,
    callback: object,
    threshold: int = _THRESHOLD,
) -> SourceManager:
    return SourceManager(
        consolidation_config=ConsolidationConfig(threshold=threshold),
        shadow_index=index,
        on_consolidation=callback,  # type: ignore[arg-type]
    )


def _post_cooldown_base(source: Source) -> datetime:
    """Return a timestamp guaranteed to be strictly after last_consolidation."""
    stamp = datetime.fromisoformat(source.metadata["last_consolidation"])
    return stamp + timedelta(seconds=1)


# ---------------------------------------------------------------------------
# TestCooldownSuppressesImmediateRetrigger
# ---------------------------------------------------------------------------


class TestCooldownSuppressesImmediateRetrigger:
    """After consolidation fires and advances the baseline, calling
    _check_consolidation again with no new entries must be suppressed because
    the post-baseline episodic count is zero.
    """

    def test_immediate_retrigger_is_suppressed(self, tmp_path: Path) -> None:
        """First call fires; second call (no new entries) stays silent."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)

        manager._check_consolidation(source)
        assert callback.call_count == 1

        # No new entries — baseline has advanced past all existing ones.
        manager._check_consolidation(source)
        assert callback.call_count == 1

        index.close()

    def test_five_successive_retrigggers_all_suppressed(self, tmp_path: Path) -> None:
        """Five successive calls after one fire all stay silent."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)

        manager._check_consolidation(source)
        assert callback.call_count == 1

        for _ in range(5):
            manager._check_consolidation(source)
        assert callback.call_count == 1

        index.close()

    def test_post_fire_episodic_count_is_zero(self, tmp_path: Path) -> None:
        """The baseline advances past all pre-fire entries, leaving count=0."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)

        baseline = datetime.fromisoformat(source.metadata["last_consolidation"])
        count_after = index.episodic_count(source.id, since=baseline)
        assert count_after == 0

        index.close()


# ---------------------------------------------------------------------------
# TestCooldownSuppressesPartialReaccumulation
# ---------------------------------------------------------------------------


class TestCooldownSuppressesPartialReaccumulation:
    """Entries that arrive after consolidation but below threshold must not
    trigger a second consolidation pass.
    """

    @pytest.mark.parametrize("new_count", range(1, _THRESHOLD))
    def test_below_threshold_new_entries_suppressed(
        self, new_count: int, tmp_path: Path
    ) -> None:
        """threshold-k new entries (1 ≤ k < threshold) after cooldown: suppressed."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)
        assert callback.call_count == 1

        post_base = _post_cooldown_base(source)
        _insert_snapshots(index, source.id, new_count, post_base)

        manager._check_consolidation(source)
        assert callback.call_count == 1  # still suppressed

        index.close()

    def test_zero_new_entries_suppressed(self, tmp_path: Path) -> None:
        """No new entries after cooldown: suppressed regardless of threshold."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)

        callback.reset_mock()
        manager._check_consolidation(source)
        callback.assert_not_called()

        index.close()

    def test_large_pre_cooldown_backlog_does_not_cause_retrigger(
        self, tmp_path: Path
    ) -> None:
        """A large backlog of pre-consolidation entries does not cause a
        second fire after the baseline advances past them.
        """
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD * 10, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)

        manager._check_consolidation(source)
        assert callback.call_count == 1

        # No new entries — the large backlog is now before the baseline.
        manager._check_consolidation(source)
        assert callback.call_count == 1

        index.close()


# ---------------------------------------------------------------------------
# TestCooldownResumesAtThreshold
# ---------------------------------------------------------------------------


class TestCooldownResumesAtThreshold:
    """Consolidation resumes once post-cooldown entries reach the threshold."""

    def test_consolidation_resumes_at_exactly_threshold(self, tmp_path: Path) -> None:
        """Exactly threshold new entries after cooldown → fires again."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)

        manager._check_consolidation(source)
        assert callback.call_count == 1

        post_base = _post_cooldown_base(source)
        _insert_snapshots(index, source.id, _THRESHOLD, post_base)

        manager._check_consolidation(source)
        assert callback.call_count == 2

        index.close()

    def test_consolidation_resumes_above_threshold(self, tmp_path: Path) -> None:
        """threshold+2 new entries after cooldown → fires once (not twice)."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)

        manager._check_consolidation(source)
        assert callback.call_count == 1

        post_base = _post_cooldown_base(source)
        _insert_snapshots(index, source.id, _THRESHOLD + 2, post_base)

        manager._check_consolidation(source)
        assert callback.call_count == 2

        index.close()

    def test_callback_receives_post_cooldown_count_not_all_time_count(
        self, tmp_path: Path
    ) -> None:
        """The second callback receives only the post-cooldown count, not the
        cumulative all-time count.
        """
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)

        post_base = _post_cooldown_base(source)
        new_count = _THRESHOLD + 1
        _insert_snapshots(index, source.id, new_count, post_base)

        manager._check_consolidation(source)

        second_call_count_arg = callback.call_args_list[1][0][1]
        assert second_call_count_arg == new_count

        index.close()

    def test_threshold_minus_one_then_one_more_resumes(self, tmp_path: Path) -> None:
        """Adding entries in two batches: threshold-1 suppressed, then +1 fires."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)
        assert callback.call_count == 1

        post_base = _post_cooldown_base(source)
        _insert_snapshots(index, source.id, _THRESHOLD - 1, post_base)

        manager._check_consolidation(source)
        assert callback.call_count == 1  # still suppressed

        # One more entry tips it over the threshold.
        later_base = post_base + timedelta(hours=_THRESHOLD + 1)
        _insert_snapshots(index, source.id, 1, later_base)

        manager._check_consolidation(source)
        assert callback.call_count == 2

        index.close()


# ---------------------------------------------------------------------------
# TestCooldownMultipleCycles
# ---------------------------------------------------------------------------


class TestCooldownMultipleCycles:
    """Consolidation fires in successive cycles; each cycle starts a new
    cooldown window and the previous cycle's entries are excluded from the
    next count.
    """

    def test_three_cycles_each_fire_exactly_once(self, tmp_path: Path) -> None:
        """Three distinct bursts each trigger exactly one consolidation."""
        index, source = _make_index_and_source(tmp_path)
        callback = MagicMock()
        manager = _make_manager(index, callback)

        burst_base = _PRE_BASE
        for cycle in range(3):
            _insert_snapshots(index, source.id, _THRESHOLD, burst_base)
            manager._check_consolidation(source)
            assert callback.call_count == cycle + 1
            # Advance base past the new cooldown stamp for the next cycle.
            burst_base = _post_cooldown_base(source)

        index.close()

    def test_each_cycle_baseline_advances(self, tmp_path: Path) -> None:
        """last_consolidation timestamp is updated after each cycle."""
        index, source = _make_index_and_source(tmp_path)
        callback = MagicMock()
        manager = _make_manager(index, callback)

        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)
        manager._check_consolidation(source)
        stamp1 = source.metadata["last_consolidation"]

        post_base = _post_cooldown_base(source)
        _insert_snapshots(index, source.id, _THRESHOLD, post_base)
        manager._check_consolidation(source)
        stamp2 = source.metadata["last_consolidation"]

        assert stamp2 > stamp1

        index.close()

    def test_entries_from_cycle_n_excluded_from_cycle_n_plus_1(
        self, tmp_path: Path
    ) -> None:
        """After cycle-1 fires, adding threshold-1 entries must still suppress."""
        index, source = _make_index_and_source(tmp_path)
        callback = MagicMock()
        manager = _make_manager(index, callback)

        # Cycle 1 fires.
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)
        manager._check_consolidation(source)
        assert callback.call_count == 1

        # Cycle 2: add exactly threshold entries and fire.
        base2 = _post_cooldown_base(source)
        _insert_snapshots(index, source.id, _THRESHOLD, base2)
        manager._check_consolidation(source)
        assert callback.call_count == 2
        stamp2 = source.metadata["last_consolidation"]

        # Post-cycle-2: only threshold-1 new entries — must be suppressed.
        base3 = datetime.fromisoformat(stamp2) + timedelta(seconds=1)
        _insert_snapshots(index, source.id, _THRESHOLD - 1, base3)
        manager._check_consolidation(source)
        assert callback.call_count == 2  # still suppressed

        index.close()


# ---------------------------------------------------------------------------
# TestCooldownBaselineIsolation
# ---------------------------------------------------------------------------


class TestCooldownBaselineIsolation:
    """Entries ingested before (or at) the consolidation timestamp must never
    contribute to the post-cooldown episodic count.
    """

    def test_entries_strictly_before_baseline_excluded(self, tmp_path: Path) -> None:
        """Pre-baseline entries return count=0 when queried with since=baseline."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)

        baseline = datetime.fromisoformat(source.metadata["last_consolidation"])
        # All snapshots have ingested_at in Jan 2026, baseline is ~Mar 2026.
        count = index.episodic_count(source.id, since=baseline)
        assert count == 0

        index.close()

    def test_entries_at_exactly_baseline_excluded(self, tmp_path: Path) -> None:
        """episodic_count uses strict >, so entries at exactly the baseline
        timestamp are excluded and must not cause a retrigger.
        """
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)
        baseline = datetime.fromisoformat(source.metadata["last_consolidation"])

        # Insert threshold entries with ingested_at == baseline exactly.
        for i in range(_THRESHOLD):
            index.add_snapshot(
                SnapshotEntry(
                    snapshot_id=str(uuid.uuid4()),
                    source_id=source.id,
                    ingested_at=baseline,  # exactly at the stamp
                    source_revision=f"boundary-rev-{i}",
                    entry_paths=[f"feeds/boundary-{i}.md"],
                    content_hash=str(uuid.uuid4()),
                    memory_type=MemoryType.EPISODIC,
                )
            )

        # Strict > means these entries are excluded → count=0 → no retrigger.
        count = index.episodic_count(source.id, since=baseline)
        assert count == 0

        manager._check_consolidation(source)
        assert callback.call_count == 1  # no second fire

        index.close()

    def test_only_entries_strictly_after_baseline_count(self, tmp_path: Path) -> None:
        """Mix of at-baseline and after-baseline entries: only the latter count."""
        index, source = _make_index_and_source(tmp_path)
        _insert_snapshots(index, source.id, _THRESHOLD, _PRE_BASE)

        callback = MagicMock()
        manager = _make_manager(index, callback)
        manager._check_consolidation(source)
        baseline = datetime.fromisoformat(source.metadata["last_consolidation"])

        # One entry exactly at baseline (excluded) + threshold entries after (included).
        index.add_snapshot(
            SnapshotEntry(
                snapshot_id=str(uuid.uuid4()),
                source_id=source.id,
                ingested_at=baseline,
                source_revision="at-boundary",
                entry_paths=["feeds/at-boundary.md"],
                content_hash=str(uuid.uuid4()),
                memory_type=MemoryType.EPISODIC,
            )
        )
        post_base = baseline + timedelta(seconds=1)
        _insert_snapshots(index, source.id, _THRESHOLD, post_base)

        count = index.episodic_count(source.id, since=baseline)
        assert count == _THRESHOLD  # boundary entry excluded, post entries included

        manager._check_consolidation(source)
        assert callback.call_count == 2  # fires again

        index.close()
