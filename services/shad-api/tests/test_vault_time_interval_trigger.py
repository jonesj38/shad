"""Tests for TimeIntervalTrigger: time-elapsed consolidation trigger.

Verifies that consolidation fires after the configured time interval has
elapsed since the last consolidation, and does not fire before the interval.

Test structure:
- TestTimeIntervalTriggerPreconditions: construction validation
- TestTimeIntervalTriggerBoundary: exact boundary at, before, and after the interval
- TestTimeIntervalTriggerNeverConsolidated: None last_consolidated_at fires immediately
- TestTimeIntervalTriggerDisabled: enabled=False always defers
- TestTimeIntervalTriggerWallClock: default now=None falls back to real clock
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.consolidation import TimeIntervalTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


class TestTimeIntervalTriggerPreconditions:
    """TimeIntervalTrigger must reject invalid construction arguments."""

    @pytest.mark.parametrize(
        "bad_interval",
        [
            timedelta(seconds=0),
            timedelta(seconds=-1),
            timedelta(days=-7),
        ],
    )
    def test_zero_or_negative_interval_raises_value_error(
        self, bad_interval: timedelta
    ) -> None:
        with pytest.raises(ValueError, match="min_interval must be > 0"):
            TimeIntervalTrigger(min_interval=bad_interval)

    def test_positive_interval_constructs_successfully(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1))
        assert trigger.min_interval == timedelta(hours=1)
        assert trigger.enabled is True

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=6))
        with pytest.raises((AttributeError, TypeError)):
            trigger.min_interval = timedelta(hours=12)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Exact boundary: at, before, and after the interval
# ---------------------------------------------------------------------------


class TestTimeIntervalTriggerBoundary:
    """Verify the precise boundary at, one second before, and one second after."""

    @pytest.mark.parametrize(
        "interval",
        [
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(days=1),
            timedelta(days=7),
        ],
    )
    def test_fires_when_exactly_interval_has_elapsed(
        self, interval: timedelta
    ) -> None:
        trigger = TimeIntervalTrigger(min_interval=interval)
        last = _BASE
        now = _BASE + interval  # exactly at the boundary
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is True

    @pytest.mark.parametrize(
        "interval",
        [
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(days=1),
            timedelta(days=7),
        ],
    )
    def test_does_not_fire_one_second_before_interval(
        self, interval: timedelta
    ) -> None:
        trigger = TimeIntervalTrigger(min_interval=interval)
        last = _BASE
        now = _BASE + interval - timedelta(seconds=1)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is False

    def test_fires_after_interval(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=3))
        last = _BASE
        now = _BASE + timedelta(hours=3, seconds=1)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is True

    def test_does_not_fire_long_before_interval(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(days=7))
        last = _BASE
        now = _BASE + timedelta(days=3)  # only half the interval elapsed
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is False

    def test_entry_count_is_ignored(self) -> None:
        """entry_count has no effect on time-based trigger."""
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1))
        last = _BASE
        now = _BASE + timedelta(hours=1)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is True
        assert trigger.should_consolidate(999, last_consolidated_at=last, now=now) is True

    def test_boundary_one_millisecond_before_does_not_fire(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=24))
        last = _BASE
        now = _BASE + timedelta(hours=24) - timedelta(milliseconds=1)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is False

    def test_boundary_one_millisecond_after_fires(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=24))
        last = _BASE
        now = _BASE + timedelta(hours=24) + timedelta(milliseconds=1)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is True


# ---------------------------------------------------------------------------
# Never consolidated: last_consolidated_at=None fires immediately
# ---------------------------------------------------------------------------


class TestTimeIntervalTriggerNeverConsolidated:
    """When last_consolidated_at is None, the trigger fires unconditionally."""

    def test_fires_when_never_consolidated(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(days=7))
        assert trigger.should_consolidate(0, last_consolidated_at=None, now=_BASE) is True

    def test_fires_when_never_consolidated_without_now(self) -> None:
        """Passes now=None (real clock) — still fires because last=None."""
        trigger = TimeIntervalTrigger(min_interval=timedelta(days=365))
        assert trigger.should_consolidate(0, last_consolidated_at=None) is True

    def test_disabled_does_not_fire_even_when_never_consolidated(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1), enabled=False)
        assert trigger.should_consolidate(0, last_consolidated_at=None, now=_BASE) is False


# ---------------------------------------------------------------------------
# Disabled trigger: always defers regardless of elapsed time
# ---------------------------------------------------------------------------


class TestTimeIntervalTriggerDisabled:
    """When enabled=False, should_consolidate always returns False."""

    def test_disabled_does_not_fire_at_exact_interval(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1), enabled=False)
        last = _BASE
        now = _BASE + timedelta(hours=1)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is False

    def test_disabled_does_not_fire_far_past_interval(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1), enabled=False)
        last = _BASE
        now = _BASE + timedelta(days=365)
        assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is False

    def test_disabled_does_not_fire_with_none_last_consolidated_at(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1), enabled=False)
        assert trigger.should_consolidate(0, last_consolidated_at=None, now=_BASE) is False


# ---------------------------------------------------------------------------
# Wall-clock fallback: now=None uses datetime.now(UTC)
# ---------------------------------------------------------------------------


class TestTimeIntervalTriggerWallClock:
    """When now is not supplied, the trigger falls back to the real clock."""

    def test_fires_after_interval_using_real_clock(self) -> None:
        """A last_consolidated_at far in the past should fire with the real clock."""
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=1))
        ancient = datetime(2000, 1, 1, tzinfo=UTC)
        # Real clock is 2026 — definitely past a 1-hour interval.
        assert trigger.should_consolidate(0, last_consolidated_at=ancient) is True

    def test_does_not_fire_with_future_last_consolidated_at(self) -> None:
        """A last_consolidated_at far in the future means interval hasn't elapsed."""
        trigger = TimeIntervalTrigger(min_interval=timedelta(days=1))
        far_future = datetime(2099, 12, 31, tzinfo=UTC)
        assert trigger.should_consolidate(0, last_consolidated_at=far_future) is False


# ---------------------------------------------------------------------------
# Parameterised: common intervals fire / defer correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval,elapsed,expected",
    [
        # Fires: elapsed >= interval
        (timedelta(hours=1), timedelta(hours=1), True),
        (timedelta(hours=6), timedelta(hours=12), True),
        (timedelta(days=1), timedelta(days=2), True),
        (timedelta(days=7), timedelta(days=7), True),
        (timedelta(minutes=30), timedelta(minutes=30), True),
        # Defers: elapsed < interval
        (timedelta(hours=1), timedelta(minutes=59), False),
        (timedelta(hours=6), timedelta(hours=5, minutes=59), False),
        (timedelta(days=1), timedelta(hours=23, minutes=59), False),
        (timedelta(days=7), timedelta(days=6, hours=23), False),
        (timedelta(minutes=30), timedelta(minutes=29), False),
    ],
)
def test_fires_and_defers_across_common_intervals(
    interval: timedelta,
    elapsed: timedelta,
    expected: bool,
) -> None:
    trigger = TimeIntervalTrigger(min_interval=interval)
    last = _BASE
    now = _BASE + elapsed
    assert trigger.should_consolidate(0, last_consolidated_at=last, now=now) is expected


# ---------------------------------------------------------------------------
# Reset semantics: advancing last_consolidated_at resets the clock
# ---------------------------------------------------------------------------


class TestTimeIntervalTriggerResetSemantics:
    """After consolidation completes and last_consolidated_at advances, the
    interval must elapse again before the next trigger fires.
    """

    def test_does_not_fire_immediately_after_advancing_baseline(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=6))
        first_baseline = _BASE
        # First consolidation fires.
        now_1 = first_baseline + timedelta(hours=6)
        assert trigger.should_consolidate(0, last_consolidated_at=first_baseline, now=now_1) is True

        # Baseline advances to now_1 (consolidation completed).
        second_baseline = now_1
        # One hour later — interval has not elapsed from new baseline.
        now_2 = second_baseline + timedelta(hours=1)
        assert trigger.should_consolidate(0, last_consolidated_at=second_baseline, now=now_2) is False

    def test_fires_again_after_full_interval_from_new_baseline(self) -> None:
        trigger = TimeIntervalTrigger(min_interval=timedelta(hours=6))
        first_baseline = _BASE
        second_baseline = first_baseline + timedelta(hours=6)

        # Full interval elapsed from the advanced baseline.
        now = second_baseline + timedelta(hours=6)
        assert trigger.should_consolidate(0, last_consolidated_at=second_baseline, now=now) is True

    def test_multiple_reset_cycles(self) -> None:
        """Trigger fires → baseline advances → defers → fires again, three times."""
        trigger = TimeIntervalTrigger(min_interval=timedelta(days=1))
        baseline = _BASE

        for _ in range(3):
            # One second before: defers.
            now_before = baseline + timedelta(days=1) - timedelta(seconds=1)
            assert trigger.should_consolidate(0, last_consolidated_at=baseline, now=now_before) is False

            # Exactly at boundary: fires.
            now_at = baseline + timedelta(days=1)
            assert trigger.should_consolidate(0, last_consolidated_at=baseline, now=now_at) is True

            # Advance baseline.
            baseline = now_at
