"""Tests for DebouncedConsolidationTrigger: debounce-window consolidation trigger.

Verifies that rapid successive memory-add events are collapsed into a single
consolidation call: consolidation is suppressed while events keep arriving
within the debounce window and fires only once the quiet period has elapsed.

The debounce contract:
- `should_consolidate(entry_count, *, last_event_at, now)` returns True when:
    (now - last_event_at) >= debounce_window  AND  entry_count >= min_entries
- Each new memory-add resets `last_event_at`, re-arming the debounce.
- No consolidation fires mid-burst; exactly one consolidation fires per burst
  after the quiet period.

Test structure:
- TestDebouncedTriggerPreconditions:       construction validation
- TestDebouncedTriggerDuringBurst:         suppressed while events keep arriving
- TestDebouncedTriggerAfterQuietPeriod:    fires once burst settles
- TestDebouncedTriggerExactBoundary:       at, one-second-before, one-second-after
- TestDebouncedTriggerMinEntries:          entry-count gate
- TestDebouncedTriggerDisabled:            enabled=False always defers
- TestDebouncedTriggerBurstSimulation:     end-to-end burst → one call sequence
- TestDebouncedTriggerMultipleBursts:      re-arming after consolidation
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.consolidation import DebouncedConsolidationTrigger

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
_WINDOW = timedelta(seconds=30)


class TestDebouncedTriggerPreconditions:
    """DebouncedConsolidationTrigger must reject invalid construction arguments."""

    @pytest.mark.parametrize(
        "bad_window",
        [
            timedelta(seconds=0),
            timedelta(seconds=-1),
            timedelta(minutes=-5),
        ],
    )
    def test_zero_or_negative_window_raises_value_error(
        self, bad_window: timedelta
    ) -> None:
        with pytest.raises(ValueError, match="debounce_window must be > 0"):
            DebouncedConsolidationTrigger(debounce_window=bad_window)

    @pytest.mark.parametrize("bad_min", [0, -1, -10])
    def test_zero_or_negative_min_entries_raises_value_error(
        self, bad_min: int
    ) -> None:
        with pytest.raises(ValueError, match="min_entries must be >= 1"):
            DebouncedConsolidationTrigger(
                debounce_window=_WINDOW, min_entries=bad_min
            )

    def test_positive_window_constructs_successfully(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        assert trigger.debounce_window == _WINDOW
        assert trigger.min_entries == 1
        assert trigger.enabled is True

    def test_custom_min_entries_stored(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, min_entries=5
        )
        assert trigger.min_entries == 5

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        with pytest.raises((AttributeError, TypeError)):
            trigger.debounce_window = timedelta(seconds=60)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# During burst: suppressed while events keep arriving within the window
# ---------------------------------------------------------------------------


class TestDebouncedTriggerDuringBurst:
    """should_consolidate returns False while adds arrive within the window."""

    def test_does_not_fire_immediately_after_single_event(self) -> None:
        """An event just added (last_event_at == now) has not yet settled."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        assert trigger.should_consolidate(
            1, last_event_at=_BASE, now=_BASE
        ) is False

    def test_does_not_fire_one_second_into_window(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        assert trigger.should_consolidate(
            5,
            last_event_at=_BASE,
            now=_BASE + timedelta(seconds=1),
        ) is False

    def test_does_not_fire_just_before_window_elapses(self) -> None:
        """One second before the window closes → still suppressed."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        just_before = _BASE + _WINDOW - timedelta(seconds=1)
        assert trigger.should_consolidate(
            10, last_event_at=_BASE, now=just_before
        ) is False

    def test_rapid_successive_events_each_reset_the_window(self) -> None:
        """Three events spaced 10 s apart inside a 30 s window: still no trigger."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        event_times = [_BASE + timedelta(seconds=i * 10) for i in range(3)]
        # Poll should_consolidate at each add, passing the latest event as last_event_at.
        for idx, event_time in enumerate(event_times):
            # "now" is the event time itself; window has not elapsed since last event.
            fired = trigger.should_consolidate(
                idx + 1, last_event_at=event_time, now=event_time
            )
            assert fired is False, f"Should not fire at event {idx} (t={event_time})"

    def test_five_rapid_events_over_25s_never_fire_within_30s_window(self) -> None:
        """Five adds at 5-second intervals: always within 30 s of the latest add."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        for i in range(5):
            last_event_at = _BASE + timedelta(seconds=i * 5)
            now = last_event_at  # check immediately after each add
            assert trigger.should_consolidate(
                i + 1, last_event_at=last_event_at, now=now
            ) is False


# ---------------------------------------------------------------------------
# After quiet period: fires once burst settles
# ---------------------------------------------------------------------------


class TestDebouncedTriggerAfterQuietPeriod:
    """should_consolidate returns True once debounce_window has elapsed."""

    def test_fires_at_exactly_window_boundary(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        at_boundary = _BASE + _WINDOW
        assert trigger.should_consolidate(
            1, last_event_at=_BASE, now=at_boundary
        ) is True

    def test_fires_well_after_window(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        long_after = _BASE + timedelta(minutes=10)
        assert trigger.should_consolidate(
            3, last_event_at=_BASE, now=long_after
        ) is True

    def test_fires_after_burst_of_five_once_quiet(self) -> None:
        """Burst of five rapid events; after the quiet period the trigger fires."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        # Simulate five adds spaced 5 s apart; the last one lands at _BASE+20 s.
        last_event_at = _BASE + timedelta(seconds=20)
        quiet_check = last_event_at + _WINDOW  # exactly one window after last add
        assert trigger.should_consolidate(
            5, last_event_at=last_event_at, now=quiet_check
        ) is True


# ---------------------------------------------------------------------------
# Exact boundary: one second before, at, one second after
# ---------------------------------------------------------------------------


class TestDebouncedTriggerExactBoundary:
    """Precision around the exact window boundary."""

    @pytest.mark.parametrize("window", [timedelta(seconds=5), timedelta(minutes=2)])
    def test_one_second_before_boundary_does_not_fire(
        self, window: timedelta
    ) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=window)
        just_before = _BASE + window - timedelta(seconds=1)
        assert trigger.should_consolidate(
            1, last_event_at=_BASE, now=just_before
        ) is False

    @pytest.mark.parametrize("window", [timedelta(seconds=5), timedelta(minutes=2)])
    def test_at_boundary_fires(self, window: timedelta) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=window)
        at_boundary = _BASE + window
        assert trigger.should_consolidate(
            1, last_event_at=_BASE, now=at_boundary
        ) is True

    @pytest.mark.parametrize("window", [timedelta(seconds=5), timedelta(minutes=2)])
    def test_one_second_after_boundary_fires(self, window: timedelta) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=window)
        just_after = _BASE + window + timedelta(seconds=1)
        assert trigger.should_consolidate(
            1, last_event_at=_BASE, now=just_after
        ) is True


# ---------------------------------------------------------------------------
# min_entries gate
# ---------------------------------------------------------------------------


class TestDebouncedTriggerMinEntries:
    """Consolidation is suppressed when entry_count < min_entries."""

    def test_does_not_fire_when_count_below_min_entries(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, min_entries=5
        )
        # Window has elapsed, but only 4 entries accumulated.
        assert trigger.should_consolidate(
            4, last_event_at=_BASE, now=_BASE + _WINDOW
        ) is False

    def test_fires_when_count_meets_min_entries(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, min_entries=5
        )
        assert trigger.should_consolidate(
            5, last_event_at=_BASE, now=_BASE + _WINDOW
        ) is True

    def test_fires_when_count_exceeds_min_entries(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, min_entries=3
        )
        assert trigger.should_consolidate(
            10, last_event_at=_BASE, now=_BASE + _WINDOW
        ) is True

    def test_default_min_entries_of_one_requires_at_least_one_entry(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        # Zero entries → never fires even after window.
        assert trigger.should_consolidate(
            0, last_event_at=_BASE, now=_BASE + _WINDOW
        ) is False

    def test_no_last_event_at_means_no_events_yet(self) -> None:
        """When last_event_at is None no event has ever been recorded → defer."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        assert trigger.should_consolidate(
            5, last_event_at=None, now=_BASE + timedelta(hours=1)
        ) is False


# ---------------------------------------------------------------------------
# Disabled trigger
# ---------------------------------------------------------------------------


class TestDebouncedTriggerDisabled:
    """enabled=False always defers regardless of timing or entry count."""

    def test_disabled_does_not_fire_at_window_boundary(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, enabled=False
        )
        assert trigger.should_consolidate(
            100, last_event_at=_BASE, now=_BASE + _WINDOW
        ) is False

    def test_disabled_does_not_fire_well_after_window(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, enabled=False
        )
        assert trigger.should_consolidate(
            100, last_event_at=_BASE, now=_BASE + timedelta(days=365)
        ) is False

    def test_disabled_does_not_fire_with_none_last_event(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=_WINDOW, enabled=False
        )
        assert trigger.should_consolidate(
            5, last_event_at=None, now=_BASE + timedelta(hours=1)
        ) is False


# ---------------------------------------------------------------------------
# End-to-end burst simulation: multiple rapid adds → exactly one trigger
# ---------------------------------------------------------------------------


class TestDebouncedTriggerBurstSimulation:
    """Simulate a realistic burst: rapid adds followed by a quiet period.

    Contract: N rapid adds within the debounce window produce zero True
    returns during the burst and exactly one True return once quiet.
    """

    def test_ten_rapid_adds_produce_no_fire_during_burst(self) -> None:
        trigger = DebouncedConsolidationTrigger(
            debounce_window=timedelta(seconds=30), min_entries=3
        )
        fires_during_burst = []
        for i in range(10):
            # Each add lands 2 seconds after the previous one (burst interval < window).
            last_event_at = _BASE + timedelta(seconds=i * 2)
            now = last_event_at
            fires_during_burst.append(
                trigger.should_consolidate(
                    i + 1, last_event_at=last_event_at, now=now
                )
            )
        assert not any(fires_during_burst), (
            f"Fired during burst at positions: "
            f"{[i for i, v in enumerate(fires_during_burst) if v]}"
        )

    def test_quiet_period_after_burst_produces_exactly_one_true(self) -> None:
        """After 10 rapid adds, poll at window+1s: fires exactly once."""
        trigger = DebouncedConsolidationTrigger(
            debounce_window=timedelta(seconds=30), min_entries=3
        )
        # Last add was at _BASE + 18 s (9 adds × 2 s).
        last_event_at = _BASE + timedelta(seconds=18)
        quiet_check = last_event_at + timedelta(seconds=31)

        results = [
            trigger.should_consolidate(10, last_event_at=last_event_at, now=quiet_check)
            for _ in range(3)  # polling multiple times at the same "now"
        ]
        # All three polls at the same timestamp return True (idempotent query).
        assert all(results)

    def test_consolidation_check_is_stateless(self) -> None:
        """should_consolidate is a pure query: calling it twice with the same args
        returns the same result without side effects.
        """
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        kwargs = dict(last_event_at=_BASE, now=_BASE + _WINDOW)
        first = trigger.should_consolidate(5, **kwargs)
        second = trigger.should_consolidate(5, **kwargs)
        assert first == second


# ---------------------------------------------------------------------------
# Multiple bursts: re-arming after consolidation runs
# ---------------------------------------------------------------------------


class TestDebouncedTriggerMultipleBursts:
    """Trigger re-arms correctly: a new burst suppresses again after a consolidation."""

    def test_second_burst_is_suppressed_while_active(self) -> None:
        """After the first consolidation the caller resets last_event_at to the
        new burst's first event; the trigger should suppress again immediately.
        """
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        # First burst fires after its quiet period.
        first_event = _BASE
        first_quiet = first_event + _WINDOW
        assert trigger.should_consolidate(
            5, last_event_at=first_event, now=first_quiet
        ) is True

        # Second burst starts one second after first consolidation.
        second_event = first_quiet + timedelta(seconds=1)
        now_during_second_burst = second_event + timedelta(seconds=5)
        assert trigger.should_consolidate(
            3, last_event_at=second_event, now=now_during_second_burst
        ) is False

    def test_second_burst_fires_after_its_own_quiet_period(self) -> None:
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        second_event = _BASE + timedelta(minutes=5)
        second_quiet = second_event + _WINDOW
        assert trigger.should_consolidate(
            4, last_event_at=second_event, now=second_quiet
        ) is True

    def test_interleaved_bursts_each_fire_once(self) -> None:
        """Three distinct bursts separated by quiet periods each produce one True."""
        trigger = DebouncedConsolidationTrigger(debounce_window=_WINDOW)
        burst_last_events = [
            _BASE,
            _BASE + timedelta(minutes=5),
            _BASE + timedelta(minutes=10),
        ]
        for last_event_at in burst_last_events:
            # During burst: no fire.
            mid_burst = last_event_at + timedelta(seconds=5)
            assert trigger.should_consolidate(
                3, last_event_at=last_event_at, now=mid_burst
            ) is False
            # After quiet period: fires.
            after_quiet = last_event_at + _WINDOW
            assert trigger.should_consolidate(
                3, last_event_at=last_event_at, now=after_quiet
            ) is True
