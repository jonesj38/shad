"""Tests for combined threshold interactions.

Verifies that count-based (ConsolidationConfig) and time-based
(TimeIntervalTrigger) thresholds interact correctly when evaluated
together as a composite gate:

  - count threshold met AND cooldown expired  → fires
  - count threshold met BUT cooldown active   → suppressed
  - count threshold not met, cooldown expired → suppressed
  - neither threshold met                     → suppressed
  - config changes to thresholds take effect on subsequent evaluations

The composite gate models the natural consolidation policy: "run only
when there is enough new data *and* the minimum rest period has passed."
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.consolidation import ConsolidationConfig, TimeIntervalTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)
_COUNT_THRESHOLD = 5
_COOLDOWN = timedelta(hours=6)


def _gate(
    count_cfg: ConsolidationConfig,
    cooldown: TimeIntervalTrigger,
    *,
    entry_count: int,
    last_consolidated_at: datetime | None,
    now: datetime,
) -> bool:
    """Composite gate: both triggers must agree to fire."""
    return count_cfg.should_consolidate(entry_count) and cooldown.should_consolidate(
        entry_count,
        last_consolidated_at=last_consolidated_at,
        now=now,
    )


# ---------------------------------------------------------------------------
# Core interaction matrix
# ---------------------------------------------------------------------------


class TestCombinedThresholdMatrix:
    """Four-quadrant matrix: count × cooldown combinations."""

    def _cfg(self, threshold: int = _COUNT_THRESHOLD) -> ConsolidationConfig:
        return ConsolidationConfig(threshold=threshold)

    def _cooldown(self, interval: timedelta = _COOLDOWN) -> TimeIntervalTrigger:
        return TimeIntervalTrigger(min_interval=interval)

    def test_count_met_cooldown_expired_fires(self) -> None:
        """Both thresholds satisfied → consolidation fires."""
        last = _BASE
        now = _BASE + _COOLDOWN  # exactly at cooldown boundary

        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD,
            last_consolidated_at=last,
            now=now,
        ) is True

    def test_count_met_but_cooldown_active_suppresses(self) -> None:
        """Count threshold reached but cooldown has not expired → suppressed."""
        last = _BASE
        now = _BASE + _COOLDOWN - timedelta(seconds=1)  # one second before cooldown expires

        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD,
            last_consolidated_at=last,
            now=now,
        ) is False

    def test_cooldown_expired_but_count_not_met_suppresses(self) -> None:
        """Enough time has passed but not enough entries yet → suppressed."""
        last = _BASE
        now = _BASE + _COOLDOWN + timedelta(hours=2)  # well past cooldown

        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD - 1,
            last_consolidated_at=last,
            now=now,
        ) is False

    def test_neither_threshold_met_suppresses(self) -> None:
        """Count below threshold AND cooldown still active → suppressed."""
        last = _BASE
        now = _BASE + timedelta(hours=1)  # well within cooldown

        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD - 1,
            last_consolidated_at=last,
            now=now,
        ) is False

    def test_count_far_above_threshold_still_blocked_by_cooldown(self) -> None:
        """A large backlog does not bypass the cooldown guard."""
        last = _BASE
        now = _BASE + _COOLDOWN - timedelta(minutes=1)

        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD * 10,
            last_consolidated_at=last,
            now=now,
        ) is False

    def test_count_exactly_at_threshold_fires_when_cooldown_expired(self) -> None:
        """The exact boundary on both axes simultaneously fires."""
        last = _BASE
        now = _BASE + _COOLDOWN

        assert _gate(
            self._cfg(_COUNT_THRESHOLD),
            self._cooldown(_COOLDOWN),
            entry_count=_COUNT_THRESHOLD,
            last_consolidated_at=last,
            now=now,
        ) is True

    def test_never_consolidated_bypasses_cooldown_check(self) -> None:
        """last_consolidated_at=None means the time trigger fires unconditionally;
        consolidation runs as soon as the count threshold is met."""
        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD,
            last_consolidated_at=None,
            now=_BASE,
        ) is True

    def test_never_consolidated_but_count_not_met_still_suppresses(self) -> None:
        """Even with last=None, the count guard still applies."""
        assert _gate(
            self._cfg(),
            self._cooldown(),
            entry_count=_COUNT_THRESHOLD - 1,
            last_consolidated_at=None,
            now=_BASE,
        ) is False


# ---------------------------------------------------------------------------
# Disabled flags interact correctly
# ---------------------------------------------------------------------------


class TestDisabledFlagsInteraction:
    """Disabled count trigger or disabled cooldown trigger each suppress independently."""

    def test_disabled_count_trigger_suppresses_regardless_of_cooldown(self) -> None:
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD, enabled=False)
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)
        last = _BASE
        now = _BASE + _COOLDOWN + timedelta(days=1)  # cooldown long expired

        assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD * 5, last_consolidated_at=last, now=now) is False

    def test_disabled_cooldown_suppresses_regardless_of_count(self) -> None:
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD)
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN, enabled=False)
        last = _BASE
        now = _BASE + _COOLDOWN + timedelta(days=1)

        assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD * 5, last_consolidated_at=last, now=now) is False

    def test_both_disabled_suppresses(self) -> None:
        cfg = ConsolidationConfig(threshold=1, enabled=False)
        cooldown = TimeIntervalTrigger(min_interval=timedelta(seconds=1), enabled=False)
        now = _BASE + timedelta(days=365)

        assert _gate(cfg, cooldown, entry_count=9999, last_consolidated_at=_BASE, now=now) is False


# ---------------------------------------------------------------------------
# Config changes take effect on subsequent evaluations
# ---------------------------------------------------------------------------


class TestConfigChangesTakeEffect:
    """Replacing a config object must affect the very next evaluation."""

    def test_lowering_count_threshold_enables_previously_blocked_run(self) -> None:
        """entry_count=7 is below threshold=10 but above threshold=5."""
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)
        last = _BASE
        now = _BASE + _COOLDOWN

        # With the original (higher) threshold: suppressed.
        high_cfg = ConsolidationConfig(threshold=10)
        assert _gate(high_cfg, cooldown, entry_count=7, last_consolidated_at=last, now=now) is False

        # After replacing with a lower threshold: fires.
        low_cfg = ConsolidationConfig(threshold=5)
        assert _gate(low_cfg, cooldown, entry_count=7, last_consolidated_at=last, now=now) is True

    def test_raising_count_threshold_suppresses_previously_firing_run(self) -> None:
        """entry_count=7 fires with threshold=5 but is blocked when raised to 10."""
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)
        last = _BASE
        now = _BASE + _COOLDOWN

        low_cfg = ConsolidationConfig(threshold=5)
        assert _gate(low_cfg, cooldown, entry_count=7, last_consolidated_at=last, now=now) is True

        high_cfg = ConsolidationConfig(threshold=10)
        assert _gate(high_cfg, cooldown, entry_count=7, last_consolidated_at=last, now=now) is False

    def test_shortening_cooldown_enables_previously_blocked_run(self) -> None:
        """Elapsed=2h is inside a 6h cooldown but outside a 1h cooldown."""
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD)
        last = _BASE
        now = _BASE + timedelta(hours=2)  # 2 h elapsed

        long_cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=6))
        assert _gate(cfg, long_cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=last, now=now) is False

        short_cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=1))
        assert _gate(cfg, short_cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=last, now=now) is True

    def test_extending_cooldown_suppresses_previously_firing_run(self) -> None:
        """Elapsed=2h fires with a 1h cooldown but is blocked by a 6h cooldown."""
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD)
        last = _BASE
        now = _BASE + timedelta(hours=2)

        short_cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=1))
        assert _gate(cfg, short_cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=last, now=now) is True

        long_cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=6))
        assert _gate(cfg, long_cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=last, now=now) is False

    def test_changing_both_thresholds_simultaneously(self) -> None:
        """Changing both count and interval configs at once affects the composite gate."""
        last = _BASE
        now = _BASE + timedelta(hours=3)  # 3 h elapsed, 8 entries

        # Original: threshold=10, cooldown=6h → blocked on both
        orig_cfg = ConsolidationConfig(threshold=10)
        orig_cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=6))
        assert _gate(orig_cfg, orig_cooldown, entry_count=8, last_consolidated_at=last, now=now) is False

        # Updated: threshold=5, cooldown=2h → both satisfied
        new_cfg = ConsolidationConfig(threshold=5)
        new_cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=2))
        assert _gate(new_cfg, new_cooldown, entry_count=8, last_consolidated_at=last, now=now) is True

    def test_config_change_does_not_affect_past_evaluations(self) -> None:
        """Changing a config does not retroactively alter a result already computed."""
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)
        last = _BASE
        now = _BASE + _COOLDOWN

        cfg_v1 = ConsolidationConfig(threshold=10)
        result_before = _gate(cfg_v1, cooldown, entry_count=7, last_consolidated_at=last, now=now)
        assert result_before is False

        # Swap in a different config — prior result is unchanged (it's a bool).
        cfg_v2 = ConsolidationConfig(threshold=5)
        result_after = _gate(cfg_v2, cooldown, entry_count=7, last_consolidated_at=last, now=now)
        assert result_after is True

        # Original result still False — changing config can't reach back in time.
        assert result_before is False


# ---------------------------------------------------------------------------
# Baseline-reset interaction: consolidation resets cooldown, not count
# ---------------------------------------------------------------------------


class TestBaselineResetInteraction:
    """After a consolidation run the cooldown clock resets; the count trigger
    evaluates fresh entries against the same threshold.
    """

    def test_after_first_consolidation_cooldown_blocks_immediate_rerun(self) -> None:
        """A second consolidation is blocked right after the first completes."""
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD)
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)

        first_last = _BASE
        fire_time = _BASE + _COOLDOWN

        # First run fires.
        assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=first_last, now=fire_time) is True

        # Immediately after (baseline advances to fire_time), new entries arrive.
        second_last = fire_time
        now_soon = fire_time + timedelta(minutes=1)
        assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=second_last, now=now_soon) is False

    def test_after_cooldown_resets_count_must_still_be_met(self) -> None:
        """After the cooldown resets, the count gate still applies independently."""
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD)
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)

        second_last = _BASE
        now = _BASE + _COOLDOWN  # cooldown expired from new baseline

        # Count not yet met → still blocked.
        assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD - 1, last_consolidated_at=second_last, now=now) is False

        # Count now met → fires.
        assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=second_last, now=now) is True

    def test_multiple_cycles_count_and_cooldown_alternate_gating(self) -> None:
        """Simulate three consolidation cycles; cooldown suppresses the interim checks."""
        cfg = ConsolidationConfig(threshold=_COUNT_THRESHOLD)
        cooldown = TimeIntervalTrigger(min_interval=_COOLDOWN)
        baseline = _BASE

        for cycle in range(3):
            # Before cooldown expires: suppressed even with plenty of entries.
            mid = baseline + _COOLDOWN - timedelta(seconds=1)
            assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD * 3, last_consolidated_at=baseline, now=mid) is False, f"cycle {cycle}: should be suppressed mid-cooldown"

            # At cooldown boundary with count met: fires.
            at_boundary = baseline + _COOLDOWN
            assert _gate(cfg, cooldown, entry_count=_COUNT_THRESHOLD, last_consolidated_at=baseline, now=at_boundary) is True, f"cycle {cycle}: should fire at boundary"

            # Advance baseline for next cycle.
            baseline = at_boundary


# ---------------------------------------------------------------------------
# Parametrised: various threshold / cooldown combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "count_threshold,cooldown_hours,entry_count,elapsed_hours,expect_fire",
    [
        # Both satisfied → fires
        (5,  6,  5,  6,  True),
        (10, 12, 15, 24, True),
        (1,  1,  1,  2,  True),
        # Count met, cooldown active → suppressed
        (5,  6,  5,  5,  False),
        (10, 24, 10, 12, False),
        # Count not met, cooldown expired → suppressed
        (5,  6,  4,  7,  False),
        (10, 6,  9,  6,  False),
        # Neither met → suppressed
        (5,  6,  3,  2,  False),
        (10, 12, 5,  6,  False),
    ],
)
def test_combined_gate_parametrised(
    count_threshold: int,
    cooldown_hours: int,
    entry_count: int,
    elapsed_hours: int,
    expect_fire: bool,
) -> None:
    cfg = ConsolidationConfig(threshold=count_threshold)
    cooldown = TimeIntervalTrigger(min_interval=timedelta(hours=cooldown_hours))
    last = _BASE
    now = _BASE + timedelta(hours=elapsed_hours)

    assert _gate(cfg, cooldown, entry_count=entry_count, last_consolidated_at=last, now=now) is expect_fire
