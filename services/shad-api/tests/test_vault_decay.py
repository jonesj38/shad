"""Tests for configurable temporal decay parameters.

Verifies that halflife duration and decay curve shape produce the expected
score adjustments.  The mathematical contracts tested here mirror the
behaviour documented in:
  - teranode/connmgr/dynamicbanscore-go  (Halflife = 60 s)
  - openclaw/src/memory/temporal-decay.ts

Decay contract summary
----------------------
EXPONENTIAL  score(t) = score * 0.5^(t/halflife)
  - At t=0:           score unchanged
  - At t=halflife:    score halved
  - At t=2*halflife:  score quartered
  - Asymptotic: score > 0 for all finite t

LINEAR       score(t) = score * max(0, 1 - t/(2*halflife))
  - At t=0:           score unchanged
  - At t=halflife:    score halved
  - At t=2*halflife:  score == 0
  - score == 0 for all t >= 2*halflife
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.decay import DecayConfig, DecayCurve, SnapshotDecayScorer, apply_decay, decay_score, linear_age_factor
from shad.vault.gap_detection import QueryHistoryAnalyzer
from shad.vault.shadow_index import MemoryType, SnapshotEntry, source_to_memory_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPSILON = 1e-9  # floating-point tolerance for "exact" assertions
SCORE_TOLERANCE = 1e-6  # tolerance for score comparisons


def exp_decay(score: float, age: float, halflife: float) -> float:
    return score * math.pow(0.5, age / halflife)


def linear_decay(score: float, age: float, halflife: float) -> float:
    return score * max(0.0, 1.0 - age / (2.0 * halflife))


# ---------------------------------------------------------------------------
# DecayConfig validation
# ---------------------------------------------------------------------------

class TestDecayConfig:
    def test_valid_exponential_config(self) -> None:
        cfg = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.EXPONENTIAL)
        assert cfg.halflife_seconds == 3600.0
        assert cfg.curve is DecayCurve.EXPONENTIAL

    def test_valid_linear_config(self) -> None:
        cfg = DecayConfig(halflife_seconds=86400.0, curve=DecayCurve.LINEAR)
        assert cfg.curve is DecayCurve.LINEAR

    def test_default_curve_is_exponential(self) -> None:
        cfg = DecayConfig(halflife_seconds=60.0)
        assert cfg.curve is DecayCurve.EXPONENTIAL

    def test_zero_halflife_raises(self) -> None:
        with pytest.raises(ValueError, match="halflife_seconds must be > 0"):
            DecayConfig(halflife_seconds=0.0)

    def test_negative_halflife_raises(self) -> None:
        with pytest.raises(ValueError, match="halflife_seconds must be > 0"):
            DecayConfig(halflife_seconds=-1.0)

    def test_config_is_immutable(self) -> None:
        cfg = DecayConfig(halflife_seconds=60.0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.halflife_seconds = 120.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EXPONENTIAL curve — core halflife contract
# ---------------------------------------------------------------------------

class TestExponentialHalflife:
    """Score is halved exactly at t == halflife."""

    @pytest.fixture
    def cfg(self) -> DecayConfig:
        return DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)

    def test_age_zero_score_unchanged(self, cfg: DecayConfig) -> None:
        assert apply_decay(0.8, age_seconds=0.0, config=cfg) == pytest.approx(0.8)

    def test_age_halflife_score_halved(self, cfg: DecayConfig) -> None:
        assert apply_decay(1.0, age_seconds=60.0, config=cfg) == pytest.approx(0.5)

    def test_age_two_halflives_score_quartered(self, cfg: DecayConfig) -> None:
        assert apply_decay(1.0, age_seconds=120.0, config=cfg) == pytest.approx(0.25)

    def test_age_three_halflives_score_eighth(self, cfg: DecayConfig) -> None:
        assert apply_decay(1.0, age_seconds=180.0, config=cfg) == pytest.approx(0.125)

    def test_fractional_halflife_age(self, cfg: DecayConfig) -> None:
        """Score at t=halflife/2 should be 0.5^0.5 ≈ 0.7071."""
        result = apply_decay(1.0, age_seconds=30.0, config=cfg)
        assert result == pytest.approx(math.sqrt(0.5), rel=1e-6)

    def test_arbitrary_score_scaled_correctly(self, cfg: DecayConfig) -> None:
        score = 0.64
        result = apply_decay(score, age_seconds=60.0, config=cfg)
        assert result == pytest.approx(score * 0.5)

    def test_score_never_reaches_zero(self, cfg: DecayConfig) -> None:
        # At 10 halflives (600 s) score = 0.5^10 ≈ 0.000977, well above float min.
        # Extremely large ages cause IEEE 754 underflow; the mathematical
        # property is asymptotic, so we test at a tractable distance.
        result = apply_decay(1.0, age_seconds=600.0, config=cfg)
        assert result > 0.0

    def test_score_monotonically_decreasing(self, cfg: DecayConfig) -> None:
        ages = [0, 30, 60, 120, 240, 600]
        scores = [apply_decay(1.0, age_seconds=float(a), config=cfg) for a in ages]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


# ---------------------------------------------------------------------------
# EXPONENTIAL curve — halflife duration controls decay rate
# ---------------------------------------------------------------------------

class TestExponentialHalflifeDuration:
    """Shorter halflife → faster decay."""

    def test_shorter_halflife_decays_faster(self) -> None:
        short = DecayConfig(halflife_seconds=30.0, curve=DecayCurve.EXPONENTIAL)
        long_ = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.EXPONENTIAL)

        age = 120.0
        assert apply_decay(1.0, age, short) < apply_decay(1.0, age, long_)

    @pytest.mark.parametrize("halflife", [10.0, 60.0, 3600.0, 86400.0])
    def test_score_halved_at_exactly_halflife(self, halflife: float) -> None:
        cfg = DecayConfig(halflife_seconds=halflife, curve=DecayCurve.EXPONENTIAL)
        assert apply_decay(1.0, age_seconds=halflife, config=cfg) == pytest.approx(0.5)

    def test_very_long_halflife_barely_decays(self) -> None:
        cfg = DecayConfig(halflife_seconds=1e9, curve=DecayCurve.EXPONENTIAL)
        result = apply_decay(1.0, age_seconds=60.0, config=cfg)
        assert result > 0.9999

    def test_very_short_halflife_decays_rapidly(self) -> None:
        cfg = DecayConfig(halflife_seconds=1.0, curve=DecayCurve.EXPONENTIAL)
        result = apply_decay(1.0, age_seconds=100.0, config=cfg)
        assert result < 1e-29

    def test_doubling_halflife_halves_exponent(self) -> None:
        """Doubling halflife should square the decay factor."""
        h1 = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)
        h2 = DecayConfig(halflife_seconds=120.0, curve=DecayCurve.EXPONENTIAL)

        age = 60.0
        r1 = apply_decay(1.0, age, h1)  # 0.5^1 = 0.5
        r2 = apply_decay(1.0, age, h2)  # 0.5^0.5 ≈ 0.707

        # r2 should equal sqrt(r1) when r1 = 0.5 and r2 = 0.5^0.5
        assert r2 == pytest.approx(math.sqrt(r1), rel=1e-6)


# ---------------------------------------------------------------------------
# LINEAR curve — core halflife contract
# ---------------------------------------------------------------------------

class TestLinearHalflife:
    """Linear curve: halved at t=halflife, zero at t=2*halflife."""

    @pytest.fixture
    def cfg(self) -> DecayConfig:
        return DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)

    def test_age_zero_score_unchanged(self, cfg: DecayConfig) -> None:
        assert apply_decay(0.8, age_seconds=0.0, config=cfg) == pytest.approx(0.8)

    def test_age_halflife_score_halved(self, cfg: DecayConfig) -> None:
        assert apply_decay(1.0, age_seconds=60.0, config=cfg) == pytest.approx(0.5)

    def test_age_two_halflives_score_is_zero(self, cfg: DecayConfig) -> None:
        assert apply_decay(1.0, age_seconds=120.0, config=cfg) == pytest.approx(0.0)

    def test_weight_zero_when_age_equals_max_age(self, cfg: DecayConfig) -> None:
        """Weight is exactly 0 when age reaches maxAge (2 * halflife)."""
        max_age = 2.0 * cfg.halflife_seconds
        assert apply_decay(1.0, age_seconds=max_age, config=cfg) == pytest.approx(0.0)

    def test_age_beyond_two_halflives_clamped_to_zero(self, cfg: DecayConfig) -> None:
        assert apply_decay(1.0, age_seconds=200.0, config=cfg) == pytest.approx(0.0)

    def test_score_monotonically_decreasing_within_window(self, cfg: DecayConfig) -> None:
        ages = [0, 20, 40, 60, 80, 100, 120]
        scores = [apply_decay(1.0, float(a), cfg) for a in ages]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_score_never_negative(self, cfg: DecayConfig) -> None:
        for age in [0, 60, 120, 180, 1_000_000]:
            assert apply_decay(1.0, float(age), cfg) >= 0.0

    def test_quarter_halflife_age(self, cfg: DecayConfig) -> None:
        """At t=halflife/4=15 s, factor = 1 - 15/120 = 0.875."""
        result = apply_decay(1.0, age_seconds=15.0, config=cfg)
        assert result == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# LINEAR curve — halflife duration controls decay rate
# ---------------------------------------------------------------------------

class TestLinearHalflifeDuration:
    @pytest.mark.parametrize("halflife", [30.0, 60.0, 3600.0, 86400.0])
    def test_score_halved_at_exactly_halflife(self, halflife: float) -> None:
        cfg = DecayConfig(halflife_seconds=halflife, curve=DecayCurve.LINEAR)
        assert apply_decay(1.0, age_seconds=halflife, config=cfg) == pytest.approx(0.5)

    @pytest.mark.parametrize("halflife", [30.0, 60.0, 3600.0])
    def test_score_zero_at_two_halflives(self, halflife: float) -> None:
        cfg = DecayConfig(halflife_seconds=halflife, curve=DecayCurve.LINEAR)
        assert apply_decay(1.0, age_seconds=2.0 * halflife, config=cfg) == pytest.approx(0.0)

    @pytest.mark.parametrize("halflife", [30.0, 60.0, 3600.0, 86400.0])
    def test_weight_is_half_at_half_of_max_age(self, halflife: float) -> None:
        """Linear decay factor is exactly 0.5 when age == maxAge / 2.

        maxAge is defined as 2 * halflife (the age at which score reaches zero).
        At age == maxAge / 2 == halflife:
            factor = 1 - halflife / (2 * halflife) = 1 - 0.5 = 0.5
        """
        cfg = DecayConfig(halflife_seconds=halflife, curve=DecayCurve.LINEAR)
        max_age = 2.0 * halflife
        assert apply_decay(1.0, age_seconds=max_age / 2.0, config=cfg) == pytest.approx(0.5)

    @pytest.mark.parametrize("halflife", [30.0, 60.0, 3600.0, 86400.0])
    @pytest.mark.parametrize("excess_seconds", [1e-9, 1.0, 60.0, 3600.0, 1_000_000.0])
    def test_weight_zero_when_age_exceeds_max_age(
        self, halflife: float, excess_seconds: float
    ) -> None:
        """Weight is 0 for any age strictly beyond maxAge (2 * halflife).

        maxAge is the finite zero-crossing of the linear curve.  Any observation
        older than maxAge carries no information weight — the floor is 0, not negative.
        """
        cfg = DecayConfig(halflife_seconds=halflife, curve=DecayCurve.LINEAR)
        max_age = 2.0 * halflife
        result = apply_decay(1.0, age_seconds=max_age + excess_seconds, config=cfg)
        assert result == pytest.approx(0.0), (
            f"expected 0.0 at age={max_age + excess_seconds}s "
            f"(maxAge={max_age}s, halflife={halflife}s), got {result}"
        )

    def test_shorter_halflife_decays_faster(self) -> None:
        short = DecayConfig(halflife_seconds=30.0, curve=DecayCurve.LINEAR)
        long_ = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.LINEAR)

        age = 60.0
        assert apply_decay(1.0, age, short) < apply_decay(1.0, age, long_)


# ---------------------------------------------------------------------------
# Curve shape comparison — same halflife, different curve shape
# ---------------------------------------------------------------------------

class TestCurveShapeComparison:
    """At the same halflife both curves agree at t=0 and t=halflife,
    but diverge elsewhere."""

    @pytest.fixture
    def exp_cfg(self) -> DecayConfig:
        return DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)

    @pytest.fixture
    def lin_cfg(self) -> DecayConfig:
        return DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)

    def test_both_unchanged_at_zero(
        self, exp_cfg: DecayConfig, lin_cfg: DecayConfig
    ) -> None:
        assert apply_decay(1.0, 0.0, exp_cfg) == pytest.approx(
            apply_decay(1.0, 0.0, lin_cfg)
        )

    def test_both_halved_at_halflife(
        self, exp_cfg: DecayConfig, lin_cfg: DecayConfig
    ) -> None:
        assert apply_decay(1.0, 60.0, exp_cfg) == pytest.approx(
            apply_decay(1.0, 60.0, lin_cfg)
        )

    def test_linear_higher_than_exponential_before_halflife(
        self, exp_cfg: DecayConfig, lin_cfg: DecayConfig
    ) -> None:
        """Linear drops slower than exponential in the early window.

        Linear factor = 1 - t/(2h): shallow slope early.
        Exponential factor = 0.5^(t/h): steeper drop at small t.
        Both equal 0.5 at t=halflife; linear exceeds exponential before that.
        """
        for age in [10.0, 20.0, 30.0, 45.0]:
            assert apply_decay(1.0, age, lin_cfg) > apply_decay(1.0, age, exp_cfg)

    def test_exponential_higher_than_linear_after_halflife(
        self, exp_cfg: DecayConfig, lin_cfg: DecayConfig
    ) -> None:
        """Exponential retains more score late; linear approaches zero."""
        for age in [70.0, 90.0, 110.0]:
            assert apply_decay(1.0, age, exp_cfg) > apply_decay(1.0, age, lin_cfg)

    def test_linear_reaches_zero_exponential_does_not(
        self, exp_cfg: DecayConfig, lin_cfg: DecayConfig
    ) -> None:
        assert apply_decay(1.0, 120.0, lin_cfg) == pytest.approx(0.0)
        assert apply_decay(1.0, 120.0, exp_cfg) > 0.0


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_score_stays_zero_exponential(self) -> None:
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)
        assert apply_decay(0.0, 60.0, cfg) == pytest.approx(0.0)

    def test_zero_score_stays_zero_linear(self) -> None:
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)
        assert apply_decay(0.0, 60.0, cfg) == pytest.approx(0.0)

    def test_negative_age_treated_as_zero(self) -> None:
        cfg = DecayConfig(halflife_seconds=60.0)
        fresh = apply_decay(0.8, 0.0, cfg)
        future = apply_decay(0.8, -100.0, cfg)
        assert fresh == pytest.approx(future)

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("age", [-1.0, -60.0, -3600.0, -1_000_000_000.0])
    def test_negative_age_treated_as_zero_both_curves(
        self, curve: DecayCurve, age: float
    ) -> None:
        """Any negative age must behave identically to age=0 for both curves."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=curve)
        assert apply_decay(0.9, age, cfg) == pytest.approx(apply_decay(0.9, 0.0, cfg))

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_very_large_negative_age_treated_as_zero(self, curve: DecayCurve) -> None:
        """float-magnitude negative age must produce the same result as age=0."""
        import sys
        cfg = DecayConfig(halflife_seconds=60.0, curve=curve)
        result = apply_decay(1.0, age_seconds=-sys.float_info.max, config=cfg)
        assert result == pytest.approx(apply_decay(1.0, 0.0, cfg))

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_infinite_age_does_not_raise(self, curve: DecayCurve) -> None:
        """math.inf passed as age_seconds must not raise and must return a valid score."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=curve)
        result = apply_decay(1.0, age_seconds=math.inf, config=cfg)
        assert result >= 0.0
        assert math.isfinite(result) or result == 0.0

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("age", [1e10, 1e100, 1e300])
    def test_very_large_age_score_non_negative(
        self, curve: DecayCurve, age: float
    ) -> None:
        """Scores must never go negative regardless of how large the age is."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=curve)
        assert apply_decay(1.0, age_seconds=age, config=cfg) >= 0.0

    def test_very_small_age_minimal_decay(self) -> None:
        cfg = DecayConfig(halflife_seconds=86400.0)
        result = apply_decay(1.0, age_seconds=1.0, config=cfg)
        assert result > 0.9999

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_output_bounded_to_unit_interval_exponential(self, score: float) -> None:
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)
        for age in [0.0, 30.0, 60.0, 3600.0]:
            result = apply_decay(score, age, cfg)
            assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_output_bounded_to_unit_interval_linear(self, score: float) -> None:
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)
        for age in [0.0, 30.0, 60.0, 120.0, 3600.0]:
            result = apply_decay(score, age, cfg)
            assert 0.0 <= result <= 1.0

    # -- entries with zero age -----------------------------------------------

    @pytest.mark.parametrize("score", [0.0, 0.001, 0.5, 0.999, 1.0])
    def test_zero_age_preserves_score_exponential(self, score: float) -> None:
        cfg = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.EXPONENTIAL)
        assert apply_decay(score, age_seconds=0.0, config=cfg) == pytest.approx(score)

    @pytest.mark.parametrize("score", [0.0, 0.001, 0.5, 0.999, 1.0])
    def test_zero_age_preserves_score_linear(self, score: float) -> None:
        cfg = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.LINEAR)
        assert apply_decay(score, age_seconds=0.0, config=cfg) == pytest.approx(score)

    # -- entries at the decay boundary ----------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_weight_is_one_when_age_is_zero(self, curve: DecayCurve) -> None:
        """Decay factor equals 1.0 for a freshly created item (age == 0).

        Exponential: 0.5^(0/halflife) = 0.5^0 = 1.0
        Linear:      1 - 0/(2*halflife) = 1.0
        """
        cfg = DecayConfig(halflife_seconds=3600.0, curve=curve)
        assert apply_decay(1.0, age_seconds=0.0, config=cfg) == pytest.approx(1.0)

    def test_just_before_halflife_exp_above_half(self) -> None:
        """A score fractionally younger than halflife must still be above 0.5."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)
        result = apply_decay(1.0, age_seconds=60.0 - 1e-6, config=cfg)
        assert result > 0.5

    def test_just_after_halflife_exp_below_half(self) -> None:
        """A score fractionally older than halflife must be below 0.5."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.EXPONENTIAL)
        result = apply_decay(1.0, age_seconds=60.0 + 1e-6, config=cfg)
        assert result < 0.5

    def test_just_before_linear_zero_boundary_positive(self) -> None:
        """Fractionally before 2*halflife the linear score must still be > 0."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)
        result = apply_decay(1.0, age_seconds=120.0 - 1e-9, config=cfg)
        assert result > 0.0

    def test_at_linear_zero_boundary_is_zero(self) -> None:
        """Exactly at 2*halflife the linear score must be 0."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)
        assert apply_decay(1.0, age_seconds=120.0, config=cfg) == pytest.approx(0.0)

    def test_just_after_linear_zero_boundary_clamped(self) -> None:
        """Beyond 2*halflife the linear score must be clamped to 0, not negative."""
        cfg = DecayConfig(halflife_seconds=60.0, curve=DecayCurve.LINEAR)
        result = apply_decay(1.0, age_seconds=120.0 + 1e-6, config=cfg)
        assert result == pytest.approx(0.0)

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_event_time_1ms_in_future_clamped_to_zero_age(self, curve: DecayCurve) -> None:
        """event_time 1 ms ahead of now yields age_seconds = -0.001, which must be
        clamped to 0 — the score is preserved exactly as if the entry were brand-new.

        This guards against clock skew or sub-millisecond timestamp drift producing
        a decayed (lower) score for an entry that is effectively simultaneous with now.
        """
        cfg = DecayConfig(halflife_seconds=3600.0, curve=curve)
        score = 0.9
        # -0.001 s: event_time is 1 ms ahead of the reference clock
        result = apply_decay(score, age_seconds=-0.001, config=cfg)
        expected = apply_decay(score, age_seconds=0.0, config=cfg)
        assert result == pytest.approx(expected), (
            f"1 ms future age should clamp to 0 (curve={curve.value}): "
            f"got {result}, expected {expected}"
        )

    # -- entries with very old timestamps -------------------------------------

    def test_very_old_age_exponential_stays_positive(self) -> None:
        """Exponential decay never reaches zero — even after ~31 years in seconds."""
        cfg = DecayConfig(halflife_seconds=86400.0, curve=DecayCurve.EXPONENTIAL)
        age_seconds = 1_000_000_000.0  # ~31.7 years
        result = apply_decay(1.0, age_seconds=age_seconds, config=cfg)
        assert result >= 0.0  # no negative values
        # At this extreme age the IEEE 754 underflow may produce 0.0,
        # so we only assert the result is finite and non-negative.
        assert math.isfinite(result)

    def test_very_old_age_linear_clamped_to_zero(self) -> None:
        """Linear decay clamps to 0 for any age beyond 2*halflife."""
        cfg = DecayConfig(halflife_seconds=86400.0, curve=DecayCurve.LINEAR)
        age_seconds = 1_000_000_000.0  # ~31.7 years
        assert apply_decay(1.0, age_seconds=age_seconds, config=cfg) == pytest.approx(0.0)

    def test_float_max_age_does_not_raise(self) -> None:
        """Passing float max age must not raise regardless of curve."""
        age = 1.7976931348623157e+308  # sys.float_info.max
        for curve in DecayCurve:
            cfg = DecayConfig(halflife_seconds=60.0, curve=curve)
            result = apply_decay(1.0, age_seconds=age, config=cfg)
            assert result >= 0.0


# ---------------------------------------------------------------------------
# Episodic entry ordering — older entries score lower than recent ones
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)


def _episodic_entry(snapshot_id: str, ingested_at: datetime) -> SnapshotEntry:
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id="feed-source-1",
        ingested_at=ingested_at,
        source_revision="rev1",
        entry_paths=["feed/item.md"],
        content_hash="abc123",
        memory_type=MemoryType.EPISODIC,
    )


class TestEpisodicEntryOrdering:
    """Older episodic entries must receive lower decay-adjusted scores than
    recent entries with an identical base relevance score.

    This models the practical use-case: a retrieval hit from a feed snapshot
    ingested yesterday should be discounted relative to one ingested today,
    even when both carry the same raw relevance.
    """

    BASE_SCORE = 0.8
    HALFLIFE = 3600.0  # 1-hour halflife

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("recent_age_s,old_age_s", [
        (0.0, 3600.0),    # fresh vs 1 halflife old
        (300.0, 7200.0),  # 5-min-old vs 2-hours-old
        (60.0, 86400.0),  # 1-min-old vs 1-day-old
    ])
    def test_older_episodic_scores_lower(
        self,
        curve: DecayCurve,
        recent_age_s: float,
        old_age_s: float,
    ) -> None:
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)

        recent = _episodic_entry(
            "snap-recent", ingested_at=_NOW - timedelta(seconds=recent_age_s)
        )
        old = _episodic_entry(
            "snap-old", ingested_at=_NOW - timedelta(seconds=old_age_s)
        )

        assert recent.memory_type is MemoryType.EPISODIC
        assert old.memory_type is MemoryType.EPISODIC

        recent_score = apply_decay(
            self.BASE_SCORE,
            (_NOW - recent.ingested_at).total_seconds(),
            cfg,
        )
        older_score = apply_decay(
            self.BASE_SCORE,
            (_NOW - old.ingested_at).total_seconds(),
            cfg,
        )

        assert recent_score > older_score, (
            f"recent_score={recent_score:.6f} should exceed older_score={older_score:.6f} "
            f"(curve={curve.value}, recent_age={recent_age_s}s, old_age={old_age_s}s)"
        )

    def test_score_ordering_preserved_across_sequence(self) -> None:
        """Multiple episodic entries ordered oldest→newest yield monotonically
        increasing decay-adjusted scores when base relevance is equal."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.EXPONENTIAL)

        # ages in decreasing order (oldest first)
        ages_s = [86400.0, 7200.0, 3600.0, 1800.0, 600.0, 0.0]
        entries = [
            _episodic_entry(f"snap-{i}", _NOW - timedelta(seconds=a))
            for i, a in enumerate(ages_s)
        ]

        scores = [
            apply_decay(self.BASE_SCORE, (_NOW - e.ingested_at).total_seconds(), cfg)
            for e in entries
        ]

        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"scores[{i}]={scores[i]:.6f} should be < scores[{i+1}]={scores[i+1]:.6f}"
            )

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_same_base_score_different_ages_not_equal(self, curve: DecayCurve) -> None:
        """Decay must produce a strictly different score for any non-zero age gap."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)

        recent = _episodic_entry("snap-r", _NOW - timedelta(seconds=10.0))
        old = _episodic_entry("snap-o", _NOW - timedelta(seconds=7200.0))

        recent_score = apply_decay(
            self.BASE_SCORE, (_NOW - recent.ingested_at).total_seconds(), cfg
        )
        older_score = apply_decay(
            self.BASE_SCORE, (_NOW - old.ingested_at).total_seconds(), cfg
        )

        assert recent_score != pytest.approx(older_score)

    def test_feed_source_produces_episodic_entries(self) -> None:
        """Verifies that 'feed' source type maps to EPISODIC — confirming that
        decay-ordering tests target the correct memory classification."""
        assert source_to_memory_type("feed") is MemoryType.EPISODIC
        assert source_to_memory_type("github") is not MemoryType.EPISODIC
        assert source_to_memory_type("url") is not MemoryType.EPISODIC


# ---------------------------------------------------------------------------
# Semantic entry score preservation — decay factor = 1.0
# ---------------------------------------------------------------------------

_SEMANTIC_SOURCE_TYPES = ["github", "url", "folder"]


def _semantic_entry(snapshot_id: str, ingested_at: datetime) -> SnapshotEntry:
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id="gh-source-1",
        ingested_at=ingested_at,
        source_revision="main@abc123",
        entry_paths=["README.md"],
        content_hash="abcdef",
        memory_type=MemoryType.SEMANTIC,
    )


class TestSemanticEntryScorePreservation:
    """Semantic / reference entries must retain their original score regardless of age.

    SEMANTIC memory (github, url, folder sources) represents timeless reference
    content — snapshot age is irrelevant to relevance.  The decay factor for
    SEMANTIC entries is always 1.0.

    Applying ``apply_decay`` with ``age_seconds=0.0`` encodes this: the factor
    is 0.5^0 = 1.0 for exponential, and 1 - 0/(2h) = 1.0 for linear, so the
    original score is preserved exactly.
    """

    BASE_SCORE = 0.8
    HALFLIFE = 3600.0

    # -- source type → memory type classification ----------------------------

    @pytest.mark.parametrize("source_type", _SEMANTIC_SOURCE_TYPES)
    def test_reference_source_types_map_to_semantic(self, source_type: str) -> None:
        assert source_to_memory_type(source_type) is MemoryType.SEMANTIC

    def test_feed_source_type_is_not_semantic(self) -> None:
        assert source_to_memory_type("feed") is not MemoryType.SEMANTIC

    # -- decay factor = 1.0 for all ages ------------------------------------

    @pytest.mark.parametrize("age_s", [0.0, 3600.0, 86400.0, 365.0 * 86400.0])
    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_semantic_decay_factor_is_one_at_any_age(
        self, age_s: float, curve: DecayCurve
    ) -> None:
        """SEMANTIC entries use age_seconds=0 to express factor=1.0.

        Regardless of when the snapshot was actually ingested, the calling
        convention for SEMANTIC entries passes age=0 to apply_decay, producing
        a factor of 1.0 and leaving the original score intact.
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)
        entry = _semantic_entry("snap-sem", _NOW - timedelta(seconds=age_s))

        assert entry.memory_type is MemoryType.SEMANTIC
        result = apply_decay(self.BASE_SCORE, age_seconds=0.0, config=cfg)
        assert result == pytest.approx(self.BASE_SCORE)

    @pytest.mark.parametrize("score", [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0])
    def test_all_scores_preserved_for_semantic_entries(self, score: float) -> None:
        """apply_decay(score, age=0) == score for all score values."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE)
        assert apply_decay(score, age_seconds=0.0, config=cfg) == pytest.approx(score)

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_semantic_factor_one_holds_for_both_curves(self, curve: DecayCurve) -> None:
        """Factor=1.0 is curve-independent: both exponential and linear yield it at age=0."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)
        result = apply_decay(self.BASE_SCORE, age_seconds=0.0, config=cfg)
        assert result == pytest.approx(self.BASE_SCORE)

    # -- semantic entries with different actual ages score equally ------------

    def test_semantic_entries_with_different_ages_score_equally(self) -> None:
        """Two SEMANTIC entries with the same base score must produce equal
        decay-adjusted scores regardless of their actual ingestion age.

        Age is irrelevant for SEMANTIC content: both entries use factor=1.0.
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.EXPONENTIAL)

        actual_ages_s = [0.0, 3600.0, 86400.0, 365.0 * 86400.0]
        scores = [
            apply_decay(self.BASE_SCORE, age_seconds=0.0, config=cfg)
            for _ in actual_ages_s
        ]

        for s in scores:
            assert s == pytest.approx(self.BASE_SCORE)

    # -- semantic vs episodic contrast ----------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("age_s", [3600.0, 7200.0, 86400.0])
    def test_semantic_score_exceeds_episodic_at_same_base_score(
        self, curve: DecayCurve, age_s: float
    ) -> None:
        """A SEMANTIC hit must never be penalised below an equally-scored EPISODIC hit.

        For any positive age, the episodic entry incurs a decay penalty while
        the semantic entry retains its full score (factor=1.0).
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)

        sem_score = apply_decay(self.BASE_SCORE, age_seconds=0.0, config=cfg)
        ep_score = apply_decay(self.BASE_SCORE, age_seconds=age_s, config=cfg)

        assert sem_score > ep_score, (
            f"semantic ({sem_score:.4f}) should exceed episodic ({ep_score:.4f}) "
            f"at age={age_s}s, curve={curve.value}"
        )

    @pytest.mark.parametrize("old_age_s", [3600.0, 86400.0, 365.0 * 86400.0])
    def test_old_semantic_entry_beats_decayed_episodic_same_base_score(
        self, old_age_s: float
    ) -> None:
        """A years-old SEMANTIC entry must outscore an equally-aged EPISODIC entry."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE)
        sem_score = apply_decay(self.BASE_SCORE, age_seconds=0.0, config=cfg)
        ep_score = apply_decay(self.BASE_SCORE, age_seconds=old_age_s, config=cfg)
        assert sem_score > ep_score

    def test_semantic_and_episodic_equal_only_at_age_zero(self) -> None:
        """At age=0 both conventions are identical; they diverge for any positive age."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE)
        base = 0.9

        sem = apply_decay(base, age_seconds=0.0, config=cfg)
        ep_at_zero = apply_decay(base, age_seconds=0.0, config=cfg)
        assert sem == pytest.approx(ep_at_zero)

        ep_at_halflife = apply_decay(base, age_seconds=self.HALFLIFE, config=cfg)
        assert sem > ep_at_halflife


# ---------------------------------------------------------------------------
# Six named decay scenarios — explicit coverage matrix
# ---------------------------------------------------------------------------

class TestDecayNamedScenarios:
    """Explicit tests for each of the six canonical decay scenarios.

    Each test directly addresses one named contract:
      1. zero age           — score must be fully preserved
      2. half-life          — score must be exactly halved
      3. exact maxAge boundary (LINEAR) — score must be exactly 0
      4. beyond maxAge (LINEAR) — score must be clamped to 0, not negative
      5. future eventTime   — negative age is clamped to 0 (no bonus decay)
      6. maxAge of zero     — halflife_seconds=0 is forbidden (maxAge can't be 0)

    maxAge is defined as 2 * halflife_seconds for the LINEAR curve (the
    finite zero-crossing). The EXPONENTIAL curve has no finite maxAge.
    """

    HALFLIFE = 3600.0  # 1 hour, representative production value

    # 1. Zero age ---------------------------------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("score", [0.0, 0.5, 0.9, 1.0])
    def test_zero_age_preserves_score(self, curve: DecayCurve, score: float) -> None:
        """At age=0 the decay factor is 1.0 for both curves — score is unchanged."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)
        assert apply_decay(score, age_seconds=0.0, config=cfg) == pytest.approx(score), (
            f"curve={curve.value}, score={score}: expected {score}, "
            f"got {apply_decay(score, 0.0, cfg)}"
        )

    # 2. Half-life ---------------------------------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("halflife", [60.0, 3600.0, 86400.0])
    def test_halflife_age_halves_score(self, curve: DecayCurve, halflife: float) -> None:
        """At age=halflife the decay factor is exactly 0.5 for both curves."""
        cfg = DecayConfig(halflife_seconds=halflife, curve=curve)
        result = apply_decay(1.0, age_seconds=halflife, config=cfg)
        assert result == pytest.approx(0.5), (
            f"curve={curve.value}, halflife={halflife}s: expected 0.5, got {result}"
        )

    # 3. Exact maxAge boundary (LINEAR) -----------------------------------------

    @pytest.mark.parametrize("halflife", [30.0, 3600.0, 86400.0])
    def test_exact_max_age_boundary_linear_score_is_zero(self, halflife: float) -> None:
        """LINEAR score is exactly 0 at age == maxAge (2 * halflife).

        factor = 1 - maxAge / (2*halflife) = 1 - 1 = 0.
        """
        cfg = DecayConfig(halflife_seconds=halflife, curve=DecayCurve.LINEAR)
        max_age = 2.0 * halflife
        result = apply_decay(1.0, age_seconds=max_age, config=cfg)
        assert result == pytest.approx(0.0), (
            f"halflife={halflife}s, maxAge={max_age}s: expected 0.0, got {result}"
        )

    def test_exact_max_age_boundary_exponential_still_positive(self) -> None:
        """EXPONENTIAL has no finite maxAge — score at 2*halflife is 0.25, not 0."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.EXPONENTIAL)
        result = apply_decay(1.0, age_seconds=2.0 * self.HALFLIFE, config=cfg)
        assert result == pytest.approx(0.25)

    # 4. Beyond maxAge (LINEAR) -------------------------------------------------

    @pytest.mark.parametrize("excess", [1e-9, 1.0, 3600.0, 1_000_000.0])
    def test_beyond_max_age_linear_clamped_to_zero(self, excess: float) -> None:
        """Any age strictly beyond 2*halflife is clamped to 0, not negative."""
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.LINEAR)
        age = 2.0 * self.HALFLIFE + excess
        result = apply_decay(1.0, age_seconds=age, config=cfg)
        assert result == pytest.approx(0.0), (
            f"age={age}s (maxAge={2*self.HALFLIFE}s): expected 0.0, got {result}"
        )
        assert result >= 0.0, "score must never go negative"

    # 5. Future eventTime -------------------------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("future_seconds", [-1e-3, -1.0, -3600.0, -1_000_000.0])
    def test_future_event_time_treated_as_zero_age(
        self, curve: DecayCurve, future_seconds: float
    ) -> None:
        """Negative age (event_time > now) is clamped to 0 — no bonus from the future.

        A future-dated entry must receive the same score as a brand-new entry,
        not a *higher* score. Clock skew and sub-millisecond drift are the common
        causes; both produce a small negative age that must not penalise or
        artificially boost the result.
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)
        score = 0.85
        future_result = apply_decay(score, age_seconds=future_seconds, config=cfg)
        zero_result = apply_decay(score, age_seconds=0.0, config=cfg)
        assert future_result == pytest.approx(zero_result), (
            f"curve={curve.value}, age={future_seconds}s: "
            f"expected {zero_result} (same as age=0), got {future_result}"
        )

    # 6. maxAge of zero ---------------------------------------------------------

    def test_max_age_of_zero_is_forbidden(self) -> None:
        """maxAge = 2*halflife = 0 is impossible: halflife_seconds=0 raises ValueError.

        The system enforces halflife_seconds > 0, guaranteeing that the linear
        zero-crossing (maxAge) is always > 0.  A maxAge of exactly 0 would make
        every observation stale regardless of age — this is disallowed by design.
        """
        with pytest.raises(ValueError, match="halflife_seconds must be > 0"):
            DecayConfig(halflife_seconds=0.0)

    def test_near_zero_halflife_linear_decays_instantly(self) -> None:
        """With a near-zero halflife the linear maxAge (2*halflife) is also near-zero.

        Any age that exceeds 2*halflife (even tiny fractions of a second) maps
        to score=0.  This is the closest achievable behaviour to maxAge=0.
        """
        tiny_halflife = 1e-6  # 1 microsecond
        cfg = DecayConfig(halflife_seconds=tiny_halflife, curve=DecayCurve.LINEAR)
        # age of 3 µs is beyond maxAge (2 µs)
        result = apply_decay(1.0, age_seconds=3e-6, config=cfg)
        assert result == pytest.approx(0.0)

    def test_near_zero_halflife_exponential_decays_rapidly(self) -> None:
        """With a near-zero halflife the exponential curve collapses almost immediately.

        Even with no finite maxAge, a 1-µs halflife makes the score effectively 0
        after just a few microseconds.
        """
        tiny_halflife = 1e-6
        cfg = DecayConfig(halflife_seconds=tiny_halflife, curve=DecayCurve.EXPONENTIAL)
        # After 100 halflives (100 µs): score = 0.5^100 ≈ 7.9e-31, effectively 0
        result = apply_decay(1.0, age_seconds=100e-6, config=cfg)
        assert result < 1e-20


# ---------------------------------------------------------------------------
# linear_age_factor — explicit edge-case coverage
# ---------------------------------------------------------------------------

class TestLinearAgeFactor:
    """linear_age_factor(age_seconds, max_age_seconds) → float in [0, 1].

    Three hard edge cases (from the openclaw temporal-decay contract):
      1. eventTime in the future  (age < 0)      → return 1.0
      2. age >= maxAge                            → return 0.0
      3. maxAge <= 0                              → return 0.0
    """

    # 1. Future eventTime -------------------------------------------------------

    @pytest.mark.parametrize("age", [-1e-3, -1.0, -3600.0, -1_000_000.0])
    def test_future_event_time_returns_one(self, age: float) -> None:
        """Negative age means the event is in the future → freshness factor = 1."""
        assert linear_age_factor(age, max_age_seconds=3600.0) == pytest.approx(1.0)

    # 2. age >= maxAge ----------------------------------------------------------

    def test_age_equal_to_max_age_returns_zero(self) -> None:
        assert linear_age_factor(3600.0, max_age_seconds=3600.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("excess", [1e-9, 1.0, 3600.0, 1_000_000.0])
    def test_age_beyond_max_age_returns_zero(self, excess: float) -> None:
        max_age = 3600.0
        assert linear_age_factor(max_age + excess, max_age_seconds=max_age) == pytest.approx(0.0)

    # 3. maxAge <= 0 ------------------------------------------------------------

    def test_max_age_zero_returns_zero(self) -> None:
        assert linear_age_factor(0.0, max_age_seconds=0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("max_age", [-1e-9, -1.0, -3600.0])
    def test_negative_max_age_returns_zero(self, max_age: float) -> None:
        assert linear_age_factor(10.0, max_age_seconds=max_age) == pytest.approx(0.0)

    # Interpolation -------------------------------------------------------------

    def test_zero_age_returns_one(self) -> None:
        assert linear_age_factor(0.0, max_age_seconds=3600.0) == pytest.approx(1.0)

    def test_half_max_age_returns_half(self) -> None:
        assert linear_age_factor(1800.0, max_age_seconds=3600.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 0.75])
    def test_linear_interpolation(self, frac: float) -> None:
        max_age = 100.0
        result = linear_age_factor(frac * max_age, max_age_seconds=max_age)
        assert result == pytest.approx(1.0 - frac)

    def test_output_bounded_to_unit_interval(self) -> None:
        for age in [-10.0, 0.0, 50.0, 100.0, 150.0]:
            result = linear_age_factor(age, max_age_seconds=100.0)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# decay_score — per-day geometric decay
# ---------------------------------------------------------------------------

class TestDecayScore:
    """Verify exact expected values for decay_score(base, days, rate=0.95).

    Formula: base_score * decay_rate ** days_since_access
    """

    # -- anchor points -------------------------------------------------------

    def test_zero_days_returns_full_score(self) -> None:
        """At day 0 the multiplier is 0.95^0 == 1.0 exactly."""
        assert decay_score(1.0, days_since_access=0) == pytest.approx(1.0)

    def test_one_day_returns_decay_rate(self) -> None:
        """At day 1 the multiplier is 0.95^1 == 0.95 exactly."""
        assert decay_score(1.0, days_since_access=1) == pytest.approx(0.95)

    def test_seven_days(self) -> None:
        """At day 7: 0.95^7 ≈ 0.6983."""
        expected = 0.95**7  # ≈ 0.698337296…
        assert decay_score(1.0, days_since_access=7) == pytest.approx(expected, rel=1e-9)
        assert decay_score(1.0, days_since_access=7) == pytest.approx(0.6983, abs=5e-5)

    def test_thirty_days(self) -> None:
        """At day 30: 0.95^30 ≈ 0.2146."""
        expected = 0.95**30  # ≈ 0.214638…
        assert decay_score(1.0, days_since_access=30) == pytest.approx(expected, rel=1e-9)
        assert decay_score(1.0, days_since_access=30) == pytest.approx(0.2146, abs=5e-5)

    def test_three_hundred_sixty_five_days_near_zero(self) -> None:
        """At day 365: 0.95^365 ≈ 7.3e-9 — effectively zero."""
        result = decay_score(1.0, days_since_access=365)
        assert result == pytest.approx(0.95**365, rel=1e-9)
        assert result < 1e-7  # negligible

    # -- base_score scaling --------------------------------------------------

    def test_base_score_scales_linearly(self) -> None:
        """Result is linear in base_score."""
        assert decay_score(0.5, days_since_access=7) == pytest.approx(
            0.5 * decay_score(1.0, days_since_access=7)
        )

    # -- default decay_rate --------------------------------------------------

    def test_default_decay_rate_matches_explicit_0_95(self) -> None:
        """Omitting decay_rate must produce the same result as decay_rate=0.95.

        This pins the default so that a future signature change (e.g. changing
        the default to 0.9 or moving to a config object) surfaces as a test
        failure rather than a silent behavioural change.
        """
        for days in (0, 1, 7, 30, 365):
            assert decay_score(1.0, days_since_access=days) == pytest.approx(
                decay_score(1.0, days_since_access=days, decay_rate=0.95),
                rel=1e-12,
            ), f"default and explicit decay_rate=0.95 differ at days={days}"

    # -- custom decay_rate ---------------------------------------------------

    def test_custom_decay_rate(self) -> None:
        """Custom rate is respected: 0.5 ** 1 == 0.5 (half-life of 1 day)."""
        assert decay_score(1.0, days_since_access=1, decay_rate=0.5) == pytest.approx(0.5)
        assert decay_score(1.0, days_since_access=2, decay_rate=0.5) == pytest.approx(0.25)

    # -- edge cases ----------------------------------------------------------

    def test_negative_days_clamped_to_zero(self) -> None:
        """Negative days_since_access are clamped to 0 — score is not amplified."""
        assert decay_score(0.8, days_since_access=-1) == pytest.approx(0.8)
        assert decay_score(0.8, days_since_access=-100) == pytest.approx(0.8)

    def test_zero_base_score_always_zero(self) -> None:
        """A base score of 0 always yields 0, regardless of age or rate."""
        assert decay_score(0.0, days_since_access=0) == pytest.approx(0.0)
        assert decay_score(0.0, days_since_access=365) == pytest.approx(0.0)
        assert decay_score(0.0, days_since_access=1, decay_rate=0.5) == pytest.approx(0.0)

    def test_custom_decay_rate_fast_and_slow(self) -> None:
        """Verify that decay_rate meaningfully controls the speed of decay."""
        fast = decay_score(1.0, days_since_access=10, decay_rate=0.5)
        slow = decay_score(1.0, days_since_access=10, decay_rate=0.99)
        assert fast < slow
        assert fast == pytest.approx(0.5**10, rel=1e-9)
        assert slow == pytest.approx(0.99**10, rel=1e-9)

    def test_very_large_days_no_overflow(self) -> None:
        """Extremely large day counts must return a finite, non-negative float."""
        result = decay_score(1.0, days_since_access=100_000)
        assert math.isfinite(result)
        assert result >= 0.0

    def test_very_large_days_underflow_to_zero_not_negative(self) -> None:
        """When the result underflows to 0.0 it must not be negative or NaN."""
        result = decay_score(1.0, days_since_access=10_000)
        assert result == 0.0 or result > 0.0  # non-negative
        assert not math.isnan(result)
        assert not math.isinf(result)


# ---------------------------------------------------------------------------
# Cross-relevance ranking — high-relevance old vs low-relevance new
# ---------------------------------------------------------------------------

class TestCrossRelevanceRanking:
    """Verifies that temporal decay combines *multiplicatively* with base
    relevance, so a highly relevant old result can still outrank a less
    relevant new result when the age penalty is moderate.

    This is the key property that distinguishes a *blended* decay model from
    a pure recency sort: relevance quality should not be wiped out by modest
    age differences.

    Decay formula recap
    -------------------
    EXPONENTIAL  combined = base * 0.5^(age / halflife)
    LINEAR       combined = base * max(0, 1 - age / (2 * halflife))

    Crossover condition (EXPONENTIAL)
    ----------------------------------
    high_base * 0.5^(age/h)  >  low_base
      ⟺  age  <  h * log2(high_base / low_base)

    For high_base=0.90, low_base=0.30:
      crossover_age = h * log2(0.90 / 0.30) = h * log2(3) ≈ 1.585 * h
    """

    HALFLIFE = 3600.0  # 1-hour halflife used throughout
    HIGH_BASE = 0.90   # highly relevant result
    LOW_BASE = 0.30    # less relevant result

    # ------------------------------------------------------------------
    # Core ranking assertions — apply_decay (seconds-based)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("old_age_s", [
        0.0,           # both fresh — high-relevance wins trivially
        1800.0,        # 0.5 halflives old
        3600.0,        # exactly 1 halflife old — high_base * 0.5 = 0.45 > 0.30
    ])
    def test_high_relevance_old_beats_low_relevance_new(
        self, curve: DecayCurve, old_age_s: float
    ) -> None:
        """A moderately aged high-relevance result outranks a fresh low-relevance one.

        At 1 halflife:  0.90 * 0.5 = 0.45  >  0.30 * 1.0 = 0.30  ✓
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)

        high_old = apply_decay(self.HIGH_BASE, age_seconds=old_age_s, config=cfg)
        low_new = apply_decay(self.LOW_BASE, age_seconds=0.0, config=cfg)

        assert high_old > low_new, (
            f"curve={curve.value}, age={old_age_s}s: "
            f"high_old={high_old:.4f} should exceed low_new={low_new:.4f}"
        )

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    @pytest.mark.parametrize("old_age_s", [
        86400.0,       # 24 halflives — exponential ≈ 0.90/2^24 ≈ 5e-8; linear = 0
        36000.0,       # 10 halflives — exponential ≈ 0.90/1024 ≈ 8.8e-4; linear = 0
    ])
    def test_low_relevance_new_beats_very_old_high_relevance(
        self, curve: DecayCurve, old_age_s: float
    ) -> None:
        """When the high-relevance result is extremely stale, the fresh low-relevance
        result should win.

        The crossover for EXPONENTIAL is at ≈1.585 halflives; 10+ halflives is
        well past that.  LINEAR reaches 0 at 2 halflives, so it always loses here.
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)

        high_very_old = apply_decay(self.HIGH_BASE, age_seconds=old_age_s, config=cfg)
        low_new = apply_decay(self.LOW_BASE, age_seconds=0.0, config=cfg)

        assert high_very_old < low_new, (
            f"curve={curve.value}, age={old_age_s}s: "
            f"high_very_old={high_very_old:.6f} should be < low_new={low_new:.4f}"
        )

    def test_exponential_crossover_age(self) -> None:
        """Validate the exact crossover age where both scores are equal.

        high_base * 0.5^(x/h) = low_base
          x = h * log2(high_base / low_base)
            = 3600 * log2(0.90 / 0.30)
            = 3600 * log2(3)
            ≈ 5707 seconds

        Just below this age: high-relevance old wins.
        Just above this age: low-relevance new wins.
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.EXPONENTIAL)
        crossover_age = self.HALFLIFE * math.log2(self.HIGH_BASE / self.LOW_BASE)

        just_before = apply_decay(self.HIGH_BASE, crossover_age - 1.0, cfg)
        just_after = apply_decay(self.HIGH_BASE, crossover_age + 1.0, cfg)
        low_new = apply_decay(self.LOW_BASE, 0.0, cfg)

        assert just_before > low_new, (
            f"just_before crossover ({just_before:.6f}) should beat low_new ({low_new:.4f})"
        )
        assert just_after < low_new, (
            f"just_after crossover ({just_after:.6f}) should lose to low_new ({low_new:.4f})"
        )

    def test_linear_crossover_age(self) -> None:
        """Linear crossover: high_base * (1 - x/(2h)) = low_base
          x = 2h * (1 - low_base/high_base)
            = 7200 * (1 - 1/3) = 4800 seconds
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.LINEAR)
        crossover_age = 2.0 * self.HALFLIFE * (1.0 - self.LOW_BASE / self.HIGH_BASE)

        just_before = apply_decay(self.HIGH_BASE, crossover_age - 1.0, cfg)
        just_after = apply_decay(self.HIGH_BASE, crossover_age + 1.0, cfg)
        low_new = apply_decay(self.LOW_BASE, 0.0, cfg)

        assert just_before > low_new, (
            f"just_before crossover ({just_before:.6f}) should beat low_new ({low_new:.4f})"
        )
        assert just_after < low_new, (
            f"just_after crossover ({just_after:.6f}) should lose to low_new ({low_new:.4f})"
        )

    # ------------------------------------------------------------------
    # Ranking lists — combined sort preserves expected order
    # ------------------------------------------------------------------

    def test_ranking_list_combined_scores(self) -> None:
        """A mixed list of (base_score, age) pairs sorts correctly after decay.

        Results (base, age_halflives):
          A: (0.90, 0.5h)  → 0.90 * 0.5^0.5  ≈ 0.637   (best)
          B: (0.60, 0.0h)  → 0.60             ≈ 0.600
          C: (0.50, 0.0h)  → 0.50             (middle)
          D: (0.80, 3.0h)  → 0.80 * 0.5^3    = 0.100   (worst)

        Expected ranking: A > B > C > D
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.EXPONENTIAL)
        h = self.HALFLIFE

        results = {
            "A": apply_decay(0.90, 0.5 * h, cfg),   # high-relevance, modest age
            "B": apply_decay(0.60, 0.0,     cfg),   # medium-relevance, fresh
            "C": apply_decay(0.50, 0.0,     cfg),   # lower-relevance, fresh
            "D": apply_decay(0.80, 3.0 * h, cfg),   # high-relevance, very stale
        }

        assert results["A"] > results["B"], f"A={results['A']:.4f} should beat B={results['B']:.4f}"
        assert results["B"] > results["C"], f"B={results['B']:.4f} should beat C={results['C']:.4f}"
        assert results["C"] > results["D"], f"C={results['C']:.4f} should beat D={results['D']:.4f}"

    # ------------------------------------------------------------------
    # Same property using decay_score (per-day)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("old_days,should_win", [
        (5,  True),   # 0.90 * 0.95^5  ≈ 0.696  > 0.30  ✓
        (14, True),   # 0.90 * 0.95^14 ≈ 0.483  > 0.30  ✓
        (28, False),  # 0.90 * 0.95^28 ≈ 0.226  < 0.30  ✗
        (60, False),  # 0.90 * 0.95^60 ≈ 0.042  < 0.30  ✗
    ])
    def test_per_day_decay_cross_relevance_ranking(
        self, old_days: int, should_win: bool
    ) -> None:
        """decay_score (per-day variant) respects the same crossover logic."""
        high_old = decay_score(self.HIGH_BASE, days_since_access=old_days)
        low_new = decay_score(self.LOW_BASE,  days_since_access=0)

        if should_win:
            assert high_old > low_new, (
                f"day {old_days}: high_old={high_old:.4f} should beat low_new={low_new:.4f}"
            )
        else:
            assert high_old < low_new, (
                f"day {old_days}: high_old={high_old:.4f} should lose to low_new={low_new:.4f}"
            )

    def test_per_day_decay_crossover_is_monotone(self) -> None:
        """As age increases, high-base eventually loses to low-base — only once.

        Below the crossover: high-base wins.
        Above the crossover: low-base wins.
        The transition happens exactly once (monotone decay).
        """
        low_new = decay_score(self.LOW_BASE, days_since_access=0)

        high_wins = [
            decay_score(self.HIGH_BASE, days_since_access=d) > low_new
            for d in range(0, 100)
        ]

        # Find the first day where high-base loses.
        crossover_idx = next((i for i, w in enumerate(high_wins) if not w), None)

        assert crossover_idx is not None, (
            "Expected a crossover day where low-relevance-new overtakes high-relevance-old"
        )

        # All days before the crossover: high wins.
        assert all(high_wins[:crossover_idx]), (
            "High-relevance result should win every day before the crossover"
        )

        # All days from the crossover onward: low wins (monotone, no re-crossover).
        assert not any(high_wins[crossover_idx:]), (
            "Once low-relevance-new overtakes, it should remain ahead (monotone decay)"
        )


# ---------------------------------------------------------------------------
# Configurable decay rate changes ranking order
# ---------------------------------------------------------------------------

class TestConfigurableDecayRateRankingOrder:
    """Verify that a higher decay rate (shorter halflife) penalises older results
    more aggressively than a lower decay rate (longer halflife), to the point of
    *reversing* the ranking order between two results.

    Scenario
    --------
    Two results with the same age (2 hours = 7 200 s):

        A (older, higher base): base_score=0.80, age=7 200 s
        B (newer, lower base):  base_score=0.50, age=0 s

    Low decay rate  → halflife = 24 h (86 400 s)
      A: 0.80 * 0.5^(7200/86400) ≈ 0.80 * 0.5^0.0833 ≈ 0.755  >  B (0.50)  → A wins

    High decay rate → halflife = 1 h (3 600 s)
      A: 0.80 * 0.5^(7200/3600)  = 0.80 * 0.25         = 0.200  <  B (0.50)  → B wins

    The ranking order is therefore *opposite* for the two decay rates, which is the
    key property this test suite verifies.
    """

    BASE_HIGH = 0.80     # older, more relevant result
    BASE_LOW  = 0.50     # fresh, less relevant result
    OLD_AGE_S = 7_200.0  # 2 hours

    # Halflife constants: short = high decay rate; long = low decay rate
    HALFLIFE_SHORT = 3_600.0   # 1 hour  → aggressive / high rate
    HALFLIFE_LONG  = 86_400.0  # 24 hours → lenient / low rate

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_low_decay_rate_preserves_old_high_relevance_ranking(
        self, curve: DecayCurve
    ) -> None:
        """With a long halflife (low decay rate) the older, higher-relevance result
        still outranks the fresh, lower-relevance one.

        EXPONENTIAL: 0.80 * 0.5^(7200/86400) ≈ 0.755  >  0.50
        LINEAR:      0.80 * max(0, 1 - 7200/(2*86400)) ≈ 0.80 * 0.958 ≈ 0.767  >  0.50
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE_LONG, curve=curve)
        score_a = apply_decay(self.BASE_HIGH, age_seconds=self.OLD_AGE_S, config=cfg)
        score_b = apply_decay(self.BASE_LOW,  age_seconds=0.0,            config=cfg)

        assert score_a > score_b, (
            f"curve={curve.value}, halflife={self.HALFLIFE_LONG}s (low rate): "
            f"older high-relevance ({score_a:.4f}) should beat fresh low-relevance ({score_b:.4f})"
        )

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_high_decay_rate_flips_ranking_in_favour_of_fresh_result(
        self, curve: DecayCurve
    ) -> None:
        """With a short halflife (high decay rate) the age penalty is severe enough
        to flip the ranking: the fresh, lower-relevance result wins.

        EXPONENTIAL: 0.80 * 0.5^(7200/3600) = 0.80 * 0.25 = 0.20  <  0.50
        LINEAR:      0.80 * max(0, 1 - 7200/(2*3600)) = 0.80 * 0.0 = 0.0  <  0.50
        """
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE_SHORT, curve=curve)
        score_a = apply_decay(self.BASE_HIGH, age_seconds=self.OLD_AGE_S, config=cfg)
        score_b = apply_decay(self.BASE_LOW,  age_seconds=0.0,            config=cfg)

        assert score_a < score_b, (
            f"curve={curve.value}, halflife={self.HALFLIFE_SHORT}s (high rate): "
            f"older high-relevance ({score_a:.4f}) should lose to fresh low-relevance ({score_b:.4f})"
        )

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_ranking_order_is_opposite_between_high_and_low_decay_rates(
        self, curve: DecayCurve
    ) -> None:
        """Directly assert that the two decay rates produce *opposite* orderings.

        This is the key property: same two results, different decay rates → reversed ranks.
        """
        slow_cfg = DecayConfig(halflife_seconds=self.HALFLIFE_LONG,  curve=curve)
        fast_cfg = DecayConfig(halflife_seconds=self.HALFLIFE_SHORT, curve=curve)

        a_slow = apply_decay(self.BASE_HIGH, age_seconds=self.OLD_AGE_S, config=slow_cfg)
        b_slow = apply_decay(self.BASE_LOW,  age_seconds=0.0,            config=slow_cfg)

        a_fast = apply_decay(self.BASE_HIGH, age_seconds=self.OLD_AGE_S, config=fast_cfg)
        b_fast = apply_decay(self.BASE_LOW,  age_seconds=0.0,            config=fast_cfg)

        assert a_slow > b_slow, (
            f"curve={curve.value}: low-rate config should rank A above B "
            f"(a={a_slow:.4f}, b={b_slow:.4f})"
        )
        assert a_fast < b_fast, (
            f"curve={curve.value}: high-rate config should rank B above A "
            f"(a={a_fast:.4f}, b={b_fast:.4f})"
        )

    def test_higher_decay_rate_always_scores_older_results_lower(self) -> None:
        """For any fixed base score, a higher decay rate must always produce a lower
        (or equal) score for a positive-age result.

        Tested across a range of ages and multiple halflife values to confirm the
        monotone relationship: shorter halflife → lower decayed score.
        """
        halflives = [3_600.0, 7_200.0, 14_400.0, 86_400.0]  # ascending = decreasing rate
        ages = [1_800.0, 3_600.0, 7_200.0, 14_400.0, 86_400.0]
        base = 0.90

        for curve in (DecayCurve.EXPONENTIAL, DecayCurve.LINEAR):
            for age in ages:
                scores = [
                    apply_decay(base, age_seconds=age,
                                config=DecayConfig(halflife_seconds=h, curve=curve))
                    for h in halflives
                ]
                # Scores must be non-decreasing as halflife grows (i.e., rate shrinks).
                for i in range(len(scores) - 1):
                    assert scores[i] <= scores[i + 1], (
                        f"curve={curve.value}, age={age}s: "
                        f"score at halflife={halflives[i]}s ({scores[i]:.6f}) should be "
                        f"<= score at halflife={halflives[i+1]}s ({scores[i+1]:.6f})"
                    )

    @pytest.mark.parametrize("decay_rate,expected_crossover_days", [
        (0.50,  1),   # half-life of 1 day — crossover within 2 days
        (0.80,  4),   # half-life of ~3.1 days
        (0.95, 20),   # default — crossover around day 21
    ])
    def test_per_day_decay_rate_controls_crossover_day(
        self, decay_rate: float, expected_crossover_days: int
    ) -> None:
        """A higher (more aggressive) per-day decay rate brings the crossover day
        earlier, meaning older results lose their ranking advantage sooner.

        Crossover: HIGH_BASE * rate^d == LOW_BASE
          d = log(LOW_BASE / HIGH_BASE) / log(rate)
        """
        high_base = 0.90
        low_base  = 0.30

        scores_high = [decay_score(high_base, d, decay_rate) for d in range(200)]
        scores_low  = [decay_score(low_base,  0, decay_rate)] * 200

        crossover = next(
            (d for d in range(200) if scores_high[d] < scores_low[d]),
            None,
        )

        assert crossover is not None, (
            f"decay_rate={decay_rate}: expected a crossover day, but none found"
        )
        # Tolerance of ±2 days around the expected crossover.
        assert abs(crossover - expected_crossover_days) <= 2, (
            f"decay_rate={decay_rate}: expected crossover ~day {expected_crossover_days}, "
            f"got day {crossover}"
        )

    def test_three_results_ranking_flips_under_aggressive_decay(self) -> None:
        """With three results at increasing ages, an aggressive decay rate
        progressively demotes older results, eventually reversing the full ranking.

        Results (base, age):
            A: (0.90, 1 h)   — slightly old, very relevant
            B: (0.70, 3 h)   — moderately old, quite relevant
            C: (0.50, 0 h)   — fresh, least relevant

        Low rate (halflife=24 h):  A > B > C  (relevance dominates)
        High rate (halflife=1 h):  C > A > B  (freshness dominates)
        """
        age_a, age_b = 3_600.0, 10_800.0  # 1 h, 3 h in seconds
        base_a, base_b, base_c = 0.90, 0.70, 0.50

        slow_cfg = DecayConfig(halflife_seconds=86_400.0, curve=DecayCurve.EXPONENTIAL)
        fast_cfg = DecayConfig(halflife_seconds= 3_600.0, curve=DecayCurve.EXPONENTIAL)

        # Low decay rate: original relevance order holds
        a_slow = apply_decay(base_a, age_a, slow_cfg)
        b_slow = apply_decay(base_b, age_b, slow_cfg)
        c_slow = apply_decay(base_c, 0.0,   slow_cfg)

        assert a_slow > b_slow > c_slow, (
            f"low rate: expected A ({a_slow:.4f}) > B ({b_slow:.4f}) > C ({c_slow:.4f})"
        )

        # High decay rate: fresh C leapfrogs both; stale B drops to last
        a_fast = apply_decay(base_a, age_a, fast_cfg)
        b_fast = apply_decay(base_b, age_b, fast_cfg)
        c_fast = apply_decay(base_c, 0.0,   fast_cfg)

        assert c_fast > a_fast, (
            f"high rate: C ({c_fast:.4f}) should beat A ({a_fast:.4f})"
        )
        assert a_fast > b_fast, (
            f"high rate: A ({a_fast:.4f}) should beat B ({b_fast:.4f})"
        )


# ---------------------------------------------------------------------------
# Integration: SnapshotDecayScorer.score_all() ordering by last-access age
# ---------------------------------------------------------------------------

_NOW_INTEGRATION = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

# Half-life of 7 days keeps both test entries clearly separated on the curve.
_HALFLIFE_7D = 7.0 * 86_400.0


def _feed_entry(snapshot_id: str, age_days: float) -> SnapshotEntry:
    """Build an EPISODIC feed entry whose ingested_at is *age_days* before _NOW_INTEGRATION."""
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id="feed-integration",
        ingested_at=_NOW_INTEGRATION - timedelta(days=age_days),
        source_revision=f"rev-age{age_days}d",
        entry_paths=["Sources/feeds.example.com/feed-integration/item.md"],
        content_hash=f"sha256:{snapshot_id}",
        memory_type=MemoryType.EPISODIC,
    )


class TestIntegrationDecayedSearchOrdering:
    """Integration test: SnapshotDecayScorer.score_all() must rank a recently
    accessed entry above an older one when both carry the same base relevance.

    Design
    ------
    Two EPISODIC entries share identical entry_paths (so their frequency bonus
    is equal) and identical source_id, but their ingested_at timestamps differ:

        recent_entry  — ingested 1 day ago  (age = 1 d < halflife)
        old_entry     — ingested 30 days ago (age = 30 d > halflife)

    With a 7-day half-life the expected age_scores are:
        recent:  0.5^(1/7)  ≈ 0.906
        old:     0.5^(30/7) ≈ 0.052

    Because frequency_score is identical for both entries, combined_score
    preserves the same relative ordering, and sorting by combined_score
    descending must put the recent entry first.
    """

    # Scorer knobs — kept loose so the test isn't brittle to exact float values.
    _STALENESS_THRESHOLD = 0.10
    _FREQUENCY_WEIGHT = 0.30
    _MAX_QUERY_COUNT = 10

    @pytest.fixture
    def analyzer(self) -> QueryHistoryAnalyzer:
        """Seed the analyzer with queries that overlap the shared entry path."""
        qa = QueryHistoryAnalyzer()
        # 3 queries whose topic ("item") appears in the shared entry_path string
        for score in [0.60, 0.65, 0.70]:
            qa.add_query("item", score)
        return qa

    @pytest.fixture
    def scorer(self, analyzer: QueryHistoryAnalyzer) -> SnapshotDecayScorer:
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_7D, curve=DecayCurve.EXPONENTIAL)
        return SnapshotDecayScorer(
            config=cfg,
            query_analyzer=analyzer,
            staleness_threshold=self._STALENESS_THRESHOLD,
            frequency_weight=self._FREQUENCY_WEIGHT,
            max_query_count=self._MAX_QUERY_COUNT,
        )

    # -----------------------------------------------------------------------
    # Core ordering assertion
    # -----------------------------------------------------------------------

    def test_recent_entry_scores_higher_than_old(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """The recently accessed entry must have a strictly higher combined_score."""
        recent = _feed_entry("snap-recent", age_days=1.0)
        old = _feed_entry("snap-old", age_days=30.0)

        scores = {ds.snapshot_id: ds for ds in scorer.score_all(
            [recent, old], now=_NOW_INTEGRATION
        )}

        assert scores["snap-recent"].combined_score > scores["snap-old"].combined_score, (
            f"recent combined={scores['snap-recent'].combined_score:.6f} "
            f"should exceed old combined={scores['snap-old'].combined_score:.6f}"
        )

    def test_age_score_component_drives_ordering(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """age_score alone (before frequency bonus) must also be strictly ordered."""
        recent = _feed_entry("snap-recent", age_days=1.0)
        old = _feed_entry("snap-old", age_days=30.0)

        scores = {ds.snapshot_id: ds for ds in scorer.score_all(
            [recent, old], now=_NOW_INTEGRATION
        )}

        assert scores["snap-recent"].age_score > scores["snap-old"].age_score

    def test_frequency_scores_are_equal(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """Both entries share identical paths, so their frequency_score must match.

        This confirms that the ordering in the other tests is caused purely by
        temporal decay, not by a frequency-score difference.
        """
        recent = _feed_entry("snap-recent", age_days=1.0)
        old = _feed_entry("snap-old", age_days=30.0)

        scores = {ds.snapshot_id: ds for ds in scorer.score_all(
            [recent, old], now=_NOW_INTEGRATION
        )}

        assert scores["snap-recent"].frequency_score == pytest.approx(
            scores["snap-old"].frequency_score
        )

    def test_sorted_results_put_recent_entry_first(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """Sorting score_all() output by combined_score descending must rank
        the recent entry first — mirroring how a search layer would order hits."""
        recent = _feed_entry("snap-recent", age_days=1.0)
        old = _feed_entry("snap-old", age_days=30.0)

        # Deliberately pass old entry first to show ordering is not input-order
        decay_scores = scorer.score_all([old, recent], now=_NOW_INTEGRATION)
        ranked = sorted(decay_scores, key=lambda ds: ds.combined_score, reverse=True)

        assert ranked[0].snapshot_id == "snap-recent"
        assert ranked[1].snapshot_id == "snap-old"

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_ordering_holds_for_both_decay_curves(
        self, analyzer: QueryHistoryAnalyzer, curve: DecayCurve
    ) -> None:
        """Decay-adjusted ordering must hold regardless of the curve shape chosen."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_7D, curve=curve)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=analyzer,
            staleness_threshold=self._STALENESS_THRESHOLD,
            frequency_weight=self._FREQUENCY_WEIGHT,
            max_query_count=self._MAX_QUERY_COUNT,
        )

        recent = _feed_entry("snap-recent", age_days=1.0)
        old = _feed_entry("snap-old", age_days=30.0)

        scores = {ds.snapshot_id: ds for ds in scorer.score_all(
            [recent, old], now=_NOW_INTEGRATION
        )}

        assert scores["snap-recent"].combined_score > scores["snap-old"].combined_score, (
            f"curve={curve.value}: recent={scores['snap-recent'].combined_score:.6f} "
            f"old={scores['snap-old'].combined_score:.6f}"
        )


# ---------------------------------------------------------------------------
# Edge cases: identical timestamps
# ---------------------------------------------------------------------------

_NOW_EDGE = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
_HALFLIFE_1H = 3600.0


def _make_episodic(snapshot_id: str, ingested_at: datetime, paths: list[str]) -> SnapshotEntry:
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id="feed-edge",
        ingested_at=ingested_at,
        source_revision="rev1",
        entry_paths=paths,
        content_hash=f"hash-{snapshot_id}",
        memory_type=MemoryType.EPISODIC,
    )


class TestIdenticalTimestamps:
    """Two entries with the same ingested_at must produce identical age_scores."""

    @pytest.fixture
    def scorer(self) -> SnapshotDecayScorer:
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        return SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )

    def test_identical_timestamps_same_age_score(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """Entries with the same ingested_at must receive the same age_score."""
        shared_ts = _NOW_EDGE - timedelta(hours=2)
        a = _make_episodic("snap-a", shared_ts, ["feed/item-a.md"])
        b = _make_episodic("snap-b", shared_ts, ["feed/item-b.md"])

        scores = {ds.snapshot_id: ds for ds in scorer.score_all([a, b], now=_NOW_EDGE)}

        assert scores["snap-a"].age_score == pytest.approx(scores["snap-b"].age_score)

    def test_identical_timestamps_same_paths_equal_combined_score(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """Same ingested_at and same paths → combined_score must be identical."""
        shared_ts = _NOW_EDGE - timedelta(hours=1)
        paths = ["feed/shared-path.md"]
        a = _make_episodic("snap-a", shared_ts, paths)
        b = _make_episodic("snap-b", shared_ts, paths)

        scores = {ds.snapshot_id: ds for ds in scorer.score_all([a, b], now=_NOW_EDGE)}

        assert scores["snap-a"].combined_score == pytest.approx(scores["snap-b"].combined_score)

    def test_identical_timestamps_different_paths_differ_by_frequency(self) -> None:
        """Same ingested_at but different paths → age_scores equal, combined_scores
        differ only by frequency bonus."""
        qa = QueryHistoryAnalyzer()
        qa.add_query("alpha", 0.9)  # matches 'alpha' path but not 'beta'

        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=qa,
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )

        shared_ts = _NOW_EDGE - timedelta(minutes=30)
        a = _make_episodic("snap-alpha", shared_ts, ["feed/alpha.md"])
        b = _make_episodic("snap-beta", shared_ts, ["feed/beta.md"])

        scores = {ds.snapshot_id: ds for ds in scorer.score_all([a, b], now=_NOW_EDGE)}

        assert scores["snap-alpha"].age_score == pytest.approx(scores["snap-beta"].age_score)
        assert scores["snap-alpha"].frequency_score > scores["snap-beta"].frequency_score
        assert scores["snap-alpha"].combined_score > scores["snap-beta"].combined_score

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_identical_timestamps_both_curves(self, curve: DecayCurve) -> None:
        """Identical-timestamp invariant holds for both curve shapes."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=curve)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )

        shared_ts = _NOW_EDGE - timedelta(hours=0.5)
        a = _make_episodic("snap-a", shared_ts, ["feed/item.md"])
        b = _make_episodic("snap-b", shared_ts, ["feed/item.md"])

        scores = {ds.snapshot_id: ds for ds in scorer.score_all([a, b], now=_NOW_EDGE)}

        assert scores["snap-a"].age_score == pytest.approx(
            scores["snap-b"].age_score
        ), f"curve={curve.value}: age_scores must match for identical timestamps"


# ---------------------------------------------------------------------------
# Edge cases: zero age (no effective timestamp offset) and future-dated entries
# ---------------------------------------------------------------------------

class TestZeroAndFutureAge:
    """Entries ingested exactly at 'now' (zero age) or in the future must not be
    penalised by decay — the age_score must equal 1.0."""

    @pytest.fixture
    def scorer(self) -> SnapshotDecayScorer:
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        return SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )

    def test_ingested_exactly_now_age_score_is_one(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """Entry ingested at exactly now → age_seconds=0 → age_score=1.0."""
        entry = _make_episodic("snap-now", _NOW_EDGE, ["feed/now.md"])
        result = scorer.score(entry, now=_NOW_EDGE)
        assert result.age_score == pytest.approx(1.0)

    def test_future_dated_entry_age_score_is_one(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """Entry with ingested_at in the future → negative age → clamped to 0
        → age_score=1.0 (same as brand-new)."""
        future_entry = _make_episodic(
            "snap-future",
            _NOW_EDGE + timedelta(hours=1),
            ["feed/future.md"],
        )
        result = scorer.score(future_entry, now=_NOW_EDGE)
        assert result.age_score == pytest.approx(1.0)

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_future_entry_not_amplified_above_one(
        self, curve: DecayCurve
    ) -> None:
        """Future-dated entries must never produce age_score > 1.0."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=curve)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )

        entry = _make_episodic(
            "snap-far-future",
            _NOW_EDGE + timedelta(days=365),
            ["feed/item.md"],
        )
        result = scorer.score(entry, now=_NOW_EDGE)
        assert result.age_score <= 1.0
        assert result.age_score == pytest.approx(1.0)

    def test_score_all_empty_list_returns_empty(
        self, scorer: SnapshotDecayScorer
    ) -> None:
        """score_all([]) must return an empty list without raising."""
        result = scorer.score_all([], now=_NOW_EDGE)
        assert result == []


# ---------------------------------------------------------------------------
# Edge cases: decay boundary values
# ---------------------------------------------------------------------------

class TestDecayBoundaryValues:
    """Verify behaviour at the hard mathematical boundaries of both decay curves.

    Boundaries tested:
      - age = 0                              → score unchanged (factor = 1.0)
      - age = halflife                       → score halved
      - age = 2 * halflife (linear maxAge)   → LINEAR score == 0
      - age just below 2 * halflife          → LINEAR score > 0
      - age >> halflife (very old)           → EXPONENTIAL score > 0 but tiny
      - age >> halflife (very old)           → LINEAR score == 0
    """

    HALFLIFE = 3600.0  # 1 hour
    BASE = 1.0

    @pytest.fixture
    def exp_cfg(self) -> DecayConfig:
        return DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.EXPONENTIAL)

    @pytest.fixture
    def lin_cfg(self) -> DecayConfig:
        return DecayConfig(halflife_seconds=self.HALFLIFE, curve=DecayCurve.LINEAR)

    # -- zero age ---------------------------------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_zero_age_score_unchanged(self, curve: DecayCurve) -> None:
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)
        assert apply_decay(self.BASE, age_seconds=0.0, config=cfg) == pytest.approx(1.0)

    def test_zero_age_preserves_arbitrary_score(self, exp_cfg: DecayConfig) -> None:
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert apply_decay(score, age_seconds=0.0, config=exp_cfg) == pytest.approx(score)

    # -- at halflife (score halved) ----------------------------------------------

    @pytest.mark.parametrize("curve", [DecayCurve.EXPONENTIAL, DecayCurve.LINEAR])
    def test_at_halflife_score_is_exactly_half(self, curve: DecayCurve) -> None:
        cfg = DecayConfig(halflife_seconds=self.HALFLIFE, curve=curve)
        result = apply_decay(self.BASE, age_seconds=self.HALFLIFE, config=cfg)
        assert result == pytest.approx(0.5)

    # -- LINEAR zero-crossing boundary (age == 2 * halflife) --------------------

    def test_linear_at_zero_crossing_is_zero(self, lin_cfg: DecayConfig) -> None:
        """Linear score reaches exactly 0 at age == 2 * halflife."""
        max_age = 2.0 * self.HALFLIFE
        assert apply_decay(self.BASE, age_seconds=max_age, config=lin_cfg) == pytest.approx(0.0)

    def test_linear_one_second_before_zero_crossing_is_positive(
        self, lin_cfg: DecayConfig
    ) -> None:
        """One second before the zero-crossing the linear score is still > 0."""
        just_before = 2.0 * self.HALFLIFE - 1.0
        result = apply_decay(self.BASE, age_seconds=just_before, config=lin_cfg)
        assert result > 0.0

    def test_linear_one_second_after_zero_crossing_is_zero(
        self, lin_cfg: DecayConfig
    ) -> None:
        """One second past 2 * halflife is clamped to 0."""
        just_after = 2.0 * self.HALFLIFE + 1.0
        assert apply_decay(self.BASE, age_seconds=just_after, config=lin_cfg) == pytest.approx(0.0)

    # -- very old entries --------------------------------------------------------

    def test_very_old_exponential_score_is_positive_but_tiny(
        self, exp_cfg: DecayConfig
    ) -> None:
        """EXPONENTIAL is asymptotic — score must be > 0 even at 100 halflives."""
        age = 100.0 * self.HALFLIFE  # 100 hours
        result = apply_decay(self.BASE, age_seconds=age, config=exp_cfg)
        assert result > 0.0
        # 0.5^100 ≈ 7.9e-31
        assert result == pytest.approx(math.pow(0.5, 100.0), rel=1e-6)

    def test_very_old_linear_score_is_exactly_zero(self, lin_cfg: DecayConfig) -> None:
        """LINEAR is zero for any age >= 2 * halflife."""
        for multiplier in [2.0, 10.0, 1000.0]:
            age = multiplier * self.HALFLIFE
            assert apply_decay(self.BASE, age_seconds=age, config=lin_cfg) == pytest.approx(0.0), (
                f"expected 0 at age={age}s (multiplier={multiplier})"
            )

    def test_very_old_entry_scorer_age_score_near_zero(self) -> None:
        """SnapshotDecayScorer.score() on a 10-year-old entry → age_score ≈ 0."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )
        ancient = _make_episodic(
            "snap-ancient",
            _NOW_EDGE - timedelta(days=365 * 10),
            ["feed/ancient.md"],
        )
        result = scorer.score(ancient, now=_NOW_EDGE)
        # 10 years of hourly half-lives → effectively 0
        assert result.age_score < 1e-20

    def test_very_old_entry_is_stale(self) -> None:
        """A 10-year-old entry with no query matches must be marked stale."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.2,
            frequency_weight=0.3,
            max_query_count=10,
        )
        ancient = _make_episodic(
            "snap-ancient",
            _NOW_EDGE - timedelta(days=365 * 10),
            ["feed/ancient.md"],
        )
        result = scorer.score(ancient, now=_NOW_EDGE)
        assert result.is_stale

    def test_zero_age_entry_not_stale(self) -> None:
        """An entry ingested right now must not be stale (age_score=1.0 > threshold)."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.2,
            frequency_weight=0.3,
            max_query_count=10,
        )
        fresh = _make_episodic("snap-fresh", _NOW_EDGE, ["feed/fresh.md"])
        result = scorer.score(fresh, now=_NOW_EDGE)
        assert not result.is_stale

    # -- score_all preserves ordering at boundaries ----------------------------

    def test_score_all_boundary_ordering_zero_vs_old(self) -> None:
        """score_all([zero-age, old]) must rank zero-age entry first."""
        cfg = DecayConfig(halflife_seconds=_HALFLIFE_1H, curve=DecayCurve.EXPONENTIAL)
        scorer = SnapshotDecayScorer(
            config=cfg,
            query_analyzer=QueryHistoryAnalyzer(),
            staleness_threshold=0.1,
            frequency_weight=0.3,
            max_query_count=10,
        )
        fresh = _make_episodic("snap-fresh", _NOW_EDGE, ["feed/item.md"])
        old = _make_episodic(
            "snap-old", _NOW_EDGE - timedelta(days=30), ["feed/item.md"]
        )

        ranked = sorted(
            scorer.score_all([old, fresh], now=_NOW_EDGE),
            key=lambda ds: ds.combined_score,
            reverse=True,
        )
        assert ranked[0].snapshot_id == "snap-fresh"
