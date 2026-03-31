"""Tests for transition precondition validators in shad.vault.taxonomy.

Covers:
  - TransitionContext construction
  - min_retention_count factory
  - staleness_threshold factory
  - confidence_gate factory
  - DEFAULT_TRANSITION_VALIDATORS registry contents
  - check_transition_validators integration (pass / fail paths)
  - Custom registry override
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.taxonomy import (
    DEFAULT_TRANSITION_VALIDATORS,
    MemoryState,
    TransitionContext,
    TransitionType,
    check_transition_validators,
    confidence_gate,
    min_retention_count,
    staleness_threshold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)


def _ctx(
    *,
    retention_count: int = 0,
    confidence: float = 1.0,
    last_accessed_at: datetime | None = None,
    ingested_at: datetime = _NOW,
) -> TransitionContext:
    return TransitionContext(
        retention_count=retention_count,
        confidence=confidence,
        last_accessed_at=last_accessed_at,
        ingested_at=ingested_at,
    )


# ---------------------------------------------------------------------------
# TransitionContext
# ---------------------------------------------------------------------------


class TestTransitionContext:
    def test_fields_stored_verbatim(self) -> None:
        t = _NOW - timedelta(days=5)
        ctx = TransitionContext(
            retention_count=7,
            confidence=0.75,
            last_accessed_at=t,
            ingested_at=_NOW,
        )
        assert ctx.retention_count == 7
        assert ctx.confidence == 0.75
        assert ctx.last_accessed_at == t
        assert ctx.ingested_at == _NOW

    def test_last_accessed_at_defaults_to_none(self) -> None:
        ctx = _ctx()
        assert ctx.last_accessed_at is None


# ---------------------------------------------------------------------------
# min_retention_count
# ---------------------------------------------------------------------------


class TestMinRetentionCount:
    def test_passes_when_count_equals_threshold(self) -> None:
        v = min_retention_count(3)
        assert v(_ctx(retention_count=3)) is True

    def test_passes_when_count_exceeds_threshold(self) -> None:
        v = min_retention_count(3)
        assert v(_ctx(retention_count=10)) is True

    def test_fails_when_count_below_threshold(self) -> None:
        v = min_retention_count(3)
        assert v(_ctx(retention_count=2)) is False

    def test_zero_threshold_always_passes(self) -> None:
        v = min_retention_count(0)
        assert v(_ctx(retention_count=0)) is True

    def test_has_descriptive_name(self) -> None:
        v = min_retention_count(5)
        assert v.__name__ == "min_retention_count(5)"

    @pytest.mark.parametrize("n,count,expected", [
        (1, 0, False),
        (1, 1, True),
        (5, 4, False),
        (5, 5, True),
        (5, 6, True),
    ])
    def test_parametrized_boundary(self, n: int, count: int, expected: bool) -> None:
        assert min_retention_count(n)(_ctx(retention_count=count)) is expected


# ---------------------------------------------------------------------------
# staleness_threshold
# ---------------------------------------------------------------------------


class TestStalenessThreshold:
    def test_passes_when_idle_longer_than_threshold(self) -> None:
        old_time = _NOW - timedelta(days=10)
        v = staleness_threshold(7.0, _now=_NOW)
        assert v(_ctx(ingested_at=old_time)) is True

    def test_fails_when_idle_shorter_than_threshold(self) -> None:
        recent = _NOW - timedelta(days=3)
        v = staleness_threshold(7.0, _now=_NOW)
        assert v(_ctx(ingested_at=recent)) is False

    def test_passes_exactly_at_threshold(self) -> None:
        exactly = _NOW - timedelta(days=7)
        v = staleness_threshold(7.0, _now=_NOW)
        assert v(_ctx(ingested_at=exactly)) is True

    def test_prefers_last_accessed_at_over_ingested_at(self) -> None:
        # ingested_at is old (stale) but last_accessed_at is recent → should fail
        old_ingested = _NOW - timedelta(days=30)
        recent_access = _NOW - timedelta(days=1)
        v = staleness_threshold(7.0, _now=_NOW)
        assert v(_ctx(ingested_at=old_ingested, last_accessed_at=recent_access)) is False

    def test_falls_back_to_ingested_at_when_last_accessed_is_none(self) -> None:
        old_ingested = _NOW - timedelta(days=30)
        v = staleness_threshold(7.0, _now=_NOW)
        assert v(_ctx(ingested_at=old_ingested, last_accessed_at=None)) is True

    def test_last_accessed_at_stale_passes_even_if_ingested_recently(self) -> None:
        recent_ingested = _NOW - timedelta(days=1)
        stale_access = _NOW - timedelta(days=60)
        v = staleness_threshold(30.0, _now=_NOW)
        assert v(_ctx(ingested_at=recent_ingested, last_accessed_at=stale_access)) is True

    def test_has_descriptive_name(self) -> None:
        v = staleness_threshold(14.0)
        assert v.__name__ == "staleness_threshold(14.0d)"

    @pytest.mark.parametrize("threshold,idle_days,expected", [
        (7.0, 6, False),
        (7.0, 7, True),
        (30.0, 29, False),
        (30.0, 30, True),
        (90.0, 89, False),
        (90.0, 91, True),
    ])
    def test_parametrized_boundaries(
        self, threshold: float, idle_days: int, expected: bool
    ) -> None:
        t = _NOW - timedelta(days=idle_days)
        v = staleness_threshold(threshold, _now=_NOW)
        assert v(_ctx(ingested_at=t)) is expected


# ---------------------------------------------------------------------------
# confidence_gate
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    def test_passes_when_confidence_equals_threshold(self) -> None:
        v = confidence_gate(0.6)
        assert v(_ctx(confidence=0.6)) is True

    def test_passes_when_confidence_above_threshold(self) -> None:
        v = confidence_gate(0.6)
        assert v(_ctx(confidence=0.99)) is True

    def test_fails_when_confidence_below_threshold(self) -> None:
        v = confidence_gate(0.6)
        assert v(_ctx(confidence=0.59)) is False

    def test_zero_threshold_always_passes(self) -> None:
        v = confidence_gate(0.0)
        assert v(_ctx(confidence=0.0)) is True

    def test_threshold_of_one_fails_below_perfect(self) -> None:
        v = confidence_gate(1.0)
        assert v(_ctx(confidence=0.99)) is False

    def test_has_descriptive_name(self) -> None:
        v = confidence_gate(0.8)
        assert v.__name__ == "confidence_gate(0.8)"

    @pytest.mark.parametrize("threshold,conf,expected", [
        (0.6, 0.59, False),
        (0.6, 0.6, True),
        (0.8, 0.79, False),
        (0.8, 0.8, True),
        (0.8, 1.0, True),
    ])
    def test_parametrized_boundary(
        self, threshold: float, conf: float, expected: bool
    ) -> None:
        assert confidence_gate(threshold)(_ctx(confidence=conf)) is expected


# ---------------------------------------------------------------------------
# DEFAULT_TRANSITION_VALIDATORS registry
# ---------------------------------------------------------------------------


class TestDefaultTransitionValidators:
    def test_episodic_promote_semantic_has_two_validators(self) -> None:
        edge = (MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC)
        assert len(DEFAULT_TRANSITION_VALIDATORS[edge]) == 2

    def test_semantic_promote_procedural_has_two_validators(self) -> None:
        edge = (MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL)
        assert len(DEFAULT_TRANSITION_VALIDATORS[edge]) == 2

    def test_episodic_demote_dormant_has_one_validator(self) -> None:
        edge = (MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT)
        assert len(DEFAULT_TRANSITION_VALIDATORS[edge]) == 1

    def test_semantic_demote_dormant_has_one_validator(self) -> None:
        edge = (MemoryState.SEMANTIC, TransitionType.DEMOTE, MemoryState.DORMANT)
        assert len(DEFAULT_TRANSITION_VALIDATORS[edge]) == 1

    def test_procedural_demote_archival_has_one_validator(self) -> None:
        edge = (MemoryState.PROCEDURAL, TransitionType.DEMOTE, MemoryState.ARCHIVAL)
        assert len(DEFAULT_TRANSITION_VALIDATORS[edge]) == 1

    def test_merge_and_split_edges_have_no_validators(self) -> None:
        for state in (MemoryState.EPISODIC, MemoryState.SEMANTIC, MemoryState.PROCEDURAL):
            for tt in (TransitionType.MERGE, TransitionType.SPLIT):
                # These edges impose no structural preconditions.
                edge = (state, tt, state)
                assert edge not in DEFAULT_TRANSITION_VALIDATORS


# ---------------------------------------------------------------------------
# check_transition_validators
# ---------------------------------------------------------------------------


class TestCheckTransitionValidators:
    """Integration tests: check_transition_validators uses DEFAULT_TRANSITION_VALIDATORS."""

    # --- PROMOTE: EPISODIC → SEMANTIC ---

    def test_episodic_to_semantic_passes_when_all_conditions_met(self) -> None:
        ctx = _ctx(retention_count=3, confidence=0.6)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert failures == []

    def test_episodic_to_semantic_fails_on_low_retention(self) -> None:
        ctx = _ctx(retention_count=2, confidence=0.9)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert len(failures) == 1
        assert "min_retention_count(3)" in failures[0]

    def test_episodic_to_semantic_fails_on_low_confidence(self) -> None:
        ctx = _ctx(retention_count=5, confidence=0.5)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert len(failures) == 1
        assert "confidence_gate(0.6)" in failures[0]

    def test_episodic_to_semantic_fails_on_both_conditions(self) -> None:
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert len(failures) == 2

    # --- PROMOTE: SEMANTIC → PROCEDURAL ---

    def test_semantic_to_procedural_passes_when_all_conditions_met(self) -> None:
        ctx = _ctx(retention_count=5, confidence=0.8)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert failures == []

    def test_semantic_to_procedural_fails_on_insufficient_retention(self) -> None:
        ctx = _ctx(retention_count=4, confidence=1.0)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert any("min_retention_count(5)" in f for f in failures)

    def test_semantic_to_procedural_fails_on_insufficient_confidence(self) -> None:
        ctx = _ctx(retention_count=10, confidence=0.79)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert any("confidence_gate(0.8)" in f for f in failures)

    # --- DEMOTE edges (use custom validator with _now override) ---

    def test_episodic_demote_dormant_passes_when_stale(self) -> None:
        stale_ingested = _NOW - timedelta(days=8)
        # Patch the registry with a _now-aware validator for deterministic behaviour.
        custom = {
            (MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT): [
                staleness_threshold(7.0, _now=_NOW),
            ]
        }
        ctx = _ctx(ingested_at=stale_ingested)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT,
            ctx, validators=custom,
        )
        assert failures == []

    def test_episodic_demote_dormant_fails_when_not_stale(self) -> None:
        fresh_ingested = _NOW - timedelta(days=2)
        custom = {
            (MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT): [
                staleness_threshold(7.0, _now=_NOW),
            ]
        }
        ctx = _ctx(ingested_at=fresh_ingested)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT,
            ctx, validators=custom,
        )
        assert len(failures) == 1
        assert "staleness_threshold(7.0d)" in failures[0]

    def test_procedural_demote_archival_passes_when_stale(self) -> None:
        ancient = _NOW - timedelta(days=91)
        custom = {
            (MemoryState.PROCEDURAL, TransitionType.DEMOTE, MemoryState.ARCHIVAL): [
                staleness_threshold(90.0, _now=_NOW),
            ]
        }
        ctx = _ctx(ingested_at=ancient)
        failures = check_transition_validators(
            MemoryState.PROCEDURAL, TransitionType.DEMOTE, MemoryState.ARCHIVAL,
            ctx, validators=custom,
        )
        assert failures == []

    # --- Edge with no registered validators ---

    def test_merge_edge_has_no_preconditions(self) -> None:
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.MERGE, MemoryState.EPISODIC, ctx
        )
        assert failures == []

    def test_split_edge_has_no_preconditions(self) -> None:
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.SPLIT, MemoryState.SEMANTIC, ctx
        )
        assert failures == []

    # --- Custom registry override ---

    def test_custom_registry_replaces_defaults(self) -> None:
        """A custom registry completely overrides the defaults for that edge."""
        custom = {
            (MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC): [
                min_retention_count(100),  # impossibly high
            ]
        }
        ctx = _ctx(retention_count=99, confidence=1.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC,
            ctx, validators=custom,
        )
        assert len(failures) == 1
        assert "min_retention_count(100)" in failures[0]

    def test_empty_custom_registry_imposes_no_preconditions(self) -> None:
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC,
            ctx, validators={},
        )
        assert failures == []

    def test_returns_empty_list_when_no_validators_registered_for_edge(self) -> None:
        """Edge absent from the registry → no preconditions → empty failure list."""
        # Use a custom registry that simply does not contain the requested edge.
        custom: dict = {}
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.DORMANT, TransitionType.PROMOTE, MemoryState.EPISODIC,
            ctx, validators=custom,
        )
        assert failures == []

    def test_returns_all_failure_messages_when_multiple_validators_fail(self) -> None:
        """Every failing validator contributes exactly one message; all are returned."""
        custom = {
            (MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC): [
                min_retention_count(10),
                confidence_gate(0.9),
                min_retention_count(5),  # third distinct validator
            ]
        }
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC,
            ctx, validators=custom,
        )
        assert len(failures) == 3
        assert any("min_retention_count(10)" in f for f in failures)
        assert any("confidence_gate(0.9)" in f for f in failures)
        assert any("min_retention_count(5)" in f for f in failures)

    # --- Failure message format ---

    def test_failure_message_contains_state_and_transition_names(self) -> None:
        ctx = _ctx(retention_count=0, confidence=0.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        for msg in failures:
            assert "episodic" in msg
            assert "promote" in msg
            assert "semantic" in msg
