"""Tests for the memory lifecycle taxonomy state machine.

Covers the invariant that terminal/suppression states (DORMANT, ARCHIVAL)
have no valid DEMOTE targets — they are floor states and cannot be lowered
further by the demotion path.

Also exhaustively verifies that every (state, transition) pair absent from
ALLOWED_TRANSITIONS returns a failed ValidationResult with a descriptive
reason string that names both the state and the transition.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.taxonomy import (
    ALLOWED_TRANSITIONS,
    MemoryState,
    StateMachine,
    TransitionContext,
    TransitionType,
    check_transition_validators,
    confidence_gate,
    is_valid_transition,
    min_retention_count,
    staleness_threshold,
)

# ---------------------------------------------------------------------------
# Build the exhaustive set of invalid (state, transition) pairs at import time
# so pytest can collect them as individual test IDs.
# ---------------------------------------------------------------------------

_ALL_PAIRS = list(itertools.product(MemoryState, TransitionType))
_INVALID_PAIRS = [
    (state, transition)
    for state, transition in _ALL_PAIRS
    if (state, transition) not in ALLOWED_TRANSITIONS
]


class TestDormantCannotBeDemoted:
    """DORMANT is a suppression floor: DEMOTE is undefined from this state."""

    def test_no_demote_key_in_allowed_transitions(self) -> None:
        assert ALLOWED_TRANSITIONS.get((MemoryState.DORMANT, TransitionType.DEMOTE)) is None

    @pytest.mark.parametrize("target", list(MemoryState))
    def test_is_valid_transition_returns_false_for_all_targets(
        self, target: MemoryState
    ) -> None:
        assert not is_valid_transition(MemoryState.DORMANT, TransitionType.DEMOTE, target)

    @pytest.mark.parametrize("target", list(MemoryState))
    def test_state_machine_validate_fails_for_all_targets(
        self, target: MemoryState
    ) -> None:
        sm = StateMachine()
        result = sm.validate(MemoryState.DORMANT, TransitionType.DEMOTE, target)
        assert not result.ok
        assert "dormant" in result.reason


class TestArchivalCannotBeDemoted:
    """ARCHIVAL is a terminal retention state: DEMOTE is undefined from this state."""

    def test_no_demote_key_in_allowed_transitions(self) -> None:
        assert ALLOWED_TRANSITIONS.get((MemoryState.ARCHIVAL, TransitionType.DEMOTE)) is None

    @pytest.mark.parametrize("target", list(MemoryState))
    def test_is_valid_transition_returns_false_for_all_targets(
        self, target: MemoryState
    ) -> None:
        assert not is_valid_transition(MemoryState.ARCHIVAL, TransitionType.DEMOTE, target)

    @pytest.mark.parametrize("target", list(MemoryState))
    def test_state_machine_validate_fails_for_all_targets(
        self, target: MemoryState
    ) -> None:
        sm = StateMachine()
        result = sm.validate(MemoryState.ARCHIVAL, TransitionType.DEMOTE, target)
        assert not result.ok
        assert "archival" in result.reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    retention_count: int = 0,
    confidence: float = 1.0,
    last_accessed_at: datetime | None = None,
    ingested_at: datetime | None = None,
) -> TransitionContext:
    now = datetime.now(UTC)
    return TransitionContext(
        retention_count=retention_count,
        confidence=confidence,
        last_accessed_at=last_accessed_at,
        ingested_at=ingested_at if ingested_at is not None else now,
    )


# ---------------------------------------------------------------------------
# min_retention_count boundary tests
# ---------------------------------------------------------------------------


class TestMinRetentionCountBoundary:
    """retention_count >= n is inclusive on the threshold."""

    @pytest.mark.parametrize("n, count, expected", [
        # threshold = 3 (EPISODIC → SEMANTIC)
        (3, 2, False),  # one below
        (3, 3, True),   # exactly at threshold
        (3, 4, True),   # one above
        # threshold = 5 (SEMANTIC → PROCEDURAL)
        (5, 4, False),  # one below
        (5, 5, True),   # exactly at threshold
        (5, 6, True),   # one above
    ])
    def test_boundary(self, n: int, count: int, expected: bool) -> None:
        validator = min_retention_count(n)
        assert validator(_ctx(retention_count=count)) is expected

    def test_name_reflects_threshold(self) -> None:
        assert min_retention_count(3).__name__ == "min_retention_count(3)"
        assert min_retention_count(5).__name__ == "min_retention_count(5)"

    def test_episodic_promote_semantic_at_count_3(self) -> None:
        ctx = _ctx(retention_count=3, confidence=1.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        # Only the confidence_gate check should pass; retention passes too → no failures.
        assert not any("min_retention_count" in f for f in failures)

    def test_episodic_promote_semantic_fails_at_count_2(self) -> None:
        ctx = _ctx(retention_count=2, confidence=1.0)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert any("min_retention_count" in f for f in failures)

    def test_semantic_promote_procedural_at_count_5(self) -> None:
        ctx = _ctx(retention_count=5, confidence=1.0)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert not any("min_retention_count" in f for f in failures)

    def test_semantic_promote_procedural_fails_at_count_4(self) -> None:
        ctx = _ctx(retention_count=4, confidence=1.0)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert any("min_retention_count" in f for f in failures)


# ---------------------------------------------------------------------------
# confidence_gate boundary tests
# ---------------------------------------------------------------------------


class TestConfidenceGateBoundary:
    """confidence >= threshold is inclusive on the gate value."""

    @pytest.mark.parametrize("threshold, conf, expected", [
        # gate = 0.6 (EPISODIC → SEMANTIC)
        (0.6, 0.5999, False),
        (0.6, 0.6,    True),
        (0.6, 0.6001, True),
        # gate = 0.8 (SEMANTIC → PROCEDURAL)
        (0.8, 0.7999, False),
        (0.8, 0.8,    True),
        (0.8, 0.8001, True),
    ])
    def test_boundary(self, threshold: float, conf: float, expected: bool) -> None:
        validator = confidence_gate(threshold)
        assert validator(_ctx(confidence=conf)) is expected

    def test_name_reflects_threshold(self) -> None:
        assert confidence_gate(0.6).__name__ == "confidence_gate(0.6)"
        assert confidence_gate(0.8).__name__ == "confidence_gate(0.8)"

    def test_episodic_promote_semantic_passes_at_0_6(self) -> None:
        ctx = _ctx(retention_count=10, confidence=0.6)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert not any("confidence_gate" in f for f in failures)

    def test_episodic_promote_semantic_fails_below_0_6(self) -> None:
        ctx = _ctx(retention_count=10, confidence=0.5999)
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC, ctx
        )
        assert any("confidence_gate" in f for f in failures)

    def test_semantic_promote_procedural_passes_at_0_8(self) -> None:
        ctx = _ctx(retention_count=10, confidence=0.8)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert not any("confidence_gate" in f for f in failures)

    def test_semantic_promote_procedural_fails_below_0_8(self) -> None:
        ctx = _ctx(retention_count=10, confidence=0.7999)
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL, ctx
        )
        assert any("confidence_gate" in f for f in failures)


# ---------------------------------------------------------------------------
# staleness_threshold boundary tests
# ---------------------------------------------------------------------------


class TestStalenessThresholdBoundary:
    """(now - last_active) >= timedelta(days=d) is inclusive: exactly d days passes."""

    @pytest.mark.parametrize("days", [7.0, 30.0, 90.0])
    def test_exactly_at_threshold_passes(self, days: float) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        last_active = now - timedelta(days=days)
        validator = staleness_threshold(days, _now=now)
        assert validator(_ctx(last_accessed_at=last_active)) is True

    @pytest.mark.parametrize("days", [7.0, 30.0, 90.0])
    def test_one_second_short_fails(self, days: float) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        last_active = now - timedelta(days=days) + timedelta(seconds=1)
        validator = staleness_threshold(days, _now=now)
        assert validator(_ctx(last_accessed_at=last_active)) is False

    @pytest.mark.parametrize("days", [7.0, 30.0, 90.0])
    def test_one_second_over_passes(self, days: float) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        last_active = now - timedelta(days=days) - timedelta(seconds=1)
        validator = staleness_threshold(days, _now=now)
        assert validator(_ctx(last_accessed_at=last_active)) is True

    def test_falls_back_to_ingested_at_when_no_last_accessed(self) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        ingested = now - timedelta(days=7)
        validator = staleness_threshold(7.0, _now=now)
        ctx = TransitionContext(
            retention_count=0,
            confidence=1.0,
            last_accessed_at=None,
            ingested_at=ingested,
        )
        assert validator(ctx) is True

    def test_episodic_demote_dormant_exactly_7_days(self) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        last_active = now - timedelta(days=7)
        ctx = _ctx(last_accessed_at=last_active)
        validators = {
            (MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT): [
                staleness_threshold(7.0, _now=now)
            ]
        }
        failures = check_transition_validators(
            MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT, ctx,
            validators=validators,
        )
        assert failures == []

    def test_semantic_demote_dormant_exactly_30_days(self) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        last_active = now - timedelta(days=30)
        ctx = _ctx(last_accessed_at=last_active)
        validators = {
            (MemoryState.SEMANTIC, TransitionType.DEMOTE, MemoryState.DORMANT): [
                staleness_threshold(30.0, _now=now)
            ]
        }
        failures = check_transition_validators(
            MemoryState.SEMANTIC, TransitionType.DEMOTE, MemoryState.DORMANT, ctx,
            validators=validators,
        )
        assert failures == []

    def test_procedural_demote_archival_exactly_90_days(self) -> None:
        now = datetime(2026, 3, 31, tzinfo=UTC)
        last_active = now - timedelta(days=90)
        ctx = _ctx(last_accessed_at=last_active)
        validators = {
            (MemoryState.PROCEDURAL, TransitionType.DEMOTE, MemoryState.ARCHIVAL): [
                staleness_threshold(90.0, _now=now)
            ]
        }
        failures = check_transition_validators(
            MemoryState.PROCEDURAL, TransitionType.DEMOTE, MemoryState.ARCHIVAL, ctx,
            validators=validators,
        )
        assert failures == []

    def test_name_reflects_days(self) -> None:
        assert staleness_threshold(7.0).__name__ == "staleness_threshold(7.0d)"
        assert staleness_threshold(30.0).__name__ == "staleness_threshold(30.0d)"
        assert staleness_threshold(90.0).__name__ == "staleness_threshold(90.0d)"


# ---------------------------------------------------------------------------
# Same-type transition tests: MERGE and SPLIT
# ---------------------------------------------------------------------------


class TestMergeSameTypeTransitions:
    """MERGE always collapses two items into one item of the *same* state.

    is_valid_transition and StateMachine.validate must accept same-state MERGE
    for every active and inactive state, and reject cross-state MERGE.
    """

    @pytest.mark.parametrize("state", [
        MemoryState.EPISODIC,
        MemoryState.SEMANTIC,
        MemoryState.PROCEDURAL,
        MemoryState.ARCHIVAL,
        MemoryState.DORMANT,
    ])
    def test_same_state_merge_is_valid(self, state: MemoryState) -> None:
        assert is_valid_transition(state, TransitionType.MERGE, state)

    @pytest.mark.parametrize("state", [
        MemoryState.EPISODIC,
        MemoryState.SEMANTIC,
        MemoryState.PROCEDURAL,
        MemoryState.ARCHIVAL,
        MemoryState.DORMANT,
    ])
    def test_state_machine_accepts_same_state_merge(self, state: MemoryState) -> None:
        sm = StateMachine()
        result = sm.validate(state, TransitionType.MERGE, state)
        assert result.ok

    @pytest.mark.parametrize("state, other", [
        (MemoryState.EPISODIC, MemoryState.SEMANTIC),
        (MemoryState.EPISODIC, MemoryState.PROCEDURAL),
        (MemoryState.SEMANTIC, MemoryState.EPISODIC),
        (MemoryState.SEMANTIC, MemoryState.PROCEDURAL),
        (MemoryState.PROCEDURAL, MemoryState.EPISODIC),
        (MemoryState.PROCEDURAL, MemoryState.SEMANTIC),
    ])
    def test_cross_state_merge_is_invalid(
        self, state: MemoryState, other: MemoryState
    ) -> None:
        assert not is_valid_transition(state, TransitionType.MERGE, other)

    @pytest.mark.parametrize("state, other", [
        (MemoryState.EPISODIC, MemoryState.SEMANTIC),
        (MemoryState.EPISODIC, MemoryState.PROCEDURAL),
        (MemoryState.SEMANTIC, MemoryState.EPISODIC),
        (MemoryState.PROCEDURAL, MemoryState.EPISODIC),
    ])
    def test_state_machine_rejects_cross_state_merge_with_reason(
        self, state: MemoryState, other: MemoryState
    ) -> None:
        sm = StateMachine()
        result = sm.validate(state, TransitionType.MERGE, other)
        assert not result.ok
        assert other.value in result.reason or state.value in result.reason

    def test_episodic_merge_target_is_exactly_episodic(self) -> None:
        allowed = ALLOWED_TRANSITIONS.get((MemoryState.EPISODIC, TransitionType.MERGE))
        assert allowed == frozenset({MemoryState.EPISODIC})


class TestSplitSameTypeTransitions:
    """SPLIT always decomposes one item into two items of the *same* state.

    Mirrors TestMergeSameTypeTransitions for the SPLIT transition.
    """

    @pytest.mark.parametrize("state", [
        MemoryState.EPISODIC,
        MemoryState.SEMANTIC,
        MemoryState.PROCEDURAL,
        MemoryState.ARCHIVAL,
        MemoryState.DORMANT,
    ])
    def test_same_state_split_is_valid(self, state: MemoryState) -> None:
        assert is_valid_transition(state, TransitionType.SPLIT, state)

    @pytest.mark.parametrize("state", [
        MemoryState.EPISODIC,
        MemoryState.SEMANTIC,
        MemoryState.PROCEDURAL,
        MemoryState.ARCHIVAL,
        MemoryState.DORMANT,
    ])
    def test_state_machine_accepts_same_state_split(self, state: MemoryState) -> None:
        sm = StateMachine()
        result = sm.validate(state, TransitionType.SPLIT, state)
        assert result.ok

    @pytest.mark.parametrize("state, other", [
        (MemoryState.EPISODIC, MemoryState.SEMANTIC),
        (MemoryState.EPISODIC, MemoryState.PROCEDURAL),
        (MemoryState.SEMANTIC, MemoryState.EPISODIC),
        (MemoryState.SEMANTIC, MemoryState.PROCEDURAL),
        (MemoryState.PROCEDURAL, MemoryState.EPISODIC),
        (MemoryState.PROCEDURAL, MemoryState.SEMANTIC),
    ])
    def test_cross_state_split_is_invalid(
        self, state: MemoryState, other: MemoryState
    ) -> None:
        assert not is_valid_transition(state, TransitionType.SPLIT, other)

    def test_episodic_split_target_is_exactly_episodic(self) -> None:
        allowed = ALLOWED_TRANSITIONS.get((MemoryState.EPISODIC, TransitionType.SPLIT))
        assert allowed == frozenset({MemoryState.EPISODIC})


# ---------------------------------------------------------------------------
# PROMOTE and DEMOTE to same state are rejected
# ---------------------------------------------------------------------------


class TestPromoteDemoteToSameStateRejected:
    """PROMOTE and DEMOTE are strictly cross-state: targeting the current state
    must be rejected by both is_valid_transition and StateMachine.validate.
    """

    @pytest.mark.parametrize("state", list(MemoryState))
    def test_promote_to_same_state_is_invalid(self, state: MemoryState) -> None:
        assert not is_valid_transition(state, TransitionType.PROMOTE, state)

    @pytest.mark.parametrize("state", list(MemoryState))
    def test_demote_to_same_state_is_invalid(self, state: MemoryState) -> None:
        assert not is_valid_transition(state, TransitionType.DEMOTE, state)

    @pytest.mark.parametrize("state", list(MemoryState))
    def test_state_machine_rejects_promote_to_same_state(self, state: MemoryState) -> None:
        sm = StateMachine()
        result = sm.validate(state, TransitionType.PROMOTE, state)
        assert not result.ok

    @pytest.mark.parametrize("state", list(MemoryState))
    def test_state_machine_rejects_demote_to_same_state(self, state: MemoryState) -> None:
        sm = StateMachine()
        result = sm.validate(state, TransitionType.DEMOTE, state)
        assert not result.ok

    def test_promote_rejection_reason_is_descriptive(self) -> None:
        sm = StateMachine()
        # EPISODIC --promote--> EPISODIC is the canonical same-state promote case
        result = sm.validate(MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.EPISODIC)
        assert not result.ok
        assert "episodic" in result.reason

    def test_demote_rejection_reason_is_descriptive(self) -> None:
        sm = StateMachine()
        # SEMANTIC --demote--> SEMANTIC is the canonical same-state demote case
        result = sm.validate(MemoryState.SEMANTIC, TransitionType.DEMOTE, MemoryState.SEMANTIC)
        assert not result.ok
        assert "semantic" in result.reason

    def test_merge_and_split_allow_same_state_where_promote_demote_do_not(self) -> None:
        """Contrast: MERGE/SPLIT accept EPISODIC→EPISODIC; PROMOTE/DEMOTE reject it."""
        sm = StateMachine()
        assert sm.validate(MemoryState.EPISODIC, TransitionType.MERGE, MemoryState.EPISODIC).ok
        assert sm.validate(MemoryState.EPISODIC, TransitionType.SPLIT, MemoryState.EPISODIC).ok
        assert not sm.validate(MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.EPISODIC).ok
        assert not sm.validate(MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.EPISODIC).ok


class TestAllInvalidPairsReturnFailureWithDescriptiveReason:
    """Exhaustive sweep: every (state, transition) pair absent from
    ALLOWED_TRANSITIONS must produce ok=False and a reason that names
    both the state and the transition.

    The parametrize list is built at module import time from the full
    Cartesian product minus ALLOWED_TRANSITIONS.keys(), so this test
    automatically stays in sync if new states or transitions are added.
    """

    @pytest.mark.parametrize(
        "state, transition",
        _INVALID_PAIRS,
        ids=[f"{s.value}__{t.value}" for s, t in _INVALID_PAIRS],
    )
    @pytest.mark.parametrize("target", list(MemoryState))
    def test_validate_returns_failure_with_descriptive_reason(
        self, state: MemoryState, transition: TransitionType, target: MemoryState
    ) -> None:
        sm = StateMachine()
        result = sm.validate(state, transition, target)

        assert not result.ok, (
            f"Expected failure for {state.value!r} --{transition.value}--> {target.value!r}"
            f" but got ok=True"
        )
        assert state.value in result.reason, (
            f"reason {result.reason!r} does not mention the current state {state.value!r}"
        )
        assert transition.value in result.reason, (
            f"reason {result.reason!r} does not mention the transition {transition.value!r}"
        )
