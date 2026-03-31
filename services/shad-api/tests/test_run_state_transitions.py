"""Exhaustive transition-matrix and validator-gate tests for run state management.

Follows the patterns in test_taxonomy.py (state machine) and
test_transition_validators.py (precondition gate factories).

Covers:
  - RunState enum: all variants, str values, invalid construction
  - RunStateManager: every valid edge in _VALID_TRANSITIONS passes
  - RunStateManager: every invalid edge raises ValueError with state names in message
  - RunStateManager: state is unchanged after a failed transition
  - RunStateManager: SUCCESS is terminal — blocks all outbound transitions
  - RunStateManager: history grows with transitions and is unchanged on failure
  - RunStateManager: is_pausable is True only in NEEDS_HUMAN
  - HITLCheckpoint.should_trigger: each gate at and below its exact threshold
  - MaxIterationsPolicy.should_continue: iteration and escalation boundaries
  - MaxIterationsPolicy.determine_final_state: all decision branches
"""

from __future__ import annotations

import pytest

from shad.refinement.manager import (
    HITLCheckpoint,
    MaxIterationsPolicy,
    RunState,
    RunStateManager,
)


# ---------------------------------------------------------------------------
# RunState enum
# ---------------------------------------------------------------------------


class TestRunStateEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in RunState}
        assert names == {"PENDING", "RUNNING", "SUCCESS", "PARTIAL", "FAILED", "NEEDS_HUMAN"}

    def test_values_are_lowercase_strings(self) -> None:
        for member in RunState:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    @pytest.mark.parametrize("member,expected_value", [
        (RunState.PENDING,      "pending"),
        (RunState.RUNNING,      "running"),
        (RunState.SUCCESS,      "success"),
        (RunState.PARTIAL,      "partial"),
        (RunState.FAILED,       "failed"),
        (RunState.NEEDS_HUMAN,  "needs_human"),
    ])
    def test_str_returns_value(self, member: RunState, expected_value: str) -> None:
        assert str(member) == expected_value

    @pytest.mark.parametrize("value,expected", [
        ("pending",     RunState.PENDING),
        ("running",     RunState.RUNNING),
        ("success",     RunState.SUCCESS),
        ("partial",     RunState.PARTIAL),
        ("failed",      RunState.FAILED),
        ("needs_human", RunState.NEEDS_HUMAN),
    ])
    def test_construction_from_value_roundtrip(self, value: str, expected: RunState) -> None:
        assert RunState(value) is expected

    @pytest.mark.parametrize("bad", [
        "PENDING", "SUCCESS", "needs-human", "invalid", "", "   ",
    ])
    def test_invalid_value_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError):
            RunState(bad)

    def test_valid_values_cover_all_members(self) -> None:
        for member in RunState:
            assert RunState(member.value) is member


# ---------------------------------------------------------------------------
# All valid / invalid edges in the transition table
# ---------------------------------------------------------------------------

# Every edge declared in _VALID_TRANSITIONS
_VALID_EDGES = [
    (RunState.PENDING,     RunState.RUNNING),
    (RunState.RUNNING,     RunState.SUCCESS),
    (RunState.RUNNING,     RunState.PARTIAL),
    (RunState.RUNNING,     RunState.FAILED),
    (RunState.RUNNING,     RunState.NEEDS_HUMAN),
    (RunState.NEEDS_HUMAN, RunState.RUNNING),
    (RunState.NEEDS_HUMAN, RunState.SUCCESS),
    (RunState.NEEDS_HUMAN, RunState.PARTIAL),
    (RunState.NEEDS_HUMAN, RunState.FAILED),
    (RunState.PARTIAL,     RunState.RUNNING),
    (RunState.FAILED,      RunState.RUNNING),
]

# A representative invalid edge from each source state (self-loops included)
_INVALID_EDGES = [
    # PENDING cannot skip to terminal or intermediate states
    (RunState.PENDING,     RunState.SUCCESS),
    (RunState.PENDING,     RunState.PARTIAL),
    (RunState.PENDING,     RunState.FAILED),
    (RunState.PENDING,     RunState.NEEDS_HUMAN),
    (RunState.PENDING,     RunState.PENDING),
    # RUNNING cannot go backwards or self-loop
    (RunState.RUNNING,     RunState.PENDING),
    (RunState.RUNNING,     RunState.RUNNING),
    # SUCCESS is terminal
    (RunState.SUCCESS,     RunState.RUNNING),
    (RunState.SUCCESS,     RunState.PENDING),
    (RunState.SUCCESS,     RunState.PARTIAL),
    (RunState.SUCCESS,     RunState.FAILED),
    (RunState.SUCCESS,     RunState.NEEDS_HUMAN),
    (RunState.SUCCESS,     RunState.SUCCESS),
    # PARTIAL can only resume to RUNNING
    (RunState.PARTIAL,     RunState.SUCCESS),
    (RunState.PARTIAL,     RunState.FAILED),
    (RunState.PARTIAL,     RunState.NEEDS_HUMAN),
    (RunState.PARTIAL,     RunState.PARTIAL),
    # FAILED can only retry to RUNNING
    (RunState.FAILED,      RunState.SUCCESS),
    (RunState.FAILED,      RunState.PARTIAL),
    (RunState.FAILED,      RunState.NEEDS_HUMAN),
    (RunState.FAILED,      RunState.FAILED),
    # NEEDS_HUMAN cannot go backwards to PENDING
    (RunState.NEEDS_HUMAN, RunState.PENDING),
    (RunState.NEEDS_HUMAN, RunState.NEEDS_HUMAN),
]


class TestRunStateManagerValidTransitions:
    """Every edge in _VALID_TRANSITIONS must succeed without raising."""

    @pytest.mark.parametrize("from_state,to_state", _VALID_EDGES)
    def test_valid_transition_does_not_raise(
        self, from_state: RunState, to_state: RunState
    ) -> None:
        sm = RunStateManager(initial_state=from_state)
        sm.transition_to(to_state)
        assert sm.state is to_state


class TestRunStateManagerInvalidTransitions:
    """Every invalid (from, to) pair must raise ValueError."""

    @pytest.mark.parametrize("from_state,to_state", _INVALID_EDGES)
    def test_invalid_transition_raises_value_error(
        self, from_state: RunState, to_state: RunState
    ) -> None:
        sm = RunStateManager(initial_state=from_state)
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition_to(to_state)

    @pytest.mark.parametrize("from_state,to_state", _INVALID_EDGES)
    def test_error_message_contains_both_state_names(
        self, from_state: RunState, to_state: RunState
    ) -> None:
        sm = RunStateManager(initial_state=from_state)
        with pytest.raises(ValueError) as exc_info:
            sm.transition_to(to_state)
        msg = str(exc_info.value)
        assert from_state.value in msg
        assert to_state.value in msg

    @pytest.mark.parametrize("from_state,to_state", _INVALID_EDGES)
    def test_state_is_unchanged_after_failed_transition(
        self, from_state: RunState, to_state: RunState
    ) -> None:
        sm = RunStateManager(initial_state=from_state)
        with pytest.raises(ValueError):
            sm.transition_to(to_state)
        assert sm.state is from_state


class TestRunStateManagerTerminalState:
    """SUCCESS is terminal — no further transitions are permitted."""

    @pytest.mark.parametrize("target", list(RunState))
    def test_success_blocks_all_outbound_transitions(self, target: RunState) -> None:
        sm = RunStateManager(initial_state=RunState.SUCCESS)
        with pytest.raises(ValueError):
            sm.transition_to(target)


# ---------------------------------------------------------------------------
# RunStateManager — history and is_pausable
# ---------------------------------------------------------------------------


class TestRunStateManagerHistory:
    def test_initial_history_contains_initial_state(self) -> None:
        sm = RunStateManager()
        assert sm._history == [RunState.PENDING]

    def test_custom_initial_state_appears_in_history(self) -> None:
        sm = RunStateManager(initial_state=RunState.RUNNING)
        assert sm._history == [RunState.RUNNING]

    def test_history_grows_with_each_valid_transition(self) -> None:
        sm = RunStateManager()
        sm.transition_to(RunState.RUNNING)
        sm.transition_to(RunState.PARTIAL)
        sm.transition_to(RunState.RUNNING)
        sm.transition_to(RunState.SUCCESS)
        assert sm._history == [
            RunState.PENDING,
            RunState.RUNNING,
            RunState.PARTIAL,
            RunState.RUNNING,
            RunState.SUCCESS,
        ]

    def test_failed_transition_does_not_append_to_history(self) -> None:
        sm = RunStateManager()
        with pytest.raises(ValueError):
            sm.transition_to(RunState.SUCCESS)
        assert sm._history == [RunState.PENDING]

    @pytest.mark.parametrize("state,expected_pausable", [
        (RunState.PENDING,     False),
        (RunState.RUNNING,     False),
        (RunState.SUCCESS,     False),
        (RunState.PARTIAL,     False),
        (RunState.FAILED,      False),
        (RunState.NEEDS_HUMAN, True),
    ])
    def test_is_pausable_only_in_needs_human(
        self, state: RunState, expected_pausable: bool
    ) -> None:
        sm = RunStateManager(initial_state=state)
        assert sm.is_pausable is expected_pausable


# ---------------------------------------------------------------------------
# HITLCheckpoint — precondition gate (threshold boundaries)
# ---------------------------------------------------------------------------


class TestHITLCheckpointGates:
    """Each trigger condition tested independently at its exact threshold boundary."""

    # --- high-impact + depth gate ---

    @pytest.mark.parametrize("depth", [0, 1])
    def test_high_impact_at_shallow_depth_triggers(self, depth: int) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=True,
            depth=depth,
            retrieval_confidence=0.9,
            generation_confidence=0.9,
        ) is True

    @pytest.mark.parametrize("depth", [2, 3, 10])
    def test_high_impact_at_deep_depth_does_not_trigger_alone(self, depth: int) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=True,
            depth=depth,
            retrieval_confidence=0.9,
            generation_confidence=0.9,
        ) is False

    def test_not_high_impact_at_shallow_depth_does_not_trigger(self) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=0,
            retrieval_confidence=0.9,
            generation_confidence=0.9,
        ) is False

    # --- retrieval confidence gate ---

    @pytest.mark.parametrize("confidence", [0.0, 0.1, 0.44])
    def test_low_retrieval_confidence_triggers(self, confidence: float) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=confidence,
            generation_confidence=0.9,
        ) is True

    def test_retrieval_confidence_at_threshold_does_not_trigger(self) -> None:
        """Exactly at 0.45 is NOT below threshold — should not trigger alone."""
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=HITLCheckpoint.LOW_RETRIEVAL_CONFIDENCE,
            generation_confidence=0.9,
        ) is False

    @pytest.mark.parametrize("confidence", [0.45, 0.5, 0.8, 1.0])
    def test_retrieval_confidence_at_or_above_threshold_does_not_trigger_alone(
        self, confidence: float
    ) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=confidence,
            generation_confidence=0.9,
        ) is False

    # --- generation confidence gate ---

    @pytest.mark.parametrize("confidence", [0.0, 0.1, 0.54])
    def test_low_generation_confidence_triggers(self, confidence: float) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=0.9,
            generation_confidence=confidence,
        ) is True

    def test_generation_confidence_at_threshold_does_not_trigger(self) -> None:
        """Exactly at 0.55 is NOT below threshold — should not trigger alone."""
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=0.9,
            generation_confidence=HITLCheckpoint.LOW_GENERATION_CONFIDENCE,
        ) is False

    @pytest.mark.parametrize("confidence", [0.55, 0.7, 0.9, 1.0])
    def test_generation_confidence_at_or_above_threshold_does_not_trigger_alone(
        self, confidence: float
    ) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=0.9,
            generation_confidence=confidence,
        ) is False

    # --- threshold constant values ---

    def test_low_retrieval_threshold_is_0_45(self) -> None:
        assert HITLCheckpoint.LOW_RETRIEVAL_CONFIDENCE == 0.45

    def test_low_generation_threshold_is_0_55(self) -> None:
        assert HITLCheckpoint.LOW_GENERATION_CONFIDENCE == 0.55

    # --- no trigger when all conditions healthy ---

    def test_no_trigger_when_all_conditions_healthy(self) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=3,
            retrieval_confidence=0.8,
            generation_confidence=0.8,
        ) is False

    # --- both confidence gates failing simultaneously ---

    def test_both_confidence_low_still_triggers(self) -> None:
        assert HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=5,
            retrieval_confidence=0.3,
            generation_confidence=0.3,
        ) is True


# ---------------------------------------------------------------------------
# MaxIterationsPolicy — should_continue and determine_final_state
# ---------------------------------------------------------------------------


class TestMaxIterationsPolicyGates:
    """Boundary tests for should_continue and all branches of determine_final_state."""

    # --- should_continue: iteration boundary ---

    @pytest.mark.parametrize("iterations,expected", [
        (0, True),
        (1, True),
        (2, False),   # == max_local_repairs_per_leaf
        (5, False),
    ])
    def test_should_continue_iteration_boundary(
        self, iterations: int, expected: bool
    ) -> None:
        policy = MaxIterationsPolicy(max_local_repairs_per_leaf=2, max_escalations_per_run=10)
        assert policy.should_continue(iterations=iterations, escalations=0) is expected

    # --- should_continue: escalation boundary ---

    @pytest.mark.parametrize("escalations,expected", [
        (0,  True),
        (9,  True),
        (10, False),  # == max_escalations_per_run
        (15, False),
    ])
    def test_should_continue_escalation_boundary(
        self, escalations: int, expected: bool
    ) -> None:
        policy = MaxIterationsPolicy(max_local_repairs_per_leaf=2, max_escalations_per_run=10)
        assert policy.should_continue(iterations=0, escalations=escalations) is expected

    def test_should_continue_stops_when_either_limit_reached(self) -> None:
        policy = MaxIterationsPolicy(max_local_repairs_per_leaf=2, max_escalations_per_run=5)
        # iterations at limit
        assert policy.should_continue(iterations=2, escalations=0) is False
        # escalations at limit
        assert policy.should_continue(iterations=0, escalations=5) is False
        # both under limit
        assert policy.should_continue(iterations=1, escalations=4) is True

    # --- determine_final_state: all branches ---

    def test_verification_passed_returns_success(self) -> None:
        policy = MaxIterationsPolicy()
        state = policy.determine_final_state(
            is_high_impact=True,
            has_artifacts=False,
            verification_passed=True,
        )
        assert state is RunState.SUCCESS

    def test_verification_passed_overrides_other_flags(self) -> None:
        """verification_passed=True always wins regardless of other args."""
        policy = MaxIterationsPolicy()
        for is_high_impact in (True, False):
            for has_artifacts in (True, False):
                state = policy.determine_final_state(
                    is_high_impact=is_high_impact,
                    has_artifacts=has_artifacts,
                    verification_passed=True,
                )
                assert state is RunState.SUCCESS

    def test_high_impact_with_artifacts_returns_needs_human(self) -> None:
        policy = MaxIterationsPolicy()
        state = policy.determine_final_state(
            is_high_impact=True,
            has_artifacts=True,
            verification_passed=False,
        )
        assert state is RunState.NEEDS_HUMAN

    def test_not_high_impact_with_artifacts_and_advisory_returns_partial(self) -> None:
        policy = MaxIterationsPolicy()
        state = policy.determine_final_state(
            is_high_impact=False,
            has_artifacts=True,
            verification_passed=False,
            verification_advisory=True,
        )
        assert state is RunState.PARTIAL

    def test_not_high_impact_with_artifacts_not_advisory_returns_needs_human(self) -> None:
        policy = MaxIterationsPolicy()
        state = policy.determine_final_state(
            is_high_impact=False,
            has_artifacts=True,
            verification_passed=False,
            verification_advisory=False,
        )
        assert state is RunState.NEEDS_HUMAN

    def test_no_artifacts_returns_failed(self) -> None:
        policy = MaxIterationsPolicy()
        state = policy.determine_final_state(
            is_high_impact=False,
            has_artifacts=False,
            verification_passed=False,
        )
        assert state is RunState.FAILED

    def test_high_impact_no_artifacts_returns_failed(self) -> None:
        """high_impact=True but no artifacts — falls through to FAILED."""
        policy = MaxIterationsPolicy()
        state = policy.determine_final_state(
            is_high_impact=True,
            has_artifacts=False,
            verification_passed=False,
        )
        assert state is RunState.FAILED
