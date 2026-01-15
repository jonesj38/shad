"""Tests for iterative refinement and run state management.

Per SPEC.md Section 2.8:
- Run states: PENDING → RUNNING → SUCCESS | PARTIAL | FAILED | NEEDS_HUMAN
- Delta verification on resume (only re-verify changed context)
- Max iterations policy with tiered fallback
- HITL checkpoints for high-impact decisions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shad.refinement.manager import (
    DeltaVerifier,
    HITLCheckpoint,
    IterationManager,
    MaxIterationsPolicy,
    NodeStaleInfo,
    RunStateManager,
    RunState,
)


class TestRunStateManager:
    """Tests for run state management."""

    def test_initial_state(self) -> None:
        """Test that runs start in PENDING state."""
        manager = RunStateManager()
        assert manager.state == RunState.PENDING

    def test_state_transitions(self) -> None:
        """Test valid state transitions."""
        manager = RunStateManager()

        # PENDING → RUNNING
        manager.transition_to(RunState.RUNNING)
        assert manager.state == RunState.RUNNING

        # RUNNING → SUCCESS
        manager.transition_to(RunState.SUCCESS)
        assert manager.state == RunState.SUCCESS

    def test_invalid_transition(self) -> None:
        """Test that invalid transitions raise error."""
        manager = RunStateManager()

        # Cannot go from PENDING to SUCCESS directly
        with pytest.raises(ValueError, match="Invalid transition"):
            manager.transition_to(RunState.SUCCESS)

    def test_needs_human_state(self) -> None:
        """Test NEEDS_HUMAN pausable state."""
        manager = RunStateManager()
        manager.transition_to(RunState.RUNNING)
        manager.transition_to(RunState.NEEDS_HUMAN)

        assert manager.state == RunState.NEEDS_HUMAN
        assert manager.is_pausable is True

    def test_resume_from_needs_human(self) -> None:
        """Test resuming from NEEDS_HUMAN state."""
        manager = RunStateManager()
        manager.transition_to(RunState.RUNNING)
        manager.transition_to(RunState.NEEDS_HUMAN)

        # Can resume to RUNNING
        manager.transition_to(RunState.RUNNING)
        assert manager.state == RunState.RUNNING


class TestDeltaVerifier:
    """Tests for delta verification on resume."""

    @pytest.fixture
    def verifier(self) -> DeltaVerifier:
        """Create delta verifier."""
        return DeltaVerifier()

    def test_track_used_notes(self, verifier: DeltaVerifier) -> None:
        """Test tracking used notes for a node."""
        verifier.track_node(
            node_id="node_123",
            used_notes=["Patterns/Auth.md", "Docs/API.md"],
            note_hashes={"Patterns/Auth.md": "abc123", "Docs/API.md": "def456"},
        )

        info = verifier.get_node_info("node_123")
        assert info is not None
        assert len(info.used_notes) == 2
        assert info.note_hashes["Patterns/Auth.md"] == "abc123"

    def test_compute_subset_fingerprint(self, verifier: DeltaVerifier) -> None:
        """Test computing subset fingerprint."""
        verifier.track_node(
            node_id="node_123",
            used_notes=["a.md", "b.md"],
            note_hashes={"a.md": "hash1", "b.md": "hash2"},
        )

        fingerprint = verifier.get_fingerprint("node_123")
        assert fingerprint is not None
        assert len(fingerprint) > 0

    def test_detect_stale_node(self, verifier: DeltaVerifier) -> None:
        """Test detecting stale nodes when context changed."""
        # Track original state
        verifier.track_node(
            node_id="node_123",
            used_notes=["a.md"],
            note_hashes={"a.md": "original_hash"},
        )

        # Check with changed hash
        is_stale = verifier.is_stale(
            node_id="node_123",
            current_hashes={"a.md": "new_hash"},
        )

        assert is_stale is True

    def test_not_stale_when_unchanged(self, verifier: DeltaVerifier) -> None:
        """Test that unchanged context is not stale."""
        verifier.track_node(
            node_id="node_123",
            used_notes=["a.md"],
            note_hashes={"a.md": "same_hash"},
        )

        is_stale = verifier.is_stale(
            node_id="node_123",
            current_hashes={"a.md": "same_hash"},
        )

        assert is_stale is False


class TestMaxIterationsPolicy:
    """Tests for max iterations policy."""

    def test_default_policy(self) -> None:
        """Test default max iterations policy."""
        policy = MaxIterationsPolicy()

        # Default should allow multiple iterations
        assert policy.max_local_repairs_per_leaf == 2
        assert policy.max_escalations_per_run == 10

    def test_should_continue_iterations(self) -> None:
        """Test checking if iterations should continue."""
        policy = MaxIterationsPolicy()

        # Should continue with low iteration count
        assert policy.should_continue(iterations=1, escalations=0) is True

        # Should stop when max reached
        assert policy.should_continue(iterations=3, escalations=0) is False

    def test_determine_final_state_high_impact(self) -> None:
        """Test final state for high-impact task with artifacts."""
        policy = MaxIterationsPolicy()

        state = policy.determine_final_state(
            is_high_impact=True,
            has_artifacts=True,
            verification_passed=False,
        )

        assert state == RunState.NEEDS_HUMAN

    def test_determine_final_state_low_risk(self) -> None:
        """Test final state for low-risk task with verification advisory."""
        policy = MaxIterationsPolicy()

        state = policy.determine_final_state(
            is_high_impact=False,
            has_artifacts=True,
            verification_passed=False,
            verification_advisory=True,
        )

        assert state == RunState.PARTIAL

    def test_determine_final_state_no_artifacts(self) -> None:
        """Test final state when no artifacts produced."""
        policy = MaxIterationsPolicy()

        state = policy.determine_final_state(
            is_high_impact=False,
            has_artifacts=False,
            verification_passed=False,
        )

        assert state == RunState.FAILED


class TestIterationManager:
    """Tests for iteration management."""

    @pytest.fixture
    def manager(self) -> IterationManager:
        """Create iteration manager."""
        return IterationManager()

    def test_track_iteration(self, manager: IterationManager) -> None:
        """Test tracking iterations."""
        manager.record_iteration(node_id="node_1", success=False)
        manager.record_iteration(node_id="node_1", success=False)

        assert manager.get_iteration_count("node_1") == 2

    def test_should_repair(self, manager: IterationManager) -> None:
        """Test deciding if repair should be attempted."""
        # First iteration - should repair
        assert manager.should_repair("node_1") is True

        # After max iterations - should not repair
        manager.record_iteration(node_id="node_1", success=False)
        manager.record_iteration(node_id="node_1", success=False)
        assert manager.should_repair("node_1") is False

    def test_error_signature_tracking(self, manager: IterationManager) -> None:
        """Test tracking error signatures to prevent infinite loops."""
        error_sig = manager.compute_error_signature(
            error_class="syntax",
            files=["src/api.ts"],
            key_lines=["line 10: unexpected token"],
        )

        # First occurrence
        is_repeat = manager.is_repeated_error(error_sig)
        assert is_repeat is False
        manager.record_error(error_sig)

        # Second occurrence
        is_repeat = manager.is_repeated_error(error_sig)
        assert is_repeat is True


class TestHITLCheckpoint:
    """Tests for human-in-the-loop checkpoints."""

    def test_checkpoint_creation(self) -> None:
        """Test creating a HITL checkpoint."""
        checkpoint = HITLCheckpoint(
            node_id="node_123",
            reason="high_impact_decision",
            summary="Decision about database schema",
            confidence=0.45,
        )

        assert checkpoint.node_id == "node_123"
        assert checkpoint.reason == "high_impact_decision"
        assert checkpoint.confidence == 0.45

    def test_should_checkpoint_high_impact(self) -> None:
        """Test that high-impact nodes trigger checkpoints."""
        should_checkpoint = HITLCheckpoint.should_trigger(
            is_high_impact=True,
            depth=0,
            retrieval_confidence=0.6,
            generation_confidence=0.7,
        )

        assert should_checkpoint is True

    def test_should_checkpoint_low_confidence(self) -> None:
        """Test that low confidence triggers checkpoints."""
        should_checkpoint = HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=2,
            retrieval_confidence=0.3,  # Below threshold
            generation_confidence=0.7,
        )

        assert should_checkpoint is True

    def test_no_checkpoint_normal_node(self) -> None:
        """Test that normal nodes don't trigger checkpoints."""
        should_checkpoint = HITLCheckpoint.should_trigger(
            is_high_impact=False,
            depth=3,
            retrieval_confidence=0.8,
            generation_confidence=0.8,
        )

        assert should_checkpoint is False

    def test_checkpoint_options(self) -> None:
        """Test checkpoint presents correct options."""
        checkpoint = HITLCheckpoint(
            node_id="node_123",
            reason="low_confidence",
            summary="Generated auth implementation",
            confidence=0.4,
        )

        options = checkpoint.get_options()

        assert "approve" in options
        assert "edit" in options
        assert "skip" in options
        assert "stop" in options


class TestSelectiveReplay:
    """Tests for selective replay on resume."""

    @pytest.fixture
    def verifier(self) -> DeltaVerifier:
        """Create verifier with tracked nodes."""
        verifier = DeltaVerifier()
        verifier.track_node("node_1", ["a.md"], {"a.md": "hash1"})
        verifier.track_node("node_2", ["b.md"], {"b.md": "hash2"})
        verifier.track_node("node_3", ["c.md"], {"c.md": "hash3"})
        return verifier

    def test_get_stale_nodes(self, verifier: DeltaVerifier) -> None:
        """Test getting list of stale nodes."""
        # node_1 changed, node_2 unchanged
        current_hashes = {
            "a.md": "new_hash",  # Changed
            "b.md": "hash2",  # Unchanged
            "c.md": "hash3",  # Unchanged
        }

        stale_nodes = verifier.get_stale_nodes(current_hashes)

        assert "node_1" in stale_nodes
        assert "node_2" not in stale_nodes
        assert "node_3" not in stale_nodes

    def test_selective_replay_specific_node(self, verifier: DeltaVerifier) -> None:
        """Test marking specific node for replay."""
        verifier.mark_for_replay("node_2")

        to_replay = verifier.get_nodes_to_replay()
        assert "node_2" in to_replay

    def test_selective_replay_subtree(self, verifier: DeltaVerifier) -> None:
        """Test marking subtree for replay."""
        # Set up node hierarchy
        verifier.set_children("node_1", ["node_2", "node_3"])

        verifier.mark_subtree_for_replay("node_1")

        to_replay = verifier.get_nodes_to_replay()
        assert "node_1" in to_replay
        assert "node_2" in to_replay
        assert "node_3" in to_replay


class TestBestEffortArtifacts:
    """Tests for always returning best-effort artifacts."""

    def test_partial_result_includes_artifacts(self) -> None:
        """Test that PARTIAL state includes best artifacts."""
        from shad.refinement.manager import create_final_result

        result = create_final_result(
            state=RunState.PARTIAL,
            artifacts={"src/api.ts": "partial implementation"},
            diagnostics=["Type check failed"],
            suggestions=["Fix type mismatch in getUser"],
        )

        assert result.state == RunState.PARTIAL
        assert len(result.artifacts) > 0
        assert len(result.diagnostics) > 0
        assert len(result.suggestions) > 0

    def test_failed_result_includes_diagnostics(self) -> None:
        """Test that FAILED state includes diagnostics."""
        from shad.refinement.manager import create_final_result

        result = create_final_result(
            state=RunState.FAILED,
            artifacts={},
            diagnostics=["Could not parse requirements", "No valid implementation generated"],
            suggestions=["Clarify the task requirements"],
        )

        assert result.state == RunState.FAILED
        assert len(result.diagnostics) > 0
        assert len(result.suggestions) > 0
