"""Iterative refinement and run state management.

Per SPEC.md Section 2.8:
- Run states: PENDING → RUNNING → SUCCESS | PARTIAL | FAILED | NEEDS_HUMAN
- Delta verification on resume (only re-verify changed context)
- Max iterations policy with tiered fallback
- HITL checkpoints for high-impact decisions
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RunState(str, Enum):
    """States of a run.

    Per SPEC.md Section 2.8.1:
    - PENDING: Not yet started
    - RUNNING: Execution in progress
    - SUCCESS: Meets acceptance criteria
    - PARTIAL: Produced artifacts but did not meet criteria
    - FAILED: Could not produce meaningful artifacts
    - NEEDS_HUMAN: Paused, context preserved, awaiting human input
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


# Valid state transitions
_VALID_TRANSITIONS: dict[RunState, list[RunState]] = {
    RunState.PENDING: [RunState.RUNNING],
    RunState.RUNNING: [RunState.SUCCESS, RunState.PARTIAL, RunState.FAILED, RunState.NEEDS_HUMAN],
    RunState.NEEDS_HUMAN: [RunState.RUNNING, RunState.SUCCESS, RunState.PARTIAL, RunState.FAILED],
    RunState.PARTIAL: [RunState.RUNNING],  # Can resume
    RunState.FAILED: [RunState.RUNNING],  # Can retry
    RunState.SUCCESS: [],  # Terminal
}


class RunStateManager:
    """Manages run state transitions.

    Enforces valid state transitions and tracks state history.
    """

    def __init__(self, initial_state: RunState = RunState.PENDING) -> None:
        self._state = initial_state
        self._history: list[RunState] = [initial_state]

    @property
    def state(self) -> RunState:
        """Get current state."""
        return self._state

    @property
    def is_pausable(self) -> bool:
        """Check if current state is pausable (can await human input)."""
        return self._state == RunState.NEEDS_HUMAN

    def transition_to(self, new_state: RunState) -> None:
        """Transition to a new state.

        Raises ValueError if transition is invalid.
        """
        valid = _VALID_TRANSITIONS.get(self._state, [])
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition from {self._state.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid]}"
            )

        logger.info(f"State transition: {self._state.value} → {new_state.value}")
        self._state = new_state
        self._history.append(new_state)


@dataclass
class NodeStaleInfo:
    """Information about a node's context dependencies."""

    node_id: str
    used_notes: list[str]
    note_hashes: dict[str, str]
    subset_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Compute fingerprint from note hashes."""
        # Sort for determinism
        sorted_items = sorted(self.note_hashes.items())
        hash_str = ":".join(f"{path}={hash_}" for path, hash_ in sorted_items)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class DeltaVerifier:
    """Manages delta verification for resume.

    Per SPEC.md Section 2.8.2:
    - Tracks used_notes[] and used_note_hashes{} for each node
    - On resume, checks current vault against stored hashes
    - Stale nodes undergo re-verification or re-execution
    """

    def __init__(self) -> None:
        self._node_info: dict[str, NodeStaleInfo] = {}
        self._children: dict[str, list[str]] = {}
        self._marked_for_replay: set[str] = set()

    def track_node(
        self,
        node_id: str,
        used_notes: list[str],
        note_hashes: dict[str, str],
    ) -> None:
        """Track context used by a node."""
        info = NodeStaleInfo(
            node_id=node_id,
            used_notes=used_notes,
            note_hashes=note_hashes,
        )
        info.subset_fingerprint = info.compute_fingerprint()
        self._node_info[node_id] = info

    def get_node_info(self, node_id: str) -> NodeStaleInfo | None:
        """Get node info."""
        return self._node_info.get(node_id)

    def get_fingerprint(self, node_id: str) -> str | None:
        """Get subset fingerprint for a node."""
        info = self._node_info.get(node_id)
        return info.subset_fingerprint if info else None

    def is_stale(
        self,
        node_id: str,
        current_hashes: dict[str, str],
    ) -> bool:
        """Check if a node is stale (context changed).

        A node is stale if any of its used note hashes differ.
        """
        info = self._node_info.get(node_id)
        if not info:
            return False  # Unknown node, not stale

        for path, stored_hash in info.note_hashes.items():
            current_hash = current_hashes.get(path)
            if current_hash and current_hash != stored_hash:
                return True

        return False

    def get_stale_nodes(self, current_hashes: dict[str, str]) -> list[str]:
        """Get list of stale node IDs."""
        stale = []
        for node_id in self._node_info:
            if self.is_stale(node_id, current_hashes):
                stale.append(node_id)
        return stale

    def set_children(self, parent_id: str, children: list[str]) -> None:
        """Set children for a node (for subtree replay)."""
        self._children[parent_id] = children

    def mark_for_replay(self, node_id: str) -> None:
        """Mark a specific node for replay."""
        self._marked_for_replay.add(node_id)

    def mark_subtree_for_replay(self, root_id: str) -> None:
        """Mark a subtree for replay (node + all descendants)."""
        self._marked_for_replay.add(root_id)

        # Add all descendants
        to_visit = list(self._children.get(root_id, []))
        while to_visit:
            node_id = to_visit.pop()
            self._marked_for_replay.add(node_id)
            to_visit.extend(self._children.get(node_id, []))

    def get_nodes_to_replay(self) -> list[str]:
        """Get list of nodes marked for replay."""
        return list(self._marked_for_replay)


class MaxIterationsPolicy:
    """Policy for max iterations and final state determination.

    Per SPEC.md Section 2.8.3:
    - max_local_repairs_per_leaf: 2
    - max_escalations_per_run: 10
    - Tiered fallback for final state
    """

    def __init__(
        self,
        max_local_repairs_per_leaf: int = 2,
        max_escalations_per_run: int = 10,
    ) -> None:
        self.max_local_repairs_per_leaf = max_local_repairs_per_leaf
        self.max_escalations_per_run = max_escalations_per_run

    def should_continue(
        self,
        iterations: int,
        escalations: int,
    ) -> bool:
        """Check if iterations should continue."""
        if iterations >= self.max_local_repairs_per_leaf:
            return False
        if escalations >= self.max_escalations_per_run:
            return False
        return True

    def determine_final_state(
        self,
        is_high_impact: bool,
        has_artifacts: bool,
        verification_passed: bool,
        verification_advisory: bool = False,
    ) -> RunState:
        """Determine final state based on conditions.

        Per SPEC.md Section 2.8.3:
        - High-impact OR substantial artifacts → NEEDS_HUMAN
        - Low-risk, verification advisory → PARTIAL
        - Cannot proceed safely OR no artifacts → FAILED
        """
        if verification_passed:
            return RunState.SUCCESS

        if is_high_impact and has_artifacts:
            return RunState.NEEDS_HUMAN

        if has_artifacts:
            if verification_advisory:
                return RunState.PARTIAL
            else:
                return RunState.NEEDS_HUMAN

        return RunState.FAILED


class IterationManager:
    """Manages iteration tracking and error signatures.

    Prevents infinite loops by tracking error signatures.
    """

    def __init__(self, max_iterations: int = 2) -> None:
        self.max_iterations = max_iterations
        self._iterations: dict[str, int] = {}
        self._error_signatures: set[str] = set()

    def record_iteration(self, node_id: str, success: bool) -> None:
        """Record an iteration for a node."""
        if node_id not in self._iterations:
            self._iterations[node_id] = 0
        self._iterations[node_id] += 1

    def get_iteration_count(self, node_id: str) -> int:
        """Get iteration count for a node."""
        return self._iterations.get(node_id, 0)

    def should_repair(self, node_id: str) -> bool:
        """Check if repair should be attempted."""
        return self.get_iteration_count(node_id) < self.max_iterations

    def compute_error_signature(
        self,
        error_class: str,
        files: list[str],
        key_lines: list[str],
    ) -> str:
        """Compute error signature for dedup.

        Per SPEC.md Section 2.6.3:
        Hash (error_class, primary_files, key_lines) to detect repeated same-error.
        """
        data = f"{error_class}:{':'.join(sorted(files))}:{':'.join(key_lines[:5])}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def is_repeated_error(self, signature: str) -> bool:
        """Check if error has been seen before."""
        return signature in self._error_signatures

    def record_error(self, signature: str) -> None:
        """Record an error signature."""
        self._error_signatures.add(signature)


@dataclass
class HITLCheckpoint:
    """Human-in-the-loop checkpoint.

    Per SPEC.md Section 2.9:
    - Triggers on high-impact decisions
    - Triggers on low confidence
    - Presents options: Approve, Edit, Skip, Stop
    """

    node_id: str
    reason: str
    summary: str
    confidence: float
    decision_to_approve: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Thresholds per SPEC.md Section 2.9.1
    LOW_RETRIEVAL_CONFIDENCE = 0.45
    LOW_GENERATION_CONFIDENCE = 0.55

    @classmethod
    def should_trigger(
        cls,
        is_high_impact: bool,
        depth: int,
        retrieval_confidence: float,
        generation_confidence: float,
    ) -> bool:
        """Determine if checkpoint should trigger.

        Per SPEC.md Section 2.9.1:
        - High-impact node at depth ≤ 1
        - Low retrieval confidence + high-impact
        - Low generation confidence + repeated repair failures
        """
        # High-impact at shallow depth
        if is_high_impact and depth <= 1:
            return True

        # Low retrieval confidence
        if retrieval_confidence < cls.LOW_RETRIEVAL_CONFIDENCE:
            return True

        # Low generation confidence
        if generation_confidence < cls.LOW_GENERATION_CONFIDENCE:
            return True

        return False

    def get_options(self) -> dict[str, str]:
        """Get checkpoint options.

        Per SPEC.md Section 2.9.2:
        - Approve: Accept as-is
        - Edit: Modify before continuing
        - Skip: Skip this node
        - Stop: Stop the entire run
        """
        return {
            "approve": "Accept and continue",
            "edit": "Edit before continuing",
            "hint": "Provide hint for improvement",
            "skip": "Skip this node",
            "stop": "Stop the run",
        }


@dataclass
class FinalResult:
    """Final result of a run, always including best-effort artifacts.

    Per SPEC.md Section 2.8.3:
    Always return:
    - Best current artifacts
    - Failure report (what criteria failed, what was tried)
    - Suggested next inputs from human
    """

    state: RunState
    artifacts: dict[str, str]
    diagnostics: list[str]
    suggestions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def create_final_result(
    state: RunState,
    artifacts: dict[str, str],
    diagnostics: list[str],
    suggestions: list[str],
    **metadata: Any,
) -> FinalResult:
    """Create a final result with all required information."""
    return FinalResult(
        state=state,
        artifacts=artifacts,
        diagnostics=diagnostics,
        suggestions=suggestions,
        metadata=dict(metadata),
    )
