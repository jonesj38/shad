"""Memory lifecycle taxonomy: states and transitions.

Defines the valid states a memory item can occupy and the transition types
that move it between states.

  ┌──────────┐  promote   ┌──────────┐  promote   ┌───────────┐
  │ episodic │ ─────────▶ │ semantic │ ─────────▶ │ procedural│
  └──────────┘            └──────────┘            └───────────┘
       │  demote                │  demote               │  demote
       ▼                        ▼                        ▼
  ┌─────────┐             ┌─────────┐            ┌──────────┐
  │ dormant │             │ dormant │            │ archival │
  └─────────┘             └─────────┘            └──────────┘

Merge collapses two items of the same state into one.
Split creates two items of the same state from one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final


class MemoryState(StrEnum):
    """Lifecycle state of a memory item.

    - EPISODIC:    Event-bound / session-scoped (raw, unvalidated).
    - SEMANTIC:    Consolidated factual knowledge (timeless, validated).
    - PROCEDURAL:  Distilled skill / executable pattern (highest confidence).
    - ARCHIVAL:    Retired from active retrieval; preserved for audit.
    - DORMANT:     Temporarily suppressed; eligible for reactivation.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ARCHIVAL = "archival"
    DORMANT = "dormant"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> MemoryState:
        """Parse a string into the matching MemoryState variant.

        Accepts the enum value (``"episodic"``) or member name
        (``"EPISODIC"``), case-insensitively.

        Raises:
            ValueError: if *value* is ``None``, empty, whitespace-only, or
                does not match any variant.
        """
        if not value or not value.strip():
            valid = ", ".join(f'"{m.value}"' for m in cls)
            raise ValueError(f"Invalid MemoryState {value!r}. Valid values: {valid}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        valid = ", ".join(f'"{m.value}"' for m in cls)
        raise ValueError(f"Invalid MemoryState {value!r}. Valid values: {valid}")

    @classmethod
    def active_states(cls) -> list[MemoryState]:
        """Return states that participate in active retrieval."""
        return [cls.EPISODIC, cls.SEMANTIC, cls.PROCEDURAL]


class MemoryTypeEnum(StrEnum):
    """Cognitive category of a memory item.

    Classifies *what kind* of knowledge a memory holds, independent of its
    lifecycle state (:class:`MemoryState`).

    - EPISODIC:    Specific events or experiences, anchored to a time and context.
    - SEMANTIC:    General facts and concepts, decoupled from when they were learned.
    - WORKING:     Actively-held, short-lived context for the current task or session.
    - PROCEDURAL:  How-to knowledge: skills, patterns, and executable workflows.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> MemoryTypeEnum:
        """Parse a string into the matching MemoryTypeEnum variant.

        Accepts the enum value (``"episodic"``) or member name
        (``"EPISODIC"``), case-insensitively.

        Raises:
            ValueError: if *value* is ``None``, empty, whitespace-only, or
                does not match any variant.
        """
        if not value or not value.strip():
            valid = ", ".join(f'"{m.value}"' for m in cls)
            raise ValueError(f"Invalid MemoryTypeEnum {value!r}. Valid values: {valid}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        valid = ", ".join(f'"{m.value}"' for m in cls)
        raise ValueError(f"Invalid MemoryTypeEnum {value!r}. Valid values: {valid}")


class TransitionType(StrEnum):
    """How a memory item moves between MemoryState values.

    - PROMOTE: Elevate to a higher-confidence / more abstract state
               (e.g. EPISODIC → SEMANTIC, SEMANTIC → PROCEDURAL).
    - DEMOTE:  Lower to a less active or less confident state
               (e.g. SEMANTIC → DORMANT, PROCEDURAL → ARCHIVAL).
    - MERGE:   Collapse two items of the same state into one consolidated item.
    - SPLIT:   Decompose one item into two more-specific items of the same state.
    """

    PROMOTE = "promote"
    DEMOTE = "demote"
    MERGE = "merge"
    SPLIT = "split"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> TransitionType:
        """Parse a string into the matching TransitionType variant.

        Accepts the enum value (``"promote"``) or member name
        (``"PROMOTE"``), case-insensitively.

        Raises:
            ValueError: if *value* is ``None``, empty, whitespace-only, or
                does not match any variant.
        """
        if not value or not value.strip():
            valid = ", ".join(f'"{m.value}"' for m in cls)
            raise ValueError(f"Invalid TransitionType {value!r}. Valid values: {valid}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        valid = ", ".join(f'"{m.value}"' for m in cls)
        raise ValueError(f"Invalid TransitionType {value!r}. Valid values: {valid}")


# ---------------------------------------------------------------------------
# Allowed-transition map
# ---------------------------------------------------------------------------

# Maps (current_state, transition_type) → frozenset of valid target states.
#
# Derived directly from the lifecycle diagram at the top of this module:
#
#   PROMOTE  EPISODIC → SEMANTIC, SEMANTIC → PROCEDURAL
#   DEMOTE   EPISODIC → DORMANT,  SEMANTIC → DORMANT,  PROCEDURAL → ARCHIVAL
#   MERGE    any state collapses two items into one item of the *same* state
#   SPLIT    any state decomposes one item into two items of the *same* state
#
# Absent keys mean no valid target exists for that (state, transition) pair.

ALLOWED_TRANSITIONS: Final[
    dict[tuple[MemoryState, TransitionType], frozenset[MemoryState]]
] = {
    # --- PROMOTE ---
    (MemoryState.EPISODIC, TransitionType.PROMOTE): frozenset({MemoryState.SEMANTIC}),
    (MemoryState.SEMANTIC, TransitionType.PROMOTE): frozenset({MemoryState.PROCEDURAL}),
    # --- DEMOTE ---
    (MemoryState.EPISODIC, TransitionType.DEMOTE): frozenset({MemoryState.DORMANT}),
    (MemoryState.SEMANTIC, TransitionType.DEMOTE): frozenset({MemoryState.DORMANT}),
    (MemoryState.PROCEDURAL, TransitionType.DEMOTE): frozenset({MemoryState.ARCHIVAL}),
    # --- MERGE (same-state collapse) ---
    (MemoryState.EPISODIC, TransitionType.MERGE): frozenset({MemoryState.EPISODIC}),
    (MemoryState.SEMANTIC, TransitionType.MERGE): frozenset({MemoryState.SEMANTIC}),
    (MemoryState.PROCEDURAL, TransitionType.MERGE): frozenset({MemoryState.PROCEDURAL}),
    (MemoryState.ARCHIVAL, TransitionType.MERGE): frozenset({MemoryState.ARCHIVAL}),
    (MemoryState.DORMANT, TransitionType.MERGE): frozenset({MemoryState.DORMANT}),
    # --- SPLIT (same-state decomposition) ---
    (MemoryState.EPISODIC, TransitionType.SPLIT): frozenset({MemoryState.EPISODIC}),
    (MemoryState.SEMANTIC, TransitionType.SPLIT): frozenset({MemoryState.SEMANTIC}),
    (MemoryState.PROCEDURAL, TransitionType.SPLIT): frozenset({MemoryState.PROCEDURAL}),
    (MemoryState.ARCHIVAL, TransitionType.SPLIT): frozenset({MemoryState.ARCHIVAL}),
    (MemoryState.DORMANT, TransitionType.SPLIT): frozenset({MemoryState.DORMANT}),
}


# ---------------------------------------------------------------------------
# Allowed-type-transition map
# ---------------------------------------------------------------------------

# Maps (current_type, transition_type) → frozenset of valid target MemoryTypeEnum values.
#
# Cognitive type hierarchy (PROMOTE direction):
#
#   WORKING → EPISODIC → SEMANTIC → PROCEDURAL
#
#   PROMOTE  WORKING → EPISODIC  (active context crystallises into a dated memory)
#            EPISODIC → SEMANTIC (repeated events generalise into timeless facts)
#            SEMANTIC → PROCEDURAL (well-understood facts become executable skills)
#   DEMOTE   PROCEDURAL → SEMANTIC (skill degrades back to knowledge)
#            SEMANTIC → EPISODIC  (a fact is re-grounded to a specific event)
#            EPISODIC → WORKING   (an old event is pulled back into active context)
#   MERGE    any type collapses two items into one item of the *same* type
#   SPLIT    any type decomposes one item into two items of the *same* type
#
# Absent keys mean no valid target exists for that (type, transition) pair.

ALLOWED_TYPE_TRANSITIONS: Final[
    dict[tuple[MemoryTypeEnum, TransitionType], frozenset[MemoryTypeEnum]]
] = {
    # --- PROMOTE ---
    (MemoryTypeEnum.WORKING, TransitionType.PROMOTE): frozenset({MemoryTypeEnum.EPISODIC}),
    (MemoryTypeEnum.EPISODIC, TransitionType.PROMOTE): frozenset({MemoryTypeEnum.SEMANTIC}),
    (MemoryTypeEnum.SEMANTIC, TransitionType.PROMOTE): frozenset({MemoryTypeEnum.PROCEDURAL}),
    # --- DEMOTE ---
    (MemoryTypeEnum.PROCEDURAL, TransitionType.DEMOTE): frozenset({MemoryTypeEnum.SEMANTIC}),
    (MemoryTypeEnum.SEMANTIC, TransitionType.DEMOTE): frozenset({MemoryTypeEnum.EPISODIC}),
    (MemoryTypeEnum.EPISODIC, TransitionType.DEMOTE): frozenset({MemoryTypeEnum.WORKING}),
    # --- MERGE (same-type collapse) ---
    (MemoryTypeEnum.WORKING, TransitionType.MERGE): frozenset({MemoryTypeEnum.WORKING}),
    (MemoryTypeEnum.EPISODIC, TransitionType.MERGE): frozenset({MemoryTypeEnum.EPISODIC}),
    (MemoryTypeEnum.SEMANTIC, TransitionType.MERGE): frozenset({MemoryTypeEnum.SEMANTIC}),
    (MemoryTypeEnum.PROCEDURAL, TransitionType.MERGE): frozenset({MemoryTypeEnum.PROCEDURAL}),
    # --- SPLIT (same-type decomposition) ---
    (MemoryTypeEnum.WORKING, TransitionType.SPLIT): frozenset({MemoryTypeEnum.WORKING}),
    (MemoryTypeEnum.EPISODIC, TransitionType.SPLIT): frozenset({MemoryTypeEnum.EPISODIC}),
    (MemoryTypeEnum.SEMANTIC, TransitionType.SPLIT): frozenset({MemoryTypeEnum.SEMANTIC}),
    (MemoryTypeEnum.PROCEDURAL, TransitionType.SPLIT): frozenset({MemoryTypeEnum.PROCEDURAL}),
}


def is_valid_type_transition(
    current: MemoryTypeEnum,
    transition: TransitionType,
    target: MemoryTypeEnum,
) -> bool:
    """Return True if *target* is a permitted result of *transition* from *current* type."""
    allowed = ALLOWED_TYPE_TRANSITIONS.get((current, transition))
    return allowed is not None and target in allowed


def is_valid_transition(
    current: MemoryState,
    transition: TransitionType,
    target: MemoryState,
) -> bool:
    """Return True if *target* is a permitted result of *transition* from *current*."""
    allowed = ALLOWED_TRANSITIONS.get((current, transition))
    return allowed is not None and target in allowed


# ---------------------------------------------------------------------------
# Transition precondition system
# ---------------------------------------------------------------------------


@dataclass
class TransitionContext:
    """Runtime context evaluated by transition validators.

    Attributes:
        retention_count:  How many times this record has been retained or
                          referenced since creation.  Drives
                          ``min_retention_count`` gates on PROMOTE edges.
        confidence:       Confidence score in [0.0, 1.0].  Higher values
                          indicate stronger retrieval or LLM evidence.
                          Drives ``confidence_gate`` guards.
        last_accessed_at: When the record was last explicitly accessed, or
                          ``None`` if it has never been accessed.  Used by
                          ``staleness_threshold`` gates on DEMOTE edges as
                          the primary staleness reference.
        ingested_at:      When the record was first written into the system.
                          Used as a fallback staleness reference when
                          ``last_accessed_at`` is ``None``.
    """

    retention_count: int
    confidence: float
    last_accessed_at: datetime | None
    ingested_at: datetime


#: Type alias for a transition validator: a callable that accepts a
#: :class:`TransitionContext` and returns ``True`` when the precondition
#: is satisfied and the transition may proceed.
TransitionValidator = Callable[[TransitionContext], bool]


def min_retention_count(n: int) -> TransitionValidator:
    """Return a validator that passes when ``ctx.retention_count >= n``.

    Use on PROMOTE edges to require that a record has been seen or
    referenced at least *n* times before it can graduate to a higher state.

    Args:
        n: Minimum number of retention/reference events required.

    Returns:
        A :data:`TransitionValidator` with a descriptive ``__name__``.
    """

    def _check(ctx: TransitionContext) -> bool:
        return ctx.retention_count >= n

    _check.__name__ = f"min_retention_count({n})"
    return _check


def staleness_threshold(days: float, *, _now: datetime | None = None) -> TransitionValidator:
    """Return a validator that passes when the record has been idle for at least *days*.

    "Idle" is measured from ``ctx.last_accessed_at`` when set, falling back
    to ``ctx.ingested_at``.  The comparison is made against *_now* (intended
    for testing); in production *_now* is ``None`` and the validator uses
    ``datetime.now(UTC)`` at evaluation time.

    Use on DEMOTE edges to require that a record is sufficiently stale
    before it can be moved to a lower-activity state.

    Args:
        days:  Minimum number of idle days required for the transition.
        _now:  Override for the current time.  Leave as ``None`` in
               production; inject a fixed ``datetime`` in tests.

    Returns:
        A :data:`TransitionValidator` with a descriptive ``__name__``.
    """

    def _check(ctx: TransitionContext) -> bool:
        now = _now if _now is not None else datetime.now(UTC)
        last_active = ctx.last_accessed_at if ctx.last_accessed_at is not None else ctx.ingested_at
        return (now - last_active) >= timedelta(days=days)

    _check.__name__ = f"staleness_threshold({days}d)"
    return _check


def confidence_gate(threshold: float) -> TransitionValidator:
    """Return a validator that passes when ``ctx.confidence >= threshold``.

    Use on PROMOTE edges to require a minimum evidence score before a
    record can graduate to a higher-confidence state.

    Args:
        threshold: Minimum confidence score in [0.0, 1.0].

    Returns:
        A :data:`TransitionValidator` with a descriptive ``__name__``.
    """

    def _check(ctx: TransitionContext) -> bool:
        return ctx.confidence >= threshold

    _check.__name__ = f"confidence_gate({threshold})"
    return _check


#: Default precondition rules indexed by ``(from_state, transition, to_state)``.
#:
#: Keys that are absent impose no preconditions (beyond the structural check
#: in :data:`ALLOWED_TRANSITIONS`).  Callers may supply their own registry to
#: :func:`check_transition_validators` to override or extend these defaults.
DEFAULT_TRANSITION_VALIDATORS: Final[
    dict[tuple[MemoryState, TransitionType, MemoryState], list[TransitionValidator]]
] = {
    # EPISODIC → PROMOTE → SEMANTIC
    # Require 3 prior episodic retention events and ≥ 60 % confidence.
    (MemoryState.EPISODIC, TransitionType.PROMOTE, MemoryState.SEMANTIC): [
        min_retention_count(3),
        confidence_gate(0.6),
    ],
    # SEMANTIC → PROMOTE → PROCEDURAL
    # Higher bar: 5 retention events and ≥ 80 % confidence.
    (MemoryState.SEMANTIC, TransitionType.PROMOTE, MemoryState.PROCEDURAL): [
        min_retention_count(5),
        confidence_gate(0.8),
    ],
    # EPISODIC → DEMOTE → DORMANT
    # Suppress episodic records that have been idle for 7 days.
    (MemoryState.EPISODIC, TransitionType.DEMOTE, MemoryState.DORMANT): [
        staleness_threshold(7.0),
    ],
    # SEMANTIC → DEMOTE → DORMANT
    # Semantic knowledge needs 30 idle days before suppression.
    (MemoryState.SEMANTIC, TransitionType.DEMOTE, MemoryState.DORMANT): [
        staleness_threshold(30.0),
    ],
    # PROCEDURAL → DEMOTE → ARCHIVAL
    # Skills are retired after 90 idle days.
    (MemoryState.PROCEDURAL, TransitionType.DEMOTE, MemoryState.ARCHIVAL): [
        staleness_threshold(90.0),
    ],
}


# ---------------------------------------------------------------------------
# Typed validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of a StateMachine.validate() call.

    Attributes:
        ok:     ``True`` when the transition is structurally allowed and all
                precondition validators (if a context was supplied) pass.
        reason: Human-readable explanation.  Always ``"ok"`` on success;
                a short description of the first blocking constraint on failure.
    """

    ok: bool
    reason: str

    @classmethod
    def success(cls) -> ValidationResult:
        return cls(ok=True, reason="ok")

    @classmethod
    def failure(cls, reason: str) -> ValidationResult:
        return cls(ok=False, reason=reason)


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------


class StateMachine:
    """Validates memory lifecycle transitions against the allowed-transition map.

    By default uses the module-level :data:`ALLOWED_TRANSITIONS` map.  A
    custom map can be supplied at construction time to support alternative
    topologies in tests or specialised deployments.

    Args:
        transitions: Optional replacement for :data:`ALLOWED_TRANSITIONS`.
    """

    def __init__(
        self,
        transitions: (
            dict[tuple[MemoryState, TransitionType], frozenset[MemoryState]] | None
        ) = None,
    ) -> None:
        self._transitions = transitions if transitions is not None else ALLOWED_TRANSITIONS

    def validate(
        self,
        current: MemoryState,
        transition: TransitionType,
        target: MemoryState,
    ) -> ValidationResult:
        """Check whether *target* is reachable from *current* via *transition*.

        Looks up ``(current, transition)`` in the allowed-transition map and
        verifies that *target* is in the resulting frozenset.

        Args:
            current:    The memory item's present state.
            transition: The transition being attempted.
            target:     The intended next state.

        Returns:
            :class:`ValidationResult` with ``ok=True`` when the edge exists,
            or ``ok=False`` with a descriptive ``reason`` when it does not.
        """
        allowed = self._transitions.get((current, transition))
        if allowed is None:
            return ValidationResult.failure(
                f"no transitions defined for {current.value!r} --{transition.value}-->"
            )
        if target not in allowed:
            valid = ", ".join(f"{s.value!r}" for s in sorted(allowed, key=lambda s: s.value))
            return ValidationResult.failure(
                f"{current.value!r} --{transition.value}--> {target.value!r} is not allowed"
                f" (valid targets: {valid})"
            )
        return ValidationResult.success()


def check_transition_validators(
    from_state: MemoryState,
    transition: TransitionType,
    to_state: MemoryState,
    context: TransitionContext,
    *,
    validators: (
        dict[tuple[MemoryState, TransitionType, MemoryState], list[TransitionValidator]] | None
    ) = None,
) -> list[str]:
    """Evaluate all precondition validators for a transition edge.

    Looks up the ``(from_state, transition, to_state)`` key in *validators*
    (falls back to :data:`DEFAULT_TRANSITION_VALIDATORS` when ``None``) and
    runs every registered :data:`TransitionValidator` against *context*.

    Args:
        from_state:  The current state of the memory item.
        transition:  The transition being attempted.
        to_state:    The intended target state.
        context:     Runtime data evaluated by the validators.
        validators:  Optional registry override.  When ``None``,
                     :data:`DEFAULT_TRANSITION_VALIDATORS` is used.

    Returns:
        A list of human-readable failure messages, one per failing validator.
        An **empty list** means all preconditions are satisfied and the
        transition may proceed (subject to :func:`is_valid_transition`).
    """
    registry = validators if validators is not None else DEFAULT_TRANSITION_VALIDATORS
    edge_validators = registry.get((from_state, transition, to_state), [])
    failures: list[str] = []
    for validator in edge_validators:
        if not validator(context):
            name = getattr(validator, "__name__", repr(validator))
            failures.append(
                f"{name} failed for {from_state.value!r} "
                f"--{transition.value}--> {to_state.value!r}"
            )
    return failures
