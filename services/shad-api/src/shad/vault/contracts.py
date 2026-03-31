"""Data contracts for the vault consolidation pipeline and feature audit.

Defines:
- Consolidation pipeline contracts: ConsolidationResult, GroupedNotes,
  PruneCandidate (and re-exports DecayScore from shad.vault.decay).
- Feature audit contracts: GapRecord, FeatureAuditEntry, FeatureAudit.
- Transition validators: TransitionContext, TransitionValidator, and the three
  built-in validator factories (min_retention_count, staleness_threshold,
  confidence_gate) plus DEFAULT_TRANSITION_VALIDATORS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from shad.vault.consolidation import ConsolidationMetrics
from shad.vault.decay import DecayScore  # re-exported as a pipeline contract
from shad.vault.shadow_index import MemoryType

__all__ = [
    "ConsolidationResult",
    "DEFAULT_TRANSITION_VALIDATORS",
    "DecayScore",
    "FeatureAudit",
    "FeatureAuditEntry",
    "GapRecord",
    "GroupedNotes",
    "MemoryRecord",
    "PruneCandidate",
    "PruneReason",
    "SynthesizedEntry",
    "TransitionContext",
    "TransitionValidator",
    "confidence_gate",
    "min_retention_count",
    "staleness_threshold",
]


@dataclass
class GapRecord:
    """Describes a single gap within a feature area.

    Attributes:
        feature_area:       Logical grouping (e.g. ``"retrieval"``, ``"cli"``,
                            ``"sources"``).
        file:               Relative path of the file that owns this gap.
        what_exists:        Short description of what is already implemented
                            (auto-generated from class/function names).
        what_must_be_added: Short description of the missing artifact or behaviour.
    """

    feature_area: str
    file: str
    what_exists: str
    what_must_be_added: str


@dataclass
class FeatureAuditEntry:
    """Contract for a single feature area.

    Attributes:
        target_file:        Canonical file that should satisfy this feature area,
                            relative to the ``services/shad-api/`` root.
        expected_artifacts: Ordered list of artifact descriptions (class names,
                            method names, or behavioural phrases) that must be
                            detectable in ``target_file``.
    """

    target_file: str
    expected_artifacts: list[str] = field(default_factory=list)


# FeatureAudit maps feature_area names to their audit contracts.
# Extend this dict as new areas are identified.
FeatureAudit: dict[str, FeatureAuditEntry] = {
    "retrieval": FeatureAuditEntry(
        target_file="src/shad/retrieval/layer.py",
        expected_artifacts=[
            "RetrievalLayer protocol",
            "RetrievalResult dataclass",
            "hybrid search mode",
            "bm25 search mode",
            "vector search mode",
        ],
    ),
    "qmd_retriever": FeatureAuditEntry(
        target_file="src/shad/retrieval/qmd.py",
        expected_artifacts=[
            "QmdRetriever class",
            "qmd CLI detection",
            "BM25 + vector hybrid search",
            "score normalisation",
        ],
    ),
    "filesystem_retriever": FeatureAuditEntry(
        target_file="src/shad/retrieval/filesystem.py",
        expected_artifacts=[
            "FilesystemRetriever class",
            "keyword fallback search",
        ],
    ),
    "gap_detection": FeatureAuditEntry(
        target_file="src/shad/vault/gap_detection.py",
        expected_artifacts=[
            "GapScore dataclass",
            "GapReport dataclass",
            "GapDetector.calculate_gap_score",
            "QueryHistoryAnalyzer",
            "GapReport.to_markdown",
        ],
    ),
    "sources": FeatureAuditEntry(
        target_file="src/shad/sources/manager.py",
        expected_artifacts=[
            "SourceManager class",
            "add source",
            "remove source",
            "list sources",
            "sync sources",
        ],
    ),
    "shadow_index": FeatureAuditEntry(
        target_file="src/shad/vault/shadow_index.py",
        expected_artifacts=[
            "ShadowIndex class",
            "SourceEntry dataclass",
            "SnapshotEntry dataclass",
            "UpdatePolicy enum",
        ],
    ),
    "cli": FeatureAuditEntry(
        target_file="src/shad/cli/main.py",
        expected_artifacts=[
            "shad run command",
            "shad status command",
            "shad search command",
            "shad ingest command",
            "shad sources subgroup",
            "shad server subgroup",
            "shad models command",
        ],
    ),
    "sandbox": FeatureAuditEntry(
        target_file="src/shad/sandbox/executor.py",
        expected_artifacts=[
            "CodeExecutor class",
            "sandboxed exec with restricted builtins",
            "__result__ extraction",
        ],
    ),
    "engine": FeatureAuditEntry(
        target_file="src/shad/engine/rlm.py",
        expected_artifacts=[
            "RLM engine",
            "DAG execution",
            "Code Mode orchestration",
            "budget system",
            "strategy selection",
        ],
    ),
    "verification": FeatureAuditEntry(
        target_file="src/shad/verification/layer.py",
        expected_artifacts=[
            "VerificationLayer",
            "syntax check",
            "type check",
            "import resolution",
            "strict / build / basic / off levels",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Memory record contract
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """Flat contract-level representation of a vault memory record.

    Used as the canonical interchange type across the consolidation,
    pruning, reranking, and audit pipeline stages.  Distinct from
    :class:`~shad.vault.shadow_index.MemoryRecord` which carries
    bi-temporal wrapping; this version exposes all fields at the top level
    for straightforward pipeline composition.

    Attributes:
        record_id:     Stable unique identifier for this record.
        content:       Plain-text payload.
        embedding:     Dense vector representation of ``content``.
        memory_type:   Classification (EPISODIC / SEMANTIC / WORKING /
                       PROCEDURAL) from :class:`~shad.vault.shadow_index.MemoryType`.
        created_at:    When this record was first created (UTC).
        updated_at:       Last modification time (UTC); equals ``created_at``
                          for records that have never been mutated.
        source_refs:      Ordered list of source identifiers (e.g. note paths,
                          URLs, or ``snapshot_id@path`` strings) that this
                          record was derived from.
        source_run_id:    ``run_id`` of the shad run that produced this record;
                          ``None`` for records ingested outside a run context.
        consolidated_into: ``entry_id`` of the :class:`SynthesizedEntry` that
                          absorbed this record during a consolidation pass;
                          ``None`` while the record is still active.
        tags:             Free-form labels used for grouping and gap detection.
        metadata:         Arbitrary extra key/value pairs (confidence scores,
                          provenance notes, etc.).
    """

    record_id: str
    content: str
    embedding: list[float]
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    source_refs: list[str] = field(default_factory=list)
    source_run_id: str | None = None
    consolidated_into: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject records with missing or invalid memory_type at construction time."""
        valid = ", ".join(f'"{m.value}"' for m in MemoryType)
        if self.memory_type is None:
            raise ValueError(
                f"memory_type is required. Valid values: {valid}"
            )
        if not isinstance(self.memory_type, MemoryType):
            raise ValueError(
                f"Invalid memory_type {self.memory_type!r}. Valid values: {valid}"
            )


@dataclass
class SynthesizedEntry:
    """A consolidated vault entry produced by folding one or more
    :class:`MemoryRecord` instances together.

    A ``SynthesizedEntry`` is the output of a consolidation merge pass.  It
    represents the synthesized, de-duplicated knowledge distilled from a
    group of related episodic or semantic records.

    Attributes:
        entry_id:        Stable unique identifier for this synthesized entry.
        content:         Synthesized plain-text payload (LLM-produced merge).
        source_refs:     Union of ``source_refs`` from all contributing
                         :class:`MemoryRecord` instances, deduplicated and
                         ordered chronologically.
        created_at:      When this entry was first created (UTC); set to the
                         timestamp of the consolidation run that produced it.
        updated_at:      Last modification time (UTC); updated whenever the
                         entry absorbs additional records in a later pass.
        source_record_ids: ``record_id`` values of every :class:`MemoryRecord`
                         that was folded into this entry.
        metadata:        Arbitrary extra key/value pairs inherited or derived
                         from the contributing records (e.g. aggregated
                         confidence, topic label).
    """

    entry_id: str
    content: str
    source_refs: list[str]
    created_at: datetime
    updated_at: datetime
    source_record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Consolidation pipeline contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedNotes:
    """A set of related memory records grouped as consolidation candidates.

    Records in the same group share a common topic and memory type; they are
    evaluated together when deciding whether to trigger a merge pass.

    Attributes:
        group_id:    Stable identifier for this candidate group (e.g. a
                     content-hash of the topic + memory_type).
        memory_type: Shared :class:`~shad.vault.shadow_index.MemoryType` of
                     every record in the group.
        record_ids:  Ordered (oldest-first) ``record_id`` values of the grouped
                     :class:`~shad.vault.shadow_index.MemoryRecord` entries.
        topic:       Human-readable label that unifies the group (e.g. the
                     shared tag or subject extracted during gap detection).
        oldest_at:   ``updated_at`` of the oldest record in the group (UTC).
        newest_at:   ``updated_at`` of the most recent record in the group (UTC).
    """

    group_id: str
    memory_type: MemoryType
    record_ids: tuple[str, ...]
    topic: str
    oldest_at: datetime
    newest_at: datetime

    @property
    def size(self) -> int:
        """Number of records in this group."""
        return len(self.record_ids)


class PruneReason(StrEnum):
    """Why a record was flagged as a prune candidate.

    - STALE:       Combined decay score fell below the staleness threshold.
    - EXPIRED:     Record has exceeded its per-state archival TTL.
    - SUPERSEDED:  Record was consumed by a consolidation merge and the
                   resulting semantic record supersedes it.
    - DUPLICATE:   Content is identical to another record; one copy is redundant.
    - REDUNDANT:   Embedding cosine similarity to an existing record exceeded
                   the configurable threshold; semantically near-duplicate.
    """

    STALE = "stale"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    DUPLICATE = "duplicate"
    REDUNDANT = "redundant"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PruneCandidate:
    """A memory record flagged for deferred pruning.

    Records are never deleted immediately; they are flagged here and only
    removed once ``eligible_at`` has passed and a human or automated
    policy has confirmed the prune.

    Attributes:
        record_id:     ``record_id`` of the candidate
                       :class:`~shad.vault.shadow_index.MemoryRecord`.
        snapshot_id:   Snapshot that contains the record.
        reason:        Why the record was flagged (see :class:`PruneReason`).
        decay_score:   Combined decay score at flag time; lower is staler.
        flagged_at:    Wall-clock time (UTC) when the candidate was identified.
        eligible_at:   Earliest time (UTC) the record may actually be pruned
                       (enforces a minimum retention window after flagging).
        superseded_by: ``record_id`` of the semantic record that replaced this
                       entry; ``None`` unless ``reason`` is ``SUPERSEDED``.
    """

    record_id: str
    snapshot_id: str
    reason: PruneReason
    decay_score: float
    flagged_at: datetime
    eligible_at: datetime
    superseded_by: str | None = None


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome of a complete consolidation pipeline run.

    Aggregates the per-merge :class:`~shad.vault.consolidation.ConsolidationMetrics`
    with higher-level pipeline bookkeeping: which note groups were evaluated,
    how many were merged vs. skipped, and which records have been queued for
    pruning.

    Attributes:
        run_id:             Unique identifier for this pipeline execution.
        groups_evaluated:   Total :class:`GroupedNotes` batches considered.
        groups_merged:      Batches where at least one merge succeeded.
        groups_skipped:     Batches that did not reach the consolidation
                            threshold or were disabled.
        metrics:            Detailed per-merge statistics from the inner
                            consolidation pass.
        prune_candidates:   Records flagged for deferred removal as a result
                            of this run.
        completed_at:       Wall-clock time (UTC) when the pipeline finished.
        success:            ``True`` if the run completed without a fatal error.
        error:              Failure description when ``success`` is ``False``;
                            ``None`` otherwise.
    """

    run_id: str
    groups_evaluated: int
    groups_merged: int
    groups_skipped: int
    metrics: ConsolidationMetrics
    prune_candidates: tuple[PruneCandidate, ...]
    completed_at: datetime
    success: bool
    error: str | None = None

    @property
    def merge_rate(self) -> float:
        """Fraction of evaluated groups that produced at least one merge."""
        if self.groups_evaluated == 0:
            return 0.0
        return self.groups_merged / self.groups_evaluated


# ---------------------------------------------------------------------------
# Transition validators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionContext:
    """Runtime context supplied to every :class:`TransitionValidator` call.

    Bundles the candidate group with the scoring signals that validators need
    without requiring them to reach into the decay or consolidation subsystems
    directly.

    Attributes:
        group:           The :class:`GroupedNotes` being evaluated for a
                         state transition.
        avg_decay_score: Mean ``combined_score`` of all records in *group*,
                         as returned by
                         :class:`~shad.vault.decay.SnapshotDecayScorer`.
                         Range [0.0, 1.0]; higher is fresher.
        avg_confidence:  Mean ``confidence`` drawn from each record's
                         :class:`~shad.vault.shadow_index.MemoryMetadata`.
                         Range [0.0, 1.0]; higher is more trusted.
    """

    group: GroupedNotes
    avg_decay_score: float
    avg_confidence: float


@runtime_checkable
class TransitionValidator(Protocol):
    """Callable gate for a memory lifecycle transition.

    Returns ``(allowed, reason)`` where *allowed* is ``True`` when the
    transition should proceed and *reason* is a short human-readable
    explanation suitable for audit logs.
    """

    def __call__(self, ctx: TransitionContext) -> tuple[bool, str]: ...


# ---------------------------------------------------------------------------
# Built-in validator factories
# ---------------------------------------------------------------------------


def min_retention_count(min_count: int) -> TransitionValidator:
    """Return a validator that passes when *group* contains at least *min_count* records.

    Args:
        min_count: Minimum number of records required to proceed.  Must be >= 0.

    Returns:
        A :class:`TransitionValidator` that evaluates ``group.size >= min_count``.
    """
    if min_count < 0:
        raise ValueError(f"min_count must be >= 0, got {min_count}")

    def _validator(ctx: TransitionContext) -> tuple[bool, str]:
        ok = ctx.group.size >= min_count
        if ok:
            return True, f"retention ok: {ctx.group.size} >= {min_count}"
        return False, f"retention low: {ctx.group.size} < {min_count} required"

    return _validator  # type: ignore[return-value]


def staleness_threshold(threshold: float) -> TransitionValidator:
    """Return a validator that passes when *avg_decay_score* is at least *threshold*.

    A higher threshold is a stricter freshness gate — it prevents stale groups
    from being promoted or consolidated.  Set *threshold* to ``0.0`` to make
    this gate always pass (useful for prune transitions where staleness is
    already determined upstream).

    Args:
        threshold: Minimum average decay score (inclusive) in [0.0, 1.0].

    Returns:
        A :class:`TransitionValidator` that evaluates ``avg_decay_score >= threshold``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    def _validator(ctx: TransitionContext) -> tuple[bool, str]:
        ok = ctx.avg_decay_score >= threshold
        if ok:
            return True, f"freshness ok: {ctx.avg_decay_score:.3f} >= {threshold}"
        return False, f"too stale: {ctx.avg_decay_score:.3f} < {threshold} required"

    return _validator  # type: ignore[return-value]


def confidence_gate(min_confidence: float) -> TransitionValidator:
    """Return a validator that passes when *avg_confidence* is at least *min_confidence*.

    Args:
        min_confidence: Minimum average confidence score (inclusive) in [0.0, 1.0].

    Returns:
        A :class:`TransitionValidator` that evaluates ``avg_confidence >= min_confidence``.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")

    def _validator(ctx: TransitionContext) -> tuple[bool, str]:
        ok = ctx.avg_confidence >= min_confidence
        if ok:
            return True, f"confidence ok: {ctx.avg_confidence:.3f} >= {min_confidence}"
        return False, f"confidence low: {ctx.avg_confidence:.3f} < {min_confidence} required"

    return _validator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Default validator registry
# ---------------------------------------------------------------------------

#: Default validators keyed by ``(from_type, transition, to_type)`` triples.
#:
#: ``to_type`` is ``None`` for destructive transitions (prune / expire) that
#: do not land on a new :class:`~shad.vault.shadow_index.MemoryType`.
#:
#: All validators in a list must pass for the transition to proceed.
#:
#: Override specific entries by copying this dict and substituting validators:
#:
#: .. code-block:: python
#:
#:     validators = dict(DEFAULT_TRANSITION_VALIDATORS)
#:     validators[(MemoryType.EPISODIC, "consolidate", MemoryType.SEMANTIC)] = [
#:         min_retention_count(3),
#:         staleness_threshold(0.3),
#:         confidence_gate(0.7),
#:     ]
DEFAULT_TRANSITION_VALIDATORS: dict[
    tuple[MemoryType, str, MemoryType | None],
    list[TransitionValidator],
] = {
    # WORKING → persist → EPISODIC
    # Require at least one record and reasonable confidence before committing
    # transient working memory as a durable episodic event.
    (MemoryType.WORKING, "persist", MemoryType.EPISODIC): [
        min_retention_count(1),
        confidence_gate(0.5),
    ],
    # EPISODIC → consolidate → SEMANTIC
    # Require at least two records (a single record has nothing to merge),
    # ensure the group is still fresh enough to be worth consolidating, and
    # demand high confidence before promoting to timeless semantic memory.
    (MemoryType.EPISODIC, "consolidate", MemoryType.SEMANTIC): [
        min_retention_count(2),
        staleness_threshold(0.2),
        confidence_gate(0.6),
    ],
    # WORKING → prune → (removed)
    # Working memory expires quickly; only require the record exists.
    # Staleness is already confirmed by the pipeline's decay pass.
    (MemoryType.WORKING, "prune", None): [
        min_retention_count(1),
        staleness_threshold(0.0),
    ],
    # EPISODIC → prune → (removed)
    # Stale episodic records flagged by the consolidation pipeline.
    (MemoryType.EPISODIC, "prune", None): [
        min_retention_count(1),
        staleness_threshold(0.0),
    ],
    # SEMANTIC → prune → (removed)
    # Semantic records are timeless; prune only confirmed near-duplicates or
    # very-low-confidence entries.
    (MemoryType.SEMANTIC, "prune", None): [
        min_retention_count(1),
        confidence_gate(0.0),
    ],
    # PROCEDURAL → prune → (removed)
    # Skills degrade slowly; staleness is sufficient justification.
    (MemoryType.PROCEDURAL, "prune", None): [
        min_retention_count(1),
        staleness_threshold(0.0),
    ],
}
