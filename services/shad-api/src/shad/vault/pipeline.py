"""Integration wiring connecting routing dispatch to the consolidation,
decay, and pruning pipelines.

Each pipeline stage receives only records of the expected memory_type:

    WORKING    → decay scoring  (1-day linear half-life; volatile)
    EPISODIC   → consolidation  (group → synthesize → prune → decay)
    SEMANTIC   → decay scoring + redundancy pruning
    PROCEDURAL → decay scoring  (365-day exponential half-life)

EPISODIC records are not sent to the top-level decay pass; their decay
is handled inside stage 4 of :func:`~shad.vault.consolidation.consolidate`.

Entry point: :func:`run_pipeline`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from shad.vault.contracts import PruneCandidate, PruneReason
from shad.vault.decay import DecayConfig, apply_decay
from shad.vault.shadow_index import (
    AnyMemoryRecord,
    EpisodicRecord,
    MemoryType,
    ProceduralRecord,
    SemanticRecord,
    WorkingRecord,
)

if TYPE_CHECKING:
    from shad.vault.consolidation import ConsolidationConfig, PostConsolidationHook
    from shad.vault.contracts import ConsolidationResult, MemoryRecord as ContractMemoryRecord
    from shad.vault.pruning import RedundancyPruner

logger = logging.getLogger(__name__)

__all__ = [
    "DecayedRecord",
    "MemoryTypePartition",
    "PipelineResult",
    "partition_by_type",
    "run_pipeline",
]

# Mirrors SnapshotDecayScorer.DEFAULT_STALENESS_THRESHOLD so the two paths
# produce consistent stale/active classifications.
_STALENESS_THRESHOLD: float = 0.2


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryTypePartition:
    """Records partitioned into per-type buckets.

    Produced by :func:`partition_by_type` and consumed by :func:`run_pipeline`
    to guarantee each pipeline stage receives only the record variant it
    expects.

    Attributes:
        working:    Records whose concrete type is :class:`~shad.vault.shadow_index.WorkingRecord`.
        episodic:   Records whose concrete type is :class:`~shad.vault.shadow_index.EpisodicRecord`.
        semantic:   Records whose concrete type is :class:`~shad.vault.shadow_index.SemanticRecord`.
        procedural: Records whose concrete type is :class:`~shad.vault.shadow_index.ProceduralRecord`.
    """

    working: tuple[WorkingRecord, ...]
    episodic: tuple[EpisodicRecord, ...]
    semantic: tuple[SemanticRecord, ...]
    procedural: tuple[ProceduralRecord, ...]

    @property
    def all_records(self) -> tuple[AnyMemoryRecord, ...]:
        """All records across every bucket in definition order."""
        return self.working + self.episodic + self.semantic + self.procedural

    def by_type(self, memory_type: MemoryType) -> tuple[WorkingRecord | EpisodicRecord | SemanticRecord | ProceduralRecord, ...]:
        """Return the bucket for *memory_type*."""
        mapping: dict[MemoryType, tuple[WorkingRecord | EpisodicRecord | SemanticRecord | ProceduralRecord, ...]] = {
            MemoryType.WORKING: self.working,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.PROCEDURAL: self.procedural,
        }
        return mapping[memory_type]


def partition_by_type(records: list[AnyMemoryRecord]) -> MemoryTypePartition:
    """Split *records* into four per-type buckets by concrete subclass.

    Every record ends up in exactly one bucket.  Routing is determined by
    ``isinstance`` checks against the concrete subclass, not the
    ``memory_type`` field, because the subclass constructors fix
    ``memory_type`` and a mismatch would indicate a construction bug rather
    than a routing ambiguity.

    Records that are not instances of any of the four known subtypes are
    routed by their ``memory_type`` field as a best-effort fallback and a
    warning is emitted.

    Args:
        records: Mixed list of any :data:`~shad.vault.shadow_index.AnyMemoryRecord` variant.

    Returns:
        A :class:`MemoryTypePartition` with each bucket populated.
    """
    working: list[WorkingRecord] = []
    episodic: list[EpisodicRecord] = []
    semantic: list[SemanticRecord] = []
    procedural: list[ProceduralRecord] = []

    for r in records:
        if isinstance(r, WorkingRecord):
            working.append(r)
        elif isinstance(r, EpisodicRecord):
            episodic.append(r)
        elif isinstance(r, SemanticRecord):
            semantic.append(r)
        elif isinstance(r, ProceduralRecord):
            procedural.append(r)
        else:
            logger.warning(
                "Unknown AnyMemoryRecord subtype %r for record %s; routing by memory_type field",
                type(r).__name__,
                r.record_id,
            )
            _fallback_route(r, working, episodic, semantic, procedural)

    return MemoryTypePartition(
        working=tuple(working),
        episodic=tuple(episodic),
        semantic=tuple(semantic),
        procedural=tuple(procedural),
    )


def _fallback_route(
    r: AnyMemoryRecord,
    working: list[WorkingRecord],
    episodic: list[EpisodicRecord],
    semantic: list[SemanticRecord],
    procedural: list[ProceduralRecord],
) -> None:
    """Route an unrecognised record to the bucket matching its memory_type field."""
    buckets: dict[MemoryType, list] = {
        MemoryType.WORKING: working,
        MemoryType.EPISODIC: episodic,
        MemoryType.SEMANTIC: semantic,
        MemoryType.PROCEDURAL: procedural,
    }
    buckets.get(r.memory_type, semantic).append(r)


# ---------------------------------------------------------------------------
# Decay scoring for non-EPISODIC records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayedRecord:
    """Decay score for a single non-EPISODIC memory record.

    Attributes:
        record_id:   Identifies the source record.
        memory_type: The type used to select the decay curve.
        decay_score: Decayed confidence in [0.0, 1.0].
        is_stale:    ``True`` when ``decay_score < _STALENESS_THRESHOLD``.
    """

    record_id: str
    memory_type: MemoryType
    decay_score: float
    is_stale: bool


def _score_decay_record(
    record: AnyMemoryRecord,
    *,
    now: datetime,
    retention_seconds: float,
) -> tuple[DecayedRecord, PruneCandidate | None]:
    """Compute a decay score for *record* using its type-specific :class:`~shad.vault.decay.DecayConfig`.

    Uses ``record.event_time`` (the real-world event time from the
    :class:`~shad.vault.shadow_index.BiTemporalRecord` wrapper) as the age
    reference so that the decay curve reflects how old the underlying event
    is, not when it was ingested.

    Args:
        record:            The record to score.  Must not be an
                           :class:`~shad.vault.shadow_index.EpisodicRecord`
                           — those are handled inside the consolidation pipeline.
        now:               Reference timestamp for age calculation.
        retention_seconds: Minimum seconds between stale flagging and prune
                           eligibility.  ``0.0`` means immediately eligible.

    Returns:
        ``(DecayedRecord, PruneCandidate | None)``; the candidate is ``None``
        unless the record is stale.
    """
    config = DecayConfig.default_for(record.memory_type)
    base_score = record.metadata.confidence if record.metadata is not None else 1.0
    age_secs = (now - record.event_time).total_seconds()
    scored = apply_decay(base_score, age_secs, config)
    is_stale = scored < _STALENESS_THRESHOLD

    candidate: PruneCandidate | None = None
    if is_stale:
        candidate = PruneCandidate(
            record_id=record.record_id,
            snapshot_id=record.record_id,
            reason=PruneReason.STALE,
            decay_score=scored,
            flagged_at=now,
            eligible_at=now + timedelta(seconds=retention_seconds),
        )

    return DecayedRecord(
        record_id=record.record_id,
        memory_type=record.memory_type,
        decay_score=scored,
        is_stale=is_stale,
    ), candidate


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregated output from a full pipeline run.

    Attributes:
        run_id:                  Fresh UUID assigned by :func:`run_pipeline`.
        consolidation_result:    Outcome of the consolidation pass over
                                 EPISODIC records; ``None`` when no EPISODIC
                                 records were present.
        decay_scores:            Per-record decay scores for WORKING, SEMANTIC,
                                 and PROCEDURAL records.  EPISODIC decay is
                                 handled inside the consolidation pipeline.
        stale_prune_candidates:  STALE :class:`~shad.vault.contracts.PruneCandidate`
                                 objects emitted for non-EPISODIC records whose
                                 decay score fell below the staleness threshold.
        redundancy_candidates:   REDUNDANT candidates returned by the
                                 :class:`~shad.vault.pruning.RedundancyPruner`
                                 pass over SEMANTIC records; empty when no pruner
                                 was configured.
        completed_at:            Wall-clock time (UTC) when :func:`run_pipeline`
                                 returned.
    """

    run_id: str
    consolidation_result: ConsolidationResult | None
    decay_scores: list[DecayedRecord] = field(default_factory=list)
    stale_prune_candidates: list[PruneCandidate] = field(default_factory=list)
    redundancy_candidates: list[PruneCandidate] = field(default_factory=list)
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def all_prune_candidates(self) -> list[PruneCandidate]:
        """Union of prune candidates across every pipeline stage, deduped by record_id.

        Order: consolidation candidates first, then stale, then redundancy.
        Within each group the original order is preserved.
        """
        seen: set[str] = set()
        merged: list[PruneCandidate] = []
        consolidation = (
            list(self.consolidation_result.prune_candidates)
            if self.consolidation_result is not None
            else []
        )
        for pc in consolidation + self.stale_prune_candidates + self.redundancy_candidates:
            if pc.record_id not in seen:
                merged.append(pc)
                seen.add(pc.record_id)
        return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    records: list[AnyMemoryRecord],
    *,
    consolidation_config: ConsolidationConfig | None = None,
    consolidation_hooks: list[PostConsolidationHook] | None = None,
    redundancy_pruner: RedundancyPruner | None = None,
    redundancy_retention_seconds: float = 0.0,
    decay_retention_seconds: float = 0.0,
    now: datetime | None = None,
) -> PipelineResult:
    """Route *records* to each pipeline stage by memory_type and aggregate results.

    Routing table
    ~~~~~~~~~~~~~
    +-------------+---------------------------------------------+
    | memory_type | pipeline stage(s)                           |
    +=============+=============================================+
    | WORKING     | decay scoring                               |
    | EPISODIC    | consolidation pipeline (→ prune → decay)   |
    | SEMANTIC    | decay scoring + redundancy pruning          |
    | PROCEDURAL  | decay scoring                               |
    +-------------+---------------------------------------------+

    Stages
    ------
    1. **Partition** — split *records* into four typed buckets via
       :func:`partition_by_type`.
    2. **Consolidation** — pass EPISODIC bucket to
       :func:`~shad.vault.consolidation.consolidate` (group → synthesize →
       prune → decay).  Skipped when the bucket is empty.
    3. **Decay** — score WORKING, SEMANTIC, and PROCEDURAL records with their
       type-specific :class:`~shad.vault.decay.DecayConfig`; flag stale ones as
       :attr:`~shad.vault.contracts.PruneReason.STALE` candidates.
    4. **Redundancy pruning** — if *redundancy_pruner* is supplied, run
       :meth:`~shad.vault.pruning.RedundancyPruner.find_redundant` over
       SEMANTIC records and collect
       :attr:`~shad.vault.contracts.PruneReason.REDUNDANT` candidates.

    Args:
        records:                      Mixed list of memory records.
        consolidation_config:         Forwarded to
                                      :func:`~shad.vault.consolidation.consolidate`.
                                      ``None`` uses :class:`~shad.vault.consolidation.ConsolidationConfig`
                                      defaults.
        consolidation_hooks:          Optional post-consolidation hooks;
                                      forwarded to :func:`~shad.vault.consolidation.consolidate`.
        redundancy_pruner:            If supplied, run near-duplicate detection
                                      over SEMANTIC records after decay scoring.
                                      ``None`` skips this step.
        redundancy_retention_seconds: Forwarded to
                                      :meth:`~shad.vault.pruning.RedundancyPruner.find_redundant`.
        decay_retention_seconds:      Minimum seconds between stale flagging and
                                      prune eligibility for WORKING / SEMANTIC /
                                      PROCEDURAL records.  ``0.0`` means
                                      immediately eligible.
        now:                          Reference timestamp (UTC).  Defaults to
                                      ``datetime.now(UTC)``.

    Returns:
        A :class:`PipelineResult` aggregating all stage outputs.
    """
    # Lazy import avoids a circular dependency between vault submodules.
    from shad.vault.consolidation import consolidate
    from shad.vault.contracts import MemoryRecord as ContractMemoryRecord

    reference = now if now is not None else datetime.now(UTC)
    run_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Stage 1 — Partition
    # ------------------------------------------------------------------
    partition = partition_by_type(records)

    # ------------------------------------------------------------------
    # Stage 2 — EPISODIC → consolidation pipeline
    # ------------------------------------------------------------------
    consolidation_result: ConsolidationResult | None = None
    if partition.episodic:
        consolidation_result = consolidate(
            list(partition.episodic),
            config=consolidation_config,
            now=reference,
            hooks=consolidation_hooks,
        )
        logger.debug(
            "Consolidation run %s: %d EPISODIC → %d merged, %d skipped",
            consolidation_result.run_id,
            len(partition.episodic),
            consolidation_result.groups_merged,
            consolidation_result.groups_skipped,
        )

    # ------------------------------------------------------------------
    # Stage 3 — WORKING + SEMANTIC + PROCEDURAL → decay scoring
    # ------------------------------------------------------------------
    decay_pool: list[WorkingRecord | SemanticRecord | ProceduralRecord] = (
        list(partition.working)
        + list(partition.semantic)
        + list(partition.procedural)
    )

    decay_scores: list[DecayedRecord] = []
    stale_candidates: list[PruneCandidate] = []

    for record in decay_pool:
        scored, candidate = _score_decay_record(
            record,
            now=reference,
            retention_seconds=decay_retention_seconds,
        )
        decay_scores.append(scored)
        if candidate is not None:
            stale_candidates.append(candidate)

    logger.debug(
        "Decay pass: %d records scored, %d stale",
        len(decay_scores),
        len(stale_candidates),
    )

    # ------------------------------------------------------------------
    # Stage 4 — SEMANTIC → redundancy pruning
    # ------------------------------------------------------------------
    redundancy_candidates: list[PruneCandidate] = []
    if redundancy_pruner is not None and partition.semantic:
        contract_records: list[ContractMemoryRecord] = [
            ContractMemoryRecord(
                record_id=r.record_id,
                content=r.content,
                embedding=(
                    r.metadata.embedding
                    if r.metadata is not None and r.metadata.embedding is not None
                    else []
                ),
                memory_type=r.memory_type,
                created_at=r.created_at,
                updated_at=r.updated_at,
                tags=r.metadata.tags if r.metadata is not None else [],
            )
            for r in partition.semantic
        ]
        redundancy_candidates = redundancy_pruner.find_redundant(
            contract_records,
            retention_seconds=redundancy_retention_seconds,
            now=reference,
        )
        logger.debug(
            "Redundancy pass: %d SEMANTIC → %d redundancy candidates",
            len(partition.semantic),
            len(redundancy_candidates),
        )

    return PipelineResult(
        run_id=run_id,
        consolidation_result=consolidation_result,
        decay_scores=decay_scores,
        stale_prune_candidates=stale_candidates,
        redundancy_candidates=redundancy_candidates,
        completed_at=datetime.now(UTC),
    )
