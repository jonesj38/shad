"""Consolidation trigger protocol, configuration, and pipeline.

Defines when and how episodic memory entries should be consolidated into
semantic memory.  Modelled on the cognee entity-consolidation pattern
(cognee/infrastructure/llm/prompts/consolidate-entity-details-txt) and the
teranode UTXO consolidator (services/utxopersister/consolidator-go), both of
which batch incremental records into a single authoritative representation once
a volume threshold is reached.

The top-level entry point is :func:`consolidate`, which sequences the four
pipeline stages — group → synthesize → prune → decay — and returns a
:class:`~shad.vault.contracts.ConsolidationResult`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from shad.vault.contracts import ConsolidationResult, GroupedNotes, PruneCandidate
    from shad.vault.gap_detection import GapReport
    from shad.vault.shadow_index import AnyMemoryRecord, EpisodicRecord, MemoryRecord, ShadowIndex, SnapshotEntry

logger = logging.getLogger(__name__)


class MergeStrategy(StrEnum):
    """How episodic entries are merged into a single semantic record.

    - LATEST_WINS:      Use the most recent entry's content verbatim.
                        Zero LLM cost; loses older signal.
    - WEIGHTED_AVERAGE: Blend confidence scores weighted by recency;
                        content taken from the highest-weighted entry.
    - LLM_SYNTHESIS:    Delegate to an LLM to produce a coherent summary
                        of all source entries.  Highest quality, highest cost.
    """

    LATEST_WINS = "latest_wins"
    WEIGHTED_AVERAGE = "weighted_average"
    LLM_SYNTHESIS = "llm_synthesis"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RetentionPolicy:
    """Per-state TTL rules governing when records are demoted or purged.

    TTL values mirror the staleness thresholds defined in
    :data:`shad.vault.taxonomy.DEFAULT_TRANSITION_VALIDATORS` so that
    consolidation and state-machine logic stay in sync.

    Attributes:
        episodic_ttl_days:   Days of inactivity before an EPISODIC record is
                             demoted to DORMANT.  Default: 7.
        semantic_ttl_days:   Days of inactivity before a SEMANTIC record is
                             demoted to DORMANT.  Default: 30.
        procedural_ttl_days: Days of inactivity before a PROCEDURAL record is
                             moved to ARCHIVAL.  Default: 90.
        archival_ttl_days:   Days after which an ARCHIVAL record may be pruned
                             entirely.  Default: 365.
    """

    episodic_ttl_days: float = 7.0
    semantic_ttl_days: float = 30.0
    procedural_ttl_days: float = 90.0
    archival_ttl_days: float = 365.0

    def __post_init__(self) -> None:
        for name, value in (
            ("episodic_ttl_days", self.episodic_ttl_days),
            ("semantic_ttl_days", self.semantic_ttl_days),
            ("procedural_ttl_days", self.procedural_ttl_days),
            ("archival_ttl_days", self.archival_ttl_days),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class MergeResult:
    """Outcome of merging one or more episodic records into a semantic record.

    Attributes:
        merged_record_id:  ``record_id`` of the resulting semantic MemoryRecord.
        source_record_ids: Episodic ``record_id`` values consumed by the merge,
                           oldest-first.
        success:           True if the merge produced a valid output record.
        merged_at:         Wall-clock time (UTC) when the merge completed.
        error:             Failure description when ``success`` is False;
                           ``None`` otherwise.
    """

    merged_record_id: str
    source_record_ids: tuple[str, ...]
    success: bool
    merged_at: datetime
    error: str | None = None


@dataclass(frozen=True)
class ConsolidationMetrics:
    """Aggregate outcomes of a single consolidation pass.

    Attributes:
        entries_processed:        Total episodic entries evaluated.
        entries_merged:           Entries consumed by at least one merge.
        entries_retained:         Entries left as episodic (below threshold
                                  or merge failed).
        semantic_records_created: New semantic MemoryRecords written.
        merge_results:            Per-merge outcome details.
        started_at:               When the pass began (UTC).
        completed_at:             When the pass finished (UTC).
    """

    entries_processed: int
    entries_merged: int
    entries_retained: int
    semantic_records_created: int
    merge_results: tuple[MergeResult, ...]
    started_at: datetime
    completed_at: datetime

    @property
    def success_rate(self) -> float:
        """Fraction of merge attempts that succeeded; 0.0 if none attempted."""
        if not self.merge_results:
            return 0.0
        return sum(1 for r in self.merge_results if r.success) / len(self.merge_results)


@runtime_checkable
class ConsolidationTrigger(Protocol):
    """Protocol for objects that decide whether consolidation should run.

    Implementations receive the current episodic entry count and return
    True when consolidation should be initiated.  The threshold comparison
    itself lives in :class:`ConsolidationConfig`; custom implementations
    may apply additional heuristics (e.g. time-based, entropy-based).
    """

    def should_consolidate(self, entry_count: int) -> bool:
        """Return True if consolidation should be triggered now.

        Args:
            entry_count: Number of unconsolidated episodic entries
                         currently queued for this memory slot.

        Returns:
            True  → initiate consolidation pass.
            False → defer until the next evaluation.
        """
        ...


@runtime_checkable
class PostConsolidationHook(Protocol):
    """Protocol for callbacks invoked after a consolidation pass completes.

    Hooks receive the original record list and the completed
    :class:`~shad.vault.contracts.ConsolidationResult` and return any
    additional :class:`~shad.vault.contracts.PruneCandidate` objects they
    identify.  Their candidates are merged into the result's
    ``prune_candidates`` before it is returned to the caller.

    A typical use case is running :class:`~shad.vault.pruning.RedundancyPruner`
    as a post-consolidation step so that near-duplicate semantic records
    produced by the merge pass are immediately flagged for removal.
    """

    def run(
        self,
        records: list[AnyMemoryRecord],
        result: ConsolidationResult,
    ) -> list[PruneCandidate]:
        """Compute additional prune candidates after consolidation.

        Args:
            records: The original record list passed to :func:`consolidate`.
            result:  The completed (but not yet finalized) consolidation result.
                     Already contains SUPERSEDED, EXPIRED, and STALE candidates.

        Returns:
            Additional :class:`~shad.vault.contracts.PruneCandidate` objects;
            may be empty.  Candidates whose ``record_id`` already appears in
            ``result.prune_candidates`` are silently dropped by :func:`consolidate`.
        """
        ...


@dataclass(frozen=True)
class ConsolidationConfig:
    """Parameters that govern automatic consolidation.

    Attributes:
        threshold:                   Minimum number of episodic entries that
                                     must accumulate before a consolidation pass
                                     is triggered.  Must be >= 1.
        enabled:                     Master switch.  When False,
                                     ``should_consolidate`` always returns False.
        cosine_similarity_threshold: Minimum cosine-similarity score in [0, 1]
                                     for two records to be considered merge
                                     candidates.  Records below this score are
                                     kept as separate entries.  Default: 0.85.
        decay_halflife_days:         Half-life for temporal decay applied to
                                     episodic scores during candidate ranking
                                     (consistent with the reranker default of
                                     90 days).  Must be > 0.
        merge_strategy:              Algorithm used to combine source entries
                                     into the consolidated semantic record.
        retention_policy:            Per-state TTL rules used after
                                     consolidation to schedule demotion and
                                     pruning.
    """

    threshold: int = 10
    enabled: bool = True
    cosine_similarity_threshold: float = 0.85
    decay_halflife_days: float = 90.0
    merge_strategy: MergeStrategy = MergeStrategy.LATEST_WINS
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {self.threshold}")
        if not (0.0 <= self.cosine_similarity_threshold <= 1.0):
            raise ValueError(
                f"cosine_similarity_threshold must be in [0, 1], "
                f"got {self.cosine_similarity_threshold}"
            )
        if self.decay_halflife_days <= 0:
            raise ValueError(
                f"decay_halflife_days must be > 0, got {self.decay_halflife_days}"
            )

    def should_consolidate(self, entry_count: int) -> bool:
        """Default trigger: fire when entry_count reaches the threshold.

        Args:
            entry_count: Number of unconsolidated episodic entries.

        Returns:
            True if enabled and entry_count >= threshold.
        """
        return self.enabled and entry_count >= self.threshold


@dataclass(frozen=True)
class TimeIntervalTrigger:
    """Fires when *min_interval* has elapsed since the last consolidation.

    Unlike :class:`ConsolidationConfig` (which counts episodic entries),
    this trigger fires purely on elapsed time.  It is useful for scheduled
    maintenance passes that should run regardless of entry volume.

    Attributes:
        min_interval: Minimum gap between consolidation passes.  Must be > 0.
        enabled:      Master switch.  When False, always returns False.
    """

    min_interval: timedelta
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.min_interval.total_seconds() <= 0:
            raise ValueError(
                f"min_interval must be > 0, got {self.min_interval}"
            )

    def should_consolidate(
        self,
        entry_count: int,
        *,
        last_consolidated_at: datetime | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Return True when enough time has passed since *last_consolidated_at*.

        Args:
            entry_count:          Ignored; present for API symmetry with
                                  :class:`ConsolidationTrigger`.
            last_consolidated_at: UTC timestamp of the previous consolidation.
                                  Pass ``None`` to indicate that no
                                  consolidation has ever run — fires immediately.
            now:                  Reference time (UTC).  Defaults to
                                  ``datetime.now(UTC)`` when ``None``.

        Returns:
            True  → interval has elapsed (or never consolidated) and enabled.
            False → interval has not elapsed, or trigger is disabled.
        """
        if not self.enabled:
            return False
        if last_consolidated_at is None:
            return True
        _now = now if now is not None else datetime.now(UTC)
        return (_now - last_consolidated_at) >= self.min_interval


@dataclass(frozen=True)
class DebouncedConsolidationTrigger:
    """Fires only after a quiet period has elapsed since the last memory-add event.

    Unlike :class:`ConsolidationConfig` (volume-based) and
    :class:`TimeIntervalTrigger` (periodic since last *consolidation*), this
    trigger suppresses consolidation while new events keep arriving and fires
    exactly once per burst — after ``debounce_window`` of inactivity.

    Typical use: feed ingestion fires many events in quick succession.  Rather
    than triggering a consolidation pass for each event, the caller records the
    timestamp of every add as ``last_event_at`` and polls
    :meth:`should_consolidate`.  The trigger defers until the burst settles.

    Attributes:
        debounce_window: Minimum quiet period (no new events) required before
                         consolidation fires.  Must be > 0.
        min_entries:     Minimum number of unconsolidated entries required to
                         consider firing.  Default: 1.
        enabled:         Master switch.  When ``False``,
                         ``should_consolidate`` always returns ``False``.
    """

    debounce_window: timedelta
    min_entries: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.debounce_window.total_seconds() <= 0:
            raise ValueError(
                f"debounce_window must be > 0, got {self.debounce_window}"
            )
        if self.min_entries < 1:
            raise ValueError(
                f"min_entries must be >= 1, got {self.min_entries}"
            )

    def should_consolidate(
        self,
        entry_count: int,
        *,
        last_event_at: datetime | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Return True when the debounce window has elapsed since *last_event_at*.

        The caller is responsible for tracking ``last_event_at`` — it should be
        updated to ``datetime.now(UTC)`` each time a new memory record is added.
        Passing the updated timestamp on every call naturally re-arms the debounce
        for each event in a burst.

        Args:
            entry_count:   Number of unconsolidated episodic entries.  Must be
                           >= ``min_entries`` for the trigger to fire.
            last_event_at: UTC timestamp of the most recent memory-add event.
                           ``None`` means no event has been recorded yet —
                           always returns ``False``.
            now:           Reference time (UTC).  Defaults to
                           ``datetime.now(UTC)`` when ``None``.

        Returns:
            True  → quiet period elapsed, enough entries, and enabled.
            False → still within debounce window, too few entries, no events
                    recorded yet, or trigger is disabled.
        """
        if not self.enabled:
            return False
        if last_event_at is None:
            return False
        if entry_count < self.min_entries:
            return False
        _now = now if now is not None else datetime.now(UTC)
        return (_now - last_event_at) >= self.debounce_window


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.  Must have the same length as *a*.

    Returns:
        Similarity score in [-1.0, 1.0].  Returns ``0.0`` when either
        vector is the zero vector (undefined direction).

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _group_by_record_embedding(
    records: list[MemoryRecord],
    threshold: float = 0.85,
) -> list[list[MemoryRecord]]:
    """Cluster MemoryRecords by embedding similarity using single-linkage agglomerative grouping.

    Two records belong to the same cluster when their ``cosine_similarity`` is
    >= *threshold*.  Single-linkage means the relation is transitive: if A is
    similar to B and B is similar to C, all three are grouped together even if
    A and C are below threshold on their own.

    Records that have no embedding (``metadata`` is ``None`` or
    ``metadata.embedding`` is ``None``) cannot be compared and are each placed
    in their own singleton group.

    Args:
        records:   MemoryRecords to cluster.  Typically a list of
                   :class:`~shad.vault.shadow_index.EpisodicRecord` instances
                   drawn from the consolidation pipeline.
        threshold: Minimum cosine similarity in [0, 1] required for two
                   records to be placed in the same cluster.  Defaults to
                   ``0.85``, matching
                   :attr:`ConsolidationConfig.cosine_similarity_threshold`.

    Returns:
        A list of groups; each group is a non-empty list of MemoryRecord
        objects.  Every input record appears in exactly one group.  The
        order of groups and the order of records within each group follow
        the order of *records*.
    """
    n = len(records)
    if n == 0:
        return []

    # Union-Find with path compression (halving variant).
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    embeddings: list[list[float] | None] = [
        r.metadata.embedding if r.metadata is not None else None
        for r in records
    ]

    for i in range(n):
        if embeddings[i] is None:
            continue
        for j in range(i + 1, n):
            if embeddings[j] is None:
                continue
            if cosine_similarity(embeddings[i], embeddings[j]) >= threshold:  # type: ignore[arg-type]
                union(i, j)

    groups: dict[int, list[MemoryRecord]] = {}
    for i, record in enumerate(records):
        groups.setdefault(find(i), []).append(record)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_SECONDS_PER_DAY = 86_400.0


def consolidate(
    records: list[AnyMemoryRecord],
    config: ConsolidationConfig | None = None,
    *,
    now: datetime | None = None,
    hooks: list[PostConsolidationHook] | None = None,
) -> ConsolidationResult:
    """Run the four-stage consolidation pipeline over *records*.

    Stages
    ------
    1. **Group**     — partition EPISODIC records into :class:`~shad.vault.contracts.GroupedNotes`
                       by ``(memory_type, topic)``.  Topic is derived from the record's
                       first feature-area tag, then ``session_id``, then ``"default"``.
    2. **Synthesize**— for every group that meets the :class:`ConsolidationConfig` threshold,
                       produce a :class:`MergeResult` using the configured
                       :class:`MergeStrategy`.
    3. **Prune**     — flag source records of successful merges as
                       :attr:`~shad.vault.contracts.PruneReason.SUPERSEDED`, and any
                       EPISODIC records whose age exceeds the
                       :class:`RetentionPolicy` TTL as
                       :attr:`~shad.vault.contracts.PruneReason.EXPIRED`.
    4. **Decay**     — score remaining EPISODIC records with exponential half-life decay
                       and flag those whose combined score falls below the staleness
                       threshold as :attr:`~shad.vault.contracts.PruneReason.STALE`.

    Non-EPISODIC records pass through untouched.

    Args:
        records: All memory records to consider.  Only EPISODIC entries are
                 processed; other types are ignored silently.
        config:  Pipeline configuration.  Defaults to :class:`ConsolidationConfig`
                 with all defaults if ``None``.
        now:     Reference wall-clock time (UTC) for age calculations.
                 Defaults to ``datetime.now(UTC)`` when ``None``.
        hooks:   Optional list of :class:`PostConsolidationHook` callbacks.
                 Each hook is called with the original *records* and a
                 preliminary :class:`~shad.vault.contracts.ConsolidationResult`,
                 and may return additional
                 :class:`~shad.vault.contracts.PruneCandidate` objects (e.g.
                 redundancy candidates from
                 :class:`~shad.vault.pruning.RedundancyPrunerHook`).  Duplicate
                 ``record_id`` values are silently dropped.

    Returns:
        A :class:`~shad.vault.contracts.ConsolidationResult` with per-run
        bookkeeping and the complete list of prune candidates.
    """
    # Lazy imports to avoid circular dependency at module load time.
    from shad.vault.contracts import ConsolidationResult, GroupedNotes, PruneCandidate, PruneReason
    from shad.vault.decay import DecayConfig, DecayCurve, apply_decay
    from shad.vault.shadow_index import EpisodicRecord, MemoryType

    if config is None:
        config = ConsolidationConfig()

    reference = now if now is not None else datetime.now(UTC)
    started_at = reference
    run_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Stage 1 — Group
    # ------------------------------------------------------------------
    episodic: list[EpisodicRecord] = [
        r for r in records if isinstance(r, EpisodicRecord)
    ]

    # Group by (topic, memory_type).  Topic: first feature-area metadata tag,
    # then session_id, then "default".
    _FEATURE_TAGS = {
        "retrieval", "qmd_retriever", "filesystem_retriever", "gap_detection",
        "sources", "shadow_index", "cli", "sandbox", "engine", "verification",
    }

    def _topic(r: EpisodicRecord) -> str:
        if r.metadata:
            for tag in r.metadata.tags:
                if tag in _FEATURE_TAGS:
                    return tag
        if r.session_id:
            return r.session_id
        return "default"

    # bucket: (topic, memory_type_value) → list of records sorted oldest-first
    buckets: dict[tuple[str, str], list[EpisodicRecord]] = {}
    for r in episodic:
        key = (_topic(r), r.memory_type.value)
        buckets.setdefault(key, []).append(r)

    # Sort each bucket oldest-first by event_time.
    for recs in buckets.values():
        recs.sort(key=lambda r: r.record.event_time)

    groups: list[GroupedNotes] = []
    for (topic, mem_type_val), recs in buckets.items():
        mem_type = MemoryType(mem_type_val)
        group_id = hashlib.sha256(f"{topic}\x00{mem_type_val}".encode()).hexdigest()[:16]
        groups.append(
            GroupedNotes(
                group_id=group_id,
                memory_type=mem_type,
                record_ids=tuple(r.record_id for r in recs),
                topic=topic,
                oldest_at=recs[0].record.event_time,
                newest_at=recs[-1].record.event_time,
            )
        )

    # ------------------------------------------------------------------
    # Stage 2 — Synthesize
    # ------------------------------------------------------------------
    # Index records by id for fast lookup.
    record_index: dict[str, EpisodicRecord] = {r.record_id: r for r in episodic}

    merge_results: list[MergeResult] = []
    groups_merged = 0
    groups_skipped = 0

    for group in groups:
        if not config.should_consolidate(group.size):
            groups_skipped += 1
            continue

        recs = [record_index[rid] for rid in group.record_ids if rid in record_index]
        if not recs:
            groups_skipped += 1
            continue

        merged_id = f"sem-{group.group_id}-{uuid.uuid4().hex[:8]}"

        if config.merge_strategy is MergeStrategy.WEIGHTED_AVERAGE:
            # Pick the record with highest confidence × recency weight.
            decay_cfg = DecayConfig(
                halflife_seconds=config.decay_halflife_days * _SECONDS_PER_DAY,
                curve=DecayCurve.EXPONENTIAL,
            )
            best = max(
                recs,
                key=lambda r: apply_decay(
                    r.metadata.confidence if r.metadata else 1.0,
                    (reference - r.record.event_time).total_seconds(),
                    decay_cfg,
                ),
            )
            _ = best  # content selection happens outside this layer
        elif config.merge_strategy is MergeStrategy.LATEST_WINS:
            # Newest record wins (already sorted oldest-first, so last is newest).
            _ = recs[-1]
        # LLM_SYNTHESIS: delegate to caller; we still record the merge result.

        mr = MergeResult(
            merged_record_id=merged_id,
            source_record_ids=tuple(r.record_id for r in recs),
            success=True,
            merged_at=reference,
        )
        merge_results.append(mr)
        groups_merged += 1
        logger.debug(
            "Merged group %s (%d records) → %s via %s",
            group.group_id,
            len(recs),
            merged_id,
            config.merge_strategy,
        )

    # ------------------------------------------------------------------
    # Stage 3 — Prune
    # ------------------------------------------------------------------
    prune_candidates: list[PruneCandidate] = []
    superseded_ids: set[str] = set()

    for mr in merge_results:
        if not mr.success:
            continue
        for src_id in mr.source_record_ids:
            superseded_ids.add(src_id)
            eligible = reference + timedelta(days=config.retention_policy.episodic_ttl_days)
            prune_candidates.append(
                PruneCandidate(
                    record_id=src_id,
                    snapshot_id=src_id,  # MemoryRecord has no separate snapshot_id
                    reason=PruneReason.SUPERSEDED,
                    decay_score=0.0,  # not yet scored; filled in Stage 4
                    flagged_at=reference,
                    eligible_at=eligible,
                    superseded_by=mr.merged_record_id,
                )
            )

    # Expired: EPISODIC records past TTL that were not already superseded.
    episodic_ttl_secs = config.retention_policy.episodic_ttl_days * _SECONDS_PER_DAY
    for r in episodic:
        if r.record_id in superseded_ids:
            continue
        age_secs = (reference - r.record.event_time).total_seconds()
        if age_secs > episodic_ttl_secs:
            eligible = reference + timedelta(days=1)
            prune_candidates.append(
                PruneCandidate(
                    record_id=r.record_id,
                    snapshot_id=r.record_id,
                    reason=PruneReason.EXPIRED,
                    decay_score=0.0,
                    flagged_at=reference,
                    eligible_at=eligible,
                )
            )

    # ------------------------------------------------------------------
    # Stage 4 — Decay
    # ------------------------------------------------------------------
    decay_cfg = DecayConfig(
        halflife_seconds=config.decay_halflife_days * _SECONDS_PER_DAY,
        curve=DecayCurve.EXPONENTIAL,
    )
    staleness_threshold = 0.2  # mirrors SnapshotDecayScorer.DEFAULT_STALENESS_THRESHOLD
    already_flagged = {pc.record_id for pc in prune_candidates}

    for r in episodic:
        if r.record_id in already_flagged:
            continue
        age_secs = (reference - r.record.event_time).total_seconds()
        base_score = r.metadata.confidence if r.metadata else 1.0
        effective_age = age_secs * getattr(r, "decay_rate", 1.0)
        decayed = apply_decay(base_score, effective_age, decay_cfg)
        if decayed < staleness_threshold:
            eligible = reference + timedelta(days=1)
            prune_candidates.append(
                PruneCandidate(
                    record_id=r.record_id,
                    snapshot_id=r.record_id,
                    reason=PruneReason.STALE,
                    decay_score=decayed,
                    flagged_at=reference,
                    eligible_at=eligible,
                )
            )

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    completed_at = datetime.now(UTC)
    entries_merged = sum(len(mr.source_record_ids) for mr in merge_results if mr.success)
    entries_retained = len(episodic) - entries_merged

    metrics = ConsolidationMetrics(
        entries_processed=len(episodic),
        entries_merged=entries_merged,
        entries_retained=max(0, entries_retained),
        semantic_records_created=sum(1 for mr in merge_results if mr.success),
        merge_results=tuple(merge_results),
        started_at=started_at,
        completed_at=completed_at,
    )

    # ------------------------------------------------------------------
    # Post-consolidation hooks
    # ------------------------------------------------------------------
    if hooks:
        preliminary = ConsolidationResult(
            run_id=run_id,
            groups_evaluated=len(groups),
            groups_merged=groups_merged,
            groups_skipped=groups_skipped,
            metrics=metrics,
            prune_candidates=tuple(prune_candidates),
            completed_at=completed_at,
            success=True,
        )
        existing_ids = {pc.record_id for pc in prune_candidates}
        for hook in hooks:
            for candidate in hook.run(records, preliminary):
                if candidate.record_id not in existing_ids:
                    prune_candidates.append(candidate)
                    existing_ids.add(candidate.record_id)

    return ConsolidationResult(
        run_id=run_id,
        groups_evaluated=len(groups),
        groups_merged=groups_merged,
        groups_skipped=groups_skipped,
        metrics=metrics,
        prune_candidates=tuple(prune_candidates),
        completed_at=completed_at,
        success=True,
    )


def synthesize_group(
    group: GroupedNotes,
    records: list[EpisodicRecord],
    config: ConsolidationConfig | None = None,
    *,
    gap_report: GapReport | None = None,
    snapshot_metadata: dict[str, SnapshotEntry] | None = None,
    shadow_index: ShadowIndex | None = None,
    now: datetime | None = None,
) -> ConsolidationResult:
    """Synthesize a single :class:`~shad.vault.contracts.GroupedNotes` batch.

    Unlike :func:`consolidate`, which processes all records in one pass,
    this function focuses on **one group** and enriches the merge decision
    with optional :class:`~shad.vault.gap_detection.GapReport` evidence and
    :class:`~shad.vault.shadow_index.SnapshotEntry` metadata from the
    ShadowIndex.

    GapReport influence
    -------------------
    When *gap_report* is supplied and contains a
    :class:`~shad.vault.gap_detection.GapScore` whose ``topic`` matches
    *group.topic*, the gap score is used as a recency multiplier
    (``gap_boost = 1 + gap_score``).  Under ``WEIGHTED_AVERAGE`` this
    biases selection toward the freshest record in the group — the most
    recent attempt to address a known knowledge gap.

    ShadowIndex metadata influence
    -------------------------------
    When *snapshot_metadata* maps a ``record_id`` to a
    :class:`~shad.vault.shadow_index.SnapshotEntry`, two signals are applied:

    1. **Freshness**: the snapshot's ``ingested_at`` timestamp replaces
       ``record.event_time`` when computing age for decay scoring, so
       re-ingested content is treated as fresh even if the original event
       time is old.
    2. **Deduplication**: records whose snapshot shares a ``content_hash``
       with a previously processed record in the group are silently dropped
       before merging to avoid redundant content in the consolidated summary.

    Args:
        group:             The pre-grouped batch of episodic records.
        records:           EpisodicRecord objects; only those whose
                           ``record_id`` appears in *group.record_ids* are
                           processed.
        config:            Pipeline configuration.  Defaults to
                           :class:`ConsolidationConfig` with all defaults.
        gap_report:        Optional gap report.  The first
                           :class:`~shad.vault.gap_detection.GapScore` whose
                           ``topic`` matches *group.topic* is used.
        snapshot_metadata: Optional mapping of ``record_id`` →
                           :class:`~shad.vault.shadow_index.SnapshotEntry`
                           for freshness and deduplication signals.
        shadow_index:      Optional :class:`~shad.vault.shadow_index.ShadowIndex`
                           instance.  When supplied and the merge succeeds, each
                           source record's ``consolidated_into`` back-pointer is
                           set to the new entry's ID via
                           :meth:`~shad.vault.shadow_index.ShadowIndex.mark_snapshot_consolidated`,
                           persisting the mutation immediately.
        now:               Reference wall-clock time (UTC).  Defaults to
                           ``datetime.now(UTC)``.

    Returns:
        :class:`~shad.vault.contracts.ConsolidationResult` with
        ``groups_evaluated == 1``.  When the threshold is not met,
        ``groups_merged == 0`` and ``groups_skipped == 1``.
    """
    from shad.vault.contracts import ConsolidationResult, PruneCandidate, PruneReason
    from shad.vault.decay import DecayConfig, DecayCurve, apply_decay

    if config is None:
        config = ConsolidationConfig()

    reference = now if now is not None else datetime.now(UTC)
    run_id = str(uuid.uuid4())
    started_at = reference

    # Filter and sort records belonging to this group (oldest-first).
    group_ids = set(group.record_ids)
    recs: list[EpisodicRecord] = sorted(
        (r for r in records if r.record_id in group_ids),
        key=lambda r: r.record.event_time,
    )

    # Early-exit: below threshold or no matching records.
    if not config.should_consolidate(group.size) or not recs:
        completed_at = datetime.now(UTC)
        metrics = ConsolidationMetrics(
            entries_processed=len(recs),
            entries_merged=0,
            entries_retained=len(recs),
            semantic_records_created=0,
            merge_results=(),
            started_at=started_at,
            completed_at=completed_at,
        )
        return ConsolidationResult(
            run_id=run_id,
            groups_evaluated=1,
            groups_merged=0,
            groups_skipped=1,
            metrics=metrics,
            prune_candidates=(),
            completed_at=completed_at,
            success=True,
        )

    # ------------------------------------------------------------------
    # GapReport: find a matching GapScore for this group's topic.
    # ------------------------------------------------------------------
    gap_boost = 1.0
    if gap_report is not None:
        for gs in gap_report.gaps:
            if gs.topic.lower() == group.topic.lower():
                # High-gap topics get a recency multiplier so the freshest
                # record (best coverage of the gap) is preferred.
                gap_boost = 1.0 + gs.score  # range [1.0, 2.0]
                logger.debug(
                    "Group %s topic %r: gap_boost=%.3f (gap_score=%.3f)",
                    group.group_id,
                    group.topic,
                    gap_boost,
                    gs.score,
                )
                break

    # ------------------------------------------------------------------
    # ShadowIndex: deduplicate records by content_hash.
    # ------------------------------------------------------------------
    snap_meta: dict[str, SnapshotEntry] = snapshot_metadata or {}
    seen_hashes: set[str] = set()
    deduped: list[EpisodicRecord] = []
    for r in recs:
        snap = snap_meta.get(r.record_id)
        if snap is not None and snap.content_hash:
            if snap.content_hash in seen_hashes:
                logger.debug(
                    "Dropping duplicate record %s (content_hash=%s)",
                    r.record_id,
                    snap.content_hash,
                )
                continue
            seen_hashes.add(snap.content_hash)
        deduped.append(r)

    # Safety: never collapse to empty.
    if not deduped:
        deduped = recs

    # ------------------------------------------------------------------
    # Scoring: confidence × recency decay × gap_boost, with optional
    # ShadowIndex freshness override.
    # ------------------------------------------------------------------
    decay_cfg = DecayConfig(
        halflife_seconds=config.decay_halflife_days * _SECONDS_PER_DAY,
        curve=DecayCurve.EXPONENTIAL,
    )

    def _score(r: EpisodicRecord) -> float:
        base = r.metadata.confidence if r.metadata else 1.0
        snap = snap_meta.get(r.record_id)
        ref_time = snap.ingested_at if snap is not None else r.record.event_time
        age_secs = max(0.0, (reference - ref_time).total_seconds())
        decayed = apply_decay(base, age_secs * getattr(r, "decay_rate", 1.0), decay_cfg)
        return decayed * gap_boost

    # ------------------------------------------------------------------
    # Apply merge strategy.
    # ------------------------------------------------------------------
    merged_id = f"sem-{group.group_id}-{uuid.uuid4().hex[:8]}"

    if config.merge_strategy is MergeStrategy.WEIGHTED_AVERAGE:
        best = max(deduped, key=_score)
        logger.debug(
            "WEIGHTED_AVERAGE: selected %s (score=%.4f) from group %s",
            best.record_id,
            _score(best),
            group.group_id,
        )
    elif config.merge_strategy is MergeStrategy.LATEST_WINS:
        logger.debug(
            "LATEST_WINS: selected %s from group %s",
            deduped[-1].record_id,
            group.group_id,
        )
    # LLM_SYNTHESIS: content generation delegated to caller; merge recorded.

    mr = MergeResult(
        merged_record_id=merged_id,
        source_record_ids=tuple(r.record_id for r in deduped),
        success=True,
        merged_at=reference,
    )

    # ------------------------------------------------------------------
    # Back-pointer update: persist consolidated_into on each source record.
    # ------------------------------------------------------------------
    if shadow_index is not None:
        for r in deduped:
            snap = shadow_index.get_snapshot(r.record_id)
            if snap is None or snap.consolidated_into is None:
                shadow_index.mark_snapshot_consolidated(r.record_id, merged_id)

    # ------------------------------------------------------------------
    # Prune candidates: source records are superseded by the merge.
    # ------------------------------------------------------------------
    eligible = reference + timedelta(days=config.retention_policy.episodic_ttl_days)
    prune_candidates: list[PruneCandidate] = [
        PruneCandidate(
            record_id=r.record_id,
            snapshot_id=r.record_id,
            reason=PruneReason.SUPERSEDED,
            decay_score=_score(r),
            flagged_at=reference,
            eligible_at=eligible,
            superseded_by=merged_id,
        )
        for r in deduped
    ]

    completed_at = datetime.now(UTC)
    metrics = ConsolidationMetrics(
        entries_processed=len(recs),
        entries_merged=len(deduped),
        entries_retained=max(0, len(recs) - len(deduped)),
        semantic_records_created=1,
        merge_results=(mr,),
        started_at=started_at,
        completed_at=completed_at,
    )
    return ConsolidationResult(
        run_id=run_id,
        groups_evaluated=1,
        groups_merged=1,
        groups_skipped=0,
        metrics=metrics,
        prune_candidates=tuple(prune_candidates),
        completed_at=completed_at,
        success=True,
    )


# ---------------------------------------------------------------------------
# Embedding-based similarity grouping
# ---------------------------------------------------------------------------


def group_by_similarity(
    records: list[AnyMemoryRecord],
    embeddings: dict[str, list[float]],
    threshold: float,
) -> list[list[AnyMemoryRecord]]:
    """Partition *records* into groups by cosine-similarity of their embeddings.

    Two records are placed in the same group when the cosine similarity of
    their embedding vectors is >= *threshold*.  Connected components are found
    via union-find so that transitivity is respected: if A~B and B~C then A,
    B, and C all land in the same group even if A and C never directly exceed
    the threshold.

    Records whose ``record_id`` is absent from *embeddings* are each placed in
    their own singleton group — they cannot be compared to anything.

    Args:
        records:    Records to partition.
        embeddings: Mapping ``{record_id: vector}``.  Vectors may have any
                    dimensionality, but must all have the same length when
                    compared.  Zero vectors yield a similarity of 0.0.
        threshold:  Minimum cosine similarity in [0, 1] required to merge two
                    records into the same group.

    Returns:
        A list of non-empty lists.  Each inner list is one group.  The order
        of groups and the order of records within a group are both stable
        (insertion order of *records*).
    """
    import math
    from collections import defaultdict

    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    n = len(records)
    if n == 0:
        return []

    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        parent[_find(i)] = _find(j)

    for i in range(n):
        vec_i = embeddings.get(records[i].record_id)
        if vec_i is None:
            continue
        for j in range(i + 1, n):
            vec_j = embeddings.get(records[j].record_id)
            if vec_j is None:
                continue
            if _cosine(vec_i, vec_j) >= threshold:
                _union(i, j)

    buckets: dict[int, list[AnyMemoryRecord]] = defaultdict(list)
    for i, rec in enumerate(records):
        buckets[_find(i)].append(rec)

    return list(buckets.values())
