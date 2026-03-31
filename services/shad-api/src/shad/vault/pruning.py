"""Snapshot pruning for the ShadowIndex.

Provides :func:`identify_snapshot_prune_candidates`, which classifies a set of
pre-scored :class:`~shad.vault.decay.DecayScore` objects against a staleness
threshold and surfaces two kinds of prune candidates:

- **STALE** — snapshots whose ``combined_score`` falls below *threshold*.
- **DUPLICATE** — non-latest snapshots that share a ``content_hash`` with a
  newer snapshot for the same source (only the most recently ingested copy is
  kept).

Pinned snapshots are always excluded from the candidate list.  Pass the
returned list to :meth:`~shad.vault.shadow_index.ShadowIndex.prune_snapshots`
to perform the actual deletions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from shad.vault.contracts import ConsolidationResult, MemoryRecord, PruneCandidate, PruneReason
from shad.vault.decay import DecayScore
from shad.vault.shadow_index import ShadowIndex, SnapshotEntry

if TYPE_CHECKING:
    from shad.vault.shadow_index import AnyMemoryRecord

__all__ = ["identify_snapshot_prune_candidates", "RedundancyPruner", "RedundancyPrunerHook"]


def identify_snapshot_prune_candidates(
    decay_scores: list[DecayScore],
    threshold: float,
    index: ShadowIndex,
    *,
    retention_seconds: float = 0.0,
    now: datetime | None = None,
) -> list[PruneCandidate]:
    """Identify snapshot prune candidates from a set of decay scores.

    Two passes are performed:

    1. **STALE pass** — any snapshot whose ``combined_score < threshold`` is
       flagged as :attr:`~shad.vault.contracts.PruneReason.STALE`.
    2. **DUPLICATE pass** — remaining snapshots (not already flagged as stale)
       are grouped by ``(source_id, content_hash)``.  Within each group all
       snapshots *except* the most recently ingested are flagged as
       :attr:`~shad.vault.contracts.PruneReason.DUPLICATE`.

    Pinned snapshots are skipped in both passes.

    Args:
        decay_scores:      Pre-computed scores, one per snapshot to consider.
        threshold:         Combined-score cutoff.  Snapshots scoring strictly
                           below this value are flagged STALE.  Must be in
                           ``[0.0, 1.0]``.
        index:             Live :class:`~shad.vault.shadow_index.ShadowIndex`
                           used to resolve pin status and snapshot metadata.
        retention_seconds: Minimum seconds that must elapse after flagging
                           before a candidate becomes eligible for removal.
                           ``0.0`` (default) means immediately eligible.
        now:               Reference timestamp.  Defaults to
                           ``datetime.now(UTC)``.

    Returns:
        List of :class:`~shad.vault.contracts.PruneCandidate` objects.  A
        snapshot appears at most once regardless of how many criteria match it
        (STALE takes precedence over DUPLICATE).  The list may be empty if no
        candidates are found.

    Raises:
        ValueError: If *threshold* is not in ``[0.0, 1.0]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be in [0.0, 1.0], got {threshold!r}"
        )

    reference = now if now is not None else datetime.now(UTC)
    eligible_at = reference + timedelta(seconds=retention_seconds)

    # Collect pinned snapshot IDs once so we never flag them.
    pinned: set[str] = {
        s.pinned_snapshot
        for s in index.list_sources()
        if s.pinned_snapshot is not None
    }

    score_map: dict[str, DecayScore] = {ds.snapshot_id: ds for ds in decay_scores}
    candidates: list[PruneCandidate] = []
    flagged_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Pass 1: STALE — score below threshold
    # ------------------------------------------------------------------
    for ds in decay_scores:
        if ds.snapshot_id in pinned:
            continue
        if ds.combined_score < threshold:
            candidates.append(
                PruneCandidate(
                    record_id=ds.snapshot_id,
                    snapshot_id=ds.snapshot_id,
                    reason=PruneReason.STALE,
                    decay_score=ds.combined_score,
                    flagged_at=reference,
                    eligible_at=eligible_at,
                )
            )
            flagged_ids.add(ds.snapshot_id)

    # ------------------------------------------------------------------
    # Pass 2: DUPLICATE — same content_hash for same source_id,
    #          keeping only the most recently ingested copy.
    # ------------------------------------------------------------------
    # Fetch full snapshot entries only for snapshots not already flagged.
    groups: dict[tuple[str, str], list[SnapshotEntry]] = defaultdict(list)
    for snapshot_id in score_map:
        if snapshot_id in flagged_ids or snapshot_id in pinned:
            continue
        entry = index.get_snapshot(snapshot_id)
        if entry is not None:
            groups[(entry.source_id, entry.content_hash)].append(entry)

    for group in groups.values():
        if len(group) <= 1:
            continue
        # Keep the newest; flag all others as duplicates.
        group.sort(key=lambda e: e.ingested_at, reverse=True)
        canonical = group[0]
        for duplicate in group[1:]:
            if duplicate.snapshot_id in flagged_ids:
                continue
            score = score_map.get(duplicate.snapshot_id)
            candidates.append(
                PruneCandidate(
                    record_id=duplicate.snapshot_id,
                    snapshot_id=duplicate.snapshot_id,
                    reason=PruneReason.DUPLICATE,
                    decay_score=score.combined_score if score else 0.0,
                    flagged_at=reference,
                    eligible_at=eligible_at,
                    superseded_by=canonical.snapshot_id,
                )
            )
            flagged_ids.add(duplicate.snapshot_id)

    return candidates


# ---------------------------------------------------------------------------
# Helpers for embedding similarity
# ---------------------------------------------------------------------------


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


# ---------------------------------------------------------------------------
# RedundancyPruner — near-duplicate detection via cosine similarity
# ---------------------------------------------------------------------------


@dataclass
class RedundancyPruner:
    """Identifies near-duplicate memory records using embedding cosine similarity.

    Records whose embedding similarity to any existing canonical record is at or
    above *similarity_threshold* are considered redundant.  Within each
    near-duplicate cluster the most recently created record is kept; all others
    are flagged as :attr:`~shad.vault.contracts.PruneReason.REDUNDANT`.

    This complements :func:`identify_snapshot_prune_candidates`, which removes
    exact content-hash duplicates.  ``RedundancyPruner`` targets semantically
    equivalent records that differ in phrasing but carry the same information.

    Attributes:
        similarity_threshold: Cosine-similarity cutoff in ``[0.0, 1.0]``.
                              Pairs scoring at or above this value are treated
                              as near-duplicates.  Defaults to ``0.95``.
    """

    similarity_threshold: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], "
                f"got {self.similarity_threshold!r}"
            )

    def find_redundant(
        self,
        records: list[MemoryRecord],
        *,
        retention_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> list[PruneCandidate]:
        """Return prune candidates for near-duplicate records in *records*.

        Algorithm (greedy, O(n²) in embedding dimension × record count):

        1. Sort records newest-first so the most recently created copy becomes
           the canonical representative of each cluster.
        2. For each record (oldest processed last), compute cosine similarity
           against every previously selected canonical.
        3. If similarity ≥ ``self.similarity_threshold``, flag the record as
           :attr:`~shad.vault.contracts.PruneReason.REDUNDANT`; otherwise add
           it to the canonical set.

        Records with a zero-norm or empty embedding are never flagged (cosine
        similarity is undefined for the zero vector).

        Args:
            records:           :class:`~shad.vault.contracts.MemoryRecord`
                               objects to scan.  All non-empty embeddings must
                               share the same dimension.
            retention_seconds: Minimum seconds between flagging and eligibility
                               for removal.  ``0.0`` means immediately eligible.
            now:               Reference timestamp.  Defaults to
                               ``datetime.now(UTC)``.

        Returns:
            List of :class:`~shad.vault.contracts.PruneCandidate` objects, one
            per redundant record.  May be empty if no near-duplicates are found.

        Raises:
            ValueError: If two records have embeddings of different dimensions.
        """
        if not records:
            return []

        reference = now if now is not None else datetime.now(UTC)
        eligible_at = reference + timedelta(seconds=retention_seconds)

        # Newest-first → first seen wins as the canonical copy.
        sorted_records = sorted(records, key=lambda r: r.created_at, reverse=True)

        # Each entry: (record_id, unit-normed embedding)
        canonicals: list[tuple[str, list[float]]] = []
        candidates: list[PruneCandidate] = []
        expected_dim: int | None = None

        for record in sorted_records:
            emb = record.embedding
            if not emb:
                continue

            if expected_dim is None:
                expected_dim = len(emb)
            elif len(emb) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim}, "
                    f"got {len(emb)} for record {record.record_id!r}"
                )

            norm = _l2_norm(emb)
            if norm == 0.0:
                # Zero vector — treat as unique, never flag.
                canonicals.append((record.record_id, emb))
                continue

            normed = [x / norm for x in emb]

            # Find the first canonical that this record is similar to.
            matched_canonical: str | None = None
            for canon_id, canon_normed in canonicals:
                if _dot(normed, canon_normed) >= self.similarity_threshold:
                    matched_canonical = canon_id
                    break

            if matched_canonical is not None:
                candidates.append(
                    PruneCandidate(
                        record_id=record.record_id,
                        snapshot_id=record.record_id,
                        reason=PruneReason.REDUNDANT,
                        decay_score=0.0,
                        flagged_at=reference,
                        eligible_at=eligible_at,
                        superseded_by=matched_canonical,
                    )
                )
            else:
                canonicals.append((record.record_id, normed))

        return candidates


# ---------------------------------------------------------------------------
# Post-consolidation hook adapter
# ---------------------------------------------------------------------------


@dataclass
class RedundancyPrunerHook:
    """PostConsolidationHook that runs RedundancyPruner after consolidation.

    Implements the :class:`~shad.vault.consolidation.PostConsolidationHook`
    protocol.  After a consolidation pass completes, this hook converts each
    :data:`~shad.vault.shadow_index.AnyMemoryRecord` to a
    :class:`~shad.vault.contracts.MemoryRecord` and delegates to
    :meth:`RedundancyPruner.find_redundant`, skipping records already flagged
    by the consolidation pipeline so that its REDUNDANT candidates are
    strictly additive.

    Attributes:
        pruner:            Underlying :class:`RedundancyPruner` instance.
                           Defaults to a new instance with
                           ``similarity_threshold=0.95``.
        retention_seconds: Forwarded to :meth:`RedundancyPruner.find_redundant`
                           as the minimum hold-off before flagged records
                           become eligible for removal.
    """

    pruner: RedundancyPruner = field(default_factory=RedundancyPruner)
    retention_seconds: float = 0.0

    def run(
        self,
        records: list[AnyMemoryRecord],
        result: ConsolidationResult,
    ) -> list[PruneCandidate]:
        """Return REDUNDANT prune candidates for near-duplicate records.

        Records whose ``record_id`` already appears in
        ``result.prune_candidates`` are excluded before similarity scoring so
        that the hook only surfaces genuinely new candidates.

        Args:
            records: Original record list from the consolidation call.
            result:  Completed consolidation result carrying existing candidates.

        Returns:
            List of :class:`~shad.vault.contracts.PruneCandidate` objects with
            ``reason == PruneReason.REDUNDANT``.  May be empty.
        """
        already_flagged = {pc.record_id for pc in result.prune_candidates}

        contract_records: list[MemoryRecord] = []
        for r in records:
            if r.record_id in already_flagged:
                continue
            embedding: list[float] = (
                r.metadata.embedding
                if r.metadata is not None and r.metadata.embedding is not None
                else []
            )
            contract_records.append(
                MemoryRecord(
                    record_id=r.record_id,
                    content=r.content,
                    embedding=embedding,
                    memory_type=r.memory_type,
                    created_at=r.created_at,
                )
            )

        if not contract_records:
            return []

        return self.pruner.find_redundant(
            contract_records,
            retention_seconds=self.retention_seconds,
        )
