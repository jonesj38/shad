"""Pairwise similarity scoring for vault memory records.

Provides :func:`score_pairwise` and :func:`build_similarity_report` to surface
near-duplicate content across a set of :class:`~shad.vault.contracts.MemoryRecord`
objects.

Built on top of the existing :func:`~shad.vault.consolidation.cosine_similarity`
primitive and the :class:`~shad.vault.pruning.RedundancyPruner` near-duplicate
detector.  The key addition is **explicit pairwise scores** — callers can inspect
exactly *how* similar any two records are, not just whether they exceed a pruning
threshold.

Typical usage::

    from shad.vault.similarity import build_similarity_report

    report = build_similarity_report(records, threshold=0.85)
    for pair in report.pairs:
        print(pair.record_id_a, pair.record_id_b, pair.similarity)
    for cluster in report.clusters:
        if len(cluster) > 1:
            print("near-duplicate cluster:", cluster)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shad.vault.contracts import MemoryRecord

__all__ = [
    "SimilarityPair",
    "SimilarityReport",
    "build_similarity_report",
    "score_pairwise",
]

# ---------------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityPair:
    """Cosine-similarity score between two vault records.

    Attributes:
        record_id_a: ``record_id`` of the first record.
        record_id_b: ``record_id`` of the second record.
        similarity:  Cosine similarity in ``[-1.0, 1.0]``; for normalised
                     embedding vectors this is effectively in ``[0.0, 1.0]``.
                     A value of ``1.0`` means the vectors are identical in
                     direction; ``0.0`` means orthogonal.
    """

    record_id_a: str
    record_id_b: str
    similarity: float

    def __post_init__(self) -> None:
        if self.record_id_a == self.record_id_b:
            raise ValueError(
                f"record_id_a and record_id_b must be different, "
                f"got {self.record_id_a!r} for both"
            )


@dataclass(frozen=True)
class SimilarityReport:
    """Aggregate result of a pairwise similarity scan.

    Attributes:
        pairs:           All pairs whose similarity is >= ``threshold``,
                         sorted by similarity descending.
        clusters:        Connected components of near-duplicate records.
                         Each inner ``frozenset`` contains the ``record_id``
                         values of all mutually reachable records (transitivity
                         via union-find).  Singleton sets are *not* included —
                         only groups of two or more.
        threshold:       The cutoff used to produce this report.
        records_scanned: Total number of records evaluated.
    """

    pairs: tuple[SimilarityPair, ...]
    clusters: tuple[frozenset[str], ...]
    threshold: float
    records_scanned: int

    @property
    def near_duplicate_count(self) -> int:
        """Number of records that belong to at least one near-duplicate cluster."""
        seen: set[str] = set()
        for cluster in self.clusters:
            seen.update(cluster)
        return len(seen)

    def pairs_for(self, record_id: str) -> list[SimilarityPair]:
        """Return all pairs that include *record_id*, sorted by similarity desc.

        Args:
            record_id: The record to query.

        Returns:
            Matching :class:`SimilarityPair` objects, highest similarity first.
        """
        return sorted(
            (
                p
                for p in self.pairs
                if p.record_id_a == record_id or p.record_id_b == record_id
            ),
            key=lambda p: p.similarity,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; returns 0.0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_pairwise(
    records: list[MemoryRecord],
    threshold: float = 0.85,
) -> list[SimilarityPair]:
    """Return all near-duplicate pairs from *records* above *threshold*.

    Iterates every unique pair (i, j) and computes cosine similarity over
    their ``embedding`` vectors.  Pairs where either record has an empty or
    zero-norm embedding are silently skipped (similarity is undefined).

    Complexity is O(n² × d) where *n* is the number of records and *d* is
    the embedding dimension.  For large collections (> 10 000 records) callers
    should pre-filter to candidate subsets using tag or memory-type buckets
    before calling this function.

    Args:
        records:   :class:`~shad.vault.contracts.MemoryRecord` objects to scan.
                   All non-empty embeddings must share the same dimension.
        threshold: Minimum cosine similarity in ``[0.0, 1.0]`` for a pair to
                   be included in the result.  Defaults to ``0.85``.

    Returns:
        :class:`SimilarityPair` list sorted by ``similarity`` descending.
        Empty list when *records* has fewer than two elements or no pair
        exceeds the threshold.

    Raises:
        ValueError: If *threshold* is not in ``[0.0, 1.0]``.
        ValueError: If two non-empty embeddings have different dimensions.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold!r}")

    n = len(records)
    if n < 2:
        return []

    # Pre-compute embeddings once; None means the record cannot be compared.
    embeddings: list[list[float] | None] = [r.embedding or None for r in records]

    # Validate dimension consistency across non-empty embeddings.
    expected_dim: int | None = None
    for emb in embeddings:
        if emb is not None:
            if expected_dim is None:
                expected_dim = len(emb)
            elif len(emb) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim}, "
                    f"got {len(emb)}"
                )

    pairs: list[SimilarityPair] = []
    for i in range(n):
        if embeddings[i] is None:
            continue
        for j in range(i + 1, n):
            if embeddings[j] is None:
                continue
            sim = _cosine(embeddings[i], embeddings[j])  # type: ignore[arg-type]
            if sim >= threshold:
                pairs.append(
                    SimilarityPair(
                        record_id_a=records[i].record_id,
                        record_id_b=records[j].record_id,
                        similarity=sim,
                    )
                )

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


def build_similarity_report(
    records: list[MemoryRecord],
    threshold: float = 0.85,
) -> SimilarityReport:
    """Build a :class:`SimilarityReport` for a collection of vault records.

    Runs :func:`score_pairwise` to find all near-duplicate pairs, then applies
    union-find to cluster transitively connected records.

    Args:
        records:   :class:`~shad.vault.contracts.MemoryRecord` objects to scan.
        threshold: Cosine-similarity cutoff.  Defaults to ``0.85``.

    Returns:
        :class:`SimilarityReport` with all pairs and connected-component
        clusters.
    """
    pairs = score_pairwise(records, threshold=threshold)

    # Union-find over record IDs to build connected components.
    parent: dict[str, str] = {r.record_id: r.record_id for r in records}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for pair in pairs:
        union(pair.record_id_a, pair.record_id_b)

    # Collect clusters; exclude singletons.
    buckets: dict[str, set[str]] = {}
    for r in records:
        root = find(r.record_id)
        buckets.setdefault(root, set()).add(r.record_id)

    clusters = tuple(
        frozenset(members) for members in buckets.values() if len(members) > 1
    )

    return SimilarityReport(
        pairs=tuple(pairs),
        clusters=clusters,
        threshold=threshold,
        records_scanned=len(records),
    )
