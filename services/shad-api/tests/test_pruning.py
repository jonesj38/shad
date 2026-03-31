"""Unit tests for similarity-threshold deduplication in RedundancyPruner.

Covers:
- Records at/above the similarity threshold are flagged REDUNDANT.
- Records strictly below the threshold are kept.
- Newest record in a near-duplicate cluster is the canonical (kept).
- All records below threshold kept when no near-duplicates exist.
- Zero-vector records are never flagged regardless of threshold.
- Empty-embedding records are skipped.
- Single-record input never produces candidates.
- Retention window is propagated to eligible_at.
- Embedding dimension mismatch raises ValueError.
- Threshold boundary validation (outside [0, 1] raises ValueError).
- Identical embeddings at threshold == 1.0 are flagged.
- Records just below threshold are not flagged.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.contracts import MemoryRecord, PruneReason
from shad.vault.pruning import RedundancyPruner
from shad.vault.shadow_index import MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _record(
    record_id: str,
    embedding: list[float],
    *,
    created_at: datetime | None = None,
) -> MemoryRecord:
    ts = created_at or _BASE_TIME
    return MemoryRecord(
        record_id=record_id,
        content=f"content for {record_id}",
        embedding=embedding,
        memory_type=MemoryType.SEMANTIC,
        created_at=ts,
        updated_at=ts,
    )


def _unit(vec: list[float]) -> list[float]:
    """Return a unit-normed copy of *vec*."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRedundancyPrunerThreshold:
    """Records at or above similarity_threshold are flagged REDUNDANT."""

    def test_identical_embeddings_flagged(self) -> None:
        """Two records with the same embedding → older one is REDUNDANT."""
        emb = _unit([1.0, 0.0, 0.0])
        newer = _record("newer", emb, created_at=_BASE_TIME + timedelta(hours=1))
        older = _record("older", emb, created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant([newer, older], now=_BASE_TIME)

        assert len(candidates) == 1
        assert candidates[0].record_id == "older"
        assert candidates[0].reason == PruneReason.REDUNDANT

    def test_at_threshold_exact_flagged(self) -> None:
        """A pair with cosine similarity exactly equal to the threshold is flagged."""
        # Build two vectors whose dot product (after normalisation) equals 0.9.
        # Use 2-D: v1 = [1, 0], v2 = [cos θ, sin θ] where cos θ = 0.9.
        cos_theta = 0.9
        sin_theta = math.sqrt(1 - cos_theta**2)
        emb_a = [1.0, 0.0]
        emb_b = [cos_theta, sin_theta]

        newer = _record("newer", emb_a, created_at=_BASE_TIME + timedelta(hours=1))
        older = _record("older", emb_b, created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.9)
        candidates = pruner.find_redundant([newer, older], now=_BASE_TIME)

        assert len(candidates) == 1
        assert candidates[0].record_id == "older"

    def test_above_threshold_flagged(self) -> None:
        """Similarity above threshold → flagged."""
        # cosine similarity ≈ 0.999 (nearly identical direction)
        emb_a = _unit([3.0, 0.1])
        emb_b = _unit([3.0, 0.2])

        newer = _record("newer", emb_a, created_at=_BASE_TIME + timedelta(hours=1))
        older = _record("older", emb_b, created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant([newer, older], now=_BASE_TIME)

        assert any(c.record_id == "older" for c in candidates)

    def test_below_threshold_not_flagged(self) -> None:
        """Similarity strictly below threshold → both records kept."""
        # Orthogonal vectors → cosine similarity = 0.0
        emb_a = [1.0, 0.0]
        emb_b = [0.0, 1.0]

        r1 = _record("r1", emb_a, created_at=_BASE_TIME + timedelta(hours=1))
        r2 = _record("r2", emb_b, created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant([r1, r2], now=_BASE_TIME)

        assert candidates == []

    def test_near_duplicate_cluster_keeps_newest(self) -> None:
        """In a cluster of near-duplicates only the newest is kept."""
        emb = _unit([1.0, 0.0])
        r1 = _record("oldest", emb, created_at=_BASE_TIME)
        r2 = _record("middle", emb, created_at=_BASE_TIME + timedelta(hours=1))
        r3 = _record("newest", emb, created_at=_BASE_TIME + timedelta(hours=2))

        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant([r1, r2, r3], now=_BASE_TIME)

        flagged_ids = {c.record_id for c in candidates}
        assert "newest" not in flagged_ids
        assert "oldest" in flagged_ids
        assert "middle" in flagged_ids
        assert len(candidates) == 2

    def test_distinct_records_none_flagged(self) -> None:
        """Records with distinct embeddings below threshold produce no candidates."""
        records = [
            _record("a", [1.0, 0.0, 0.0], created_at=_BASE_TIME + timedelta(hours=3)),
            _record("b", [0.0, 1.0, 0.0], created_at=_BASE_TIME + timedelta(hours=2)),
            _record("c", [0.0, 0.0, 1.0], created_at=_BASE_TIME + timedelta(hours=1)),
        ]
        pruner = RedundancyPruner(similarity_threshold=0.95)
        assert pruner.find_redundant(records, now=_BASE_TIME) == []

    def test_single_record_never_flagged(self) -> None:
        """A single record can never be a near-duplicate of anything."""
        r = _record("only", [1.0, 0.0])
        pruner = RedundancyPruner(similarity_threshold=0.95)
        assert pruner.find_redundant([r], now=_BASE_TIME) == []

    def test_empty_list_returns_empty(self) -> None:
        pruner = RedundancyPruner(similarity_threshold=0.95)
        assert pruner.find_redundant([], now=_BASE_TIME) == []


class TestRedundancyPrunerEdgeCases:
    """Edge-case and invariant tests."""

    def test_zero_vector_never_flagged(self) -> None:
        """Zero-norm embeddings are treated as unique and never pruned."""
        zero = [0.0, 0.0, 0.0]
        unit = _unit([1.0, 0.0, 0.0])

        r_zero = _record("zero", zero, created_at=_BASE_TIME + timedelta(hours=1))
        r_unit = _record("unit", unit, created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.0)  # flag everything
        candidates = pruner.find_redundant([r_zero, r_unit], now=_BASE_TIME)

        flagged_ids = {c.record_id for c in candidates}
        assert "zero" not in flagged_ids

    def test_empty_embedding_skipped(self) -> None:
        """Records with an empty embedding list are ignored."""
        r_empty = _record("empty", [], created_at=_BASE_TIME + timedelta(hours=1))
        r_valid = _record("valid", [1.0, 0.0], created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant([r_empty, r_valid], now=_BASE_TIME)

        assert all(c.record_id != "empty" for c in candidates)

    def test_retention_seconds_sets_eligible_at(self) -> None:
        """eligible_at = now + retention_seconds."""
        emb = _unit([1.0, 0.0])
        newer = _record("newer", emb, created_at=_BASE_TIME + timedelta(hours=1))
        older = _record("older", emb, created_at=_BASE_TIME)

        retention = 3600.0
        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant(
            [newer, older], retention_seconds=retention, now=_BASE_TIME
        )

        assert len(candidates) == 1
        expected_eligible = _BASE_TIME + timedelta(seconds=retention)
        assert candidates[0].eligible_at == expected_eligible
        assert candidates[0].flagged_at == _BASE_TIME

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched embedding dimensions raise ValueError."""
        r1 = _record("r1", [1.0, 0.0], created_at=_BASE_TIME + timedelta(hours=1))
        r2 = _record("r2", [1.0, 0.0, 0.0], created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.95)
        with pytest.raises(ValueError, match="dimension mismatch"):
            pruner.find_redundant([r1, r2], now=_BASE_TIME)

    def test_invalid_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold"):
            RedundancyPruner(similarity_threshold=1.1)

    def test_invalid_threshold_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold"):
            RedundancyPruner(similarity_threshold=-0.01)

    def test_threshold_boundary_zero_flags_all_nonzero_duplicates(self) -> None:
        """threshold=0.0 treats any non-negative cosine similarity as redundant."""
        # Two non-parallel vectors still have cosine ≥ 0 if components are positive.
        r1 = _record("r1", [1.0, 0.5], created_at=_BASE_TIME + timedelta(hours=1))
        r2 = _record("r2", [0.5, 1.0], created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=0.0)
        candidates = pruner.find_redundant([r1, r2], now=_BASE_TIME)

        # r1 is canonical (newer); r2 has non-negative cosine with r1 → flagged.
        assert len(candidates) == 1
        assert candidates[0].record_id == "r2"

    def test_threshold_one_only_flags_identical_direction(self) -> None:
        """threshold=1.0 only flags vectors with cosine similarity == 1.0."""
        emb = _unit([1.0, 2.0, 3.0])
        slightly_off = _unit([1.0, 2.0, 3.001])  # cosine < 1.0

        newer_same = _record("same", list(emb), created_at=_BASE_TIME + timedelta(hours=2))
        older_same = _record("same_old", list(emb), created_at=_BASE_TIME + timedelta(hours=1))
        different = _record("diff", slightly_off, created_at=_BASE_TIME)

        pruner = RedundancyPruner(similarity_threshold=1.0)
        candidates = pruner.find_redundant(
            [newer_same, older_same, different], now=_BASE_TIME
        )

        flagged_ids = {c.record_id for c in candidates}
        assert "same_old" in flagged_ids
        assert "diff" not in flagged_ids

    def test_candidate_reason_is_redundant(self) -> None:
        """All candidates produced by RedundancyPruner carry PruneReason.REDUNDANT."""
        emb = _unit([1.0, 0.0])
        records = [
            _record("new", emb, created_at=_BASE_TIME + timedelta(hours=2)),
            _record("mid", emb, created_at=_BASE_TIME + timedelta(hours=1)),
            _record("old", emb, created_at=_BASE_TIME),
        ]
        pruner = RedundancyPruner(similarity_threshold=0.95)
        candidates = pruner.find_redundant(records, now=_BASE_TIME)

        assert all(c.reason == PruneReason.REDUNDANT for c in candidates)
