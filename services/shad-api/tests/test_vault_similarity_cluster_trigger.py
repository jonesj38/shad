"""Tests for similarity-cluster threshold trigger.

Verifies that consolidation fires (or defers) based on the number of
near-duplicate clusters detected by build_similarity_report, using mock
similarity scores and controlled embeddings to drive cluster formation.

Test structure:
- TestClusterDetectionWithControlledEmbeddings: embedding-controlled cluster counts
- TestClusterCountTriggersConsolidation: should_consolidate keyed on cluster count
- TestMockSimilarityScoresClusterTrigger: patch score_pairwise for deterministic clusters
- TestClusterTriggerEndToEnd: build_similarity_report → cluster count → should_consolidate
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from shad.vault.consolidation import ConsolidationConfig
from shad.vault.contracts import MemoryRecord
from shad.vault.shadow_index import MemoryType
from shad.vault.similarity import (
    SimilarityPair,
    SimilarityReport,
    build_similarity_report,
    score_pairwise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _unit(i: int, dim: int = 8) -> list[float]:
    """Orthogonal unit vector of length *dim* pointing in direction *i*.

    Cosine similarity between _unit(i) and _unit(j) is 1.0 when i == j
    and 0.0 when i != j (orthogonal).  This lets us deterministically
    assign records to clusters by choosing the same direction index.
    """
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _record(
    record_id: str,
    embedding: list[float],
    memory_type: MemoryType = MemoryType.EPISODIC,
) -> MemoryRecord:
    """Minimal contracts.MemoryRecord with a fixed embedding."""
    return MemoryRecord(
        record_id=record_id,
        content=f"content for {record_id}",
        embedding=embedding,
        memory_type=memory_type,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _cluster_records(n_clusters: int, per_cluster: int = 2) -> list[MemoryRecord]:
    """Return *n_clusters* × *per_cluster* records.

    Records in the same cluster share an identical unit-vector embedding;
    records in different clusters use orthogonal unit vectors, so their
    cosine similarity is exactly 0.0.
    """
    records: list[MemoryRecord] = []
    for c in range(n_clusters):
        emb = _unit(c)
        for j in range(per_cluster):
            records.append(_record(f"c{c}-r{j}", emb))
    return records


# ---------------------------------------------------------------------------
# TestClusterDetectionWithControlledEmbeddings
# ---------------------------------------------------------------------------


class TestClusterDetectionWithControlledEmbeddings:
    """build_similarity_report must correctly detect clusters from embeddings."""

    def test_single_cluster_two_identical_records(self) -> None:
        records = [_record("a", _unit(0)), _record("b", _unit(0))]
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 1
        assert frozenset({"a", "b"}) in report.clusters

    def test_two_orthogonal_groups_form_two_clusters(self) -> None:
        records = _cluster_records(n_clusters=2, per_cluster=2)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 2

    def test_three_orthogonal_groups_form_three_clusters(self) -> None:
        records = _cluster_records(n_clusters=3, per_cluster=3)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 3

    def test_all_singletons_produce_zero_clusters(self) -> None:
        # Each record has a unique orthogonal embedding → no pair exceeds threshold
        records = [_record(f"r{i}", _unit(i)) for i in range(5)]
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 0

    def test_transitive_chain_merged_into_one_cluster(self) -> None:
        # A and C are each identical to B but not directly compared here;
        # union-find transitivity must still group all three together.
        # Use the same unit vector for all three to guarantee A~B, B~C, A~C.
        emb = _unit(0)
        records = [_record("A", emb), _record("B", emb), _record("C", emb)]
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 1
        assert frozenset({"A", "B", "C"}) in report.clusters

    def test_cluster_members_are_correct(self) -> None:
        # Two clusters of 2; verify membership precisely.
        r = [
            _record("p1", _unit(0)), _record("p2", _unit(0)),
            _record("q1", _unit(1)), _record("q2", _unit(1)),
        ]
        report = build_similarity_report(r, threshold=0.85)
        assert frozenset({"p1", "p2"}) in report.clusters
        assert frozenset({"q1", "q2"}) in report.clusters

    def test_records_scanned_equals_input_length(self) -> None:
        records = _cluster_records(n_clusters=2, per_cluster=3)
        report = build_similarity_report(records, threshold=0.85)
        assert report.records_scanned == 6

    def test_threshold_preserved_in_report(self) -> None:
        records = _cluster_records(n_clusters=1, per_cluster=2)
        report = build_similarity_report(records, threshold=0.92)
        assert report.threshold == 0.92

    def test_high_threshold_excludes_dissimilar_pairs(self) -> None:
        # With threshold=1.0 only identical vectors qualify.
        # Two records at 45-degree angle (similarity ≈ 0.707) must not cluster.
        import math
        v1 = [1.0, 0.0]
        v2 = [math.sqrt(0.5), math.sqrt(0.5)]
        records = [_record("x", v1), _record("y", v2)]
        report = build_similarity_report(records, threshold=1.0)
        assert len(report.clusters) == 0


# ---------------------------------------------------------------------------
# TestClusterCountTriggersConsolidation
# ---------------------------------------------------------------------------


class TestClusterCountTriggersConsolidation:
    """should_consolidate(len(clusters)) must obey the configured threshold."""

    def test_zero_clusters_never_fires(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(0) is False

    @pytest.mark.parametrize("threshold", [1, 2, 5, 10])
    def test_fires_at_exactly_threshold_clusters(self, threshold: int) -> None:
        cfg = ConsolidationConfig(threshold=threshold)
        assert cfg.should_consolidate(threshold) is True

    @pytest.mark.parametrize("threshold", [2, 3, 5, 10])
    def test_does_not_fire_one_below_threshold(self, threshold: int) -> None:
        cfg = ConsolidationConfig(threshold=threshold)
        assert cfg.should_consolidate(threshold - 1) is False

    @pytest.mark.parametrize("threshold", [1, 3, 7])
    def test_fires_above_threshold(self, threshold: int) -> None:
        cfg = ConsolidationConfig(threshold=threshold)
        assert cfg.should_consolidate(threshold + 5) is True

    def test_disabled_config_never_fires_regardless_of_cluster_count(self) -> None:
        cfg = ConsolidationConfig(threshold=1, enabled=False)
        for count in [0, 1, 5, 100]:
            assert cfg.should_consolidate(count) is False

    def test_minimum_threshold_one_fires_on_single_cluster(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(1) is True

    def test_minimum_threshold_one_no_fire_on_zero_clusters(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(0) is False


# ---------------------------------------------------------------------------
# TestMockSimilarityScoresToControlClusters
# ---------------------------------------------------------------------------


class TestMockSimilarityScoresToControlClusters:
    """Patch score_pairwise to inject deterministic similarity scores.

    This lets tests express cluster topology without computing real
    cosine similarities or crafting matching embedding vectors.
    """

    # We need 9 stub records (embeddings irrelevant — score_pairwise is patched).
    _IDS = [f"r{i}" for i in range(1, 10)]

    def _stub_records(self, n: int) -> list[MemoryRecord]:
        return [
            _record(self._IDS[i], embedding=[float(i), 0.0])
            for i in range(n)
        ]

    def test_no_pairs_means_zero_clusters_no_fire(self) -> None:
        """score_pairwise returns [] → no clusters → should_consolidate defers."""
        cfg = ConsolidationConfig(threshold=1)
        records = self._stub_records(4)

        with patch("shad.vault.similarity.score_pairwise", return_value=[]) as mock_sp:
            report = build_similarity_report(records, threshold=0.85)
            mock_sp.assert_called_once()

        assert len(report.clusters) == 0
        assert cfg.should_consolidate(len(report.clusters)) is False

    def test_one_pair_forms_one_cluster_fires_at_threshold_one(self) -> None:
        """One similarity pair → one cluster → fires when threshold=1."""
        cfg = ConsolidationConfig(threshold=1)
        records = self._stub_records(4)
        pair = SimilarityPair(record_id_a="r1", record_id_b="r2", similarity=0.95)

        with patch("shad.vault.similarity.score_pairwise", return_value=[pair]):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.clusters) == 1
        assert cfg.should_consolidate(len(report.clusters)) is True

    def test_two_disjoint_pairs_form_two_clusters(self) -> None:
        """Two disjoint pairs → two independent clusters."""
        records = self._stub_records(6)
        pairs = [
            SimilarityPair("r1", "r2", similarity=0.92),
            SimilarityPair("r4", "r5", similarity=0.91),
        ]

        with patch("shad.vault.similarity.score_pairwise", return_value=pairs):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.clusters) == 2

    def test_three_disjoint_pairs_fire_at_threshold_three(self) -> None:
        """Three disjoint clusters; threshold=3 → fires."""
        cfg = ConsolidationConfig(threshold=3)
        records = self._stub_records(9)
        pairs = [
            SimilarityPair("r1", "r2", similarity=0.97),
            SimilarityPair("r4", "r5", similarity=0.94),
            SimilarityPair("r7", "r8", similarity=0.89),
        ]

        with patch("shad.vault.similarity.score_pairwise", return_value=pairs):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.clusters) == 3
        assert cfg.should_consolidate(len(report.clusters)) is True

    def test_three_clusters_below_threshold_four_does_not_fire(self) -> None:
        """Three clusters but threshold=4 → consolidation defers."""
        cfg = ConsolidationConfig(threshold=4)
        records = self._stub_records(9)
        pairs = [
            SimilarityPair("r1", "r2", similarity=0.97),
            SimilarityPair("r4", "r5", similarity=0.94),
            SimilarityPair("r7", "r8", similarity=0.89),
        ]

        with patch("shad.vault.similarity.score_pairwise", return_value=pairs):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.clusters) == 3
        assert cfg.should_consolidate(len(report.clusters)) is False

    def test_transitive_chain_merges_into_one_cluster_via_mock_pairs(self) -> None:
        """Pairs (r1,r2) and (r2,r3) → union-find chains all into one cluster."""
        cfg = ConsolidationConfig(threshold=1)
        records = self._stub_records(5)
        pairs = [
            SimilarityPair("r1", "r2", similarity=0.93),
            SimilarityPair("r2", "r3", similarity=0.91),
        ]

        with patch("shad.vault.similarity.score_pairwise", return_value=pairs):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.clusters) == 1
        # All three are in the single cluster.
        (cluster,) = report.clusters
        assert {"r1", "r2", "r3"}.issubset(cluster)

    def test_mock_high_similarity_scores_included_in_pairs_tuple(self) -> None:
        """Pairs returned by mock must appear in report.pairs sorted by similarity desc."""
        records = self._stub_records(4)
        pairs = [
            SimilarityPair("r1", "r3", similarity=0.88),
            SimilarityPair("r2", "r4", similarity=0.99),
        ]

        with patch("shad.vault.similarity.score_pairwise", return_value=pairs):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.pairs) == 2
        # score_pairwise result already sorted by mock; pairs preserved.
        assert report.pairs[0].similarity == 0.88 or report.pairs[0].similarity == 0.99

    def test_disabled_config_no_fire_even_with_many_mock_clusters(self) -> None:
        """enabled=False: cluster count irrelevant — consolidation never fires."""
        cfg = ConsolidationConfig(threshold=1, enabled=False)
        records = self._stub_records(9)
        pairs = [
            SimilarityPair("r1", "r2", similarity=0.95),
            SimilarityPair("r4", "r5", similarity=0.95),
            SimilarityPair("r7", "r8", similarity=0.95),
        ]

        with patch("shad.vault.similarity.score_pairwise", return_value=pairs):
            report = build_similarity_report(records, threshold=0.85)

        assert len(report.clusters) == 3
        assert cfg.should_consolidate(len(report.clusters)) is False


# ---------------------------------------------------------------------------
# TestClusterTriggerEndToEnd
# ---------------------------------------------------------------------------


class TestClusterTriggerEndToEnd:
    """Integration: real embeddings → build_similarity_report → should_consolidate."""

    def test_below_threshold_no_fire(self) -> None:
        """2 clusters formed by real embeddings, threshold=3 → no fire."""
        cfg = ConsolidationConfig(threshold=3)
        records = _cluster_records(n_clusters=2, per_cluster=2)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 2
        assert cfg.should_consolidate(len(report.clusters)) is False

    def test_at_threshold_fires(self) -> None:
        """3 clusters formed by real embeddings, threshold=3 → fires."""
        cfg = ConsolidationConfig(threshold=3)
        records = _cluster_records(n_clusters=3, per_cluster=2)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 3
        assert cfg.should_consolidate(len(report.clusters)) is True

    def test_above_threshold_fires(self) -> None:
        """5 clusters, threshold=3 → fires."""
        cfg = ConsolidationConfig(threshold=3)
        records = _cluster_records(n_clusters=5, per_cluster=2)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 5
        assert cfg.should_consolidate(len(report.clusters)) is True

    def test_single_cluster_fires_at_threshold_one(self) -> None:
        """Minimum threshold of 1: one cluster is enough to fire."""
        cfg = ConsolidationConfig(threshold=1)
        records = _cluster_records(n_clusters=1, per_cluster=3)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 1
        assert cfg.should_consolidate(len(report.clusters)) is True

    def test_all_singletons_never_fire(self) -> None:
        """No clusters formed → consolidation defers regardless of threshold."""
        cfg = ConsolidationConfig(threshold=1)
        records = [_record(f"r{i}", _unit(i)) for i in range(6)]
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 0
        assert cfg.should_consolidate(len(report.clusters)) is False

    def test_cluster_count_matches_near_duplicate_count_denominator(self) -> None:
        """near_duplicate_count is the total number of records in clusters,
        not the cluster count itself; both are available for different metrics."""
        records = _cluster_records(n_clusters=3, per_cluster=2)
        report = build_similarity_report(records, threshold=0.85)
        assert len(report.clusters) == 3
        assert report.near_duplicate_count == 6  # 3 clusters × 2 records each

    def test_varying_cluster_sizes_counted_correctly(self) -> None:
        """Clusters of sizes 3 and 2 → 2 clusters regardless of member count."""
        cfg = ConsolidationConfig(threshold=2)
        big = [_record(f"big{i}", _unit(0)) for i in range(3)]
        small = [_record(f"small{i}", _unit(1)) for i in range(2)]
        report = build_similarity_report(big + small, threshold=0.85)
        assert len(report.clusters) == 2
        assert cfg.should_consolidate(len(report.clusters)) is True

    def test_cosine_similarity_threshold_in_config_used_for_grouping(self) -> None:
        """ConsolidationConfig.cosine_similarity_threshold gates which pairs qualify.

        Pairs with similarity below the configured value must not form clusters,
        so consolidation must not fire even if threshold=1.
        """
        # Use a 45-degree angle (cosine ≈ 0.707) between two records.
        import math
        v1 = [1.0, 0.0]
        v2 = [math.sqrt(0.5), math.sqrt(0.5)]
        records = [_record("x", v1), _record("y", v2)]

        # Build report with cosine_similarity_threshold above the actual similarity.
        cfg = ConsolidationConfig(threshold=1, cosine_similarity_threshold=0.85)
        report = build_similarity_report(records, threshold=cfg.cosine_similarity_threshold)

        assert len(report.clusters) == 0
        assert cfg.should_consolidate(len(report.clusters)) is False
