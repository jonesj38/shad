"""Tests for vault similarity scoring — empty-input contracts.

Verifies that score_pairwise and build_similarity_report handle an empty
record list gracefully: returning empty results with no errors raised.
"""

from __future__ import annotations

from shad.vault.similarity import SimilarityReport, build_similarity_report, score_pairwise


class TestEmptyInput:
    """Passing no records must return empty results without raising."""

    def test_score_pairwise_empty_returns_empty_list(self) -> None:
        result = score_pairwise([])
        assert result == []

    def test_score_pairwise_empty_no_error(self) -> None:
        # Must not raise any exception
        score_pairwise([])

    def test_build_similarity_report_empty_returns_report(self) -> None:
        report = build_similarity_report([])
        assert isinstance(report, SimilarityReport)

    def test_build_similarity_report_empty_pairs(self) -> None:
        report = build_similarity_report([])
        assert report.pairs == ()

    def test_build_similarity_report_empty_clusters(self) -> None:
        report = build_similarity_report([])
        assert report.clusters == ()

    def test_build_similarity_report_records_scanned_zero(self) -> None:
        report = build_similarity_report([])
        assert report.records_scanned == 0

    def test_build_similarity_report_near_duplicate_count_zero(self) -> None:
        report = build_similarity_report([])
        assert report.near_duplicate_count == 0

    def test_build_similarity_report_threshold_preserved(self) -> None:
        report = build_similarity_report([], threshold=0.9)
        assert report.threshold == 0.9
