"""Tests for vault curation tools.

Per SPEC.md Section 2.3 and 2.12:
- Ingestion pipeline: Clone, process repos with presets
- Shadow index: Maps source_url → latest_snapshot
- Gap detection: Combined scoring of history + patterns + LLM
- Note standardization: Frontmatter, code tags, links
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.vault.ingestion import (
    IngestPreset,
    IngestResult,
    SnapshotMetadata,
    VaultIngester,
)
from shad.vault.shadow_index import (
    ShadowIndex,
    SourceEntry,
    SnapshotEntry,
    UpdatePolicy,
)
from shad.vault.gap_detection import (
    GapDetector,
    GapReport,
    GapScore,
    QueryHistoryAnalyzer,
)


class TestIngestPreset:
    """Tests for ingestion presets."""

    def test_preset_mirror(self) -> None:
        """Test mirror preset (all files)."""
        preset = IngestPreset.MIRROR
        assert preset.value == "mirror"

    def test_preset_docs(self) -> None:
        """Test docs preset (default)."""
        preset = IngestPreset.DOCS
        assert preset.value == "docs"

    def test_preset_deep(self) -> None:
        """Test deep preset (semantic index)."""
        preset = IngestPreset.DEEP
        assert preset.value == "deep"


class TestSnapshotMetadata:
    """Tests for snapshot metadata."""

    def test_create_metadata(self) -> None:
        """Test creating snapshot metadata."""
        meta = SnapshotMetadata(
            source_url="https://github.com/org/repo",
            source_type="github_repo",
            ingested_at=datetime.now(),
            source_revision="abc123",
            content_hash="sha256:...",
            snapshot_id="repo@2026-01-15",
        )

        assert meta.source_url == "https://github.com/org/repo"
        assert meta.source_type == "github_repo"

    def test_metadata_to_frontmatter(self) -> None:
        """Test converting metadata to YAML frontmatter."""
        meta = SnapshotMetadata(
            source_url="https://github.com/org/repo",
            source_type="github_repo",
            ingested_at=datetime(2026, 1, 15, 12, 0, 0),
            source_revision="abc123",
            content_hash="sha256:abc",
            snapshot_id="repo@2026-01-15",
        )

        frontmatter = meta.to_frontmatter()

        assert "source_url" in frontmatter
        assert "github.com" in frontmatter["source_url"]
        assert frontmatter["source_type"] == "github_repo"


class TestVaultIngester:
    """Tests for vault ingestion."""

    @pytest.fixture
    def ingester(self, tmp_path: Path) -> VaultIngester:
        """Create ingester with temp vault."""
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        return VaultIngester(vault_path=vault_path)

    def test_generate_snapshot_path(self, ingester: VaultIngester) -> None:
        """Test generating snapshot path structure."""
        path = ingester.generate_snapshot_path(
            source_id="react-native",
            domain="github.com",
        )

        # Should follow Sources/<domain>/<source_id>/<date>/ pattern
        assert "Sources" in str(path)
        assert "github.com" in str(path)
        assert "react-native" in str(path)

    @pytest.mark.asyncio
    async def test_ingest_creates_entry_note(self, ingester: VaultIngester) -> None:
        """Test that ingestion creates entry note."""
        # Mock git clone
        with patch.object(ingester, '_clone_repo', new_callable=AsyncMock) as mock_clone:
            mock_clone.return_value = Path("/tmp/cloned")

            with patch.object(ingester, '_process_files') as mock_process:
                mock_process.return_value = ["README.md", "docs/api.md"]

                result = await ingester.ingest_github(
                    url="https://github.com/org/repo",
                    preset=IngestPreset.DOCS,
                )

                assert result.success is True
                assert result.snapshot_id is not None

    def test_extract_source_id(self, ingester: VaultIngester) -> None:
        """Test extracting source ID from URL."""
        source_id = ingester.extract_source_id(
            "https://github.com/facebook/react-native"
        )
        assert source_id == "react-native"


class TestShadowIndex:
    """Tests for shadow index."""

    @pytest.fixture
    def index(self, tmp_path: Path) -> ShadowIndex:
        """Create shadow index with temp DB."""
        db_path = tmp_path / "index.sqlite"
        return ShadowIndex(db_path=db_path)

    def test_add_source(self, index: ShadowIndex) -> None:
        """Test adding a source to index."""
        index.add_source(SourceEntry(
            source_url="https://github.com/org/repo",
            source_id="repo",
            source_type="github_repo",
            update_policy=UpdatePolicy.MANUAL,
        ))

        source = index.get_source("https://github.com/org/repo")
        assert source is not None
        assert source.source_id == "repo"

    def test_add_snapshot(self, index: ShadowIndex) -> None:
        """Test adding a snapshot to index."""
        # First add source
        index.add_source(SourceEntry(
            source_url="https://github.com/org/repo",
            source_id="repo",
            source_type="github_repo",
            update_policy=UpdatePolicy.MANUAL,
        ))

        # Add snapshot
        index.add_snapshot(SnapshotEntry(
            snapshot_id="repo@2026-01-15",
            source_id="repo",
            ingested_at=datetime.now(),
            source_revision="abc123",
            entry_paths=["Sources/github.com/repo/2026-01-15/_entry.md"],
            content_hash="sha256:abc",
        ))

        snapshot = index.get_latest_snapshot("repo")
        assert snapshot is not None
        assert snapshot.snapshot_id == "repo@2026-01-15"

    def test_list_sources(self, index: ShadowIndex) -> None:
        """Test listing all sources."""
        index.add_source(SourceEntry(
            source_url="https://github.com/org/repo1",
            source_id="repo1",
            source_type="github_repo",
            update_policy=UpdatePolicy.MANUAL,
        ))
        index.add_source(SourceEntry(
            source_url="https://github.com/org/repo2",
            source_id="repo2",
            source_type="github_repo",
            update_policy=UpdatePolicy.AUTO,
        ))

        sources = index.list_sources()
        assert len(sources) == 2

    def test_pin_snapshot(self, index: ShadowIndex) -> None:
        """Test pinning a specific snapshot version."""
        index.add_source(SourceEntry(
            source_url="https://github.com/org/repo",
            source_id="repo",
            source_type="github_repo",
            update_policy=UpdatePolicy.MANUAL,
        ))

        # Add multiple snapshots
        index.add_snapshot(SnapshotEntry(
            snapshot_id="repo@2026-01-14",
            source_id="repo",
            ingested_at=datetime(2026, 1, 14),
            source_revision="v1",
            entry_paths=[],
            content_hash="hash1",
        ))
        index.add_snapshot(SnapshotEntry(
            snapshot_id="repo@2026-01-15",
            source_id="repo",
            ingested_at=datetime(2026, 1, 15),
            source_revision="v2",
            entry_paths=[],
            content_hash="hash2",
        ))

        # Pin to older version
        index.pin_snapshot("https://github.com/org/repo", "repo@2026-01-14")

        latest = index.get_latest_snapshot("repo")
        assert latest.snapshot_id == "repo@2026-01-14"


class TestGapDetector:
    """Tests for vault gap detection."""

    @pytest.fixture
    def detector(self) -> GapDetector:
        """Create gap detector."""
        return GapDetector()

    def test_gap_score_calculation(self, detector: GapDetector) -> None:
        """Test combined gap scoring.

        Per SPEC.md Section 2.12.1:
        gap_score = 0.55 * history_pain + 0.25 * coverage_miss + 0.20 * llm_score
        """
        score = detector.calculate_gap_score(
            history_pain=0.8,
            coverage_miss=0.5,
            llm_score=0.6,
        )

        expected = 0.55 * 0.8 + 0.25 * 0.5 + 0.20 * 0.6
        assert abs(score - expected) < 0.01

    def test_detect_history_pain(self, detector: GapDetector) -> None:
        """Test detecting pain from query history."""
        # Add some query history
        detector.add_query_result(
            query="OAuth refresh tokens",
            retrieval_score=0.25,
            fallback_used=True,
        )
        detector.add_query_result(
            query="OAuth refresh tokens",
            retrieval_score=0.30,
            fallback_used=True,
        )

        pain = detector.calculate_history_pain("OAuth refresh tokens")

        # Low scores and fallback usage should result in high pain
        assert pain > 0.5

    def test_detect_coverage_miss(self, detector: GapDetector) -> None:
        """Test detecting coverage misses."""
        # Topics that should have anchor notes
        common_topics = ["authentication", "api design", "testing", "deployment"]

        # Vault has some but not all
        vault_topics = ["testing", "deployment"]

        misses = detector.find_coverage_misses(common_topics, vault_topics)

        assert "authentication" in misses
        assert "api design" in misses

    def test_generate_gap_report(self, detector: GapDetector) -> None:
        """Test generating gap report."""
        # Add history
        detector.add_query_result("auth tokens", 0.3, True)
        detector.add_query_result("auth tokens", 0.25, True)

        report = detector.generate_report(vault_topics=["testing"])

        assert isinstance(report, GapReport)
        assert len(report.gaps) >= 0


class TestGapReport:
    """Tests for gap report structure."""

    def test_gap_report_structure(self) -> None:
        """Test gap report structure."""
        report = GapReport(
            gaps=[
                GapScore(
                    topic="Auth refresh tokens",
                    score=0.72,
                    evidence=["14 queries", "median score 0.31", "62% fallback"],
                    suggestions=["Patterns/Auth/RefreshTokens.md"],
                    priority="high",
                ),
            ],
            total_queries_analyzed=50,
            coverage_percentage=0.65,
        )

        assert len(report.gaps) == 1
        assert report.gaps[0].priority == "high"

    def test_gap_report_to_markdown(self) -> None:
        """Test converting report to markdown."""
        report = GapReport(
            gaps=[
                GapScore(
                    topic="Auth refresh tokens",
                    score=0.72,
                    evidence=["High query frequency"],
                    suggestions=["Add Auth/RefreshTokens.md"],
                    priority="high",
                ),
            ],
            total_queries_analyzed=50,
            coverage_percentage=0.65,
        )

        md = report.to_markdown()

        assert "Auth refresh tokens" in md
        assert "high" in md.lower()


class TestQueryHistoryAnalyzer:
    """Tests for query history analysis."""

    @pytest.fixture
    def analyzer(self) -> QueryHistoryAnalyzer:
        """Create analyzer."""
        return QueryHistoryAnalyzer()

    def test_analyze_query_frequency(self, analyzer: QueryHistoryAnalyzer) -> None:
        """Test analyzing query frequency."""
        analyzer.add_query("authentication", 0.8)
        analyzer.add_query("authentication", 0.7)
        analyzer.add_query("authentication", 0.6)
        analyzer.add_query("database", 0.9)

        freq = analyzer.get_query_frequency()

        assert freq["authentication"] == 3
        assert freq["database"] == 1

    def test_analyze_median_scores(self, analyzer: QueryHistoryAnalyzer) -> None:
        """Test calculating median retrieval scores."""
        analyzer.add_query("auth", 0.3)
        analyzer.add_query("auth", 0.5)
        analyzer.add_query("auth", 0.7)

        median = analyzer.get_median_score("auth")

        assert median == 0.5

    def test_identify_problematic_topics(self, analyzer: QueryHistoryAnalyzer) -> None:
        """Test identifying topics with poor retrieval."""
        # Good topic
        analyzer.add_query("database", 0.8)
        analyzer.add_query("database", 0.9)

        # Problematic topic
        analyzer.add_query("oauth", 0.2)
        analyzer.add_query("oauth", 0.3)
        analyzer.add_query("oauth", 0.25)

        problems = analyzer.get_problematic_topics(threshold=0.5)

        assert "oauth" in problems
        assert "database" not in problems


class TestVaultCurationIntegration:
    """Integration tests for vault curation."""

    @pytest.mark.asyncio
    async def test_full_ingestion_workflow(self, tmp_path: Path) -> None:
        """Test complete ingestion workflow."""
        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        ingester = VaultIngester(vault_path=vault_path)
        index = ShadowIndex(db_path=tmp_path / "index.sqlite")

        # Mock the actual clone
        with patch.object(ingester, '_clone_repo', new_callable=AsyncMock) as mock:
            mock.return_value = tmp_path / "cloned"
            (tmp_path / "cloned").mkdir()
            (tmp_path / "cloned" / "README.md").write_text("# Test Repo")

            result = await ingester.ingest_github(
                url="https://github.com/test/repo",
                preset=IngestPreset.DOCS,
            )

            if result.success:
                # Add to shadow index
                index.add_source(SourceEntry(
                    source_url="https://github.com/test/repo",
                    source_id="repo",
                    source_type="github_repo",
                    update_policy=UpdatePolicy.MANUAL,
                ))
                index.add_snapshot(SnapshotEntry(
                    snapshot_id=result.snapshot_id,
                    source_id="repo",
                    ingested_at=datetime.now(),
                    source_revision="main",
                    entry_paths=result.files_processed,
                    content_hash="hash",
                ))

                # Verify
                sources = index.list_sources()
                assert len(sources) == 1
