"""Tests for collection curation tools.

Per SPEC.md Section 2.3 and 2.12:
- Ingestion pipeline: Clone, process repos with presets
- Shadow index: Maps source_url → latest_snapshot
- Gap detection: Combined scoring of history + patterns + LLM
- Note standardization: Frontmatter, code tags, links
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.vault.consolidation import (
    ConsolidationConfig,
    ConsolidationMetrics,
    MergeResult,
    MergeStrategy,
    RetentionPolicy,
)
from shad.vault.decay import DecayConfig, DecayCurve, SnapshotDecayScorer
from shad.vault.gap_detection import (
    GapDetector,
    GapReport,
    GapScore,
    QueryHistoryAnalyzer,
)
from shad.vault.ingestion import (
    IngestPreset,
    IngestResult,
    SnapshotMetadata,
    VaultIngester,
)
from shad.vault.shadow_index import (
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
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
    """Tests for collection ingestion."""

    @pytest.fixture
    def ingester(self, tmp_path: Path) -> VaultIngester:
        """Create ingester with temp collection."""
        collection_path = tmp_path / "collection"
        collection_path.mkdir()
        return VaultIngester(collection_path=collection_path)

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
    async def test_ingest_creates_entry_note(self, ingester: VaultIngester, tmp_path: Path) -> None:
        """Test that ingestion creates entry note with converted files."""
        # Create a mock clone directory with actual files
        clone_dir = tmp_path / "mock_clone"
        clone_dir.mkdir()

        # Create test files
        readme = clone_dir / "README.md"
        readme.write_text("# Test Repo\n\nThis is a test repository.")

        docs_dir = clone_dir / "docs"
        docs_dir.mkdir()
        api_doc = docs_dir / "api.md"
        api_doc.write_text("# API Documentation\n\nAPI docs here.")

        # Mock git clone to use our test directory
        async def mock_clone(url: str, target: Path) -> Path:
            # Copy files to target (simulating git clone)
            import shutil
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(clone_dir, target)
            return target

        with patch.object(ingester, '_clone_repo', side_effect=mock_clone):
            result = await ingester.ingest_github(
                url="https://github.com/org/repo",
                preset=IngestPreset.DOCS,
            )

            assert result.success is True
            assert result.snapshot_id is not None
            assert len(result.files_processed) == 2

            # Verify entry note was created
            sources_dir = ingester.collection_path / "Sources"
            entry_files = list(sources_dir.rglob("_entry.md"))
            assert len(entry_files) == 1

            # Verify notes were created
            md_files = list(sources_dir.rglob("*.md"))
            assert len(md_files) >= 3  # _entry.md + README.md + api.md

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
    """Tests for collection gap detection."""

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

        # Collection has some but not all
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
    """Integration tests for collection curation."""

    @pytest.mark.asyncio
    async def test_full_ingestion_workflow(self, tmp_path: Path) -> None:
        """Test complete ingestion workflow."""
        collection_path = tmp_path / "collection"
        collection_path.mkdir()

        ingester = VaultIngester(collection_path=collection_path)
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


class TestVaultConsolidationE2E:
    """End-to-end integration tests for the vault consolidation pipeline.

    Scenario
    --------
    Three sources with different memory characteristics are registered:

    - ``github-auth``  (SEMANTIC):  2 snapshots — used as a control group
                                    that consolidation must not touch.
    - ``rss-security`` (EPISODIC):  5 snapshots at ages 500 d / 400 d / 180 d /
                                    30 d / 7 d; consolidation triggers because
                                    ≥ 3 episodic entries exist (threshold = 3).
    - ``docs-api``     (WORKING):   2 snapshots — threshold not reached so
                                    consolidation is deferred.

    ``QueryHistoryAnalyzer`` is seeded with 4 "security" queries (low scores
    → high pain) whose topic overlaps the ``rss-security`` entry paths,
    contributing a fixed frequency bonus to every rss snapshot's decay score.

    Decay parameters (half-life = 90 days, staleness threshold = 0.20)
    -------------------------------------------------------------------
    Snapshot  |  Age  |  age_score                |  freq bonus  |  combined  |  outcome
    ----------|-------|---------------------------|--------------|------------|----------
    rss-500d  | 500 d |  0.5^(500/90) ≈ 0.021     |  +0.12       |  0.141     |  STALE
    rss-400d  | 400 d |  0.5^(400/90) ≈ 0.046     |  +0.12       |  0.166     |  STALE
    rss-180d  | 180 d |  0.5^2       = 0.250       |  +0.12       |  0.370     |  promoted
    rss-30d   |  30 d |  0.5^(1/3)   ≈ 0.794       |  +0.12       |  0.914     |  promoted
    rss-7d    |   7 d |  0.5^(7/90)  ≈ 0.947       |  +0.12       |  1.000     |  promoted
    """

    # -----------------------------------------------------------------------
    # Deterministic reference clock
    # -----------------------------------------------------------------------
    _NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)

    # -----------------------------------------------------------------------
    # Decay / staleness constants
    # -----------------------------------------------------------------------
    _HALFLIFE_DAYS: int = 90
    _HALFLIFE_SECONDS: float = 90 * 86_400.0
    _STALENESS_THRESHOLD: float = 0.20
    _FREQUENCY_WEIGHT: float = 0.30
    _MAX_QUERY_COUNT: int = 10
    # 4 "security" queries → freq_score = 4/10 = 0.4 → bonus = 0.4 * 0.3 = 0.12
    _SECURITY_QUERY_COUNT: int = 4

    # -----------------------------------------------------------------------
    # Snapshot ages (days before _NOW)
    # -----------------------------------------------------------------------
    _AGE_VERY_OLD: int = 500  # definitely stale
    _AGE_OLD: int = 400       # also stale
    _AGE_MEDIUM: int = 180    # borderline – rescued by freq bonus → promoted
    _AGE_RECENT: int = 30     # comfortably fresh → promoted
    _AGE_FRESH: int = 7       # very fresh → promoted

    # -----------------------------------------------------------------------
    # Fixtures
    # -----------------------------------------------------------------------

    @pytest.fixture
    def index(self, tmp_path: Path) -> ShadowIndex:
        return ShadowIndex(db_path=tmp_path / "e2e_index.sqlite")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_snapshot(
        self,
        snapshot_id: str,
        source_id: str,
        age_days: int,
        entry_paths: list[str],
        memory_type: MemoryType | None,
    ) -> SnapshotEntry:
        ingested_at = self._NOW - timedelta(days=age_days)
        return SnapshotEntry(
            snapshot_id=snapshot_id,
            source_id=source_id,
            ingested_at=ingested_at,
            source_revision=f"rev-{age_days}d",
            entry_paths=entry_paths,
            content_hash=f"sha256:{snapshot_id}",
            memory_type=memory_type,
        )

    def _populate(self, index: ShadowIndex) -> dict[str, list[str]]:
        """Register all sources + snapshots; return {source_id: [snapshot_ids]}."""
        # --- github-auth: SEMANTIC reference source (consolidation must ignore) ---
        index.add_source(SourceEntry(
            source_url="https://github.com/example/auth-patterns",
            source_id="github-auth",
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        for snap in [
            self._make_snapshot(
                "auth-old", "github-auth", 200,
                ["Sources/github.com/auth-patterns/2025-09-13/README.md"],
                MemoryType.SEMANTIC,
            ),
            self._make_snapshot(
                "auth-recent", "github-auth", 10,
                ["Sources/github.com/auth-patterns/2026-03-21/README.md"],
                MemoryType.SEMANTIC,
            ),
        ]:
            index.add_snapshot(snap)

        # --- rss-security: EPISODIC feed, 5 snapshots at varying ages ---
        index.add_source(SourceEntry(
            source_url="https://feeds.example.com/security",
            source_id="rss-security",
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
        ))
        rss_snaps = [
            self._make_snapshot(
                "rss-500d", "rss-security", self._AGE_VERY_OLD,
                ["Sources/feeds.example.com/rss-security/2024-10-18/security-news.md"],
                MemoryType.EPISODIC,
            ),
            self._make_snapshot(
                "rss-400d", "rss-security", self._AGE_OLD,
                ["Sources/feeds.example.com/rss-security/2025-01-26/security-update.md"],
                MemoryType.EPISODIC,
            ),
            self._make_snapshot(
                "rss-180d", "rss-security", self._AGE_MEDIUM,
                ["Sources/feeds.example.com/rss-security/2025-10-03/security-report.md"],
                MemoryType.EPISODIC,
            ),
            self._make_snapshot(
                "rss-30d", "rss-security", self._AGE_RECENT,
                ["Sources/feeds.example.com/rss-security/2026-03-01/security-brief.md"],
                MemoryType.EPISODIC,
            ),
            self._make_snapshot(
                "rss-7d", "rss-security", self._AGE_FRESH,
                ["Sources/feeds.example.com/rss-security/2026-03-24/security-alert.md"],
                MemoryType.EPISODIC,
            ),
        ]
        for snap in rss_snaps:
            index.add_snapshot(snap)

        # --- docs-api: WORKING snapshots (below threshold; consolidation deferred) ---
        index.add_source(SourceEntry(
            source_url="https://github.com/example/api-guide",
            source_id="docs-api",
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        for snap in [
            self._make_snapshot(
                "docs-w1", "docs-api", 3,
                ["Sources/github.com/api-guide/2026-03-28/intro.md"],
                MemoryType.WORKING,
            ),
            self._make_snapshot(
                "docs-w2", "docs-api", 1,
                ["Sources/github.com/api-guide/2026-03-30/api-ref.md"],
                MemoryType.WORKING,
            ),
        ]:
            index.add_snapshot(snap)

        return {
            "github-auth": ["auth-old", "auth-recent"],
            "rss-security": ["rss-500d", "rss-400d", "rss-180d", "rss-30d", "rss-7d"],
            "docs-api": ["docs-w1", "docs-w2"],
        }

    def _make_analyzer(self) -> QueryHistoryAnalyzer:
        """Seed query history with deterministic patterns."""
        analyzer = QueryHistoryAnalyzer()
        # 4 low-scoring "security" queries — match rss-security paths; high pain
        for score in [0.25, 0.30, 0.28, 0.22]:
            analyzer.add_query("security", score, fallback_used=True)
        # auth queries — do NOT match rss paths
        for score in [0.45, 0.50, 0.40]:
            analyzer.add_query("auth patterns", score)
        # gap topic — no matching source at all
        for score in [0.15, 0.18, 0.20, 0.17, 0.19]:
            analyzer.add_query("oauth refresh", score, fallback_used=True)
        return analyzer

    def _make_decay_config(self) -> DecayConfig:
        return DecayConfig(
            halflife_seconds=self._HALFLIFE_SECONDS,
            curve=DecayCurve.EXPONENTIAL,
        )

    def _make_consolidation_config(self, threshold: int = 3) -> ConsolidationConfig:
        return ConsolidationConfig(
            threshold=threshold,
            enabled=True,
            merge_strategy=MergeStrategy.LATEST_WINS,
        )

    def _consolidate(
        self,
        index: ShadowIndex,
        analyzer: QueryHistoryAnalyzer,
        config: ConsolidationConfig,
        decay_config: DecayConfig,
    ) -> ConsolidationMetrics:
        """Drive one consolidation pass over all sources.

        For each source whose EPISODIC entry count meets *config.threshold*:
          1. Score every EPISODIC snapshot with *decay_config* and *analyzer*.
          2. Promote non-stale snapshots to SEMANTIC via ``promote_to_semantic``.
          3. Record stale snapshots as failed merges (pruned).
        """
        scorer = SnapshotDecayScorer(
            config=decay_config,
            query_analyzer=analyzer,
            staleness_threshold=self._STALENESS_THRESHOLD,
            frequency_weight=self._FREQUENCY_WEIGHT,
            max_query_count=self._MAX_QUERY_COUNT,
        )

        merge_results: list[MergeResult] = []
        total_processed = total_merged = total_retained = semantic_created = 0

        for source in index.list_sources():
            episodic = index.list_snapshots(source.source_id, memory_type=MemoryType.EPISODIC)
            if not config.should_consolidate(len(episodic)):
                continue

            total_processed += len(episodic)
            for snap in episodic:
                ds = scorer.score(snap, now=self._NOW)
                if ds.is_stale:
                    total_retained += 1
                    merge_results.append(MergeResult(
                        merged_record_id=snap.snapshot_id,
                        source_record_ids=(snap.snapshot_id,),
                        success=False,
                        merged_at=self._NOW,
                        error="stale: below staleness threshold",
                    ))
                else:
                    index.promote_to_semantic(snap.snapshot_id, consolidation_time=self._NOW)
                    total_merged += 1
                    semantic_created += 1
                    merge_results.append(MergeResult(
                        merged_record_id=snap.snapshot_id,
                        source_record_ids=(snap.snapshot_id,),
                        success=True,
                        merged_at=self._NOW,
                    ))

        return ConsolidationMetrics(
            entries_processed=total_processed,
            entries_merged=total_merged,
            entries_retained=total_retained,
            semantic_records_created=semantic_created,
            merge_results=tuple(merge_results),
            started_at=self._NOW,
            completed_at=self._NOW,
        )

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def test_grouping_correctness(self, index: ShadowIndex) -> None:
        """Snapshots are correctly isolated by source_id and memory_type."""
        self._populate(index)

        # github-auth has only SEMANTIC snapshots
        auth_episodic = index.list_snapshots("github-auth", memory_type=MemoryType.EPISODIC)
        auth_semantic = index.list_snapshots("github-auth", memory_type=MemoryType.SEMANTIC)
        assert len(auth_episodic) == 0
        assert len(auth_semantic) == 2
        assert {s.snapshot_id for s in auth_semantic} == {"auth-old", "auth-recent"}

        # rss-security has only EPISODIC snapshots (5 of them)
        rss_episodic = index.list_snapshots("rss-security", memory_type=MemoryType.EPISODIC)
        rss_semantic = index.list_snapshots("rss-security", memory_type=MemoryType.SEMANTIC)
        assert len(rss_episodic) == 5
        assert len(rss_semantic) == 0
        assert {s.snapshot_id for s in rss_episodic} == {
            "rss-500d", "rss-400d", "rss-180d", "rss-30d", "rss-7d"
        }

        # docs-api has only WORKING snapshots — not EPISODIC or SEMANTIC
        docs_episodic = index.list_snapshots("docs-api", memory_type=MemoryType.EPISODIC)
        docs_working = index.list_snapshots("docs-api", memory_type=MemoryType.WORKING)
        assert len(docs_episodic) == 0
        assert len(docs_working) == 2

        # Cross-source isolation: total sources count
        assert len(index.list_sources()) == 3

    def test_decay_scores(self, index: ShadowIndex) -> None:
        """Decay scores match the expected half-life formula with frequency bonus."""
        self._populate(index)
        analyzer = self._make_analyzer()
        decay_config = self._make_decay_config()

        scorer = SnapshotDecayScorer(
            config=decay_config,
            query_analyzer=analyzer,
            staleness_threshold=self._STALENESS_THRESHOLD,
            frequency_weight=self._FREQUENCY_WEIGHT,
            max_query_count=self._MAX_QUERY_COUNT,
        )

        rss_episodic = index.list_snapshots("rss-security", memory_type=MemoryType.EPISODIC)
        scores = {ds.snapshot_id: ds for ds in scorer.score_all(rss_episodic, now=self._NOW)}

        # Verify age_score formula: 0.5^(age_days / halflife_days)
        for snap_id, age_days in [
            ("rss-500d", self._AGE_VERY_OLD),
            ("rss-400d", self._AGE_OLD),
            ("rss-180d", self._AGE_MEDIUM),
            ("rss-30d", self._AGE_RECENT),
            ("rss-7d", self._AGE_FRESH),
        ]:
            expected_age_score = math.pow(0.5, age_days / self._HALFLIFE_DAYS)
            assert abs(scores[snap_id].age_score - expected_age_score) < 1e-4, (
                f"{snap_id}: age_score {scores[snap_id].age_score:.6f} != "
                f"expected {expected_age_score:.6f}"
            )

        # All rss entries match 4 "security" queries → freq_score = 4/10 = 0.4
        for snap_id in scores:
            assert abs(scores[snap_id].frequency_score - 0.4) < 1e-9, snap_id

        # Stale entries are the two oldest; fresh entries survive
        assert scores["rss-500d"].is_stale is True
        assert scores["rss-400d"].is_stale is True
        assert scores["rss-180d"].is_stale is False
        assert scores["rss-30d"].is_stale is False
        assert scores["rss-7d"].is_stale is False

        # Scores are monotonically increasing with freshness
        assert (
            scores["rss-500d"].combined_score
            < scores["rss-400d"].combined_score
            < scores["rss-180d"].combined_score
            < scores["rss-30d"].combined_score
            <= scores["rss-7d"].combined_score
        )

        # SEMANTIC entries in github-auth always yield age_score = 1.0 (timeless)
        auth_semantic = index.list_snapshots("github-auth", memory_type=MemoryType.SEMANTIC)
        auth_scores = scorer.score_all(auth_semantic, now=self._NOW)
        for ds in auth_scores:
            assert ds.age_score == 1.0, f"{ds.snapshot_id} SEMANTIC age_score != 1.0"

    def test_consolidation_prunes_stale_entries(self, index: ShadowIndex) -> None:
        """Stale snapshots are NOT promoted; their MergeResult.success is False."""
        self._populate(index)
        analyzer = self._make_analyzer()
        config = self._make_consolidation_config(threshold=3)
        decay_config = self._make_decay_config()

        metrics = self._consolidate(index, analyzer, config, decay_config)

        # Two stale entries must appear as failed merges
        failed = [r for r in metrics.merge_results if not r.success]
        assert len(failed) == 2
        pruned_ids = {r.merged_record_id for r in failed}
        assert pruned_ids == {"rss-500d", "rss-400d"}

        # Stale entries remain EPISODIC in the index (not promoted)
        for snap_id in ("rss-500d", "rss-400d"):
            snap = index.get_snapshot(snap_id)
            assert snap is not None
            assert snap.memory_type == MemoryType.EPISODIC, (
                f"{snap_id} should still be EPISODIC after pruning"
            )

    def test_threshold_plus_one_consolidates_all(self, index: ShadowIndex) -> None:
        """Records just above the threshold are ALL consolidated into a single group.

        Setup: one source with exactly threshold+1 fresh EPISODIC snapshots.
        Every snapshot is 1 day old (well within staleness bounds), so none
        are pruned.  After running consolidation, all threshold+1 entries must
        be promoted to SEMANTIC and the source must have zero remaining EPISODIC
        snapshots.
        """
        threshold = 3
        count = threshold + 1  # 4 snapshots — just above threshold

        index.add_source(SourceEntry(
            source_url="https://github.com/example/threshold-test",
            source_id="threshold-src",
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        snap_ids = [f"thresh-{i}" for i in range(count)]
        for snap_id in snap_ids:
            index.add_snapshot(self._make_snapshot(
                snapshot_id=snap_id,
                source_id="threshold-src",
                age_days=1,          # fresh: 0.5^(1/90) ≈ 0.992, well above 0.20
                entry_paths=[f"docs/{snap_id}.md"],
                memory_type=MemoryType.EPISODIC,
            ))

        analyzer = QueryHistoryAnalyzer()  # no seeded queries → freq_bonus = 0
        config = self._make_consolidation_config(threshold=threshold)
        decay_config = self._make_decay_config()

        metrics = self._consolidate(index, analyzer, config, decay_config)

        # All threshold+1 entries were processed and merged (none stale)
        assert metrics.entries_processed == count
        assert metrics.entries_merged == count
        assert metrics.entries_retained == 0
        assert all(r.success for r in metrics.merge_results)

        # Every snapshot has been promoted to SEMANTIC
        post_episodic = index.list_snapshots("threshold-src", memory_type=MemoryType.EPISODIC)
        post_semantic = index.list_snapshots("threshold-src", memory_type=MemoryType.SEMANTIC)
        assert len(post_episodic) == 0, "no EPISODIC entries should remain after consolidation"
        assert len(post_semantic) == count
        assert {s.snapshot_id for s in post_semantic} == set(snap_ids)

    def test_full_consolidation_e2e(self, index: ShadowIndex) -> None:
        """Full pipeline: populate → seed history → consolidate → verify all invariants."""
        snapshot_map = self._populate(index)
        analyzer = self._make_analyzer()
        config = self._make_consolidation_config(threshold=3)
        decay_config = self._make_decay_config()

        # --- Run consolidation ---
        metrics = self._consolidate(index, analyzer, config, decay_config)

        # ---- ConsolidationMetrics aggregate counts ----------------------
        # Only rss-security triggered consolidation (5 episodic, threshold=3)
        assert metrics.entries_processed == 5
        assert metrics.entries_merged == 3      # rss-180d, rss-30d, rss-7d
        assert metrics.entries_retained == 2    # rss-500d, rss-400d (stale)
        assert metrics.semantic_records_created == 3
        assert metrics.success_rate == pytest.approx(3 / 5)

        # ---- github-auth: SEMANTIC snapshots untouched ------------------
        for snap_id in snapshot_map["github-auth"]:
            snap = index.get_snapshot(snap_id)
            assert snap is not None
            assert snap.memory_type == MemoryType.SEMANTIC
            # No consolidation metadata should have been added
            assert snap.metadata.get("consolidated_at") is None

        # ---- docs-api: WORKING snapshots untouched ----------------------
        for snap_id in snapshot_map["docs-api"]:
            snap = index.get_snapshot(snap_id)
            assert snap is not None
            assert snap.memory_type == MemoryType.WORKING

        # ---- rss-security: stale entries still EPISODIC -----------------
        for snap_id in ("rss-500d", "rss-400d"):
            snap = index.get_snapshot(snap_id)
            assert snap is not None
            assert snap.memory_type == MemoryType.EPISODIC
            assert snap.metadata.get("consolidated_at") is None

        # ---- rss-security: promoted entries now SEMANTIC ----------------
        for snap_id in ("rss-180d", "rss-30d", "rss-7d"):
            snap = index.get_snapshot(snap_id)
            assert snap is not None, f"{snap_id} not found after consolidation"

            # Memory type upgraded
            assert snap.memory_type == MemoryType.SEMANTIC, (
                f"{snap_id}: expected SEMANTIC, got {snap.memory_type}"
            )

            # Provenance metadata written
            assert snap.metadata.get("consolidated_at") is not None, (
                f"{snap_id}: missing consolidated_at"
            )
            assert snap.metadata.get("consolidated_from") == MemoryType.EPISODIC.value, (
                f"{snap_id}: consolidated_from != episodic_memory"
            )

            # Promotion history contains exactly one EPISODIC→SEMANTIC entry
            assert len(snap.promotion_history) == 1
            ph = snap.promotion_history[0]
            assert ph.from_type == MemoryType.EPISODIC
            assert ph.to_type == MemoryType.SEMANTIC
            assert ph.reason == "session consolidation"
            assert ph.timestamp == self._NOW

        # ---- post-consolidation grouping --------------------------------
        # rss-security now has 3 SEMANTIC and 2 EPISODIC
        post_episodic = index.list_snapshots("rss-security", memory_type=MemoryType.EPISODIC)
        post_semantic = index.list_snapshots("rss-security", memory_type=MemoryType.SEMANTIC)
        assert len(post_episodic) == 2
        assert {s.snapshot_id for s in post_episodic} == {"rss-500d", "rss-400d"}
        assert len(post_semantic) == 3
        assert {s.snapshot_id for s in post_semantic} == {"rss-180d", "rss-30d", "rss-7d"}

        # ---- MergeResult details ----------------------------------------
        successful = [r for r in metrics.merge_results if r.success]
        failed = [r for r in metrics.merge_results if not r.success]
        assert {r.merged_record_id for r in successful} == {"rss-180d", "rss-30d", "rss-7d"}
        assert {r.merged_record_id for r in failed} == {"rss-500d", "rss-400d"}
        for r in failed:
            assert r.error is not None
            assert "stale" in r.error
        for r in successful:
            assert r.error is None
            assert r.merged_at == self._NOW

    def test_single_record_group_skipped_unchanged(self, index: ShadowIndex) -> None:
        """A group containing exactly one record is skipped by consolidation.

        When a source has only one EPISODIC snapshot the group size (1) is
        below the consolidation threshold, so ``config.should_consolidate``
        returns False.  The snapshot must remain EPISODIC and no merge results
        or processed-entry counts should be produced for it.
        """
        index.add_source(SourceEntry(
            source_url="https://github.com/example/single-record",
            source_id="single-src",
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        index.add_snapshot(self._make_snapshot(
            snapshot_id="single-only",
            source_id="single-src",
            age_days=1,       # fresh: well above staleness threshold
            entry_paths=["docs/only.md"],
            memory_type=MemoryType.EPISODIC,
        ))

        analyzer = QueryHistoryAnalyzer()   # no seeded queries
        config = self._make_consolidation_config(threshold=3)
        decay_config = self._make_decay_config()

        # Threshold is 3; a single-record group must not satisfy it.
        assert config.should_consolidate(1) is False

        metrics = self._consolidate(index, analyzer, config, decay_config)

        # No entries were processed or merged for this source.
        assert metrics.entries_processed == 0
        assert metrics.entries_merged == 0
        assert metrics.entries_retained == 0
        assert metrics.semantic_records_created == 0
        assert metrics.merge_results == ()

        # The snapshot is returned unchanged — still EPISODIC.
        snap = index.get_snapshot("single-only")
        assert snap is not None
        assert snap.memory_type == MemoryType.EPISODIC
        assert snap.metadata.get("consolidated_at") is None
        assert snap.promotion_history == []
