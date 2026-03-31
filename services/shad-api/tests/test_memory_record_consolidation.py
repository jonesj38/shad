"""Integration-style type tests: MemoryRecord[] collections × ConsolidationConfig.

Verifies that lists of MemoryRecord (and its subtypes) interact correctly with
ConsolidationConfig, MergeResult, ConsolidationMetrics, and apply_decay for
realistic merge/decay pipeline scenarios.

These tests do NOT mock an LLM or consolidation engine; they verify that the
types compose correctly and that the invariants hold end-to-end across the
consolidation data model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.consolidation import (
    ConsolidationConfig,
    ConsolidationMetrics,
    MergeResult,
    MergeStrategy,
    RetentionPolicy,
    consolidate,
)
from shad.vault.contracts import ConsolidationResult, PruneReason
from shad.vault.decay import DecayConfig, DecayCurve, apply_decay
from shad.vault.shadow_index import (
    AnyMemoryRecord,
    BiTemporalRecord,
    EpisodicRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
    ProceduralRecord,
    SemanticRecord,
    WorkingRecord,
    make_bitemporal_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _btr(content: str, *, age_days: float = 0.0) -> BiTemporalRecord[str]:
    event_time = _BASE - timedelta(days=age_days)
    return make_bitemporal_record(event_time, content, ingestion_time=_BASE)


def _episodic(
    record_id: str,
    content: str = "event",
    *,
    age_days: float = 0.0,
    confidence: float = 1.0,
    session_id: str | None = None,
    decay_rate: float = 1.0,
) -> EpisodicRecord:
    return EpisodicRecord(
        record_id=record_id,
        record=_btr(content, age_days=age_days),
        metadata=MemoryMetadata(source="test", confidence=confidence),
        session_id=session_id,
        decay_rate=decay_rate,
    )


def _semantic(record_id: str, content: str = "fact", *, concept_name: str | None = None) -> SemanticRecord:
    return SemanticRecord(
        record_id=record_id,
        record=_btr(content),
        metadata=MemoryMetadata(source="test", confidence=0.9),
        concept_name=concept_name,
    )


# ---------------------------------------------------------------------------
# 1. Collection threshold triggering
# ---------------------------------------------------------------------------


class TestCollectionThresholdTriggering:
    """should_consolidate reflects the size of a real MemoryRecord list."""

    def test_empty_list_does_not_trigger(self) -> None:
        records: list[EpisodicRecord] = []
        cfg = ConsolidationConfig(threshold=5)
        assert cfg.should_consolidate(len(records)) is False

    def test_list_below_threshold_does_not_trigger(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(4)]
        cfg = ConsolidationConfig(threshold=5)
        assert cfg.should_consolidate(len(records)) is False

    def test_list_at_threshold_triggers(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(5)]
        cfg = ConsolidationConfig(threshold=5)
        assert cfg.should_consolidate(len(records)) is True

    def test_list_above_threshold_triggers(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(12)]
        cfg = ConsolidationConfig(threshold=5)
        assert cfg.should_consolidate(len(records)) is True

    def test_disabled_config_never_triggers_regardless_of_list_size(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(100)]
        cfg = ConsolidationConfig(threshold=1, enabled=False)
        assert cfg.should_consolidate(len(records)) is False

    def test_threshold_one_triggers_on_single_record(self) -> None:
        records = [_episodic("ep-0")]
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(len(records)) is True

    def test_only_episodic_records_are_counted_for_threshold(self) -> None:
        """Only EPISODIC entries should be fed to should_consolidate."""
        all_records: list[AnyMemoryRecord] = [
            _episodic("ep-0"),
            _episodic("ep-1"),
            _semantic("sem-0"),  # should not count
            WorkingRecord(record_id="w-0", record=_btr("tmp")),
        ]
        episodic_count = sum(1 for r in all_records if r.memory_type is MemoryType.EPISODIC)
        cfg = ConsolidationConfig(threshold=2)
        assert episodic_count == 2
        assert cfg.should_consolidate(episodic_count) is True


# ---------------------------------------------------------------------------
# 2. MergeResult built from MemoryRecord IDs
# ---------------------------------------------------------------------------


class TestMergeResultFromMemoryRecordIDs:
    """MergeResult.source_record_ids must exactly reflect the source collection."""

    def test_source_ids_match_episodic_record_ids(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(3)]
        result = MergeResult(
            merged_record_id="sem-out",
            source_record_ids=tuple(r.record_id for r in records),
            success=True,
            merged_at=_BASE,
        )
        assert result.source_record_ids == ("ep-0", "ep-1", "ep-2")

    def test_source_ids_length_matches_collection(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(7)]
        result = MergeResult(
            merged_record_id="sem-out",
            source_record_ids=tuple(r.record_id for r in records),
            success=True,
            merged_at=_BASE,
        )
        assert len(result.source_record_ids) == len(records)

    def test_merged_record_id_not_in_source_ids(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(4)]
        merged_id = "sem-new"
        result = MergeResult(
            merged_record_id=merged_id,
            source_record_ids=tuple(r.record_id for r in records),
            success=True,
            merged_at=_BASE,
        )
        assert merged_id not in result.source_record_ids

    def test_failed_merge_still_carries_source_ids(self) -> None:
        records = [_episodic("ep-a"), _episodic("ep-b")]
        result = MergeResult(
            merged_record_id="sem-fail",
            source_record_ids=tuple(r.record_id for r in records),
            success=False,
            merged_at=_BASE,
            error="LLM timeout",
        )
        assert result.source_record_ids == ("ep-a", "ep-b")
        assert result.success is False

    def test_single_record_merge(self) -> None:
        record = _episodic("ep-solo")
        result = MergeResult(
            merged_record_id="sem-solo",
            source_record_ids=(record.record_id,),
            success=True,
            merged_at=_BASE,
        )
        assert result.source_record_ids == ("ep-solo",)


# ---------------------------------------------------------------------------
# 3. ConsolidationMetrics from a simulated merge pass
# ---------------------------------------------------------------------------


class TestConsolidationMetricsFromMergePass:
    """Metrics computed from a real collection must have internally consistent counts."""

    def _run_pass(
        self,
        records: list[EpisodicRecord],
        cfg: ConsolidationConfig,
        *,
        group_size: int = 3,
    ) -> ConsolidationMetrics:
        """Simulate a merge pass: group records, build MergeResults, emit metrics."""
        started = _BASE
        groups = [records[i : i + group_size] for i in range(0, len(records), group_size)]
        merge_results: list[MergeResult] = []
        merged_count = 0
        semantic_created = 0

        for idx, group in enumerate(groups):
            if len(group) < group_size:
                # partial group — retain
                continue
            mr = MergeResult(
                merged_record_id=f"sem-{idx}",
                source_record_ids=tuple(r.record_id for r in group),
                success=True,
                merged_at=_BASE,
            )
            merge_results.append(mr)
            merged_count += len(group)
            semantic_created += 1

        retained = len(records) - merged_count
        completed = _BASE + timedelta(seconds=1)

        return ConsolidationMetrics(
            entries_processed=len(records),
            entries_merged=merged_count,
            entries_retained=retained,
            semantic_records_created=semantic_created,
            merge_results=tuple(merge_results),
            started_at=started,
            completed_at=completed,
        )

    def test_entries_processed_equals_collection_length(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(9)]
        cfg = ConsolidationConfig(threshold=3)
        m = self._run_pass(records, cfg, group_size=3)
        assert m.entries_processed == 9

    def test_merged_plus_retained_equals_processed(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(10)]
        cfg = ConsolidationConfig(threshold=3)
        m = self._run_pass(records, cfg, group_size=3)
        assert m.entries_merged + m.entries_retained == m.entries_processed

    def test_semantic_records_created_equals_merge_result_count(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(9)]
        cfg = ConsolidationConfig(threshold=3)
        m = self._run_pass(records, cfg, group_size=3)
        assert m.semantic_records_created == len(m.merge_results)

    def test_all_successful_merges_give_success_rate_one(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(6)]
        cfg = ConsolidationConfig(threshold=3)
        m = self._run_pass(records, cfg, group_size=3)
        assert m.success_rate == 1.0

    def test_empty_collection_yields_zero_metrics(self) -> None:
        records: list[EpisodicRecord] = []
        cfg = ConsolidationConfig(threshold=5)
        started = _BASE
        m = ConsolidationMetrics(
            entries_processed=0,
            entries_merged=0,
            entries_retained=0,
            semantic_records_created=0,
            merge_results=(),
            started_at=started,
            completed_at=started + timedelta(seconds=1),
        )
        assert m.entries_processed == 0
        assert m.success_rate == 0.0

    def test_merge_result_source_ids_cover_all_merged_records(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(6)]
        cfg = ConsolidationConfig(threshold=3)
        m = self._run_pass(records, cfg, group_size=3)
        all_source_ids = {sid for mr in m.merge_results for sid in mr.source_record_ids}
        expected_ids = {r.record_id for r in records}
        assert all_source_ids == expected_ids


# ---------------------------------------------------------------------------
# 4. Decay applied to MemoryRecord confidence scores
# ---------------------------------------------------------------------------


class TestDecayAppliedToMemoryRecordConfidence:
    """apply_decay integrates correctly with MemoryRecord.metadata.confidence."""

    def _age_seconds(self, age_days: float) -> float:
        return age_days * 86_400.0

    def test_fresh_record_confidence_unchanged(self) -> None:
        r = _episodic("ep-0", confidence=0.9)
        cfg = DecayConfig(halflife_seconds=self._age_seconds(90))
        assert r.metadata is not None
        decayed = apply_decay(r.metadata.confidence, 0.0, cfg)
        assert decayed == pytest.approx(0.9, rel=1e-6)

    def test_record_at_halflife_confidence_halved(self) -> None:
        halflife_days = 90.0
        r = _episodic("ep-0", confidence=0.8)
        cfg = DecayConfig(halflife_seconds=self._age_seconds(halflife_days))
        assert r.metadata is not None
        decayed = apply_decay(r.metadata.confidence, self._age_seconds(halflife_days), cfg)
        assert decayed == pytest.approx(0.4, rel=1e-4)

    def test_very_old_record_approaches_zero(self) -> None:
        r = _episodic("ep-0", confidence=1.0)
        cfg = DecayConfig(halflife_seconds=self._age_seconds(90))
        assert r.metadata is not None
        decayed = apply_decay(r.metadata.confidence, self._age_seconds(9000), cfg)
        assert decayed < 0.001

    def test_decay_rate_multiplier_affects_effective_age(self) -> None:
        """A higher decay_rate should yield lower effective confidence."""
        r_fast = _episodic("ep-fast", confidence=1.0, decay_rate=2.0)
        r_slow = _episodic("ep-slow", confidence=1.0, decay_rate=0.5)
        cfg = DecayConfig(halflife_seconds=self._age_seconds(90))
        age = self._age_seconds(90)
        assert r_fast.metadata and r_slow.metadata
        # Simulate effective age scaled by decay_rate
        decayed_fast = apply_decay(r_fast.metadata.confidence, age * r_fast.decay_rate, cfg)
        decayed_slow = apply_decay(r_slow.metadata.confidence, age * r_slow.decay_rate, cfg)
        assert decayed_fast < decayed_slow

    def test_decay_halflife_days_from_config_consistent_with_decay_module(self) -> None:
        """ConsolidationConfig.decay_halflife_days maps directly to DecayConfig.halflife_seconds."""
        cfg = ConsolidationConfig(decay_halflife_days=30.0)
        decay_cfg = DecayConfig(halflife_seconds=cfg.decay_halflife_days * 86_400.0)
        r = _episodic("ep-0", confidence=1.0)
        assert r.metadata is not None
        decayed = apply_decay(r.metadata.confidence, 30.0 * 86_400.0, decay_cfg)
        assert decayed == pytest.approx(0.5, rel=1e-4)

    def test_linear_curve_reaches_zero_at_two_halflives(self) -> None:
        r = _episodic("ep-0", confidence=1.0)
        cfg = DecayConfig(halflife_seconds=86_400.0, curve=DecayCurve.LINEAR)
        assert r.metadata is not None
        # At 2 × halflife, factor = 0 → decayed score = 0
        decayed = apply_decay(r.metadata.confidence, 2.0 * 86_400.0, cfg)
        assert decayed == pytest.approx(0.0, abs=1e-9)

    def test_collection_decayed_scores_are_all_in_range(self) -> None:
        records = [_episodic(f"ep-{i}", confidence=1.0, age_days=float(i * 30)) for i in range(5)]
        cfg = DecayConfig(halflife_seconds=90.0 * 86_400.0)
        for r in records:
            assert r.metadata is not None
            age = (r.record.event_time - _BASE).total_seconds() * -1  # age from _BASE
            decayed = apply_decay(r.metadata.confidence, max(0.0, age), cfg)
            assert 0.0 <= decayed <= 1.0


# ---------------------------------------------------------------------------
# 5. Retention policy alignment with MemoryRecord ages
# ---------------------------------------------------------------------------


class TestRetentionPolicyVsMemoryRecordAge:
    """RetentionPolicy TTLs should be interpretable against MemoryRecord timestamps."""

    def test_episodic_record_within_ttl_is_not_expired(self) -> None:
        policy = RetentionPolicy(episodic_ttl_days=7.0)
        r = _episodic("ep-0")
        age_days = (_BASE - r.record.event_time).days
        assert age_days < policy.episodic_ttl_days

    def test_episodic_record_beyond_ttl_is_expired(self) -> None:
        policy = RetentionPolicy(episodic_ttl_days=7.0)
        r = _episodic("ep-old", age_days=8.0)
        age_days = (_BASE - r.record.event_time).total_seconds() / 86_400.0
        assert age_days > policy.episodic_ttl_days

    def test_semantic_record_within_ttl_is_not_expired(self) -> None:
        policy = RetentionPolicy(semantic_ttl_days=30.0)
        r = _semantic("sem-0")
        age_days = (_BASE - r.record.event_time).days
        assert age_days < policy.semantic_ttl_days

    def test_collection_expired_episodic_records_identified(self) -> None:
        policy = RetentionPolicy(episodic_ttl_days=7.0)
        records = [_episodic(f"ep-{i}", age_days=float(i)) for i in range(15)]
        expired = [
            r for r in records
            if (_BASE - r.record.event_time).total_seconds() / 86_400.0 > policy.episodic_ttl_days
        ]
        assert len(expired) == 7  # ages 8–14 inclusive

    def test_config_retention_policy_ttls_are_positive(self) -> None:
        cfg = ConsolidationConfig()
        p = cfg.retention_policy
        assert p.episodic_ttl_days > 0
        assert p.semantic_ttl_days > 0
        assert p.procedural_ttl_days > 0
        assert p.archival_ttl_days > 0

    def test_ttl_ordering_episodic_lt_semantic_lt_procedural_lt_archival(self) -> None:
        p = RetentionPolicy()
        assert p.episodic_ttl_days < p.semantic_ttl_days
        assert p.semantic_ttl_days < p.procedural_ttl_days
        assert p.procedural_ttl_days < p.archival_ttl_days


# ---------------------------------------------------------------------------
# 6. MergeStrategy variants over MemoryRecord collections
# ---------------------------------------------------------------------------


class TestMergeStrategyOverCollections:
    """Each MergeStrategy produces a valid ConsolidationConfig × MergeResult pair."""

    @pytest.mark.parametrize("strategy", list(MergeStrategy))
    def test_config_accepts_all_merge_strategies(self, strategy: MergeStrategy) -> None:
        cfg = ConsolidationConfig(merge_strategy=strategy)
        assert cfg.merge_strategy is strategy

    def test_latest_wins_selects_most_recent_record(self) -> None:
        """Simulate LATEST_WINS: pick the record with the greatest event_time."""
        records = [_episodic(f"ep-{i}", f"content-{i}", age_days=float(10 - i)) for i in range(5)]
        # ep-4 has age_days=6, ep-0 has age_days=10 → ep-4 is most recent
        latest = max(records, key=lambda r: r.record.event_time)
        assert latest.record_id == "ep-4"

    def test_merge_result_content_source_matches_strategy(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(5)]
        cfg = ConsolidationConfig(threshold=5, merge_strategy=MergeStrategy.WEIGHTED_AVERAGE)
        assert cfg.should_consolidate(len(records)) is True
        result = MergeResult(
            merged_record_id="sem-weighted",
            source_record_ids=tuple(r.record_id for r in records),
            success=True,
            merged_at=_BASE,
        )
        assert len(result.source_record_ids) == 5

    def test_llm_synthesis_failure_captured_in_merge_result(self) -> None:
        records = [_episodic(f"ep-{i}") for i in range(3)]
        result = MergeResult(
            merged_record_id="sem-llm",
            source_record_ids=tuple(r.record_id for r in records),
            success=False,
            merged_at=_BASE,
            error="LLM rate-limited",
        )
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# 7. Metadata consolidation provenance on SemanticRecord
# ---------------------------------------------------------------------------


class TestMetadataConsolidationProvenance:
    """consolidated_at / consolidated_from fields are correctly typed and accessible."""

    def test_semantic_record_metadata_consolidated_at_set(self) -> None:
        r = _semantic("sem-0")
        assert r.metadata is not None
        r.metadata.consolidated_at = _BASE.isoformat()
        assert r.metadata.consolidated_at == _BASE.isoformat()

    def test_semantic_record_metadata_consolidated_from_set(self) -> None:
        r = _semantic("sem-0")
        assert r.metadata is not None
        r.metadata.consolidated_from = MemoryType.EPISODIC.value
        assert r.metadata.consolidated_from == "episodic_memory"

    def test_consolidation_provenance_chain_episodic_to_semantic(self) -> None:
        source_records = [_episodic(f"ep-{i}") for i in range(3)]
        semantic = _semantic("sem-merged")
        assert semantic.metadata is not None
        semantic.metadata.consolidated_at = _BASE.isoformat()
        semantic.metadata.consolidated_from = MemoryType.EPISODIC.value

        assert semantic.memory_type is MemoryType.SEMANTIC
        assert semantic.metadata.consolidated_from == "episodic_memory"
        assert semantic.metadata.consolidated_at is not None

    def test_source_episodic_records_retain_original_type_after_merge(self) -> None:
        """Source records are not mutated during merge; type stays EPISODIC."""
        records = [_episodic(f"ep-{i}") for i in range(4)]
        # Simulate a merge (just build the result; sources are unchanged)
        _ = MergeResult(
            merged_record_id="sem-out",
            source_record_ids=tuple(r.record_id for r in records),
            success=True,
            merged_at=_BASE,
        )
        for r in records:
            assert r.memory_type is MemoryType.EPISODIC

    def test_metadata_tags_include_memory_type_label(self) -> None:
        r = _episodic("ep-0")
        assert r.metadata is not None
        r.metadata.tags.append("episodic_memory")
        assert "episodic_memory" in r.metadata.tags


# ---------------------------------------------------------------------------
# 8. Heterogeneous collection filtering
# ---------------------------------------------------------------------------


class TestHeterogeneousCollectionFiltering:
    """Mixed AnyMemoryRecord collections can be partitioned by type for consolidation."""

    def _make_mixed_collection(self) -> list[AnyMemoryRecord]:
        return [
            _episodic("ep-0"),
            _episodic("ep-1"),
            _semantic("sem-0"),
            WorkingRecord(record_id="w-0", record=_btr("tmp")),
            ProceduralRecord(record_id="proc-0", record=_btr("how-to")),
            _episodic("ep-2"),
        ]

    def test_filter_yields_only_episodic_records(self) -> None:
        records = self._make_mixed_collection()
        episodic = [r for r in records if r.memory_type is MemoryType.EPISODIC]
        assert len(episodic) == 3
        assert all(isinstance(r, EpisodicRecord) for r in episodic)

    def test_non_episodic_records_excluded_from_threshold_count(self) -> None:
        records = self._make_mixed_collection()
        episodic_count = sum(1 for r in records if r.memory_type is MemoryType.EPISODIC)
        cfg = ConsolidationConfig(threshold=3)
        assert cfg.should_consolidate(episodic_count) is True

    def test_semantic_records_not_consumed_by_consolidation(self) -> None:
        records = self._make_mixed_collection()
        semantic_before = [r for r in records if r.memory_type is MemoryType.SEMANTIC]
        # After consolidation the semantic records should still be present
        assert len(semantic_before) == 1
        assert semantic_before[0].record_id == "sem-0"

    def test_all_memory_types_present_in_collection(self) -> None:
        records = self._make_mixed_collection()
        types_present = {r.memory_type for r in records}
        assert MemoryType.EPISODIC in types_present
        assert MemoryType.SEMANTIC in types_present
        assert MemoryType.WORKING in types_present
        assert MemoryType.PROCEDURAL in types_present

    def test_working_records_have_none_session_id(self) -> None:
        records = self._make_mixed_collection()
        working = [r for r in records if isinstance(r, WorkingRecord)]
        for r in working:
            assert r.context_window_id is None

    def test_episodic_records_preserve_session_id_through_collection(self) -> None:
        records: list[AnyMemoryRecord] = [
            _episodic("ep-0", session_id="sess-abc"),
            _episodic("ep-1", session_id="sess-abc"),
            _episodic("ep-2", session_id="sess-xyz"),
        ]
        session_groups: dict[str | None, list[EpisodicRecord]] = {}
        for r in records:
            if isinstance(r, EpisodicRecord):
                session_groups.setdefault(r.session_id, []).append(r)
        assert len(session_groups["sess-abc"]) == 2
        assert len(session_groups["sess-xyz"]) == 1


# ---------------------------------------------------------------------------
# 9. ConsolidationConfig cosine_similarity_threshold as a collection filter
# ---------------------------------------------------------------------------


class TestCosineSimilarityThresholdFiltering:
    """Simulated cosine similarity gating with ConsolidationConfig threshold."""

    def test_records_above_threshold_are_merge_candidates(self) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=0.8)
        # Pretend each record has a pre-computed similarity to a centroid
        similarities = {"ep-0": 0.95, "ep-1": 0.82, "ep-2": 0.70, "ep-3": 0.60}
        records = [_episodic(rid) for rid in similarities]
        candidates = [
            r for r in records if similarities[r.record_id] >= cfg.cosine_similarity_threshold
        ]
        assert len(candidates) == 2
        assert {r.record_id for r in candidates} == {"ep-0", "ep-1"}

    def test_records_below_threshold_are_retained(self) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=0.9)
        similarities = {"ep-0": 0.95, "ep-1": 0.85, "ep-2": 0.80}
        records = [_episodic(rid) for rid in similarities]
        retained = [
            r for r in records if similarities[r.record_id] < cfg.cosine_similarity_threshold
        ]
        assert len(retained) == 2

    def test_zero_threshold_accepts_all_records(self) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=0.0)
        similarities = {"ep-0": 0.0, "ep-1": 0.1, "ep-2": 0.5}
        records = [_episodic(rid) for rid in similarities]
        candidates = [
            r for r in records if similarities[r.record_id] >= cfg.cosine_similarity_threshold
        ]
        assert len(candidates) == len(records)

    def test_threshold_one_accepts_only_perfect_similarity(self) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=1.0)
        similarities = {"ep-0": 1.0, "ep-1": 0.999, "ep-2": 0.98}
        records = [_episodic(rid) for rid in similarities]
        candidates = [
            r for r in records if similarities[r.record_id] >= cfg.cosine_similarity_threshold
        ]
        assert len(candidates) == 1
        assert candidates[0].record_id == "ep-0"


# ---------------------------------------------------------------------------
# 10. All-identical records — single representative after consolidation
# ---------------------------------------------------------------------------


class TestAllIdenticalRecordsConsolidation:
    """N duplicate records share the same topic and collapse into one MergeResult."""

    _N = 5
    _SESSION = "sess-dup"

    def _records(self, n: int = _N, session: str = _SESSION) -> list[EpisodicRecord]:
        return [_episodic(f"dup-{i}", content="same content", session_id=session) for i in range(n)]

    def test_produces_exactly_one_merge_result(self) -> None:
        """All identical records collapse into a single MergeResult."""
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N, merge_strategy=MergeStrategy.LATEST_WINS)
        result: ConsolidationResult = consolidate(records, cfg, now=_BASE)

        assert result.success is True
        assert len(result.metrics.merge_results) == 1

    def test_single_merge_result_is_successful(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        mr = result.metrics.merge_results[0]
        assert mr.success is True
        assert mr.error is None

    def test_merge_result_contains_all_source_ids(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        mr = result.metrics.merge_results[0]
        assert set(mr.source_record_ids) == {r.record_id for r in records}
        assert len(mr.source_record_ids) == self._N

    def test_merged_record_id_is_distinct_from_all_source_ids(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        mr = result.metrics.merge_results[0]
        assert isinstance(mr.merged_record_id, str)
        assert mr.merged_record_id not in mr.source_record_ids

    def test_entries_merged_equals_n_retained_is_zero(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.entries_processed == self._N
        assert result.metrics.entries_merged == self._N
        assert result.metrics.entries_retained == 0

    def test_semantic_records_created_is_one(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.semantic_records_created == 1

    def test_all_sources_flagged_superseded_in_prune_candidates(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        mr = result.metrics.merge_results[0]
        superseded = [pc for pc in result.prune_candidates if pc.reason == PruneReason.SUPERSEDED]
        assert {pc.record_id for pc in superseded} == {r.record_id for r in records}

    def test_each_prune_candidate_points_to_merged_record(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        mr = result.metrics.merge_results[0]
        superseded = [pc for pc in result.prune_candidates if pc.reason == PruneReason.SUPERSEDED]
        for pc in superseded:
            assert pc.superseded_by == mr.merged_record_id

    def test_groups_merged_one_skipped_zero(self) -> None:
        records = self._records()
        cfg = ConsolidationConfig(threshold=self._N)
        result = consolidate(records, cfg, now=_BASE)

        assert result.groups_merged == 1
        assert result.groups_skipped == 0


# ---------------------------------------------------------------------------
# 11. Records just below the consolidation threshold remain separate
# ---------------------------------------------------------------------------


class TestBelowThresholdRecordsRemainSeparate:
    """threshold-1 records in a single group must NOT be consolidated.

    Verifies the Stage-2 skip path: when a group's size < threshold,
    consolidate() must not produce a MergeResult, must not mark any record as
    SUPERSEDED, and must report all records as retained.
    """

    _THRESHOLD = 5  # one more than what we'll insert

    def _records(self, n: int, *, session: str = "sess-below") -> list[EpisodicRecord]:
        """Create *n* fresh episodic records sharing the same session (one group)."""
        return [_episodic(f"ep-below-{i}", content=f"event-{i}", session_id=session) for i in range(n)]

    def test_no_merge_results_produced(self) -> None:
        """No MergeResult when the group is one short of the threshold."""
        records = self._records(self._THRESHOLD - 1)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.merge_results == ()

    def test_entries_merged_is_zero(self) -> None:
        records = self._records(self._THRESHOLD - 1)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.entries_merged == 0

    def test_entries_retained_equals_record_count(self) -> None:
        n = self._THRESHOLD - 1
        records = self._records(n)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.entries_retained == n

    def test_entries_processed_equals_record_count(self) -> None:
        n = self._THRESHOLD - 1
        records = self._records(n)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.entries_processed == n

    def test_no_superseded_prune_candidates(self) -> None:
        """Source record IDs must not appear as SUPERSEDED prune candidates."""
        records = self._records(self._THRESHOLD - 1)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        record_ids = {r.record_id for r in records}
        superseded_ids = {
            pc.record_id
            for pc in result.prune_candidates
            if pc.reason == PruneReason.SUPERSEDED
        }
        assert superseded_ids.isdisjoint(record_ids)

    def test_groups_skipped_is_one(self) -> None:
        records = self._records(self._THRESHOLD - 1)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.groups_skipped == 1
        assert result.groups_merged == 0

    def test_semantic_records_created_is_zero(self) -> None:
        records = self._records(self._THRESHOLD - 1)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.semantic_records_created == 0

    def test_result_is_still_successful(self) -> None:
        """A below-threshold run completes successfully — it is not an error."""
        records = self._records(self._THRESHOLD - 1)
        cfg = ConsolidationConfig(threshold=self._THRESHOLD)
        result = consolidate(records, cfg, now=_BASE)

        assert result.success is True

    @pytest.mark.parametrize("threshold", [2, 5, 10, 50])
    def test_threshold_minus_one_never_consolidates(self, threshold: int) -> None:
        """Parametrised: for any threshold, threshold-1 records are never merged."""
        records = self._records(threshold - 1)
        cfg = ConsolidationConfig(threshold=threshold)
        result = consolidate(records, cfg, now=_BASE)

        assert result.metrics.merge_results == ()
        assert result.metrics.entries_merged == 0
        assert result.metrics.entries_retained == threshold - 1
