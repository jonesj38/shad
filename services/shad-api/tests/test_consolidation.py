"""Type-level tests for consolidation.py contracts.

Verifies that MergeResult, ConsolidationMetrics, ConsolidationConfig, and the
ConsolidationTrigger protocol enforce required fields, reject invalid shapes,
and honour their invariants.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shad.vault.consolidation import (
    ConsolidationConfig,
    ConsolidationMetrics,
    ConsolidationTrigger,
    MergeResult,
    MergeStrategy,
    RetentionPolicy,
    cosine_similarity,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)


def _merge_result(
    *,
    merged_record_id: str = "sem-001",
    source_record_ids: tuple[str, ...] = ("ep-001", "ep-002"),
    success: bool = True,
    merged_at: datetime = _T0,
    error: str | None = None,
) -> MergeResult:
    return MergeResult(
        merged_record_id=merged_record_id,
        source_record_ids=source_record_ids,
        success=success,
        merged_at=merged_at,
        error=error,
    )


def _metrics(
    *,
    entries_processed: int = 5,
    entries_merged: int = 4,
    entries_retained: int = 1,
    semantic_records_created: int = 2,
    merge_results: tuple[MergeResult, ...] = (),
    started_at: datetime = _T0,
    completed_at: datetime = _T1,
) -> ConsolidationMetrics:
    return ConsolidationMetrics(
        entries_processed=entries_processed,
        entries_merged=entries_merged,
        entries_retained=entries_retained,
        semantic_records_created=semantic_records_created,
        merge_results=merge_results,
        started_at=started_at,
        completed_at=completed_at,
    )


# ---------------------------------------------------------------------------
# MergeResult — required fields
# ---------------------------------------------------------------------------


class TestMergeResultRequiredFields:
    """Valid MergeResult exposes all required fields with correct types."""

    def test_merged_record_id_present(self) -> None:
        r = _merge_result(merged_record_id="sem-xyz")
        assert r.merged_record_id == "sem-xyz"

    def test_merged_record_id_is_str(self) -> None:
        assert isinstance(_merge_result().merged_record_id, str)

    def test_source_record_ids_present(self) -> None:
        ids = ("a", "b", "c")
        r = _merge_result(source_record_ids=ids)
        assert r.source_record_ids == ids

    def test_source_record_ids_is_tuple(self) -> None:
        assert isinstance(_merge_result().source_record_ids, tuple)

    def test_success_present(self) -> None:
        assert _merge_result(success=True).success is True
        assert _merge_result(success=False).success is False

    def test_success_is_bool(self) -> None:
        assert isinstance(_merge_result().success, bool)

    def test_merged_at_present(self) -> None:
        r = _merge_result(merged_at=_T0)
        assert r.merged_at == _T0

    def test_merged_at_is_datetime(self) -> None:
        assert isinstance(_merge_result().merged_at, datetime)

    def test_error_defaults_to_none(self) -> None:
        assert _merge_result().error is None

    def test_error_accepts_str(self) -> None:
        r = _merge_result(error="merge failed")
        assert r.error == "merge failed"

    def test_error_is_str_or_none(self) -> None:
        assert isinstance(_merge_result(error="oops").error, str)
        assert _merge_result(error=None).error is None


class TestMergeResultMissingRequired:
    """Omitting any required field raises TypeError."""

    def test_missing_merged_record_id(self) -> None:
        with pytest.raises(TypeError):
            MergeResult(  # type: ignore[call-arg]
                source_record_ids=("ep-001",),
                success=True,
                merged_at=_T0,
            )

    def test_missing_source_record_ids(self) -> None:
        with pytest.raises(TypeError):
            MergeResult(  # type: ignore[call-arg]
                merged_record_id="sem-001",
                success=True,
                merged_at=_T0,
            )

    def test_missing_success(self) -> None:
        with pytest.raises(TypeError):
            MergeResult(  # type: ignore[call-arg]
                merged_record_id="sem-001",
                source_record_ids=("ep-001",),
                merged_at=_T0,
            )

    def test_missing_merged_at(self) -> None:
        with pytest.raises(TypeError):
            MergeResult(  # type: ignore[call-arg]
                merged_record_id="sem-001",
                source_record_ids=("ep-001",),
                success=True,
            )

    def test_no_arguments(self) -> None:
        with pytest.raises(TypeError):
            MergeResult()  # type: ignore[call-arg]


class TestMergeResultImmutability:
    """MergeResult is frozen — field reassignment must raise FrozenInstanceError."""

    def test_cannot_reassign_merged_record_id(self) -> None:
        r = _merge_result()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError is AttributeError subclass
            r.merged_record_id = "other"  # type: ignore[misc]

    def test_cannot_reassign_success(self) -> None:
        r = _merge_result()
        with pytest.raises(Exception):
            r.success = False  # type: ignore[misc]

    def test_cannot_reassign_error(self) -> None:
        r = _merge_result()
        with pytest.raises(Exception):
            r.error = "late error"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConsolidationMetrics — required fields and success_rate
# ---------------------------------------------------------------------------


class TestConsolidationMetricsRequiredFields:
    """Valid ConsolidationMetrics exposes all required fields with correct types."""

    def test_entries_processed_present(self) -> None:
        m = _metrics(entries_processed=10)
        assert m.entries_processed == 10

    def test_entries_processed_is_int(self) -> None:
        assert isinstance(_metrics().entries_processed, int)

    def test_entries_merged_present(self) -> None:
        m = _metrics(entries_merged=3)
        assert m.entries_merged == 3

    def test_entries_retained_present(self) -> None:
        m = _metrics(entries_retained=2)
        assert m.entries_retained == 2

    def test_semantic_records_created_present(self) -> None:
        m = _metrics(semantic_records_created=1)
        assert m.semantic_records_created == 1

    def test_merge_results_is_tuple(self) -> None:
        assert isinstance(_metrics().merge_results, tuple)

    def test_started_at_is_datetime(self) -> None:
        assert isinstance(_metrics().started_at, datetime)

    def test_completed_at_is_datetime(self) -> None:
        assert isinstance(_metrics().completed_at, datetime)


class TestConsolidationMetricsMissingRequired:
    """Omitting required fields raises TypeError."""

    def test_missing_entries_processed(self) -> None:
        with pytest.raises(TypeError):
            ConsolidationMetrics(  # type: ignore[call-arg]
                entries_merged=0,
                entries_retained=0,
                semantic_records_created=0,
                merge_results=(),
                started_at=_T0,
                completed_at=_T1,
            )

    def test_missing_started_at(self) -> None:
        with pytest.raises(TypeError):
            ConsolidationMetrics(  # type: ignore[call-arg]
                entries_processed=0,
                entries_merged=0,
                entries_retained=0,
                semantic_records_created=0,
                merge_results=(),
                completed_at=_T1,
            )

    def test_missing_completed_at(self) -> None:
        with pytest.raises(TypeError):
            ConsolidationMetrics(  # type: ignore[call-arg]
                entries_processed=0,
                entries_merged=0,
                entries_retained=0,
                semantic_records_created=0,
                merge_results=(),
                started_at=_T0,
            )

    def test_no_arguments(self) -> None:
        with pytest.raises(TypeError):
            ConsolidationMetrics()  # type: ignore[call-arg]


class TestConsolidationMetricsImmutability:
    """ConsolidationMetrics is frozen."""

    def test_cannot_reassign_entries_processed(self) -> None:
        m = _metrics()
        with pytest.raises(Exception):
            m.entries_processed = 99  # type: ignore[misc]


class TestConsolidationMetricsSuccessRate:
    """success_rate computed property returns the correct fraction."""

    def test_all_success(self) -> None:
        results = tuple(_merge_result(success=True) for _ in range(4))
        m = _metrics(merge_results=results)
        assert m.success_rate == 1.0

    def test_all_failure(self) -> None:
        results = tuple(_merge_result(success=False, error="err") for _ in range(3))
        m = _metrics(merge_results=results)
        assert m.success_rate == 0.0

    def test_half_success(self) -> None:
        results = (
            _merge_result(success=True),
            _merge_result(success=False, error="e"),
        )
        m = _metrics(merge_results=results)
        assert m.success_rate == 0.5

    def test_empty_merge_results_returns_zero(self) -> None:
        m = _metrics(merge_results=())
        assert m.success_rate == 0.0

    def test_success_rate_is_float(self) -> None:
        assert isinstance(_metrics(merge_results=(_merge_result(),)).success_rate, float)


# ---------------------------------------------------------------------------
# ConsolidationConfig — defaults and invariants
# ---------------------------------------------------------------------------


class TestConsolidationConfigDefaults:
    """Default ConsolidationConfig is valid with expected field values."""

    def test_default_threshold(self) -> None:
        assert ConsolidationConfig().threshold == 10

    def test_default_enabled(self) -> None:
        assert ConsolidationConfig().enabled is True

    def test_default_cosine_similarity_threshold(self) -> None:
        assert ConsolidationConfig().cosine_similarity_threshold == 0.85

    def test_default_decay_halflife_days(self) -> None:
        assert ConsolidationConfig().decay_halflife_days == 90.0

    def test_default_merge_strategy(self) -> None:
        assert ConsolidationConfig().merge_strategy is MergeStrategy.LATEST_WINS

    def test_default_retention_policy(self) -> None:
        assert ConsolidationConfig().retention_policy == RetentionPolicy()

    def test_custom_threshold(self) -> None:
        assert ConsolidationConfig(threshold=5).threshold == 5

    def test_disabled(self) -> None:
        assert ConsolidationConfig(enabled=False).enabled is False

    def test_threshold_is_int(self) -> None:
        assert isinstance(ConsolidationConfig().threshold, int)

    def test_enabled_is_bool(self) -> None:
        assert isinstance(ConsolidationConfig().enabled, bool)

    def test_cosine_similarity_threshold_is_float(self) -> None:
        assert isinstance(ConsolidationConfig().cosine_similarity_threshold, float)

    def test_decay_halflife_days_is_float(self) -> None:
        assert isinstance(ConsolidationConfig().decay_halflife_days, float)


class TestConsolidationConfigInvalidThreshold:
    """threshold < 1 is rejected at construction time."""

    @pytest.mark.parametrize("bad_threshold", [0, -1, -100])
    def test_threshold_below_one_raises_value_error(self, bad_threshold: int) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ConsolidationConfig(threshold=bad_threshold)

    def test_threshold_of_one_is_valid(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.threshold == 1


class TestConsolidationConfigImmutability:
    """ConsolidationConfig is frozen."""

    def test_cannot_reassign_threshold(self) -> None:
        cfg = ConsolidationConfig()
        with pytest.raises(Exception):
            cfg.threshold = 99  # type: ignore[misc]

    def test_cannot_reassign_enabled(self) -> None:
        cfg = ConsolidationConfig()
        with pytest.raises(Exception):
            cfg.enabled = False  # type: ignore[misc]


class TestConsolidationConfigShouldConsolidate:
    """should_consolidate returns True iff enabled and entry_count >= threshold."""

    def test_at_threshold_triggers(self) -> None:
        cfg = ConsolidationConfig(threshold=10)
        assert cfg.should_consolidate(10) is True

    def test_above_threshold_triggers(self) -> None:
        cfg = ConsolidationConfig(threshold=10)
        assert cfg.should_consolidate(11) is True

    def test_below_threshold_does_not_trigger(self) -> None:
        cfg = ConsolidationConfig(threshold=10)
        assert cfg.should_consolidate(9) is False

    def test_zero_count_does_not_trigger(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(0) is False

    def test_disabled_never_triggers(self) -> None:
        cfg = ConsolidationConfig(threshold=1, enabled=False)
        assert cfg.should_consolidate(1) is False
        assert cfg.should_consolidate(1000) is False

    def test_returns_bool(self) -> None:
        cfg = ConsolidationConfig()
        result = cfg.should_consolidate(0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# ConsolidationConfig — cosine_similarity_threshold validation
# ---------------------------------------------------------------------------


class TestConsolidationConfigCosineSimilarityThreshold:
    """cosine_similarity_threshold must be in [0.0, 1.0]."""

    @pytest.mark.parametrize("valid", [0.0, 0.5, 0.85, 1.0])
    def test_valid_values_accepted(self, valid: float) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=valid)
        assert cfg.cosine_similarity_threshold == valid

    @pytest.mark.parametrize("bad", [-0.001, -1.0, 1.001, 2.0])
    def test_out_of_range_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="cosine_similarity_threshold"):
            ConsolidationConfig(cosine_similarity_threshold=bad)

    def test_exactly_zero_accepted(self) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=0.0)
        assert cfg.cosine_similarity_threshold == 0.0

    def test_exactly_one_accepted(self) -> None:
        cfg = ConsolidationConfig(cosine_similarity_threshold=1.0)
        assert cfg.cosine_similarity_threshold == 1.0


# ---------------------------------------------------------------------------
# ConsolidationConfig — decay_halflife_days validation
# ---------------------------------------------------------------------------


class TestConsolidationConfigDecayHalflifeDays:
    """decay_halflife_days must be strictly positive."""

    @pytest.mark.parametrize("valid", [0.001, 1.0, 90.0, 365.0])
    def test_valid_values_accepted(self, valid: float) -> None:
        cfg = ConsolidationConfig(decay_halflife_days=valid)
        assert cfg.decay_halflife_days == valid

    @pytest.mark.parametrize("bad", [0.0, -0.001, -90.0])
    def test_non_positive_raises_value_error(self, bad: float) -> None:
        with pytest.raises(ValueError, match="decay_halflife_days"):
            ConsolidationConfig(decay_halflife_days=bad)


# ---------------------------------------------------------------------------
# RetentionPolicy — defaults and validation
# ---------------------------------------------------------------------------


class TestRetentionPolicyDefaults:
    """Default RetentionPolicy has expected TTL values."""

    def test_default_episodic_ttl(self) -> None:
        assert RetentionPolicy().episodic_ttl_days == 7.0

    def test_default_semantic_ttl(self) -> None:
        assert RetentionPolicy().semantic_ttl_days == 30.0

    def test_default_procedural_ttl(self) -> None:
        assert RetentionPolicy().procedural_ttl_days == 90.0

    def test_default_archival_ttl(self) -> None:
        assert RetentionPolicy().archival_ttl_days == 365.0

    def test_all_fields_are_float(self) -> None:
        p = RetentionPolicy()
        assert all(
            isinstance(v, float)
            for v in (
                p.episodic_ttl_days,
                p.semantic_ttl_days,
                p.procedural_ttl_days,
                p.archival_ttl_days,
            )
        )


class TestRetentionPolicyValidation:
    """All TTL fields must be strictly positive."""

    @pytest.mark.parametrize(
        "field_name",
        ["episodic_ttl_days", "semantic_ttl_days", "procedural_ttl_days", "archival_ttl_days"],
    )
    def test_zero_raises_value_error(self, field_name: str) -> None:
        with pytest.raises(ValueError, match=field_name):
            RetentionPolicy(**{field_name: 0.0})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field_name",
        ["episodic_ttl_days", "semantic_ttl_days", "procedural_ttl_days", "archival_ttl_days"],
    )
    def test_negative_raises_value_error(self, field_name: str) -> None:
        with pytest.raises(ValueError, match=field_name):
            RetentionPolicy(**{field_name: -1.0})  # type: ignore[arg-type]

    def test_small_positive_value_accepted(self) -> None:
        p = RetentionPolicy(episodic_ttl_days=0.001)
        assert p.episodic_ttl_days == 0.001

    def test_immutable(self) -> None:
        p = RetentionPolicy()
        with pytest.raises(Exception):
            p.episodic_ttl_days = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MergeStrategy — enum values
# ---------------------------------------------------------------------------


class TestMergeStrategy:
    """MergeStrategy enum has the expected members and str representation."""

    def test_latest_wins_value(self) -> None:
        assert MergeStrategy.LATEST_WINS.value == "latest_wins"

    def test_weighted_average_value(self) -> None:
        assert MergeStrategy.WEIGHTED_AVERAGE.value == "weighted_average"

    def test_llm_synthesis_value(self) -> None:
        assert MergeStrategy.LLM_SYNTHESIS.value == "llm_synthesis"

    def test_str_returns_value(self) -> None:
        assert str(MergeStrategy.LATEST_WINS) == "latest_wins"
        assert str(MergeStrategy.LLM_SYNTHESIS) == "llm_synthesis"

    def test_all_members_present(self) -> None:
        names = {m.name for m in MergeStrategy}
        assert names == {"LATEST_WINS", "WEIGHTED_AVERAGE", "LLM_SYNTHESIS"}


# ---------------------------------------------------------------------------
# ConsolidationTrigger — protocol structural checking
# ---------------------------------------------------------------------------


class TestConsolidationTriggerProtocol:
    """Objects satisfying the protocol are accepted; non-conforming ones are not."""

    def test_config_satisfies_protocol(self) -> None:
        cfg = ConsolidationConfig()
        assert isinstance(cfg, ConsolidationTrigger)

    def test_custom_class_satisfies_protocol(self) -> None:
        class AlwaysTrigger:
            def should_consolidate(self, entry_count: int) -> bool:
                return True

        assert isinstance(AlwaysTrigger(), ConsolidationTrigger)

    def test_missing_method_does_not_satisfy_protocol(self) -> None:
        class NoMethod:
            pass

        assert not isinstance(NoMethod(), ConsolidationTrigger)

    def test_wrong_signature_still_satisfies_structural_check(self) -> None:
        """Protocol runtime_checkable only checks method presence, not signature."""

        class WrongSig:
            def should_consolidate(self) -> bool:  # missing entry_count
                return False

        # isinstance only validates name presence at runtime
        assert isinstance(WrongSig(), ConsolidationTrigger)

    def test_lambda_does_not_satisfy_protocol(self) -> None:
        """A bare callable without should_consolidate does not satisfy the protocol."""
        fn = lambda n: n > 5  # noqa: E731
        assert not isinstance(fn, ConsolidationTrigger)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        neg = [-x for x in v]
        assert cosine_similarity(v, neg) == pytest.approx(-1.0)

    def test_mismatched_dimensions_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
