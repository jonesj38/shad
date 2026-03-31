"""Edge-case tests for MemoryRecord field constraints.

Covers gaps not addressed by the existing edge-case test files:

  1. Type mismatches in memory_record_from_dict (non-enum fields):
       - integer / None for record_id
       - malformed ISO-8601 strings for datetime fields
       - `record` sub-dict replaced by a scalar
       - metadata.tags supplied as a plain string (silent list corruption risk)
       - memory_type as None or integer

  2. MemoryMetadata.confidence out-of-range values:
       - The dataclass performs no range validation; this suite documents the
         existing behaviour so any future validator is a deliberate change.

  3. Subclass-specific field edge cases:
       - WorkingRecord: ttl_seconds=0, ttl_seconds negative
       - EpisodicRecord: decay_rate=0.0, decay_rate negative
       - SemanticRecord: abstraction_level=0, abstraction_level negative
       - ProceduralRecord: invocation_count=0, invocation_count negative
       - Fixed memory_type discriminator cannot be overridden by callers

  4. Timezone-naive datetimes through serialization:
       - isoformat() of a naive datetime lacks +00:00; fromisoformat() round-
         trip must not silently discard timezone information.

  5. contracts.MemoryRecord embedding constraints:
       - empty embedding list
       - NaN and Inf float values (stored without error by the dataclass)
       - created_at > updated_at (logically backwards; no validation gate)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timezone, timedelta

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
    WorkingRecord,
    EpisodicRecord,
    SemanticRecord,
    ProceduralRecord,
    memory_record_from_dict,
    memory_record_to_dict,
    memory_record_from_json,
    memory_record_to_json,
)
from shad.vault.contracts import MemoryRecord as ContractsMemoryRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _make(
    *,
    record_id: str = "rec-001",
    content: str = "content",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    metadata: MemoryMetadata | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        memory_type=memory_type,
        record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content=content),
        created_at=_T0,
        updated_at=_T0,
        metadata=metadata,
    )


def _valid_dict() -> dict:
    return memory_record_to_dict(_make())


# ---------------------------------------------------------------------------
# 1. Type mismatches in memory_record_from_dict
# ---------------------------------------------------------------------------


class TestFromDictTypeMismatches:
    """memory_record_from_dict must not silently coerce wrong-typed values."""

    # --- memory_type ---

    def test_none_memory_type_raises(self) -> None:
        d = _valid_dict()
        d["memory_type"] = None
        with pytest.raises((ValueError, TypeError)):
            memory_record_from_dict(d)

    def test_integer_memory_type_raises(self) -> None:
        d = _valid_dict()
        d["memory_type"] = 42
        with pytest.raises((ValueError, TypeError)):
            memory_record_from_dict(d)

    def test_list_memory_type_raises(self) -> None:
        d = _valid_dict()
        d["memory_type"] = ["semantic_memory"]
        with pytest.raises((ValueError, TypeError)):
            memory_record_from_dict(d)

    # --- datetime fields ---

    @pytest.mark.parametrize("field_name", ["created_at", "updated_at"])
    def test_malformed_datetime_string_raises(self, field_name: str) -> None:
        d = _valid_dict()
        d[field_name] = "not-a-datetime"
        with pytest.raises((ValueError, TypeError)):
            memory_record_from_dict(d)

    def test_integer_created_at_raises(self) -> None:
        d = _valid_dict()
        d["created_at"] = 1234567890
        with pytest.raises((ValueError, TypeError, AttributeError)):
            memory_record_from_dict(d)

    def test_malformed_last_accessed_at_raises(self) -> None:
        d = _valid_dict()
        d["last_accessed_at"] = "2026-99-99T00:00:00"
        with pytest.raises((ValueError, OverflowError)):
            memory_record_from_dict(d)

    # --- record sub-dict ---

    def test_record_as_string_raises(self) -> None:
        d = _valid_dict()
        d["record"] = "not-a-dict"
        with pytest.raises((TypeError, KeyError)):
            memory_record_from_dict(d)

    def test_record_as_list_raises(self) -> None:
        d = _valid_dict()
        d["record"] = ["event_time", "ingested_at", "content"]
        with pytest.raises((TypeError, KeyError)):
            memory_record_from_dict(d)

    def test_record_event_time_malformed_raises(self) -> None:
        d = _valid_dict()
        d["record"]["event_time"] = "banana"
        with pytest.raises(ValueError):
            memory_record_from_dict(d)

    # --- metadata.tags as a plain string (silent corruption risk) ---

    def test_metadata_tags_as_string_does_not_silently_corrupt(self) -> None:
        """Supplying tags as a bare string rather than a list of strings is a
        common mistake.  The deserializer wraps the value in ``list()``, which
        turns a string into a list of individual characters.  This test
        documents the current behaviour so the corruption is explicit rather
        than invisible.

        If validation is added later, update this test to expect an error
        instead.
        """
        d = _valid_dict()
        d["metadata"] = {
            "source": "https://example.com",
            "confidence": 0.9,
            "tags": "github",  # string, not list
            "promoted_at": None,
            "promoted_from": None,
            "consolidated_at": None,
            "consolidated_from": None,
        }
        rec = memory_record_from_dict(d)
        # ``list("github")`` == ['g', 'i', 't', 'h', 'u', 'b']
        assert rec.metadata is not None
        assert rec.metadata.tags == list("github"), (
            "tags-as-string is silently split into chars; if this assertion "
            "fails, validation has been added — update the test accordingly"
        )


# ---------------------------------------------------------------------------
# 2. MemoryMetadata.confidence out-of-range values
# ---------------------------------------------------------------------------


class TestMetadataConfidenceOutOfRange:
    """MemoryMetadata has no range validation on confidence.

    These tests document the *current* permissive behaviour.  If a validator
    is added (e.g., ``__post_init__`` raising ValueError for values outside
    [0.0, 1.0]), change the assertions to expect the error.
    """

    @pytest.mark.parametrize("value", [-0.001, -1.0, -100.0])
    def test_negative_confidence_is_stored_as_is(self, value: float) -> None:
        m = MemoryMetadata(source="src", confidence=value)
        assert m.confidence == value

    @pytest.mark.parametrize("value", [1.001, 2.0, 100.0])
    def test_above_one_confidence_is_stored_as_is(self, value: float) -> None:
        m = MemoryMetadata(source="src", confidence=value)
        assert m.confidence == value

    def test_confidence_at_lower_boundary_accepted(self) -> None:
        m = MemoryMetadata(source="src", confidence=0.0)
        assert m.confidence == 0.0

    def test_confidence_at_upper_boundary_accepted(self) -> None:
        m = MemoryMetadata(source="src", confidence=1.0)
        assert m.confidence == 1.0

    def test_nan_confidence_is_stored_without_error(self) -> None:
        m = MemoryMetadata(source="src", confidence=float("nan"))
        assert math.isnan(m.confidence)

    def test_inf_confidence_is_stored_without_error(self) -> None:
        m = MemoryMetadata(source="src", confidence=float("inf"))
        assert math.isinf(m.confidence)


# ---------------------------------------------------------------------------
# 3. Subclass-specific field edge cases
# ---------------------------------------------------------------------------


class TestWorkingRecordFields:
    """WorkingRecord field constraints."""

    def test_memory_type_is_always_working(self) -> None:
        r = WorkingRecord(
            record_id="w-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.memory_type is MemoryType.WORKING

    def test_memory_type_cannot_be_changed_at_construction(self) -> None:
        """memory_type is init=False; passing it as a kwarg is a TypeError."""
        with pytest.raises(TypeError):
            WorkingRecord(  # type: ignore[call-arg]
                record_id="w-2",
                record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
                memory_type=MemoryType.SEMANTIC,
            )

    def test_ttl_seconds_zero_is_accepted(self) -> None:
        r = WorkingRecord(
            record_id="w-3",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            ttl_seconds=0,
        )
        assert r.ttl_seconds == 0

    def test_ttl_seconds_negative_is_stored_without_error(self) -> None:
        """No validation prevents negative TTL; documents current behaviour."""
        r = WorkingRecord(
            record_id="w-4",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            ttl_seconds=-1,
        )
        assert r.ttl_seconds == -1

    def test_ttl_seconds_defaults_to_none(self) -> None:
        r = WorkingRecord(
            record_id="w-5",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.ttl_seconds is None

    def test_context_window_id_none_by_default(self) -> None:
        r = WorkingRecord(
            record_id="w-6",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.context_window_id is None

    def test_context_window_id_empty_string_accepted(self) -> None:
        r = WorkingRecord(
            record_id="w-7",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            context_window_id="",
        )
        assert r.context_window_id == ""


class TestEpisodicRecordFields:
    """EpisodicRecord field constraints."""

    def test_memory_type_is_always_episodic(self) -> None:
        r = EpisodicRecord(
            record_id="e-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.memory_type is MemoryType.EPISODIC

    def test_decay_rate_default_is_one(self) -> None:
        r = EpisodicRecord(
            record_id="e-2",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.decay_rate == 1.0

    def test_decay_rate_zero_is_accepted(self) -> None:
        r = EpisodicRecord(
            record_id="e-3",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            decay_rate=0.0,
        )
        assert r.decay_rate == 0.0

    def test_decay_rate_negative_is_stored_without_error(self) -> None:
        """No validation prevents negative decay; documents current behaviour."""
        r = EpisodicRecord(
            record_id="e-4",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            decay_rate=-0.5,
        )
        assert r.decay_rate == -0.5

    def test_session_id_none_by_default(self) -> None:
        r = EpisodicRecord(
            record_id="e-5",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.session_id is None

    def test_session_id_empty_string_accepted(self) -> None:
        r = EpisodicRecord(
            record_id="e-6",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            session_id="",
        )
        assert r.session_id == ""


class TestSemanticRecordFields:
    """SemanticRecord field constraints."""

    def test_memory_type_is_always_semantic(self) -> None:
        r = SemanticRecord(
            record_id="s-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.memory_type is MemoryType.SEMANTIC

    def test_abstraction_level_default_is_zero(self) -> None:
        r = SemanticRecord(
            record_id="s-2",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.abstraction_level == 0

    def test_abstraction_level_zero_is_accepted(self) -> None:
        r = SemanticRecord(
            record_id="s-3",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            abstraction_level=0,
        )
        assert r.abstraction_level == 0

    def test_abstraction_level_negative_is_stored_without_error(self) -> None:
        """No validation prevents negative abstraction levels; documents current behaviour."""
        r = SemanticRecord(
            record_id="s-4",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            abstraction_level=-1,
        )
        assert r.abstraction_level == -1

    def test_concept_name_none_by_default(self) -> None:
        r = SemanticRecord(
            record_id="s-5",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.concept_name is None

    def test_concept_name_empty_string_accepted(self) -> None:
        r = SemanticRecord(
            record_id="s-6",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            concept_name="",
        )
        assert r.concept_name == ""


class TestProceduralRecordFields:
    """ProceduralRecord field constraints."""

    def test_memory_type_is_always_procedural(self) -> None:
        r = ProceduralRecord(
            record_id="p-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.memory_type is MemoryType.PROCEDURAL

    def test_invocation_count_default_is_zero(self) -> None:
        r = ProceduralRecord(
            record_id="p-2",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.invocation_count == 0

    def test_invocation_count_negative_is_stored_without_error(self) -> None:
        """No validation prevents negative counts; documents current behaviour."""
        r = ProceduralRecord(
            record_id="p-3",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            invocation_count=-1,
        )
        assert r.invocation_count == -1

    def test_skill_name_none_by_default(self) -> None:
        r = ProceduralRecord(
            record_id="p-4",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
        )
        assert r.skill_name is None


# ---------------------------------------------------------------------------
# 4. Timezone-naive datetimes through serialization
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    """MemoryRecord datetime fields should carry UTC timezone info.

    Tests document the round-trip behaviour for both UTC and other offsets,
    and alert on naive datetime inputs that lose timezone on re-read.
    """

    def test_utc_datetimes_survive_roundtrip(self) -> None:
        rec = _make()
        restored = memory_record_from_json(memory_record_to_json(rec))
        assert restored.created_at.tzinfo is not None
        assert restored.created_at == _T0

    def test_non_utc_offset_survives_roundtrip(self) -> None:
        """A +05:30 (IST) datetime round-trips with offset preserved."""
        ist = timezone(timedelta(hours=5, minutes=30))
        t = datetime(2026, 1, 1, 5, 30, 0, tzinfo=ist)
        rec = MemoryRecord(
            record_id="tz-001",
            memory_type=MemoryType.SEMANTIC,
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            created_at=t,
            updated_at=t,
        )
        restored = memory_record_from_json(memory_record_to_json(rec))
        # isoformat/fromisoformat preserves the numeric offset; equality holds
        # because both sides represent the same instant.
        assert restored.created_at == t

    def test_naive_datetime_loses_timezone_on_roundtrip(self) -> None:
        """A timezone-naive created_at has no tzinfo after isoformat/fromisoformat.

        This test *documents* the data-loss risk so callers know to always
        pass UTC-aware datetimes.  If validation is added to reject naive
        datetimes, change this to expect a ValueError.
        """
        naive = datetime(2026, 1, 1, 0, 0, 0)  # no tzinfo
        rec = MemoryRecord(
            record_id="tz-002",
            memory_type=MemoryType.SEMANTIC,
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="x"),
            created_at=naive,
            updated_at=naive,
        )
        raw = memory_record_to_json(rec)
        restored = memory_record_from_json(raw)
        # Naive datetime has no tzinfo — documents the information loss.
        assert restored.created_at.tzinfo is None


# ---------------------------------------------------------------------------
# 5. contracts.MemoryRecord embedding edge cases
# ---------------------------------------------------------------------------

_CNOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)


def _valid_contracts(**overrides) -> ContractsMemoryRecord:
    defaults = dict(
        record_id="rec-001",
        content="payload",
        embedding=[0.1, 0.2, 0.3],
        memory_type=MemoryType.SEMANTIC,
        created_at=_CNOW,
        updated_at=_CNOW,
    )
    defaults.update(overrides)
    return ContractsMemoryRecord(**defaults)


class TestContractsMemoryRecordEmbeddingConstraints:
    """contracts.MemoryRecord embedding field edge cases."""

    def test_empty_embedding_is_accepted(self) -> None:
        rec = _valid_contracts(embedding=[])
        assert rec.embedding == []

    def test_single_element_embedding_accepted(self) -> None:
        rec = _valid_contracts(embedding=[1.0])
        assert rec.embedding == [1.0]

    def test_nan_in_embedding_stored_without_error(self) -> None:
        """NaN in a float list is stored as-is; no validation gate exists."""
        rec = _valid_contracts(embedding=[0.1, float("nan"), 0.3])
        assert math.isnan(rec.embedding[1])

    def test_inf_in_embedding_stored_without_error(self) -> None:
        rec = _valid_contracts(embedding=[float("inf"), 0.0])
        assert math.isinf(rec.embedding[0])

    def test_large_embedding_accepted(self) -> None:
        vec = [float(i) / 1536 for i in range(1536)]
        rec = _valid_contracts(embedding=vec)
        assert len(rec.embedding) == 1536


class TestContractsMemoryRecordTemporalConstraints:
    """contracts.MemoryRecord temporal field edge cases."""

    def test_created_at_after_updated_at_is_stored_without_error(self) -> None:
        """No invariant prevents created_at > updated_at; documents current behaviour."""
        later = _CNOW + timedelta(days=1)
        rec = _valid_contracts(created_at=later, updated_at=_CNOW)
        assert rec.created_at > rec.updated_at

    def test_empty_record_id_accepted_by_contracts_record(self) -> None:
        rec = _valid_contracts(record_id="")
        assert rec.record_id == ""

    def test_empty_content_accepted_by_contracts_record(self) -> None:
        rec = _valid_contracts(content="")
        assert rec.content == ""

    def test_source_refs_duplicates_preserved(self) -> None:
        refs = ["path/a.md", "path/a.md", "path/b.md"]
        rec = _valid_contracts(source_refs=refs)
        assert rec.source_refs == refs

    def test_tags_duplicates_preserved(self) -> None:
        tags = ["retrieval", "retrieval", "github"]
        rec = _valid_contracts(tags=tags)
        assert rec.tags == tags
