"""Unit tests verifying last_accessed_at is correctly updated on vault entry access.

Contract under test:
  - last_accessed_at starts as None (never accessed)
  - Setting last_accessed_at to a UTC datetime records the access time
  - Subsequent accesses update to a strictly newer timestamp
  - last_accessed_at survives JSON serialization/deserialization
  - All four MemoryRecord subtypes inherit the same behaviour
  - last_accessed_at is independent of updated_at (read ≠ write)
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    EpisodicRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
    ProceduralRecord,
    SemanticRecord,
    WorkingRecord,
    memory_record_from_dict,
    memory_record_from_json,
    memory_record_to_dict,
    memory_record_to_json,
)

# ---------------------------------------------------------------------------
# Shared timestamps (all UTC-aware, strictly ordered)
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)       # event time
_T1 = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)       # ingested / created
_T2 = datetime(2026, 1, 2, 8, 0, 0, tzinfo=UTC)       # updated
_FIRST_ACCESS = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)
_SECOND_ACCESS = datetime(2026, 1, 4, 9, 0, 0, 500000, tzinfo=UTC)


def _make(
    *,
    record_id: str = "rec-001",
    memory_type: MemoryType = MemoryType.EPISODIC,
    content: str = "test content",
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        memory_type=memory_type,
        record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content=content),
        created_at=_T1,
        updated_at=_T2,
    )


# ---------------------------------------------------------------------------
# Initial state: never accessed
# ---------------------------------------------------------------------------


class TestInitialState:
    """Before any access, last_accessed_at must be None."""

    def test_new_record_last_accessed_at_is_none(self) -> None:
        assert _make().last_accessed_at is None

    def test_explicit_none_constructor(self) -> None:
        rec = MemoryRecord(
            record_id="x",
            memory_type=MemoryType.WORKING,
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content=""),
            created_at=_T1,
            updated_at=_T2,
            last_accessed_at=None,
        )
        assert rec.last_accessed_at is None


# ---------------------------------------------------------------------------
# First access: None → timestamp
# ---------------------------------------------------------------------------


class TestFirstAccess:
    """Setting last_accessed_at from None to a datetime represents first access."""

    def test_first_access_sets_timestamp(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.last_accessed_at == _FIRST_ACCESS

    def test_first_access_timestamp_is_utc_aware(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.last_accessed_at is not None
        assert rec.last_accessed_at.utcoffset() == timedelta(0)

    def test_first_access_timestamp_is_after_ingestion(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.last_accessed_at > rec.ingested_at

    def test_first_access_does_not_mutate_updated_at(self) -> None:
        rec = _make()
        original_updated_at = rec.updated_at
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.updated_at == original_updated_at

    def test_first_access_does_not_mutate_created_at(self) -> None:
        rec = _make()
        original_created_at = rec.created_at
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.created_at == original_created_at


# ---------------------------------------------------------------------------
# Subsequent access: old timestamp → newer timestamp
# ---------------------------------------------------------------------------


class TestSubsequentAccess:
    """Re-accessing a record updates last_accessed_at to a strictly newer time."""

    def test_second_access_overwrites_first(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        rec.last_accessed_at = _SECOND_ACCESS
        assert rec.last_accessed_at == _SECOND_ACCESS

    def test_second_access_is_strictly_newer(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        rec.last_accessed_at = _SECOND_ACCESS
        assert rec.last_accessed_at > _FIRST_ACCESS

    def test_access_with_microsecond_precision(self) -> None:
        ts = datetime(2026, 2, 15, 10, 30, 0, 123456, tzinfo=UTC)
        rec = _make()
        rec.last_accessed_at = ts
        assert rec.last_accessed_at == ts
        assert rec.last_accessed_at.microsecond == 123456

    def test_access_can_be_cleared_back_to_none(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        rec.last_accessed_at = None
        assert rec.last_accessed_at is None


# ---------------------------------------------------------------------------
# JSON serialization roundtrip
# ---------------------------------------------------------------------------


class TestSerializationRoundtrip:
    """last_accessed_at must survive memory_record_to_json / memory_record_from_json."""

    def test_none_roundtrips_as_none(self) -> None:
        rec = _make()
        assert memory_record_from_json(memory_record_to_json(rec)).last_accessed_at is None

    def test_timestamp_roundtrips_exactly(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        restored = memory_record_from_json(memory_record_to_json(rec))
        assert restored.last_accessed_at == _FIRST_ACCESS

    def test_updated_timestamp_roundtrips(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        rec.last_accessed_at = _SECOND_ACCESS
        restored = memory_record_from_json(memory_record_to_json(rec))
        assert restored.last_accessed_at == _SECOND_ACCESS

    def test_microseconds_preserved_through_serialization(self) -> None:
        ts = datetime(2026, 3, 1, 6, 0, 0, 999999, tzinfo=UTC)
        rec = _make()
        rec.last_accessed_at = ts
        restored = memory_record_from_json(memory_record_to_json(rec))
        assert restored.last_accessed_at == ts

    def test_utc_offset_preserved_through_serialization(self) -> None:
        rec = _make()
        rec.last_accessed_at = _SECOND_ACCESS
        restored = memory_record_from_json(memory_record_to_json(rec))
        assert restored.last_accessed_at is not None
        assert restored.last_accessed_at.utcoffset() == timedelta(0)

    def test_dict_serialization_emits_none_when_not_accessed(self) -> None:
        payload = memory_record_to_dict(_make())
        assert payload["last_accessed_at"] is None

    def test_dict_serialization_emits_iso_string_when_accessed(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        payload = memory_record_to_dict(rec)
        assert isinstance(payload["last_accessed_at"], str)
        assert datetime.fromisoformat(payload["last_accessed_at"]) == _FIRST_ACCESS

    def test_dict_deserialization_absent_key_yields_none(self) -> None:
        d = memory_record_to_dict(_make())
        d.pop("last_accessed_at", None)
        restored = memory_record_from_dict(d)
        assert restored.last_accessed_at is None

    def test_dict_deserialization_null_value_yields_none(self) -> None:
        d = memory_record_to_dict(_make())
        d["last_accessed_at"] = None
        restored = memory_record_from_dict(d)
        assert restored.last_accessed_at is None


# ---------------------------------------------------------------------------
# Subtype inheritance
# ---------------------------------------------------------------------------


class TestSubtypeInheritance:
    """All four MemoryRecord subtypes must honour the same last_accessed_at contract."""

    def _working(self) -> WorkingRecord:
        return WorkingRecord(
            record_id="w-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content="w"),
            created_at=_T1,
            updated_at=_T2,
        )

    def _episodic(self) -> EpisodicRecord:
        return EpisodicRecord(
            record_id="e-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content="e"),
            created_at=_T1,
            updated_at=_T2,
        )

    def _semantic(self) -> SemanticRecord:
        return SemanticRecord(
            record_id="s-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content="s"),
            created_at=_T1,
            updated_at=_T2,
        )

    def _procedural(self) -> ProceduralRecord:
        return ProceduralRecord(
            record_id="p-1",
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content="p"),
            created_at=_T1,
            updated_at=_T2,
        )

    @pytest.mark.parametrize("factory", [
        "_working", "_episodic", "_semantic", "_procedural"
    ])
    def test_defaults_to_none(self, factory: str) -> None:
        rec = getattr(self, factory)()
        assert rec.last_accessed_at is None

    @pytest.mark.parametrize("factory", [
        "_working", "_episodic", "_semantic", "_procedural"
    ])
    def test_update_sets_timestamp(self, factory: str) -> None:
        rec = getattr(self, factory)()
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.last_accessed_at == _FIRST_ACCESS

    @pytest.mark.parametrize("factory", [
        "_working", "_episodic", "_semantic", "_procedural"
    ])
    def test_subsequent_update_overwrites(self, factory: str) -> None:
        rec = getattr(self, factory)()
        rec.last_accessed_at = _FIRST_ACCESS
        rec.last_accessed_at = _SECOND_ACCESS
        assert rec.last_accessed_at == _SECOND_ACCESS

    @pytest.mark.parametrize("factory", [
        "_working", "_episodic", "_semantic", "_procedural"
    ])
    def test_roundtrip_preserves_access_time(self, factory: str) -> None:
        rec = getattr(self, factory)()
        rec.last_accessed_at = _FIRST_ACCESS
        restored = memory_record_from_json(memory_record_to_json(rec))
        assert restored.last_accessed_at == _FIRST_ACCESS


# ---------------------------------------------------------------------------
# Temporal ordering invariants
# ---------------------------------------------------------------------------


class TestTemporalOrdering:
    """last_accessed_at must be >= created_at and >= ingested_at when set."""

    def test_access_time_not_before_creation(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.last_accessed_at >= rec.created_at

    def test_access_time_not_before_ingestion(self) -> None:
        rec = _make()
        rec.last_accessed_at = _FIRST_ACCESS
        assert rec.last_accessed_at >= rec.ingested_at

    def test_multiple_accesses_are_non_decreasing(self) -> None:
        rec = _make()
        accesses = [
            datetime(2026, 1, 3, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 3, 0, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 3, 0, 0, 2, tzinfo=UTC),
        ]
        prev = None
        for ts in accesses:
            rec.last_accessed_at = ts
            if prev is not None:
                assert rec.last_accessed_at >= prev
            prev = rec.last_accessed_at

    def test_access_independent_of_updated_at_ordering(self) -> None:
        # A read (access) can happen well after the last write (updated_at).
        late_access = _T2 + timedelta(days=30)
        rec = _make()
        rec.last_accessed_at = late_access
        assert rec.last_accessed_at > rec.updated_at
