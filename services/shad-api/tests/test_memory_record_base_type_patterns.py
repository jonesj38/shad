"""Structural contract tests: MemoryRecord as a base record type.

Verifies that MemoryRecord satisfies the three base-type patterns that every
vault record must honour:

  1. Identity pattern  — a stable ``record_id: str`` field
  2. Timestamp pattern — ``created_at``, ``updated_at`` (required, UTC-aware
     datetimes) and ``last_accessed_at`` (optional, defaults to ``None``)
  3. Discriminated-union pattern — ``memory_type: MemoryType`` acts as the
     type discriminator; ``MemoryType`` is a ``str`` subtype whose values are
     unique, stable, and non-overlapping across all variants

The bi-temporal axes (``event_time``, ``ingested_at``) live inside the
composed ``BiTemporalRecord`` but are promoted as read-only properties on
``MemoryRecord`` itself; this is also verified here as part of the "inherits
timestamp fields" contract.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
_T2 = datetime(2026, 1, 2, 8, 0, 0, tzinfo=UTC)
_T3 = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)


def _make(
    *,
    record_id: str = "base-001",
    memory_type: MemoryType = MemoryType.EPISODIC,
    content: str = "sample content",
    created_at: datetime = _T1,
    updated_at: datetime = _T2,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        memory_type=memory_type,
        record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content=content),
        created_at=created_at,
        updated_at=updated_at,
    )


# ---------------------------------------------------------------------------
# 1. Identity pattern: record_id is a required str field
# ---------------------------------------------------------------------------

class TestIdentityField:
    """record_id must exist, be a str, and be preserved verbatim."""

    def test_record_id_field_exists(self) -> None:
        rec = _make(record_id="test-id")
        assert hasattr(rec, "record_id")

    def test_record_id_is_str(self) -> None:
        rec = _make(record_id="test-id")
        assert isinstance(rec.record_id, str)

    def test_record_id_preserved(self) -> None:
        rec = _make(record_id="unique-record-42")
        assert rec.record_id == "unique-record-42"

    def test_record_id_empty_string_allowed(self) -> None:
        rec = _make(record_id="")
        assert rec.record_id == ""

    def test_record_id_unicode_allowed(self) -> None:
        rec = _make(record_id="запись-001")
        assert rec.record_id == "запись-001"

    @pytest.mark.parametrize("rid", [
        "simple",
        "with-dashes",
        "with_underscores",
        "CamelCase",
        "uuid-like-1234-5678",
    ])
    def test_record_id_various_formats(self, rid: str) -> None:
        assert _make(record_id=rid).record_id == rid

    def test_record_id_is_in_dataclass_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(MemoryRecord)}
        assert "record_id" in fields

    def test_record_id_field_type_annotation_is_str(self) -> None:
        fields = {f.name: f for f in dataclasses.fields(MemoryRecord)}
        assert fields["record_id"].type is str or fields["record_id"].type == "str"

    def test_two_records_with_different_ids_are_independent(self) -> None:
        r1 = _make(record_id="alpha")
        r2 = _make(record_id="beta")
        assert r1.record_id != r2.record_id


# ---------------------------------------------------------------------------
# 2. Timestamp pattern: created_at and updated_at (required, UTC-aware)
# ---------------------------------------------------------------------------

class TestRequiredTimestampFields:
    """created_at and updated_at must be present and UTC-aware datetime instances."""

    def test_created_at_field_exists(self) -> None:
        assert hasattr(_make(), "created_at")

    def test_updated_at_field_exists(self) -> None:
        assert hasattr(_make(), "updated_at")

    def test_created_at_is_datetime(self) -> None:
        assert isinstance(_make().created_at, datetime)

    def test_updated_at_is_datetime(self) -> None:
        assert isinstance(_make().updated_at, datetime)

    def test_created_at_is_timezone_aware(self) -> None:
        rec = _make(created_at=_T1)
        assert rec.created_at.tzinfo is not None

    def test_updated_at_is_timezone_aware(self) -> None:
        rec = _make(updated_at=_T2)
        assert rec.updated_at.tzinfo is not None

    def test_created_at_utc_offset_is_zero(self) -> None:
        rec = _make(created_at=_T1)
        assert rec.created_at.utcoffset() == timedelta(0)

    def test_updated_at_utc_offset_is_zero(self) -> None:
        rec = _make(updated_at=_T2)
        assert rec.updated_at.utcoffset() == timedelta(0)

    def test_created_at_preserved(self) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        assert _make(created_at=ts).created_at == ts

    def test_updated_at_preserved(self) -> None:
        ts = datetime(2025, 9, 15, 8, 30, 0, tzinfo=UTC)
        assert _make(updated_at=ts).updated_at == ts

    def test_created_at_and_updated_at_are_independent_fields(self) -> None:
        rec = _make(created_at=_T1, updated_at=_T2)
        assert rec.created_at != rec.updated_at

    def test_created_at_and_updated_at_can_be_equal(self) -> None:
        same = datetime(2026, 3, 31, tzinfo=UTC)
        rec = _make(created_at=same, updated_at=same)
        assert rec.created_at == rec.updated_at

    def test_created_at_in_dataclass_fields(self) -> None:
        assert "created_at" in {f.name for f in dataclasses.fields(MemoryRecord)}

    def test_updated_at_in_dataclass_fields(self) -> None:
        assert "updated_at" in {f.name for f in dataclasses.fields(MemoryRecord)}


# ---------------------------------------------------------------------------
# 3. Optional timestamp: last_accessed_at defaults to None
# ---------------------------------------------------------------------------

class TestOptionalTimestampField:
    """last_accessed_at is optional: None by default, datetime when set."""

    def test_last_accessed_at_field_exists(self) -> None:
        assert hasattr(_make(), "last_accessed_at")

    def test_last_accessed_at_defaults_to_none(self) -> None:
        assert _make().last_accessed_at is None

    def test_last_accessed_at_accepts_datetime(self) -> None:
        rec = _make()
        rec.last_accessed_at = _T3
        assert rec.last_accessed_at == _T3

    def test_last_accessed_at_accepts_none_explicitly(self) -> None:
        rec = MemoryRecord(
            record_id="x",
            memory_type=MemoryType.WORKING,
            record=BiTemporalRecord(event_time=_T0, ingested_at=_T1, content=""),
            created_at=_T1,
            updated_at=_T2,
            last_accessed_at=None,
        )
        assert rec.last_accessed_at is None

    def test_last_accessed_at_is_utc_aware_when_set(self) -> None:
        rec = _make()
        rec.last_accessed_at = _T3
        assert rec.last_accessed_at is not None
        assert rec.last_accessed_at.utcoffset() == timedelta(0)

    def test_last_accessed_at_in_dataclass_fields(self) -> None:
        assert "last_accessed_at" in {f.name for f in dataclasses.fields(MemoryRecord)}


# ---------------------------------------------------------------------------
# 4. Bi-temporal properties: event_time, ingested_at, content
# ---------------------------------------------------------------------------

class TestBiTemporalProperties:
    """event_time, ingested_at, and content are accessible as properties
    delegating into the composed BiTemporalRecord."""

    def test_event_time_property_exists(self) -> None:
        assert hasattr(MemoryRecord, "event_time")

    def test_ingested_at_property_exists(self) -> None:
        assert hasattr(MemoryRecord, "ingested_at")

    def test_content_property_exists(self) -> None:
        assert hasattr(MemoryRecord, "content")

    def test_event_time_matches_inner_record(self) -> None:
        rec = _make()
        assert rec.event_time == rec.record.event_time

    def test_ingested_at_matches_inner_record(self) -> None:
        rec = _make()
        assert rec.ingested_at == rec.record.ingested_at

    def test_content_matches_inner_record(self) -> None:
        rec = _make(content="hello property")
        assert rec.content == rec.record.content

    def test_event_time_is_a_property(self) -> None:
        assert isinstance(MemoryRecord.__dict__["event_time"], property)

    def test_ingested_at_is_a_property(self) -> None:
        assert isinstance(MemoryRecord.__dict__["ingested_at"], property)

    def test_content_is_a_property(self) -> None:
        assert isinstance(MemoryRecord.__dict__["content"], property)

    def test_event_time_is_datetime(self) -> None:
        assert isinstance(_make().event_time, datetime)

    def test_ingested_at_is_datetime(self) -> None:
        assert isinstance(_make().ingested_at, datetime)

    def test_content_is_str(self) -> None:
        assert isinstance(_make(content="payload").content, str)


# ---------------------------------------------------------------------------
# 5. Discriminated-union pattern: memory_type as the type discriminator
# ---------------------------------------------------------------------------

class TestDiscriminatedUnionPattern:
    """memory_type is the discriminator field.  MemoryType is a str-subtype
    enum whose values are unique and stable."""

    def test_memory_type_field_exists(self) -> None:
        assert hasattr(_make(), "memory_type")

    def test_memory_type_is_memory_type_enum(self) -> None:
        assert isinstance(_make().memory_type, MemoryType)

    def test_memory_type_is_str_subtype(self) -> None:
        """MemoryType inherits from str — safe to use as a dict key or JSON value."""
        assert isinstance(_make().memory_type, str)

    def test_memory_type_in_dataclass_fields(self) -> None:
        assert "memory_type" in {f.name for f in dataclasses.fields(MemoryRecord)}

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_all_variants_constructible(self, mt: MemoryType) -> None:
        rec = _make(memory_type=mt)
        assert rec.memory_type is mt

    def test_all_variant_values_are_unique(self) -> None:
        values = [mt.value for mt in MemoryType]
        assert len(values) == len(set(values)), "Discriminator values must be unique"

    def test_all_variant_names_are_unique(self) -> None:
        names = [mt.name for mt in MemoryType]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("mt,expected", [
        (MemoryType.WORKING,    "working_memory"),
        (MemoryType.EPISODIC,   "episodic_memory"),
        (MemoryType.SEMANTIC,   "semantic_memory"),
        (MemoryType.PROCEDURAL, "procedural_memory"),
    ])
    def test_discriminator_value_strings_are_stable(
        self, mt: MemoryType, expected: str
    ) -> None:
        assert mt.value == expected

    def test_records_with_different_types_have_different_discriminators(self) -> None:
        r_episodic  = _make(memory_type=MemoryType.EPISODIC)
        r_semantic  = _make(memory_type=MemoryType.SEMANTIC)
        r_working   = _make(memory_type=MemoryType.WORKING)
        r_procedural = _make(memory_type=MemoryType.PROCEDURAL)

        types = {r_episodic.memory_type, r_semantic.memory_type,
                 r_working.memory_type, r_procedural.memory_type}
        assert len(types) == 4

    def test_str_of_memory_type_equals_value(self) -> None:
        for mt in MemoryType:
            assert str(mt) == mt.value

    def test_memory_type_can_be_used_as_dict_key(self) -> None:
        """Discriminator must be hashable (str subtype satisfies this)."""
        mapping = {mt: f"record-for-{mt.value}" for mt in MemoryType}
        assert len(mapping) == len(list(MemoryType))

    def test_memory_type_equality_is_identity(self) -> None:
        """Each variant is a singleton — identity and equality agree."""
        for mt in MemoryType:
            assert MemoryType(mt.value) is mt


# ---------------------------------------------------------------------------
# 6. MemoryRecord is a dataclass
# ---------------------------------------------------------------------------

class TestMemoryRecordIsDataclass:
    """Verify MemoryRecord is a proper dataclass with expected field order."""

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(MemoryRecord)

    def test_is_not_a_dataclass_instance_of_wrong_type(self) -> None:
        rec = _make()
        assert dataclasses.is_dataclass(rec)

    def test_required_fields_are_present(self) -> None:
        field_names = {f.name for f in dataclasses.fields(MemoryRecord)}
        for required in ("record_id", "memory_type", "record", "created_at", "updated_at"):
            assert required in field_names, f"Missing required field: {required!r}"

    def test_optional_fields_are_present(self) -> None:
        field_names = {f.name for f in dataclasses.fields(MemoryRecord)}
        for optional in ("last_accessed_at", "metadata"):
            assert optional in field_names, f"Missing optional field: {optional!r}"

    def test_record_field_is_bitemporal_record(self) -> None:
        rec = _make()
        assert isinstance(rec.record, BiTemporalRecord)

    def test_metadata_defaults_to_none(self) -> None:
        rec = _make()
        assert rec.metadata is None

    def test_metadata_accepts_memory_metadata(self) -> None:
        rec = _make()
        rec.metadata = MemoryMetadata(source="https://example.com")
        assert isinstance(rec.metadata, MemoryMetadata)
