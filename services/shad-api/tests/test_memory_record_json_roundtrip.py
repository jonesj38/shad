"""Round-trip JSON serialization tests for MemoryRecord.

Contract: every field in a MemoryRecord must survive the cycle
    record → memory_record_to_json() → memory_record_from_json() → record

Specifically verified:
  - All scalar string/float fields
  - datetime objects (event_time, ingested_at, created_at, updated_at,
    last_accessed_at) — including timezone and microsecond precision
  - MemoryType enum (stored as value string, reconstructed as enum)
  - Nested MemoryMetadata with all sub-fields
  - Optional fields absent (None) and present
  - All four MemoryType variants
  - The raw JSON text is valid JSON and contains the expected keys
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
    memory_record_from_json,
    memory_record_to_json,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EVENT_TIME = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
_INGEST_TIME = datetime(2025, 6, 1, 12, 0, 5, tzinfo=UTC)
_CREATED_AT = datetime(2025, 6, 1, 12, 0, 6, tzinfo=UTC)
_UPDATED_AT = datetime(2025, 6, 2, 8, 30, 0, tzinfo=UTC)
_ACCESSED_AT = datetime(2025, 6, 3, 14, 45, 0, 123456, tzinfo=UTC)


def _minimal_record(
    *,
    record_id: str = "rec-001",
    memory_type: MemoryType = MemoryType.EPISODIC,
    content: str = "Hello, world.",
) -> MemoryRecord:
    """Return a MemoryRecord with only required fields set (optionals are None)."""
    return MemoryRecord(
        record_id=record_id,
        memory_type=memory_type,
        record=BiTemporalRecord(
            event_time=_EVENT_TIME,
            ingested_at=_INGEST_TIME,
            content=content,
        ),
        created_at=_CREATED_AT,
        updated_at=_UPDATED_AT,
    )


def _full_record() -> MemoryRecord:
    """Return a MemoryRecord with every optional field populated."""
    return MemoryRecord(
        record_id="rec-full",
        memory_type=MemoryType.SEMANTIC,
        record=BiTemporalRecord(
            event_time=_EVENT_TIME,
            ingested_at=_INGEST_TIME,
            content="Full record payload.",
        ),
        created_at=_CREATED_AT,
        updated_at=_UPDATED_AT,
        last_accessed_at=_ACCESSED_AT,
        metadata=MemoryMetadata(
            source="github.com/example/repo",
            confidence=0.87,
            tags=["semantic_memory", "github", "retrieval"],
            promoted_at="2025-06-02T10:00:00+00:00",
            promoted_from="working_memory",
            consolidated_at="2025-06-03T08:00:00+00:00",
            consolidated_from="episodic_memory",
        ),
    )


def roundtrip(record: MemoryRecord) -> MemoryRecord:
    return memory_record_from_json(memory_record_to_json(record))


# ---------------------------------------------------------------------------
# Scalar identity fields
# ---------------------------------------------------------------------------

class TestRecordIdSurvives:
    def test_record_id_preserved(self) -> None:
        rec = _minimal_record(record_id="unique-id-42")
        assert roundtrip(rec).record_id == "unique-id-42"

    def test_record_id_empty_string(self) -> None:
        rec = _minimal_record(record_id="")
        assert roundtrip(rec).record_id == ""

    def test_record_id_unicode(self) -> None:
        rec = _minimal_record(record_id="ид-записи-001")
        assert roundtrip(rec).record_id == "ид-записи-001"


# ---------------------------------------------------------------------------
# MemoryType round-trip
# ---------------------------------------------------------------------------

class TestMemoryTypeSurvives:
    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_all_variants_survive(self, mt: MemoryType) -> None:
        rec = _minimal_record(memory_type=mt)
        restored = roundtrip(rec)
        assert restored.memory_type is mt

    def test_memory_type_is_enum_not_string(self) -> None:
        rec = _minimal_record(memory_type=MemoryType.WORKING)
        restored = roundtrip(rec)
        assert isinstance(restored.memory_type, MemoryType)

    def test_memory_type_stored_as_value_in_json(self) -> None:
        rec = _minimal_record(memory_type=MemoryType.PROCEDURAL)
        payload = json.loads(memory_record_to_json(rec))
        assert payload["memory_type"] == MemoryType.PROCEDURAL.value


# ---------------------------------------------------------------------------
# BiTemporalRecord inner fields
# ---------------------------------------------------------------------------

class TestBiTemporalRecordSurvives:
    def test_event_time_preserved(self) -> None:
        rec = _minimal_record()
        assert roundtrip(rec).record.event_time == _EVENT_TIME

    def test_ingested_at_preserved(self) -> None:
        rec = _minimal_record()
        assert roundtrip(rec).record.ingested_at == _INGEST_TIME

    def test_content_preserved(self) -> None:
        rec = _minimal_record(content="payload with\nnewlines\tand tabs")
        assert roundtrip(rec).record.content == "payload with\nnewlines\tand tabs"

    def test_empty_content_preserved(self) -> None:
        rec = _minimal_record(content="")
        assert roundtrip(rec).record.content == ""

    def test_event_time_with_microseconds(self) -> None:
        t = datetime(2026, 1, 1, 0, 0, 0, 999999, tzinfo=UTC)
        rec = MemoryRecord(
            record_id="us-test",
            memory_type=MemoryType.EPISODIC,
            record=BiTemporalRecord(event_time=t, ingested_at=t, content="x"),
            created_at=t,
            updated_at=t,
        )
        assert roundtrip(rec).record.event_time == t

    def test_event_time_timezone_preserved(self) -> None:
        rec = _minimal_record()
        restored = roundtrip(rec)
        assert restored.record.event_time.utcoffset() == timedelta(0)
        assert restored.record.ingested_at.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Top-level datetime fields
# ---------------------------------------------------------------------------

class TestTopLevelDatetimesSurvive:
    def test_created_at_preserved(self) -> None:
        assert roundtrip(_minimal_record()).created_at == _CREATED_AT

    def test_updated_at_preserved(self) -> None:
        assert roundtrip(_minimal_record()).updated_at == _UPDATED_AT

    def test_created_at_timezone_is_utc(self) -> None:
        assert roundtrip(_minimal_record()).created_at.utcoffset() == timedelta(0)

    def test_updated_at_timezone_is_utc(self) -> None:
        assert roundtrip(_minimal_record()).updated_at.utcoffset() == timedelta(0)

    def test_created_at_microseconds_preserved(self) -> None:
        t = datetime(2025, 11, 30, 23, 59, 59, 500001, tzinfo=UTC)
        rec = MemoryRecord(
            record_id="us",
            memory_type=MemoryType.SEMANTIC,
            record=BiTemporalRecord(event_time=t, ingested_at=t, content=""),
            created_at=t,
            updated_at=t,
        )
        assert roundtrip(rec).created_at == t


# ---------------------------------------------------------------------------
# Optional field: last_accessed_at
# ---------------------------------------------------------------------------

class TestLastAccessedAt:
    def test_none_survives(self) -> None:
        rec = _minimal_record()
        assert rec.last_accessed_at is None
        assert roundtrip(rec).last_accessed_at is None

    def test_datetime_value_survives(self) -> None:
        rec = _minimal_record()
        rec.last_accessed_at = _ACCESSED_AT
        assert roundtrip(rec).last_accessed_at == _ACCESSED_AT

    def test_microseconds_in_last_accessed_at(self) -> None:
        t = datetime(2026, 3, 31, 15, 0, 0, 123456, tzinfo=UTC)
        rec = _minimal_record()
        rec.last_accessed_at = t
        assert roundtrip(rec).last_accessed_at == t

    def test_last_accessed_at_timezone_preserved(self) -> None:
        rec = _minimal_record()
        rec.last_accessed_at = _ACCESSED_AT
        restored = roundtrip(rec)
        assert restored.last_accessed_at is not None
        assert restored.last_accessed_at.utcoffset() == timedelta(0)

    def test_stored_as_null_in_json_when_none(self) -> None:
        payload = json.loads(memory_record_to_json(_minimal_record()))
        assert payload["last_accessed_at"] is None

    def test_stored_as_iso_string_in_json_when_set(self) -> None:
        rec = _minimal_record()
        rec.last_accessed_at = _ACCESSED_AT
        payload = json.loads(memory_record_to_json(rec))
        assert isinstance(payload["last_accessed_at"], str)
        assert datetime.fromisoformat(payload["last_accessed_at"]) == _ACCESSED_AT


# ---------------------------------------------------------------------------
# Optional field: metadata (None case)
# ---------------------------------------------------------------------------

class TestMetadataAbsent:
    def test_none_metadata_survives(self) -> None:
        rec = _minimal_record()
        assert rec.metadata is None
        assert roundtrip(rec).metadata is None

    def test_stored_as_null_in_json(self) -> None:
        payload = json.loads(memory_record_to_json(_minimal_record()))
        assert payload["metadata"] is None


# ---------------------------------------------------------------------------
# Optional field: metadata (present case)
# ---------------------------------------------------------------------------

class TestMetadataPresent:
    def _rec_with_meta(self, **overrides: object) -> MemoryRecord:
        base = {
            "source": "https://example.com",
            "confidence": 0.9,
            "tags": ["github", "semantic_memory"],
            "promoted_at": None,
            "promoted_from": None,
            "consolidated_at": None,
            "consolidated_from": None,
        }
        base.update(overrides)
        rec = _minimal_record()
        rec.metadata = MemoryMetadata(**base)  # type: ignore[arg-type]
        return rec

    def test_source_survives(self) -> None:
        rec = self._rec_with_meta(source="https://github.com/owner/repo")
        assert roundtrip(rec).metadata is not None
        assert roundtrip(rec).metadata.source == "https://github.com/owner/repo"

    def test_confidence_survives(self) -> None:
        rec = self._rec_with_meta(confidence=0.42)
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert restored.metadata.confidence == pytest.approx(0.42)

    def test_tags_list_survives(self) -> None:
        tags = ["semantic_memory", "github", "retrieval"]
        rec = self._rec_with_meta(tags=tags)
        assert roundtrip(rec).metadata.tags == tags  # type: ignore[union-attr]

    def test_empty_tags_list_survives(self) -> None:
        rec = self._rec_with_meta(tags=[])
        assert roundtrip(rec).metadata.tags == []  # type: ignore[union-attr]

    def test_promoted_at_none_survives(self) -> None:
        rec = self._rec_with_meta(promoted_at=None)
        assert roundtrip(rec).metadata.promoted_at is None  # type: ignore[union-attr]

    def test_promoted_at_string_survives(self) -> None:
        ts = "2025-06-02T10:00:00+00:00"
        rec = self._rec_with_meta(promoted_at=ts)
        assert roundtrip(rec).metadata.promoted_at == ts  # type: ignore[union-attr]

    def test_promoted_from_survives(self) -> None:
        rec = self._rec_with_meta(promoted_from="working_memory")
        assert roundtrip(rec).metadata.promoted_from == "working_memory"  # type: ignore[union-attr]

    def test_consolidated_at_none_survives(self) -> None:
        rec = self._rec_with_meta(consolidated_at=None)
        assert roundtrip(rec).metadata.consolidated_at is None  # type: ignore[union-attr]

    def test_consolidated_at_string_survives(self) -> None:
        ts = "2025-06-03T08:00:00+00:00"
        rec = self._rec_with_meta(consolidated_at=ts)
        assert roundtrip(rec).metadata.consolidated_at == ts  # type: ignore[union-attr]

    def test_consolidated_from_survives(self) -> None:
        rec = self._rec_with_meta(consolidated_from="episodic_memory")
        assert roundtrip(rec).metadata.consolidated_from == "episodic_memory"  # type: ignore[union-attr]

    def test_metadata_type_is_memory_metadata(self) -> None:
        rec = self._rec_with_meta()
        restored = roundtrip(rec)
        assert isinstance(restored.metadata, MemoryMetadata)

    def test_full_metadata_all_fields_populated(self) -> None:
        original = _full_record()
        restored = roundtrip(original)
        m = restored.metadata
        assert m is not None
        assert m.source == "github.com/example/repo"
        assert m.confidence == pytest.approx(0.87)
        assert m.tags == ["semantic_memory", "github", "retrieval"]
        assert m.promoted_at == "2025-06-02T10:00:00+00:00"
        assert m.promoted_from == "working_memory"
        assert m.consolidated_at == "2025-06-03T08:00:00+00:00"
        assert m.consolidated_from == "episodic_memory"


# ---------------------------------------------------------------------------
# Full record round-trip
# ---------------------------------------------------------------------------

class TestFullRecordRoundTrip:
    def test_full_record_all_fields_survive(self) -> None:
        original = _full_record()
        restored = roundtrip(original)

        assert restored.record_id == original.record_id
        assert restored.memory_type is original.memory_type
        assert restored.record.event_time == original.record.event_time
        assert restored.record.ingested_at == original.record.ingested_at
        assert restored.record.content == original.record.content
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at
        assert restored.last_accessed_at == original.last_accessed_at

    def test_two_independent_roundtrips_produce_equal_records(self) -> None:
        original = _full_record()
        r1 = roundtrip(original)
        r2 = roundtrip(original)
        # Fields must match; identity is irrelevant
        assert r1.record_id == r2.record_id
        assert r1.memory_type is r2.memory_type
        assert r1.record.event_time == r2.record.event_time
        assert r1.last_accessed_at == r2.last_accessed_at

    def test_double_roundtrip_is_stable(self) -> None:
        """Encoding the already-restored record again must produce the same JSON."""
        original = _full_record()
        once = memory_record_to_json(original)
        twice = memory_record_to_json(roundtrip(original))
        assert once == twice


# ---------------------------------------------------------------------------
# JSON structure assertions
# ---------------------------------------------------------------------------

class TestJsonStructure:
    def test_output_is_valid_json(self) -> None:
        raw = memory_record_to_json(_minimal_record())
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_required_top_level_keys_present(self) -> None:
        payload = json.loads(memory_record_to_json(_minimal_record()))
        for key in ("record_id", "memory_type", "record", "created_at", "updated_at",
                    "last_accessed_at", "metadata"):
            assert key in payload, f"Missing key: {key!r}"

    def test_record_subkeys_present(self) -> None:
        payload = json.loads(memory_record_to_json(_minimal_record()))
        for key in ("event_time", "ingested_at", "content"):
            assert key in payload["record"], f"Missing record sub-key: {key!r}"

    def test_memory_type_is_string_in_json(self) -> None:
        payload = json.loads(memory_record_to_json(_minimal_record()))
        assert isinstance(payload["memory_type"], str)

    def test_datetime_fields_are_strings_in_json(self) -> None:
        payload = json.loads(memory_record_to_json(_minimal_record()))
        for field in ("created_at", "updated_at"):
            assert isinstance(payload[field], str), f"{field} should be str in JSON"
        for field in ("event_time", "ingested_at"):
            assert isinstance(payload["record"][field], str)

    def test_metadata_keys_present_when_set(self) -> None:
        rec = _full_record()
        payload = json.loads(memory_record_to_json(rec))
        meta = payload["metadata"]
        for key in ("source", "confidence", "tags", "promoted_at", "promoted_from",
                    "consolidated_at", "consolidated_from"):
            assert key in meta, f"Missing metadata key: {key!r}"
