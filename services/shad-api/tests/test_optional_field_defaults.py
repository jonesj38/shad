"""Tests for optional field defaults on MemoryMetadata, SnapshotEntry, and MemoryRecord.

Verifies two behavioural contracts:
  1. Optional fields are set to their documented defaults when omitted.
  2. When a value is supplied it is stored with the correct type and value.

Fields under test
-----------------
MemoryMetadata:
    source (required str), confidence (float, default 1.0),
    tags (list[str], default []), promoted_at / promoted_from /
    consolidated_at / consolidated_from (str | None, default None).

SnapshotEntry:
    metadata (dict, default {}), memory_type (MemoryType | None, default None),
    promotion_history (list, default []).

MemoryRecord:
    last_accessed_at (datetime | None, default None),
    metadata (MemoryMetadata | None, default None).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
    PromotionHistoryEntry,
    SnapshotEntry,
    make_bitemporal_record,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EVENT_TIME = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)
_INGESTED_AT = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)


def _minimal_snapshot(
    snapshot_id: str = "snap-001",
    source_id: str = "src-001",
) -> SnapshotEntry:
    """Return a SnapshotEntry with only the required fields."""
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id=source_id,
        ingested_at=_INGESTED_AT,
        source_revision="abc123",
        entry_paths=["a/b.md"],
        content_hash="deadbeef",
    )


def _minimal_memory_record(record_id: str = "rec-001") -> MemoryRecord:
    """Return a MemoryRecord with only required fields."""
    btr: BiTemporalRecord[str] = make_bitemporal_record(_EVENT_TIME, "content")
    return MemoryRecord(
        record_id=record_id,
        memory_type=MemoryType.WORKING,
        record=btr,
    )


# ---------------------------------------------------------------------------
# MemoryMetadata — absence defaults
# ---------------------------------------------------------------------------


class TestMemoryMetadataAbsenceDefaults:
    """Optional MemoryMetadata fields must carry documented defaults when absent."""

    def test_confidence_defaults_to_one(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert meta.confidence == 1.0

    def test_confidence_is_float(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert isinstance(meta.confidence, float)

    def test_tags_defaults_to_empty_list(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert meta.tags == []

    def test_tags_is_list(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert isinstance(meta.tags, list)

    def test_promoted_at_defaults_to_none(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert meta.promoted_at is None

    def test_promoted_from_defaults_to_none(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert meta.promoted_from is None

    def test_consolidated_at_defaults_to_none(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert meta.consolidated_at is None

    def test_consolidated_from_defaults_to_none(self) -> None:
        meta = MemoryMetadata(source="https://example.com")
        assert meta.consolidated_from is None

    def test_all_optional_fields_absent_simultaneously(self) -> None:
        meta = MemoryMetadata(source="x")
        assert meta.confidence == 1.0
        assert meta.tags == []
        assert meta.promoted_at is None
        assert meta.promoted_from is None
        assert meta.consolidated_at is None
        assert meta.consolidated_from is None


# ---------------------------------------------------------------------------
# MemoryMetadata — tags default independence (no shared mutable state)
# ---------------------------------------------------------------------------


class TestMemoryMetadataTagsIsolation:
    """Each instance must get its own tags list — not a shared mutable default."""

    def test_two_instances_have_independent_tags(self) -> None:
        a = MemoryMetadata(source="a")
        b = MemoryMetadata(source="b")
        a.tags.append("x")
        assert b.tags == [], "Mutating one instance must not affect another"

    def test_tags_list_is_not_class_level_singleton(self) -> None:
        a = MemoryMetadata(source="a")
        b = MemoryMetadata(source="b")
        assert a.tags is not b.tags


# ---------------------------------------------------------------------------
# MemoryMetadata — presence: correct types and values
# ---------------------------------------------------------------------------


class TestMemoryMetadataPresenceTyping:
    """When optional fields are provided they must be stored with the correct type."""

    def test_source_is_str(self) -> None:
        meta = MemoryMetadata(source="https://github.com/org/repo")
        assert isinstance(meta.source, str)
        assert meta.source == "https://github.com/org/repo"

    def test_confidence_stored_as_float(self) -> None:
        meta = MemoryMetadata(source="x", confidence=0.75)
        assert isinstance(meta.confidence, float)
        assert meta.confidence == 0.75

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_confidence_boundary_values(self, value: float) -> None:
        meta = MemoryMetadata(source="x", confidence=value)
        assert meta.confidence == value

    def test_tags_stored_as_list_of_str(self) -> None:
        meta = MemoryMetadata(source="x", tags=["semantic_memory", "github"])
        assert isinstance(meta.tags, list)
        assert all(isinstance(t, str) for t in meta.tags)
        assert meta.tags == ["semantic_memory", "github"]

    def test_promoted_at_stored_as_str(self) -> None:
        ts = "2026-01-15T09:00:00+00:00"
        meta = MemoryMetadata(source="x", promoted_at=ts)
        assert isinstance(meta.promoted_at, str)
        assert meta.promoted_at == ts

    def test_promoted_from_stored_as_str(self) -> None:
        meta = MemoryMetadata(source="x", promoted_from="working_memory")
        assert isinstance(meta.promoted_from, str)
        assert meta.promoted_from == "working_memory"

    def test_consolidated_at_stored_as_str(self) -> None:
        ts = "2026-02-01T00:00:00+00:00"
        meta = MemoryMetadata(source="x", consolidated_at=ts)
        assert isinstance(meta.consolidated_at, str)
        assert meta.consolidated_at == ts

    def test_consolidated_from_stored_as_str(self) -> None:
        meta = MemoryMetadata(source="x", consolidated_from="episodic_memory")
        assert isinstance(meta.consolidated_from, str)
        assert meta.consolidated_from == "episodic_memory"

    def test_all_fields_provided_simultaneously(self) -> None:
        meta = MemoryMetadata(
            source="https://example.com",
            confidence=0.9,
            tags=["retrieval"],
            promoted_at="2026-01-01T00:00:00+00:00",
            promoted_from="working_memory",
            consolidated_at="2026-02-01T00:00:00+00:00",
            consolidated_from="episodic_memory",
        )
        assert meta.source == "https://example.com"
        assert meta.confidence == 0.9
        assert meta.tags == ["retrieval"]
        assert meta.promoted_at == "2026-01-01T00:00:00+00:00"
        assert meta.promoted_from == "working_memory"
        assert meta.consolidated_at == "2026-02-01T00:00:00+00:00"
        assert meta.consolidated_from == "episodic_memory"


# ---------------------------------------------------------------------------
# SnapshotEntry — optional field absence defaults
# ---------------------------------------------------------------------------


class TestSnapshotEntryAbsenceDefaults:
    """Optional SnapshotEntry fields must carry documented defaults when absent."""

    def test_metadata_defaults_to_empty_dict(self) -> None:
        snap = _minimal_snapshot()
        assert snap.metadata == {}

    def test_metadata_is_dict(self) -> None:
        snap = _minimal_snapshot()
        assert isinstance(snap.metadata, dict)

    def test_memory_type_defaults_to_none(self) -> None:
        snap = _minimal_snapshot()
        assert snap.memory_type is None

    def test_promotion_history_defaults_to_empty_list(self) -> None:
        snap = _minimal_snapshot()
        assert snap.promotion_history == []

    def test_promotion_history_is_list(self) -> None:
        snap = _minimal_snapshot()
        assert isinstance(snap.promotion_history, list)

    def test_metadata_instances_are_independent(self) -> None:
        a = _minimal_snapshot("a")
        b = _minimal_snapshot("b")
        a.metadata["key"] = "value"
        assert b.metadata == {}, "Mutable dict default must not be shared"

    def test_promotion_history_instances_are_independent(self) -> None:
        a = _minimal_snapshot("a")
        b = _minimal_snapshot("b")
        a.promotion_history.append(
            PromotionHistoryEntry(
                from_type=None,
                to_type=MemoryType.EPISODIC,
                timestamp=_INGESTED_AT,
                reason="test",
            )
        )
        assert b.promotion_history == []


# ---------------------------------------------------------------------------
# SnapshotEntry — optional field presence typing
# ---------------------------------------------------------------------------


class TestSnapshotEntryPresenceTyping:
    """When optional SnapshotEntry fields are provided they must be properly typed."""

    def test_metadata_stored_as_dict(self) -> None:
        snap = SnapshotEntry(
            snapshot_id="s1",
            source_id="src",
            ingested_at=_INGESTED_AT,
            source_revision="rev",
            entry_paths=["p.md"],
            content_hash="hash",
            metadata={"branch": "main", "score": 0.9},
        )
        assert isinstance(snap.metadata, dict)
        assert snap.metadata == {"branch": "main", "score": 0.9}

    def test_memory_type_stored_as_memory_type_enum(self) -> None:
        snap = SnapshotEntry(
            snapshot_id="s1",
            source_id="src",
            ingested_at=_INGESTED_AT,
            source_revision="rev",
            entry_paths=[],
            content_hash="hash",
            memory_type=MemoryType.SEMANTIC,
        )
        assert isinstance(snap.memory_type, MemoryType)
        assert snap.memory_type == MemoryType.SEMANTIC

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_all_memory_type_variants_accepted(self, mt: MemoryType) -> None:
        snap = SnapshotEntry(
            snapshot_id="s1",
            source_id="src",
            ingested_at=_INGESTED_AT,
            source_revision="rev",
            entry_paths=[],
            content_hash="hash",
            memory_type=mt,
        )
        assert snap.memory_type is mt

    def test_promotion_history_stored_as_list(self) -> None:
        entry = PromotionHistoryEntry(
            from_type=MemoryType.WORKING,
            to_type=MemoryType.EPISODIC,
            timestamp=_INGESTED_AT,
            reason="persist event",
        )
        snap = SnapshotEntry(
            snapshot_id="s1",
            source_id="src",
            ingested_at=_INGESTED_AT,
            source_revision="rev",
            entry_paths=[],
            content_hash="hash",
            promotion_history=[entry],
        )
        assert isinstance(snap.promotion_history, list)
        assert len(snap.promotion_history) == 1
        assert snap.promotion_history[0] is entry


# ---------------------------------------------------------------------------
# MemoryRecord — optional field absence defaults
# ---------------------------------------------------------------------------


class TestMemoryRecordAbsenceDefaults:
    """Optional MemoryRecord fields must carry documented defaults when absent."""

    def test_last_accessed_at_defaults_to_none(self) -> None:
        rec = _minimal_memory_record()
        assert rec.last_accessed_at is None

    def test_metadata_defaults_to_none(self) -> None:
        rec = _minimal_memory_record()
        assert rec.metadata is None

    def test_both_optionals_absent_simultaneously(self) -> None:
        rec = _minimal_memory_record()
        assert rec.last_accessed_at is None
        assert rec.metadata is None


# ---------------------------------------------------------------------------
# MemoryRecord — optional field presence typing
# ---------------------------------------------------------------------------


class TestMemoryRecordPresenceTyping:
    """When optional MemoryRecord fields are provided they must be properly typed."""

    def test_last_accessed_at_stored_as_datetime(self) -> None:
        ts = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        btr: BiTemporalRecord[str] = make_bitemporal_record(_EVENT_TIME, "content")
        rec = MemoryRecord(
            record_id="r1",
            memory_type=MemoryType.SEMANTIC,
            record=btr,
            last_accessed_at=ts,
        )
        assert isinstance(rec.last_accessed_at, datetime)
        assert rec.last_accessed_at == ts

    def test_metadata_stored_as_memory_metadata(self) -> None:
        meta = MemoryMetadata(source="https://example.com", confidence=0.8)
        btr: BiTemporalRecord[str] = make_bitemporal_record(_EVENT_TIME, "content")
        rec = MemoryRecord(
            record_id="r1",
            memory_type=MemoryType.SEMANTIC,
            record=btr,
            metadata=meta,
        )
        assert isinstance(rec.metadata, MemoryMetadata)
        assert rec.metadata is meta

    def test_metadata_fields_accessible_through_record(self) -> None:
        meta = MemoryMetadata(
            source="https://github.com/org/repo",
            confidence=0.95,
            tags=["semantic_memory"],
        )
        btr: BiTemporalRecord[str] = make_bitemporal_record(_EVENT_TIME, "content")
        rec = MemoryRecord(
            record_id="r1",
            memory_type=MemoryType.SEMANTIC,
            record=btr,
            metadata=meta,
        )
        assert rec.metadata is not None
        assert rec.metadata.source == "https://github.com/org/repo"
        assert rec.metadata.confidence == 0.95
        assert rec.metadata.tags == ["semantic_memory"]

    def test_both_optional_fields_present_simultaneously(self) -> None:
        ts = datetime(2026, 3, 15, 8, 0, 0, tzinfo=UTC)
        meta = MemoryMetadata(source="x")
        btr: BiTemporalRecord[str] = make_bitemporal_record(_EVENT_TIME, "content")
        rec = MemoryRecord(
            record_id="r1",
            memory_type=MemoryType.WORKING,
            record=btr,
            last_accessed_at=ts,
            metadata=meta,
        )
        assert rec.last_accessed_at == ts
        assert rec.metadata is meta
