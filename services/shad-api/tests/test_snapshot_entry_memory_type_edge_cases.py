"""Edge-case tests for SnapshotEntry memory_type handling.

Three specific edge cases:

1. SnapshotEntry with memory_type=None resolves to the correct MemoryType via
   the source_type_map fallback inside group_snapshots().

2. promote_to_episodic() on a SEMANTIC snapshot raises ValueError (wrong state).

3. Round-trip serialization via memory_record_to_dict / memory_record_from_dict
   preserves memory_type for every MemoryType variant.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryRecord,
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
    group_snapshots,
    memory_record_from_dict,
    memory_record_to_dict,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_SOURCE_ID = "src-edge-001"


def _make_record(memory_type: MemoryType = MemoryType.SEMANTIC) -> MemoryRecord:
    return MemoryRecord(
        record_id="rec-" + str(uuid.uuid4()),
        memory_type=memory_type,
        record=BiTemporalRecord(event_time=_T0, ingested_at=_T0, content="test content"),
        created_at=_T0,
        updated_at=_T0,
    )


def _make_snapshot(
    *,
    memory_type: MemoryType | None,
    snapshot_id: str | None = None,
    source_id: str = _SOURCE_ID,
) -> SnapshotEntry:
    return SnapshotEntry(
        snapshot_id=snapshot_id or str(uuid.uuid4()),
        source_id=source_id,
        ingested_at=_T0,
        source_revision="rev-1",
        entry_paths=["docs/overview.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
    )


def _make_index(tmp_path: Path, *, source_type: str = "github") -> ShadowIndex:
    index = ShadowIndex(tmp_path / "test.sqlite")
    index.add_source(
        SourceEntry(
            source_url=f"https://example.com/{source_type}",
            source_id=_SOURCE_ID,
            source_type=source_type,
            update_policy=UpdatePolicy.MANUAL,
        )
    )
    return index


def _make_index_with_snapshot(
    tmp_path: Path,
    memory_type: MemoryType,
) -> tuple[ShadowIndex, SnapshotEntry]:
    index = _make_index(tmp_path)
    snap = _make_snapshot(memory_type=memory_type)
    index.add_snapshot(snap)
    return index, snap


# ---------------------------------------------------------------------------
# 1. None memory_type → source type fallback in group_snapshots()
# ---------------------------------------------------------------------------


class TestNoneMemoryTypeFallbackViaSourceTypeMap:
    """group_snapshots() resolves memory_type=None via source_type_map."""

    def test_none_with_github_source_resolves_to_semantic(self) -> None:
        snap = _make_snapshot(memory_type=None)
        groups = group_snapshots([snap], source_type_map={_SOURCE_ID: "github"})
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.SEMANTIC

    def test_none_with_feed_source_resolves_to_episodic(self) -> None:
        snap = _make_snapshot(memory_type=None)
        groups = group_snapshots([snap], source_type_map={_SOURCE_ID: "feed"})
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.EPISODIC

    def test_none_with_sessions_source_resolves_to_episodic(self) -> None:
        snap = _make_snapshot(memory_type=None)
        groups = group_snapshots([snap], source_type_map={_SOURCE_ID: "sessions"})
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.EPISODIC

    def test_none_with_url_source_resolves_to_semantic(self) -> None:
        snap = _make_snapshot(memory_type=None)
        groups = group_snapshots([snap], source_type_map={_SOURCE_ID: "url"})
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.SEMANTIC

    def test_none_without_source_type_map_defaults_to_semantic(self) -> None:
        """When no source_type_map is provided, None → SEMANTIC."""
        snap = _make_snapshot(memory_type=None)
        groups = group_snapshots([snap])
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.SEMANTIC

    def test_none_source_id_absent_from_map_defaults_to_semantic(self) -> None:
        """source_id not present in source_type_map → SEMANTIC fallback."""
        snap = _make_snapshot(memory_type=None)
        groups = group_snapshots([snap], source_type_map={"other-src": "feed"})
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.SEMANTIC

    def test_explicit_memory_type_not_overridden_by_source_type_map(self) -> None:
        """An already-typed snapshot must ignore source_type_map entirely."""
        snap = _make_snapshot(memory_type=MemoryType.EPISODIC)
        # source_type_map would resolve to SEMANTIC for "github"
        groups = group_snapshots([snap], source_type_map={_SOURCE_ID: "github"})
        assert len(groups) == 1
        assert groups[0].memory_type is MemoryType.EPISODIC

    def test_mixed_none_and_typed_snapshots_each_resolved_independently(self) -> None:
        """Two snapshots with the same source: one None, one explicit type."""
        src_id = "src-mixed"
        untyped = SnapshotEntry(
            snapshot_id="snap-untyped",
            source_id=src_id,
            ingested_at=_T0,
            source_revision="rev-1",
            entry_paths=["a/note.md"],
            content_hash="aaa",
            memory_type=None,
        )
        typed = SnapshotEntry(
            snapshot_id="snap-typed",
            source_id=src_id,
            ingested_at=_T0,
            source_revision="rev-2",
            entry_paths=["a/other.md"],
            content_hash="bbb",
            memory_type=MemoryType.EPISODIC,
        )
        groups = group_snapshots(
            [untyped, typed],
            source_type_map={src_id: "github"},  # would give SEMANTIC for None
        )
        types_in_groups = {g.memory_type for g in groups}
        # untyped → SEMANTIC (via map), typed stays EPISODIC
        assert MemoryType.SEMANTIC in types_in_groups
        assert MemoryType.EPISODIC in types_in_groups


# ---------------------------------------------------------------------------
# 2. promote_to_episodic() on wrong state raises ValueError
# ---------------------------------------------------------------------------


class TestPromoteToEpisodicOnSemanticSnapshot:
    """promote_to_episodic() requires WORKING; SEMANTIC snapshot → ValueError."""

    def test_raises_value_error(self, tmp_path: Path) -> None:
        index, snap = _make_index_with_snapshot(tmp_path, MemoryType.SEMANTIC)
        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_cites_required_working_state(self, tmp_path: Path) -> None:
        index, snap = _make_index_with_snapshot(tmp_path, MemoryType.SEMANTIC)
        with pytest.raises(ValueError, match=MemoryType.WORKING.value):
            index.promote_to_episodic(snap.snapshot_id)

    def test_error_message_contains_snapshot_id(self, tmp_path: Path) -> None:
        index, snap = _make_index_with_snapshot(tmp_path, MemoryType.SEMANTIC)
        with pytest.raises(ValueError, match=snap.snapshot_id):
            index.promote_to_episodic(snap.snapshot_id)

    def test_snapshot_type_unchanged_after_rejection(self, tmp_path: Path) -> None:
        index, snap = _make_index_with_snapshot(tmp_path, MemoryType.SEMANTIC)
        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)
        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type is MemoryType.SEMANTIC

    def test_no_promotion_provenance_written_on_rejection(self, tmp_path: Path) -> None:
        """promoted_at / promoted_from must not appear in metadata after rejection."""
        index, snap = _make_index_with_snapshot(tmp_path, MemoryType.SEMANTIC)
        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)
        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert "promoted_at" not in fetched.metadata
        assert "promoted_from" not in fetched.metadata

    def test_promotion_history_unchanged_on_rejection(self, tmp_path: Path) -> None:
        """promotion_history must remain empty after a rejected call."""
        index, snap = _make_index_with_snapshot(tmp_path, MemoryType.SEMANTIC)
        with pytest.raises(ValueError):
            index.promote_to_episodic(snap.snapshot_id)
        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.promotion_history == []


# ---------------------------------------------------------------------------
# 3. memory_record_to_dict / memory_record_from_dict preserves memory_type
# ---------------------------------------------------------------------------


class TestMemoryRecordDictRoundTripPreservesMemoryType:
    """memory_record_to_dict/from_dict cycle preserves memory_type for every variant."""

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_all_variants_survive_dict_roundtrip(self, mt: MemoryType) -> None:
        restored = memory_record_from_dict(memory_record_to_dict(_make_record(mt)))
        assert restored.memory_type is mt

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_dict_stores_value_string_not_enum(self, mt: MemoryType) -> None:
        d = memory_record_to_dict(_make_record(mt))
        assert d["memory_type"] == mt.value
        assert isinstance(d["memory_type"], str)

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_restored_memory_type_is_enum_instance(self, mt: MemoryType) -> None:
        restored = memory_record_from_dict(memory_record_to_dict(_make_record(mt)))
        assert isinstance(restored.memory_type, MemoryType)

    def test_double_dict_roundtrip_is_stable(self) -> None:
        record = _make_record(MemoryType.EPISODIC)
        once = memory_record_from_dict(memory_record_to_dict(record))
        twice = memory_record_from_dict(memory_record_to_dict(once))
        assert twice.memory_type is MemoryType.EPISODIC

    def test_dict_roundtrip_does_not_mutate_other_fields(self) -> None:
        """memory_type round-trip must not disturb record_id or content."""
        record = _make_record(MemoryType.WORKING)
        restored = memory_record_from_dict(memory_record_to_dict(record))
        assert restored.record_id == record.record_id
        assert restored.record.content == record.record.content
        assert restored.created_at == record.created_at

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_memory_type_value_in_dict_matches_enum_value(self, mt: MemoryType) -> None:
        """The string stored in the dict must equal the canonical MemoryType value."""
        d = memory_record_to_dict(_make_record(mt))
        assert d["memory_type"] == mt.value

    def test_invalid_memory_type_string_in_dict_raises_value_error(self) -> None:
        """Feeding a bad memory_type string into from_dict raises ValueError."""
        d = memory_record_to_dict(_make_record(MemoryType.SEMANTIC))
        d["memory_type"] = "not_a_real_memory_type"
        with pytest.raises(ValueError):
            memory_record_from_dict(d)
