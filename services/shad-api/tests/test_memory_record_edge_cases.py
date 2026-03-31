"""Edge-case tests for MemoryRecord and related dataclasses.

Covers:
  - Empty strings for all str fields (record_id, source, promoted_at, etc.)
  - Empty arrays for tags
  - None vs. absent key in memory_record_from_dict for optional fields
  - Special characters in content (control chars, null byte, JSON-special,
    Unicode supplementary plane, emoji)
  - Large metadata objects (many tags, large dicts, large content)
"""

from __future__ import annotations

import json
import string
from datetime import UTC, datetime

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryMetadata,
    MemoryRecord,
    MemoryType,
    SnapshotEntry,
    memory_record_from_dict,
    memory_record_from_json,
    memory_record_to_dict,
    memory_record_to_json,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


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


def roundtrip(record: MemoryRecord) -> MemoryRecord:
    return memory_record_from_json(memory_record_to_json(record))


# ---------------------------------------------------------------------------
# 1. Empty strings
# ---------------------------------------------------------------------------


class TestEmptyStringFields:
    """Empty-string values on str fields are legal and must survive roundtrip."""

    def test_empty_record_id_survives(self) -> None:
        rec = _make(record_id="")
        assert roundtrip(rec).record_id == ""

    def test_empty_content_survives(self) -> None:
        rec = _make(content="")
        assert roundtrip(rec).record.content == ""

    def test_empty_source_in_metadata_survives(self) -> None:
        rec = _make(metadata=MemoryMetadata(source=""))
        assert roundtrip(rec).metadata is not None
        assert roundtrip(rec).metadata.source == ""

    def test_empty_promoted_at_survives(self) -> None:
        """Empty string is distinct from None for promoted_at."""
        rec = _make(metadata=MemoryMetadata(source="x", promoted_at=""))
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert restored.metadata.promoted_at == ""

    def test_empty_promoted_from_survives(self) -> None:
        rec = _make(metadata=MemoryMetadata(source="x", promoted_from=""))
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert restored.metadata.promoted_from == ""

    def test_empty_consolidated_at_survives(self) -> None:
        rec = _make(metadata=MemoryMetadata(source="x", consolidated_at=""))
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert restored.metadata.consolidated_at == ""

    def test_empty_consolidated_from_survives(self) -> None:
        rec = _make(metadata=MemoryMetadata(source="x", consolidated_from=""))
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert restored.metadata.consolidated_from == ""

    def test_empty_string_not_treated_as_none_for_promoted_at(self) -> None:
        """``""`` and ``None`` are different values and must not be conflated."""
        none_rec = _make(metadata=MemoryMetadata(source="x", promoted_at=None))
        empty_rec = _make(metadata=MemoryMetadata(source="x", promoted_at=""))
        assert roundtrip(none_rec).metadata.promoted_at is None  # type: ignore[union-attr]
        assert roundtrip(empty_rec).metadata.promoted_at == ""  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 2. Empty arrays for tags
# ---------------------------------------------------------------------------


class TestEmptyTagsArray:
    """An empty tags list is valid and must survive serialisation."""

    def test_empty_tags_survives_roundtrip(self) -> None:
        rec = _make(metadata=MemoryMetadata(source="x", tags=[]))
        assert roundtrip(rec).metadata.tags == []  # type: ignore[union-attr]

    def test_empty_tags_serialised_as_json_array(self) -> None:
        rec = _make(metadata=MemoryMetadata(source="x", tags=[]))
        payload = json.loads(memory_record_to_json(rec))
        assert payload["metadata"]["tags"] == []

    def test_empty_tags_is_list_not_none(self) -> None:
        rec = _make(metadata=MemoryMetadata(source="x", tags=[]))
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert isinstance(restored.metadata.tags, list)

    def test_tags_with_single_empty_string(self) -> None:
        """A tag that is itself an empty string is preserved verbatim."""
        rec = _make(metadata=MemoryMetadata(source="x", tags=[""]))
        restored = roundtrip(rec)
        assert restored.metadata is not None
        assert restored.metadata.tags == [""]

    def test_tags_with_duplicate_values(self) -> None:
        """Duplicate tags are not de-duplicated — order and count are preserved."""
        tags = ["github", "github", "retrieval"]
        rec = _make(metadata=MemoryMetadata(source="x", tags=tags))
        assert roundtrip(rec).metadata.tags == tags  # type: ignore[union-attr]

    def test_tags_with_special_characters(self) -> None:
        tags = ['tag/with/slash', 'tag with spaces', 'tag\twith\ttab', 'tag\nnewline']
        rec = _make(metadata=MemoryMetadata(source="x", tags=tags))
        assert roundtrip(rec).metadata.tags == tags  # type: ignore[union-attr]

    def test_tags_with_non_taxonomy_values_accepted(self) -> None:
        """Tags not in MEMORY_TAG_TAXONOMY are still stored without error."""
        tags = ["custom_label", "future_feature_xyz", ""]
        rec = _make(metadata=MemoryMetadata(source="x", tags=tags))
        assert roundtrip(rec).metadata.tags == tags  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 3. None vs. absent key in memory_record_from_dict for optional fields
# ---------------------------------------------------------------------------


class TestNoneVsAbsentOptionalKeys:
    """For optional fields, explicit None and absent key must both resolve to None."""

    def _base_dict(self) -> dict:
        return memory_record_to_dict(_make())

    def test_absent_last_accessed_at_resolves_to_none(self) -> None:
        d = self._base_dict()
        d.pop("last_accessed_at", None)
        rec = memory_record_from_dict(d)
        assert rec.last_accessed_at is None

    def test_null_last_accessed_at_resolves_to_none(self) -> None:
        d = self._base_dict()
        d["last_accessed_at"] = None
        rec = memory_record_from_dict(d)
        assert rec.last_accessed_at is None

    def test_absent_metadata_resolves_to_none(self) -> None:
        d = self._base_dict()
        d.pop("metadata", None)
        rec = memory_record_from_dict(d)
        assert rec.metadata is None

    def test_null_metadata_resolves_to_none(self) -> None:
        d = self._base_dict()
        d["metadata"] = None
        rec = memory_record_from_dict(d)
        assert rec.metadata is None

    def test_absent_and_null_last_accessed_at_produce_equal_records(self) -> None:
        base = self._base_dict()

        d_absent = {k: v for k, v in base.items() if k != "last_accessed_at"}
        d_null = {**base, "last_accessed_at": None}

        rec_absent = memory_record_from_dict(d_absent)
        rec_null = memory_record_from_dict(d_null)
        assert rec_absent.last_accessed_at == rec_null.last_accessed_at


# ---------------------------------------------------------------------------
# 4. Special characters in content
# ---------------------------------------------------------------------------


class TestSpecialCharactersInContent:
    """Content field must survive all character classes after JSON roundtrip."""

    @pytest.mark.parametrize("special", [
        # JSON-special
        '"quoted"',
        'back\\slash',
        'forward/slash',
        '<html>&amp;</html>',
        # Whitespace control chars
        "tab\there",
        "newline\nhere",
        "carriage\rreturn",
        "null\x00byte",
        # Other control characters
        "".join(chr(i) for i in range(1, 32)),
        # Unicode supplementary plane (emoji, CJK ext)
        "emoji: \U0001F600 \U0001F4A9",
        "CJK ext: \U00020000",
        "combining: e\u0301",    # 'é' as base + combining accent
        "replacement char: \uFFFD",
        # Very long ASCII
        string.ascii_letters * 200,
    ])
    def test_special_content_survives(self, special: str) -> None:
        rec = _make(content=special)
        assert roundtrip(rec).record.content == special

    def test_json_special_chars_produce_valid_json(self) -> None:
        """Content with quotes and backslashes must still yield parseable JSON."""
        rec = _make(content='"key": "value", \\escaped\\')
        raw = memory_record_to_json(rec)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_null_byte_in_content_roundtrips(self) -> None:
        payload = "before\x00after"
        rec = _make(content=payload)
        assert roundtrip(rec).record.content == payload


# ---------------------------------------------------------------------------
# 5. Large metadata objects
# ---------------------------------------------------------------------------


class TestLargeMetadataObjects:
    """Serialisation must handle arbitrarily large payloads without truncation."""

    def test_many_tags_survive(self) -> None:
        tags = [f"tag_{i:04d}" for i in range(500)]
        rec = _make(metadata=MemoryMetadata(source="x", tags=tags))
        assert roundtrip(rec).metadata.tags == tags  # type: ignore[union-attr]

    def test_large_content_survives(self) -> None:
        big = "x" * 100_000
        rec = _make(content=big)
        assert roundtrip(rec).record.content == big

    def test_large_source_url_survives(self) -> None:
        long_url = "https://example.com/" + "a" * 4096
        rec = _make(metadata=MemoryMetadata(source=long_url))
        assert roundtrip(rec).metadata.source == long_url  # type: ignore[union-attr]

    def test_snapshot_entry_with_large_metadata_dict(self) -> None:
        """SnapshotEntry.metadata accepts large, nested dict payloads."""
        large_meta = {f"key_{i}": {"nested": "value", "index": i} for i in range(200)}
        snap = SnapshotEntry(
            snapshot_id="snap-big",
            source_id="src-001",
            ingested_at=_T0,
            source_revision="rev1",
            entry_paths=["a.md"],
            content_hash="hash1",
            metadata=large_meta,
        )
        assert snap.metadata == large_meta

    def test_snapshot_with_many_entry_paths(self) -> None:
        paths = [f"docs/page_{i:04d}.md" for i in range(1000)]
        snap = SnapshotEntry(
            snapshot_id="snap-paths",
            source_id="src-001",
            ingested_at=_T0,
            source_revision="rev1",
            entry_paths=paths,
            content_hash="hash1",
        )
        assert snap.entry_paths == paths

    def test_memory_record_with_all_large_fields(self) -> None:
        """Combining many large fields must still produce a valid, parseable JSON."""
        tags = [f"tag_{i}" for i in range(300)]
        big_content = "\U0001F600" * 10_000  # 10k emoji
        rec = _make(
            record_id="r-" + "x" * 500,
            content=big_content,
            metadata=MemoryMetadata(
                source="https://example.com/" + "path/" * 100,
                confidence=0.5,
                tags=tags,
                promoted_at="2026-01-01T00:00:00+00:00",
                promoted_from="working_memory",
            ),
        )
        raw = memory_record_to_json(rec)
        restored = memory_record_from_json(raw)
        assert restored.record_id == rec.record_id
        assert restored.record.content == big_content
        assert restored.metadata is not None
        assert restored.metadata.tags == tags
