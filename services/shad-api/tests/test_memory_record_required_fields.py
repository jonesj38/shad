"""Tests that MemoryRecord enforces its required fields.

Contract under test:
  - A valid MemoryRecord carries all five observable fields:
      record_id (str), memory_type (MemoryType), content (str),
      created_at (datetime), updated_at (datetime)
  - Constructing a MemoryRecord with any required positional field absent
    raises TypeError.
  - memory_record_from_dict raises KeyError when any required dict key is
    missing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shad.vault.shadow_index import (
    BiTemporalRecord,
    MemoryRecord,
    MemoryType,
    memory_record_from_dict,
    memory_record_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_TIME = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)
_INGESTED_AT = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)


def _make_record(
    *,
    record_id: str = "rec-001",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    content: str = "hello vault",
) -> MemoryRecord:
    """Build a minimal valid MemoryRecord."""
    return MemoryRecord(
        record_id=record_id,
        memory_type=memory_type,
        record=BiTemporalRecord(
            event_time=_EVENT_TIME,
            ingested_at=_INGESTED_AT,
            content=content,
        ),
    )


def _valid_dict() -> dict:
    """Return a valid serialised MemoryRecord dict."""
    return memory_record_to_dict(_make_record())


# ---------------------------------------------------------------------------
# Field presence and types on a valid instance
# ---------------------------------------------------------------------------

class TestRequiredFieldsPresent:
    """A valid MemoryRecord exposes all required fields with the correct types."""

    def test_record_id_present(self) -> None:
        r = _make_record(record_id="abc-123")
        assert r.record_id == "abc-123"

    def test_record_id_is_str(self) -> None:
        r = _make_record()
        assert isinstance(r.record_id, str)

    def test_memory_type_present(self) -> None:
        r = _make_record(memory_type=MemoryType.EPISODIC)
        assert r.memory_type is MemoryType.EPISODIC

    def test_memory_type_is_memory_type(self) -> None:
        r = _make_record()
        assert isinstance(r.memory_type, MemoryType)

    def test_content_present(self) -> None:
        r = _make_record(content="important fact")
        assert r.content == "important fact"

    def test_content_is_str(self) -> None:
        r = _make_record()
        assert isinstance(r.content, str)

    def test_created_at_present(self) -> None:
        r = _make_record()
        assert r.created_at is not None

    def test_created_at_is_datetime(self) -> None:
        r = _make_record()
        assert isinstance(r.created_at, datetime)

    def test_updated_at_present(self) -> None:
        r = _make_record()
        assert r.updated_at is not None

    def test_updated_at_is_datetime(self) -> None:
        r = _make_record()
        assert isinstance(r.updated_at, datetime)

    def test_created_at_is_utc(self) -> None:
        r = _make_record()
        from datetime import timedelta
        assert r.created_at.utcoffset() == timedelta(0)

    def test_updated_at_is_utc(self) -> None:
        r = _make_record()
        from datetime import timedelta
        assert r.updated_at.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Field values match what was passed in
# ---------------------------------------------------------------------------

class TestFieldValues:
    @pytest.mark.parametrize("record_id", ["x", "uuid-abc", "1", "r" * 128])
    def test_record_id_round_trips(self, record_id: str) -> None:
        r = _make_record(record_id=record_id)
        assert r.record_id == record_id

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_memory_type_round_trips(self, mt: MemoryType) -> None:
        r = _make_record(memory_type=mt)
        assert r.memory_type is mt

    @pytest.mark.parametrize("content", [
        "short",
        "",
        "multi\nline\ncontent",
        "unicode: café 日本語 🚀",
        "a" * 10_000,
    ])
    def test_content_round_trips(self, content: str) -> None:
        r = _make_record(content=content)
        assert r.content == content


# ---------------------------------------------------------------------------
# Missing required fields raise TypeError on construction
# ---------------------------------------------------------------------------

class TestMissingRequiredFieldRaisesTypeError:
    """Omitting any positional required field must raise TypeError."""

    def test_missing_record_id(self) -> None:
        with pytest.raises(TypeError):
            MemoryRecord(  # type: ignore[call-arg]
                memory_type=MemoryType.SEMANTIC,
                record=BiTemporalRecord(
                    event_time=_EVENT_TIME,
                    ingested_at=_INGESTED_AT,
                    content="content",
                ),
            )

    def test_missing_memory_type(self) -> None:
        with pytest.raises(TypeError):
            MemoryRecord(  # type: ignore[call-arg]
                record_id="r-001",
                record=BiTemporalRecord(
                    event_time=_EVENT_TIME,
                    ingested_at=_INGESTED_AT,
                    content="content",
                ),
            )

    def test_missing_record(self) -> None:
        with pytest.raises(TypeError):
            MemoryRecord(  # type: ignore[call-arg]
                record_id="r-001",
                memory_type=MemoryType.SEMANTIC,
            )

    def test_no_arguments(self) -> None:
        with pytest.raises(TypeError):
            MemoryRecord()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# memory_record_from_dict rejects dicts missing required keys
# ---------------------------------------------------------------------------

_REQUIRED_DICT_KEYS = ["record_id", "memory_type", "record", "created_at", "updated_at"]


class TestFromDictMissingKeyRaisesKeyError:
    """memory_record_from_dict must raise KeyError for each absent required key."""

    @pytest.mark.parametrize("missing_key", _REQUIRED_DICT_KEYS)
    def test_missing_key_raises_key_error(self, missing_key: str) -> None:
        d = _valid_dict()
        del d[missing_key]
        with pytest.raises(KeyError):
            memory_record_from_dict(d)

    @pytest.mark.parametrize("missing_record_subkey", ["event_time", "ingested_at", "content"])
    def test_missing_record_subkey_raises_key_error(self, missing_record_subkey: str) -> None:
        d = _valid_dict()
        del d["record"][missing_record_subkey]
        with pytest.raises(KeyError):
            memory_record_from_dict(d)


# ---------------------------------------------------------------------------
# Dict representation carries all five fields
# ---------------------------------------------------------------------------

class TestDictRepresentationFields:
    """memory_record_to_dict output must include all required keys."""

    @pytest.mark.parametrize("key", ["record_id", "memory_type", "created_at", "updated_at"])
    def test_top_level_key_present(self, key: str) -> None:
        d = _valid_dict()
        assert key in d

    def test_content_accessible_via_record_subkey(self) -> None:
        d = _valid_dict()
        assert "content" in d["record"]

    def test_record_id_type_in_dict(self) -> None:
        d = _valid_dict()
        assert isinstance(d["record_id"], str)

    def test_memory_type_type_in_dict(self) -> None:
        d = _valid_dict()
        assert isinstance(d["memory_type"], str)

    def test_content_type_in_dict(self) -> None:
        d = _valid_dict()
        assert isinstance(d["record"]["content"], str)

    def test_created_at_type_in_dict(self) -> None:
        d = _valid_dict()
        assert isinstance(d["created_at"], str)
        # Must be parseable as ISO-8601
        datetime.fromisoformat(d["created_at"])

    def test_updated_at_type_in_dict(self) -> None:
        d = _valid_dict()
        assert isinstance(d["updated_at"], str)
        datetime.fromisoformat(d["updated_at"])
