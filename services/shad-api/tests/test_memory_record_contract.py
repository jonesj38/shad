"""Tests validating that a well-formed MemoryRecord satisfies the interface shape.

Contract under test (shad.vault.contracts.MemoryRecord):
  - All required fields are present and carry the correct types after construction
  - Optional fields default to the correct sentinel values
  - __post_init__ rejects a missing or invalid memory_type at construction time
  - Mutable default collections are independent across instances (no shared state)
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from shad.vault.contracts import MemoryRecord
from shad.vault.shadow_index import MemoryType

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_EMBEDDING = [0.1, 0.2, 0.3]


def _make(**overrides: object) -> MemoryRecord:
    """Return a minimal well-formed MemoryRecord, with optional field overrides."""
    defaults: dict[str, object] = {
        "record_id": "rec-001",
        "content": "test content",
        "embedding": list(_EMBEDDING),
        "memory_type": MemoryType.EPISODIC,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Required fields: presence and types
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """A well-formed MemoryRecord has every required field with the expected type."""

    def test_record_id_is_str(self) -> None:
        rec = _make(record_id="abc-123")
        assert isinstance(rec.record_id, str)
        assert rec.record_id == "abc-123"

    def test_content_is_str(self) -> None:
        rec = _make(content="some knowledge")
        assert isinstance(rec.content, str)
        assert rec.content == "some knowledge"

    def test_embedding_is_list_of_float(self) -> None:
        embedding = [0.5, -0.3, 1.0]
        rec = _make(embedding=embedding)
        assert isinstance(rec.embedding, list)
        assert all(isinstance(v, float) for v in rec.embedding)
        assert rec.embedding == embedding

    def test_memory_type_is_memory_type_enum(self) -> None:
        for mt in MemoryType:
            rec = _make(memory_type=mt)
            assert isinstance(rec.memory_type, MemoryType)
            assert rec.memory_type is mt

    def test_created_at_is_datetime(self) -> None:
        rec = _make(created_at=_NOW)
        assert isinstance(rec.created_at, datetime)
        assert rec.created_at == _NOW

    def test_updated_at_is_datetime(self) -> None:
        rec = _make(updated_at=_NOW)
        assert isinstance(rec.updated_at, datetime)
        assert rec.updated_at == _NOW

    def test_all_required_fields_present_on_dataclass(self) -> None:
        """dataclasses.fields() must include every required field name."""
        field_names = {f.name for f in dataclasses.fields(MemoryRecord)}
        required = {"record_id", "content", "embedding", "memory_type", "created_at", "updated_at"}
        assert required.issubset(field_names)


# ---------------------------------------------------------------------------
# Optional fields: default values and types
# ---------------------------------------------------------------------------


class TestOptionalFieldDefaults:
    """Optional fields default to the correct sentinel types."""

    def test_source_refs_defaults_to_empty_list(self) -> None:
        rec = _make()
        assert isinstance(rec.source_refs, list)
        assert rec.source_refs == []

    def test_source_run_id_defaults_to_none(self) -> None:
        rec = _make()
        assert rec.source_run_id is None

    def test_consolidated_into_defaults_to_none(self) -> None:
        rec = _make()
        assert rec.consolidated_into is None

    def test_tags_defaults_to_empty_list(self) -> None:
        rec = _make()
        assert isinstance(rec.tags, list)
        assert rec.tags == []

    def test_metadata_defaults_to_empty_dict(self) -> None:
        rec = _make()
        assert isinstance(rec.metadata, dict)
        assert rec.metadata == {}

    def test_optional_fields_accept_non_default_values(self) -> None:
        rec = _make(
            source_refs=["note://a", "note://b"],
            source_run_id="run-xyz",
            consolidated_into="entry-999",
            tags=["retrieval", "python"],
            metadata={"confidence": 0.9},
        )
        assert rec.source_refs == ["note://a", "note://b"]
        assert rec.source_run_id == "run-xyz"
        assert rec.consolidated_into == "entry-999"
        assert rec.tags == ["retrieval", "python"]
        assert rec.metadata == {"confidence": 0.9}


# ---------------------------------------------------------------------------
# Mutable default isolation (no shared state between instances)
# ---------------------------------------------------------------------------


class TestMutableDefaultIsolation:
    """Default-factory collections must not be shared across distinct instances."""

    def test_source_refs_are_independent(self) -> None:
        a, b = _make(), _make()
        a.source_refs.append("x")
        assert b.source_refs == []

    def test_tags_are_independent(self) -> None:
        a, b = _make(), _make()
        a.tags.append("tag")
        assert b.tags == []

    def test_metadata_are_independent(self) -> None:
        a, b = _make(), _make()
        a.metadata["key"] = "value"
        assert b.metadata == {}


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


class TestMemoryTypeValidation:
    """__post_init__ must reject missing or invalid memory_type at construction time."""

    def test_none_memory_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="memory_type is required"):
            _make(memory_type=None)  # type: ignore[arg-type]

    def test_string_memory_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid memory_type"):
            _make(memory_type="episodic")  # type: ignore[arg-type]

    def test_int_memory_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid memory_type"):
            _make(memory_type=0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("mt", list(MemoryType))
    def test_all_valid_memory_types_are_accepted(self, mt: MemoryType) -> None:
        rec = _make(memory_type=mt)
        assert rec.memory_type is mt


# ---------------------------------------------------------------------------
# Round-trip: dataclasses.asdict preserves shape
# ---------------------------------------------------------------------------


class TestDictRoundTrip:
    """dataclasses.asdict() must produce the canonical field set."""

    def test_asdict_contains_all_fields(self) -> None:
        rec = _make()
        d = dataclasses.asdict(rec)
        expected_keys = {
            "record_id", "content", "embedding", "memory_type",
            "created_at", "updated_at", "source_refs", "source_run_id",
            "consolidated_into", "tags", "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_asdict_preserves_required_field_values(self) -> None:
        rec = _make(record_id="rec-42", content="hello", memory_type=MemoryType.SEMANTIC)
        d = dataclasses.asdict(rec)
        assert d["record_id"] == "rec-42"
        assert d["content"] == "hello"
        # StrEnum values serialise to their string value via asdict
        assert d["memory_type"] == MemoryType.SEMANTIC
