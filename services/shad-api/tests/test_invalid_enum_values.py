"""Tests for rejection of invalid enum values across MemoryType and UpdatePolicy.

Covers three rejection surfaces:

  1. Direct enum constructor  — ``MemoryType("bad")`` / ``UpdatePolicy("bad")``
  2. ``MemoryType.from_string()`` — ``None`` and other non-string inputs not
     already covered by test_memory_type_serialization.py
  3. Deserialization — ``deserialize_promotion_history`` with corrupted
     ``from_type`` / ``to_type`` values
  4. DB round-trip — ``ShadowIndex.get_snapshot`` when the stored
     ``memory_type`` column contains a value that is not a valid MemoryType
  5. ``contracts.MemoryRecord`` construction — ``__post_init__`` gate rejects
     ``None`` and non-MemoryType values before the record enters the pipeline

Each test asserts that an appropriate exception (``ValueError`` or
``TypeError``) is raised and that no silent coercion occurs.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shad.vault.contracts import MemoryRecord as ContractsMemoryRecord
from shad.vault.shadow_index import (
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
    deserialize_promotion_history,
)

# ---------------------------------------------------------------------------
# 1. MemoryType direct constructor  (str-enum value lookup)
# ---------------------------------------------------------------------------


class TestMemoryTypeDirectConstructor:
    """MemoryType(value) uses Python's enum value lookup — invalid values must
    raise ValueError, not silently produce a new member."""

    @pytest.mark.parametrize("bad_value", [
        "invalid",
        "INVALID",
        "",
        "   ",
        "working memory",   # space instead of underscore
        "semantic",         # short name, not the full value string
        "episodic_memo",    # truncated
        "working_memoryX",  # suffix
    ])
    def test_raises_value_error_for_bad_string(self, bad_value: str) -> None:
        with pytest.raises(ValueError):
            MemoryType(bad_value)

    def test_raises_for_none(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            MemoryType(None)  # type: ignore[arg-type]

    def test_raises_for_integer(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            MemoryType(42)  # type: ignore[arg-type]

    def test_valid_values_do_not_raise(self) -> None:
        """Sanity-check: all declared values must construct successfully."""
        for member in MemoryType:
            assert MemoryType(member.value) is member


# ---------------------------------------------------------------------------
# 2. UpdatePolicy direct constructor
# ---------------------------------------------------------------------------


class TestUpdatePolicyDirectConstructor:
    @pytest.mark.parametrize("bad_value", [
        "invalid",
        "MANUAL",       # upper-case name, not the value "manual"
        "AUTO",         # same
        "",
        "   ",
        "auto_policy",
    ])
    def test_raises_value_error_for_bad_string(self, bad_value: str) -> None:
        with pytest.raises(ValueError):
            UpdatePolicy(bad_value)

    def test_raises_for_none(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            UpdatePolicy(None)  # type: ignore[arg-type]

    def test_valid_values_do_not_raise(self) -> None:
        for member in UpdatePolicy:
            assert UpdatePolicy(member.value) is member


# ---------------------------------------------------------------------------
# 3. MemoryType.from_string() — None / non-string inputs
# ---------------------------------------------------------------------------


class TestMemoryTypeFromStringNonStringInputs:
    """None and non-string inputs must not be silently coerced."""

    def test_none_raises(self) -> None:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            MemoryType.from_string(None)  # type: ignore[arg-type]

    def test_integer_raises(self) -> None:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            MemoryType.from_string(42)  # type: ignore[arg-type]

    def test_list_raises(self) -> None:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            MemoryType.from_string(["semantic_memory"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. deserialize_promotion_history — invalid enum strings in JSON
# ---------------------------------------------------------------------------


class TestDeserializePromotionHistoryInvalidEnums:
    """When the stored JSON contains a value that is not a valid MemoryType
    the deserializer must propagate the ValueError from MemoryType(), not
    return a record with a corrupt type."""

    def _make_entry(
        self,
        *,
        from_type: str | None,
        to_type: str,
    ) -> str:
        return json.dumps([
            {
                "from_type": from_type,
                "to_type": to_type,
                "timestamp": "2026-03-01T10:00:00+00:00",
                "reason": "test",
            }
        ])

    @pytest.mark.parametrize("bad_to_type", [
        "invalid",
        "",
        "EPISODIC",  # upper-case name, not value
        "episodic",  # short name
    ])
    def test_invalid_to_type_raises(self, bad_to_type: str) -> None:
        raw = self._make_entry(from_type=None, to_type=bad_to_type)
        with pytest.raises(ValueError):
            deserialize_promotion_history(raw)

    @pytest.mark.parametrize("bad_from_type", [
        "invalid",
        "",
        "WORKING",   # upper-case name, not value
        "working",   # short name
    ])
    def test_invalid_from_type_raises(self, bad_from_type: str) -> None:
        raw = self._make_entry(from_type=bad_from_type, to_type="episodic_memory")
        with pytest.raises(ValueError):
            deserialize_promotion_history(raw)

    def test_null_from_type_is_accepted(self) -> None:
        """from_type=null is legal — represents the initial (unclassified) state."""
        raw = self._make_entry(from_type=None, to_type="episodic_memory")
        history = deserialize_promotion_history(raw)
        assert len(history) == 1
        assert history[0].from_type is None
        assert history[0].to_type is MemoryType.EPISODIC

    def test_empty_string_raw_returns_empty_list(self) -> None:
        """Falsy raw values (empty string, None) yield an empty list — not an error."""
        assert deserialize_promotion_history("") == []
        assert deserialize_promotion_history(None) == []


# ---------------------------------------------------------------------------
# 5. ShadowIndex DB round-trip — corrupted memory_type column
# ---------------------------------------------------------------------------


def _make_index_with_source(tmp_path: Path) -> tuple[ShadowIndex, str]:
    """Return an index and a source_id pre-populated with one source."""
    index = ShadowIndex(tmp_path / "test.sqlite")
    source_id = "corrupt-enum-src-001"
    index.add_source(
        SourceEntry(
            source_url="sessions://local/corrupt-enum-test",
            source_id=source_id,
            source_type="sessions",
            update_policy=UpdatePolicy.MANUAL,
        )
    )
    return index, source_id


def _insert_snapshot_with_raw_memory_type(
    db_path: Path,
    source_id: str,
    raw_memory_type: str,
) -> str:
    """Bypass the ORM and write a snapshot row with a raw (possibly invalid)
    memory_type value directly into SQLite.  Returns the snapshot_id."""
    snapshot_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO snapshots
          (snapshot_id, source_id, ingested_at, source_revision,
           entry_paths, content_hash, metadata, memory_type, promotion_history)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            source_id,
            datetime.now(UTC).isoformat(),
            "rev-corrupt",
            json.dumps(["sessions/note.md"]),
            "deadbeef",
            json.dumps({}),
            raw_memory_type,
            "[]",
        ),
    )
    conn.commit()
    conn.close()
    return snapshot_id


class TestShadowIndexCorruptedEnumOnRead:
    """ShadowIndex.get_snapshot must not silently swallow a corrupted
    memory_type value; it should propagate a ValueError from MemoryType()."""

    @pytest.mark.parametrize("bad_value", [
        "invalid_memory",
        "EPISODIC",     # upper-case name, not the stored value string
        "garbage",
    ])
    def test_get_snapshot_raises_on_corrupted_memory_type(
        self, tmp_path: Path, bad_value: str
    ) -> None:
        db_path = tmp_path / "test.sqlite"
        index, source_id = _make_index_with_source(tmp_path)

        snapshot_id = _insert_snapshot_with_raw_memory_type(
            db_path, source_id, bad_value
        )

        with pytest.raises(ValueError):
            index.get_snapshot(snapshot_id)

    def test_empty_string_in_db_treated_as_null(self, tmp_path: Path) -> None:
        """The ORM uses ``if row["memory_type"]`` so an empty string stored in
        the column is coerced to None rather than raising ValueError.
        This test documents the existing behaviour so any change is explicit."""
        db_path = tmp_path / "test.sqlite"
        index, source_id = _make_index_with_source(tmp_path)

        snapshot_id = _insert_snapshot_with_raw_memory_type(db_path, source_id, "")

        fetched = index.get_snapshot(snapshot_id)
        assert fetched is not None
        assert fetched.memory_type is None

    def test_get_snapshot_succeeds_for_null_memory_type(
        self, tmp_path: Path
    ) -> None:
        """NULL in the DB column is valid — maps to memory_type=None."""
        index = ShadowIndex(tmp_path / "test.sqlite")
        source_id = "null-type-src"
        index.add_source(
            SourceEntry(
                source_url="sessions://local/null-type",
                source_id=source_id,
                source_type="sessions",
                update_policy=UpdatePolicy.MANUAL,
            )
        )
        snap = SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=source_id,
            ingested_at=datetime(2026, 3, 1, tzinfo=UTC),
            source_revision="rev-null",
            entry_paths=["note.md"],
            content_hash="abc123",
            memory_type=None,  # explicitly unclassified
        )
        index.add_snapshot(snap)

        fetched = index.get_snapshot(snap.snapshot_id)
        assert fetched is not None
        assert fetched.memory_type is None


# ---------------------------------------------------------------------------
# 5. contracts.MemoryRecord — __post_init__ validation gate
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)


def _valid_contracts_record(**overrides) -> ContractsMemoryRecord:
    """Build a minimal valid contracts.MemoryRecord, with optional overrides."""
    defaults = dict(
        record_id="rec-001",
        content="hello pipeline",
        embedding=[0.1, 0.2, 0.3],
        memory_type=MemoryType.SEMANTIC,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(overrides)
    return ContractsMemoryRecord(**defaults)


class TestContractsMemoryRecordTypeGate:
    """contracts.MemoryRecord.__post_init__ must reject None and raw strings."""

    def test_valid_memory_type_accepted(self) -> None:
        """All four MemoryType variants must construct without error."""
        for mt in MemoryType:
            record = _valid_contracts_record(memory_type=mt)
            assert record.memory_type is mt

    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="memory_type is required"):
            _valid_contracts_record(memory_type=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value", [
        "semantic_memory",   # valid value string — but not a MemoryType instance
        "episodic_memory",
        "working_memory",
        "procedural_memory",
        "SEMANTIC",          # member name, not the value
        "invalid",
        "",
        42,
    ])
    def test_non_enum_instance_raises_value_error(self, bad_value) -> None:
        with pytest.raises(ValueError, match="Invalid memory_type"):
            _valid_contracts_record(memory_type=bad_value)  # type: ignore[arg-type]

    def test_error_message_lists_valid_variants(self) -> None:
        """The ValueError message must enumerate all valid MemoryType values."""
        with pytest.raises(ValueError) as exc_info:
            _valid_contracts_record(memory_type="bad_value")  # type: ignore[arg-type]

        message = str(exc_info.value)
        for mt in MemoryType:
            assert mt.value in message, f'"{mt.value}" missing from error message'

    def test_error_message_for_none_lists_valid_variants(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _valid_contracts_record(memory_type=None)  # type: ignore[arg-type]

        message = str(exc_info.value)
        for mt in MemoryType:
            assert mt.value in message, f'"{mt.value}" missing from error message'


# ---------------------------------------------------------------------------
# 6. contracts.MemoryRecord — missing required fields raise TypeError
# ---------------------------------------------------------------------------


class TestContractsMemoryRecordMissingRequiredFields:
    """Omitting any required positional field must raise TypeError immediately,
    before __post_init__ is even reached."""

    def test_missing_record_id_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord(  # type: ignore[call-arg]
                content="hello",
                embedding=[0.1, 0.2],
                memory_type=MemoryType.SEMANTIC,
                created_at=_NOW,
                updated_at=_NOW,
            )

    def test_missing_content_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord(  # type: ignore[call-arg]
                record_id="rec-001",
                embedding=[0.1, 0.2],
                memory_type=MemoryType.SEMANTIC,
                created_at=_NOW,
                updated_at=_NOW,
            )

    def test_missing_embedding_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord(  # type: ignore[call-arg]
                record_id="rec-001",
                content="hello",
                memory_type=MemoryType.SEMANTIC,
                created_at=_NOW,
                updated_at=_NOW,
            )

    def test_missing_memory_type_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord(  # type: ignore[call-arg]
                record_id="rec-001",
                content="hello",
                embedding=[0.1, 0.2],
                created_at=_NOW,
                updated_at=_NOW,
            )

    def test_missing_created_at_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord(  # type: ignore[call-arg]
                record_id="rec-001",
                content="hello",
                embedding=[0.1, 0.2],
                memory_type=MemoryType.SEMANTIC,
                updated_at=_NOW,
            )

    def test_missing_updated_at_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord(  # type: ignore[call-arg]
                record_id="rec-001",
                content="hello",
                embedding=[0.1, 0.2],
                memory_type=MemoryType.SEMANTIC,
                created_at=_NOW,
            )

    def test_no_arguments_raises(self) -> None:
        with pytest.raises(TypeError):
            ContractsMemoryRecord()  # type: ignore[call-arg]
