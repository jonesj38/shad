"""Tests for snapshot promotion idempotency and edge cases.

'Promotion' means upgrading a SnapshotEntry from MemoryType.EPISODIC (transient,
time-bound event data) to MemoryType.SEMANTIC (permanent, factual knowledge).
This happens when the consolidation threshold is reached and episodic entries are
absorbed into the semantic store.

The promotion mechanism is ShadowIndex.add_snapshot with INSERT OR REPLACE:
the same snapshot_id with an updated memory_type overwrites the prior DB row.

Contracts under test:
  1. Idempotency  — re-promoting an already-SEMANTIC snapshot leaves the store
                    unchanged (no duplicate rows, same field values).
  2. Malformed    — promoting with missing or invalid required state fields raises.
  3. Rapid writes — successive promotions for the same snapshot_id converge to the
                    last-written state; no phantom rows accumulate.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shad.vault.shadow_index import (
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TIME = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
_SOURCE_ID = "src-001"
_SOURCE_URL = "https://feeds.example.com/shad-test"


def _source(source_id: str = _SOURCE_ID) -> SourceEntry:
    return SourceEntry(
        source_url=f"https://feeds.example.com/{source_id}",
        source_id=source_id,
        source_type="feed",
        update_policy=UpdatePolicy.AUTO,
    )


def _snapshot(
    *,
    snapshot_id: str | None = None,
    source_id: str = _SOURCE_ID,
    memory_type: MemoryType | None = MemoryType.EPISODIC,
    ingested_at: datetime = _BASE_TIME,
) -> SnapshotEntry:
    """Minimal valid SnapshotEntry with configurable identity and type fields."""
    return SnapshotEntry(
        snapshot_id=snapshot_id or str(uuid.uuid4()),
        source_id=source_id,
        ingested_at=ingested_at,
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["feeds/item.md"],
        content_hash=str(uuid.uuid4()),
        memory_type=memory_type,
    )


def _promote(snap: SnapshotEntry) -> SnapshotEntry:
    """Return a copy of *snap* with memory_type set to SEMANTIC."""
    return SnapshotEntry(
        snapshot_id=snap.snapshot_id,
        source_id=snap.source_id,
        ingested_at=snap.ingested_at,
        source_revision=snap.source_revision,
        entry_paths=snap.entry_paths,
        content_hash=snap.content_hash,
        memory_type=MemoryType.SEMANTIC,
    )


@pytest.fixture
def index(tmp_path: Path) -> ShadowIndex:
    """ShadowIndex backed by a temp SQLite file with one feed source registered."""
    idx = ShadowIndex(tmp_path / "index.sqlite")
    idx.add_source(_source())
    return idx


# ---------------------------------------------------------------------------
# 1. Idempotency — re-promoting an already-SEMANTIC snapshot is a no-op
# ---------------------------------------------------------------------------


class TestPromotionIdempotency:
    """Contracts:
    - A second promotion call for a SEMANTIC snapshot leaves the row unchanged.
    - No duplicate rows appear in the snapshots table.
    - All non-memory_type fields are preserved verbatim across re-promotion.
    """

    def test_single_promotion_changes_memory_type(self, index: ShadowIndex) -> None:
        """EPISODIC → SEMANTIC: memory_type on the stored row is updated."""
        snap = _snapshot(memory_type=MemoryType.EPISODIC)
        index.add_snapshot(snap)
        assert index.get_snapshot(snap.snapshot_id).memory_type is MemoryType.EPISODIC

        index.add_snapshot(_promote(snap))

        assert index.get_snapshot(snap.snapshot_id).memory_type is MemoryType.SEMANTIC

    def test_re_promoting_already_semantic_is_idempotent(
        self, index: ShadowIndex
    ) -> None:
        """Calling promotion twice on an already-SEMANTIC snapshot is a no-op."""
        snap_id = str(uuid.uuid4())
        snap = _snapshot(snapshot_id=snap_id, memory_type=MemoryType.SEMANTIC)
        index.add_snapshot(snap)

        # Promote again — should be a no-op
        index.add_snapshot(snap)

        result = index.get_snapshot(snap_id)
        assert result is not None
        assert result.memory_type is MemoryType.SEMANTIC

    def test_re_promoting_does_not_accumulate_rows(self, index: ShadowIndex) -> None:
        """INSERT OR REPLACE must not create extra rows on re-promotion."""
        snap_id = str(uuid.uuid4())
        snap = _snapshot(snapshot_id=snap_id, memory_type=MemoryType.SEMANTIC)
        for _ in range(5):
            index.add_snapshot(snap)

        all_snaps = index.list_snapshots(_SOURCE_ID)
        assert [s.snapshot_id for s in all_snaps].count(snap_id) == 1

    def test_re_promotion_preserves_source_id(self, index: ShadowIndex) -> None:
        snap = _snapshot(memory_type=MemoryType.SEMANTIC)
        index.add_snapshot(snap)
        index.add_snapshot(snap)  # re-promote

        result = index.get_snapshot(snap.snapshot_id)
        assert result.source_id == snap.source_id

    def test_re_promotion_preserves_content_hash(self, index: ShadowIndex) -> None:
        snap = _snapshot(memory_type=MemoryType.SEMANTIC)
        index.add_snapshot(snap)
        index.add_snapshot(snap)

        result = index.get_snapshot(snap.snapshot_id)
        assert result.content_hash == snap.content_hash

    def test_re_promotion_preserves_ingested_at(self, index: ShadowIndex) -> None:
        fixed_time = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)
        snap = _snapshot(ingested_at=fixed_time, memory_type=MemoryType.SEMANTIC)
        index.add_snapshot(snap)
        index.add_snapshot(snap)

        result = index.get_snapshot(snap.snapshot_id)
        assert result.ingested_at == fixed_time

    def test_re_promotion_preserves_entry_paths(self, index: ShadowIndex) -> None:
        snap = SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=_SOURCE_ID,
            ingested_at=_BASE_TIME,
            source_revision="rev-x",
            entry_paths=["feeds/a.md", "feeds/b.md", "feeds/c.md"],
            content_hash="abc",
            memory_type=MemoryType.SEMANTIC,
        )
        index.add_snapshot(snap)
        index.add_snapshot(snap)

        result = index.get_snapshot(snap.snapshot_id)
        assert result.entry_paths == ["feeds/a.md", "feeds/b.md", "feeds/c.md"]

    def test_promoted_snapshot_excluded_from_episodic_count(
        self, index: ShadowIndex
    ) -> None:
        """After EPISODIC → SEMANTIC promotion, episodic_count drops to zero."""
        snap = _snapshot(memory_type=MemoryType.EPISODIC)
        index.add_snapshot(snap)

        since = _BASE_TIME - timedelta(hours=1)
        assert index.episodic_count(_SOURCE_ID, since=since) == 1

        index.add_snapshot(_promote(snap))

        assert index.episodic_count(_SOURCE_ID, since=since) == 0

    def test_triple_promotion_leaves_single_semantic_row(
        self, index: ShadowIndex
    ) -> None:
        """Promoting three times: exactly one SEMANTIC row, no EPISODIC rows."""
        snap = _snapshot(memory_type=MemoryType.EPISODIC)
        index.add_snapshot(snap)

        promoted = _promote(snap)
        for _ in range(3):
            index.add_snapshot(promoted)

        semantic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.SEMANTIC)
        episodic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.EPISODIC)
        assert len(semantic) == 1
        assert len(episodic) == 0

    def test_promoting_none_memory_type_to_semantic(self, index: ShadowIndex) -> None:
        """A snapshot stored without a memory_type can be promoted to SEMANTIC."""
        snap = _snapshot(memory_type=None)
        index.add_snapshot(snap)

        assert index.get_snapshot(snap.snapshot_id).memory_type is None

        index.add_snapshot(_promote(snap))

        assert index.get_snapshot(snap.snapshot_id).memory_type is MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# 2. Malformed state raises
# ---------------------------------------------------------------------------


class TestPromotionMalformedState:
    """Contracts:
    - Promotion with None ingested_at raises AttributeError (can't call .isoformat()).
    - Promotion with None source_id raises IntegrityError (NOT NULL constraint).
    - Promotion with None content_hash raises IntegrityError (NOT NULL constraint).
    - A corrupted memory_type value written directly to the DB raises ValueError
      when the row is read back via get_snapshot.
    """

    def test_none_ingested_at_raises_attribute_error(
        self, index: ShadowIndex
    ) -> None:
        """`ingested_at=None` triggers AttributeError on `.isoformat()` in add_snapshot."""
        snap = SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=_SOURCE_ID,
            ingested_at=None,  # type: ignore[arg-type]
            source_revision="rev-bad",
            entry_paths=["feeds/item.md"],
            content_hash="abc123",
            memory_type=MemoryType.SEMANTIC,
        )
        with pytest.raises(AttributeError):
            index.add_snapshot(snap)

    def test_none_source_id_raises_integrity_error(
        self, index: ShadowIndex
    ) -> None:
        """`source_id=None` violates the NOT NULL constraint on the snapshots table."""
        snap = SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=None,  # type: ignore[arg-type]
            ingested_at=_BASE_TIME,
            source_revision="rev-bad",
            entry_paths=["feeds/item.md"],
            content_hash="abc123",
            memory_type=MemoryType.SEMANTIC,
        )
        with pytest.raises(sqlite3.IntegrityError):
            index.add_snapshot(snap)

    def test_none_content_hash_raises_integrity_error(
        self, index: ShadowIndex
    ) -> None:
        """`content_hash=None` violates the NOT NULL constraint on content_hash."""
        snap = SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=_SOURCE_ID,
            ingested_at=_BASE_TIME,
            source_revision="rev-bad",
            entry_paths=["feeds/item.md"],
            content_hash=None,  # type: ignore[arg-type]
            memory_type=MemoryType.SEMANTIC,
        )
        with pytest.raises(sqlite3.IntegrityError):
            index.add_snapshot(snap)

    def test_invalid_memory_type_in_db_raises_on_read(self, tmp_path: Path) -> None:
        """A corrupted memory_type string in the DB raises ValueError on get_snapshot.

        This guards against direct DB mutations or schema migrations that leave
        memory_type values outside the MemoryType enum.
        """
        db_path = tmp_path / "corrupt.sqlite"
        snap_id = str(uuid.uuid4())

        # Bootstrap the schema, then close cleanly before a raw write.
        idx = ShadowIndex(db_path)
        idx.add_source(_source())
        idx.close()

        # Inject a row with an invalid memory_type directly, bypassing the ORM.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            INSERT INTO snapshots
            (snapshot_id, source_id, ingested_at, source_revision,
             entry_paths, content_hash, metadata, memory_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snap_id,
                _SOURCE_ID,
                _BASE_TIME.isoformat(),
                "rev-x",
                '["feeds/item.md"]',
                "hash-abc",
                "{}",
                "not_a_valid_memory_type",  # not in MemoryType enum
            ),
        )
        conn.commit()
        conn.close()

        # Re-open through ShadowIndex and attempt to read — must raise ValueError.
        idx2 = ShadowIndex(db_path)
        try:
            with pytest.raises(ValueError):
                idx2.get_snapshot(snap_id)
        finally:
            idx2.close()

    def test_invalid_memory_type_in_db_raises_on_list_snapshots(
        self, tmp_path: Path
    ) -> None:
        """Corrupted memory_type also raises when the row is returned by list_snapshots."""
        db_path = tmp_path / "corrupt_list.sqlite"

        idx = ShadowIndex(db_path)
        idx.add_source(_source())
        idx.close()

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            INSERT INTO snapshots
            (snapshot_id, source_id, ingested_at, source_revision,
             entry_paths, content_hash, metadata, memory_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                _SOURCE_ID,
                _BASE_TIME.isoformat(),
                "rev-x",
                '["feeds/item.md"]',
                "hash-abc",
                "{}",
                "garbage_type",
            ),
        )
        conn.commit()
        conn.close()

        idx2 = ShadowIndex(db_path)
        try:
            with pytest.raises(ValueError):
                idx2.list_snapshots(_SOURCE_ID)
        finally:
            idx2.close()

    def test_none_source_revision_raises_integrity_error(
        self, index: ShadowIndex
    ) -> None:
        """`source_revision=None` violates the NOT NULL constraint."""
        snap = SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=_SOURCE_ID,
            ingested_at=_BASE_TIME,
            source_revision=None,  # type: ignore[arg-type]
            entry_paths=["feeds/item.md"],
            content_hash="abc123",
            memory_type=MemoryType.SEMANTIC,
        )
        with pytest.raises(sqlite3.IntegrityError):
            index.add_snapshot(snap)


# ---------------------------------------------------------------------------
# 3. Rapid successive promotions
# ---------------------------------------------------------------------------


class TestRapidSuccessivePromotions:
    """Contracts:
    - INSERT OR REPLACE semantics: the last write wins.
    - No phantom / duplicate rows accumulate after N rapid writes.
    - Final episodic_count reflects the last-written memory_type.
    - Multiple independent snapshots can be promoted in sequence without
      cross-contamination.
    """

    def test_last_write_wins_alternating_types(self, index: ShadowIndex) -> None:
        """Alternating EPISODIC/SEMANTIC writes: the final call's type persists."""
        snap = _snapshot(memory_type=MemoryType.EPISODIC)
        index.add_snapshot(snap)

        writes = [
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,  # ← last write
        ]
        for mt in writes:
            updated = SnapshotEntry(
                snapshot_id=snap.snapshot_id,
                source_id=snap.source_id,
                ingested_at=snap.ingested_at,
                source_revision=snap.source_revision,
                entry_paths=snap.entry_paths,
                content_hash=snap.content_hash,
                memory_type=mt,
            )
            index.add_snapshot(updated)

        result = index.get_snapshot(snap.snapshot_id)
        assert result.memory_type is MemoryType.SEMANTIC  # matches last write

    def test_no_duplicate_rows_after_rapid_writes(self, index: ShadowIndex) -> None:
        """Rapid INSERT OR REPLACE calls for one snapshot_id produce exactly one row."""
        snap = _snapshot(memory_type=MemoryType.EPISODIC)
        for _ in range(10):
            index.add_snapshot(snap)

        all_snaps = index.list_snapshots(_SOURCE_ID)
        assert [s.snapshot_id for s in all_snaps].count(snap.snapshot_id) == 1

    def test_episodic_count_zero_after_rapid_semantic_writes(
        self, index: ShadowIndex
    ) -> None:
        """After several SEMANTIC writes, episodic_count must be 0."""
        snap = _snapshot(memory_type=MemoryType.EPISODIC)
        index.add_snapshot(snap)

        since = _BASE_TIME - timedelta(hours=1)
        assert index.episodic_count(_SOURCE_ID, since=since) == 1

        promoted = _promote(snap)
        for _ in range(4):
            index.add_snapshot(promoted)

        assert index.episodic_count(_SOURCE_ID, since=since) == 0

    def test_rapid_promotion_of_multiple_independent_snapshots(
        self, index: ShadowIndex
    ) -> None:
        """Promoting N distinct snapshots in sequence leaves N SEMANTIC rows, 0 EPISODIC."""
        n = 8
        snaps = [
            _snapshot(
                snapshot_id=str(uuid.uuid4()),
                ingested_at=_BASE_TIME + timedelta(minutes=i),
                memory_type=MemoryType.EPISODIC,
            )
            for i in range(n)
        ]
        for s in snaps:
            index.add_snapshot(s)

        for s in snaps:
            index.add_snapshot(_promote(s))

        since = _BASE_TIME - timedelta(hours=1)
        assert index.episodic_count(_SOURCE_ID, since=since) == 0

        all_snaps = index.list_snapshots(_SOURCE_ID)
        assert len(all_snaps) == n
        assert all(s.memory_type is MemoryType.SEMANTIC for s in all_snaps)

    def test_partial_promotion_leaves_correct_counts(
        self, index: ShadowIndex
    ) -> None:
        """Promoting half of N snapshots leaves N/2 SEMANTIC and N/2 EPISODIC."""
        n = 6
        snaps = [
            _snapshot(
                snapshot_id=str(uuid.uuid4()),
                ingested_at=_BASE_TIME + timedelta(minutes=i),
                memory_type=MemoryType.EPISODIC,
            )
            for i in range(n)
        ]
        for s in snaps:
            index.add_snapshot(s)

        for s in snaps[::2]:  # promote even-indexed snapshots
            index.add_snapshot(_promote(s))

        semantic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.SEMANTIC)
        episodic = index.list_snapshots(_SOURCE_ID, memory_type=MemoryType.EPISODIC)

        assert len(semantic) == n // 2
        assert len(episodic) == n - n // 2

    def test_rapid_writes_do_not_alter_other_snapshots(
        self, index: ShadowIndex
    ) -> None:
        """Rapid writes for snapshot A must not modify co-resident snapshot B."""
        snap_a = _snapshot(
            snapshot_id=str(uuid.uuid4()),
            ingested_at=_BASE_TIME,
            memory_type=MemoryType.EPISODIC,
        )
        snap_b = _snapshot(
            snapshot_id=str(uuid.uuid4()),
            ingested_at=_BASE_TIME + timedelta(minutes=5),
            memory_type=MemoryType.EPISODIC,
        )
        index.add_snapshot(snap_a)
        index.add_snapshot(snap_b)

        # Rapidly promote A
        promoted_a = _promote(snap_a)
        for _ in range(6):
            index.add_snapshot(promoted_a)

        # B must remain untouched
        result_b = index.get_snapshot(snap_b.snapshot_id)
        assert result_b.memory_type is MemoryType.EPISODIC
        assert result_b.content_hash == snap_b.content_hash

    @pytest.mark.parametrize("n_writes", [1, 2, 5, 20])
    def test_any_number_of_identical_writes_leaves_one_row(
        self, index: ShadowIndex, n_writes: int
    ) -> None:
        """Regardless of how many times the same snapshot is written, one row persists."""
        snap = _snapshot(memory_type=MemoryType.SEMANTIC)
        for _ in range(n_writes):
            index.add_snapshot(snap)

        all_snaps = index.list_snapshots(_SOURCE_ID)
        assert [s.snapshot_id for s in all_snaps].count(snap.snapshot_id) == 1
