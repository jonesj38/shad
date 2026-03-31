"""Unit tests for source_to_memory_type mapper."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shad.vault.shadow_index import (
    MemoryType,
    SnapshotEntry,
    SourceEntry,
    ShadowIndex,
    UpdatePolicy,
    source_to_memory_type,
)


@pytest.mark.parametrize("source_type,expected", [
    # Known semantic sources
    ("github", MemoryType.SEMANTIC),
    ("url", MemoryType.SEMANTIC),
    ("folder", MemoryType.SEMANTIC),
    # Known episodic sources
    ("feed", MemoryType.EPISODIC),
    ("sessions", MemoryType.EPISODIC),
])
def test_known_source_types(source_type: str, expected: MemoryType) -> None:
    assert source_to_memory_type(source_type) == expected


def test_unknown_source_type_falls_back_to_semantic() -> None:
    assert source_to_memory_type("unknown") == MemoryType.SEMANTIC


def test_empty_string_falls_back_to_semantic() -> None:
    assert source_to_memory_type("") == MemoryType.SEMANTIC


def test_case_insensitive() -> None:
    assert source_to_memory_type("GITHUB") == MemoryType.SEMANTIC
    assert source_to_memory_type("Feed") == MemoryType.EPISODIC
    assert source_to_memory_type("FEED") == MemoryType.EPISODIC


def test_return_type_is_memory_type_enum() -> None:
    result = source_to_memory_type("github")
    assert isinstance(result, MemoryType)


# --- boundary / edge-case inputs ---

def test_whitespace_only_falls_back_to_semantic() -> None:
    """A whitespace-only string is not a known source type; default routing applies."""
    assert source_to_memory_type("   ") == MemoryType.SEMANTIC


def test_padded_known_type_falls_back_to_semantic() -> None:
    """Leading/trailing whitespace is NOT stripped — ' github ' is unknown, so default applies."""
    assert source_to_memory_type(" github ") == MemoryType.SEMANTIC
    assert source_to_memory_type(" feed ") == MemoryType.SEMANTIC


def test_none_raises_attribute_error() -> None:
    """Passing None is a type violation; the function does not guard against it."""
    with pytest.raises(AttributeError):
        source_to_memory_type(None)  # type: ignore[arg-type]


# --- integration: unknown source_type propagates to SEMANTIC via add_snapshot ---

@pytest.fixture
def shadow_index(tmp_path: Path) -> ShadowIndex:
    idx = ShadowIndex(tmp_path / "index.sqlite")
    yield idx
    idx.close()


def test_index_file_unknown_source_falls_back_to_semantic(shadow_index: ShadowIndex) -> None:
    """add_snapshot with memory_type derived from an unknown source_type stores SEMANTIC.

    This mirrors the indexFile path in TypeScript codebase: when the source tag is
    unrecognised, source_to_memory_type returns SEMANTIC, and that value must survive
    the round-trip through the SQLite index unchanged.
    """
    unknown_source_type = "rss_widget"  # not in _SOURCE_TYPE_TO_MEMORY_TYPE

    source_id = str(uuid.uuid4())
    shadow_index.add_source(SourceEntry(
        source_url="https://example.com/feed",
        source_id=source_id,
        source_type=unknown_source_type,
        update_policy=UpdatePolicy.MANUAL,
    ))

    # Derive memory_type exactly as the ingestion pipeline would
    derived_type = source_to_memory_type(unknown_source_type)
    assert derived_type == MemoryType.SEMANTIC, "precondition: unknown type must map to SEMANTIC"

    snapshot_id = str(uuid.uuid4())
    shadow_index.add_snapshot(SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id=source_id,
        ingested_at=datetime.now(UTC),
        source_revision="abc123",
        entry_paths=["docs/index.md"],
        content_hash="deadbeef",
        memory_type=derived_type,
    ))

    retrieved = shadow_index.get_snapshot(snapshot_id)
    assert retrieved is not None
    assert retrieved.memory_type == MemoryType.SEMANTIC


# --- list_snapshots memory_type filtering ---

class TestListSnapshotsMemoryTypeFilter:
    """Verify that list_snapshots filters by memory_type when provided
    and returns all snapshots when the parameter is omitted.
    """

    @pytest.fixture
    def populated_index(self, tmp_path: Path) -> ShadowIndex:
        """Index pre-loaded with one SEMANTIC and one EPISODIC snapshot under the same source."""
        idx = ShadowIndex(tmp_path / "filter_index.sqlite")

        source_id = "src-filter"
        idx.add_source(SourceEntry(
            source_url="https://example.com/source",
            source_id=source_id,
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))

        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-semantic",
            source_id=source_id,
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_revision="v1",
            entry_paths=["docs/overview.md"],
            content_hash="aaaa",
            memory_type=MemoryType.SEMANTIC,
        ))
        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-episodic",
            source_id=source_id,
            ingested_at=datetime(2026, 1, 2, tzinfo=UTC),
            source_revision="v2",
            entry_paths=["sessions/chat.md"],
            content_hash="bbbb",
            memory_type=MemoryType.EPISODIC,
        ))
        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-none",
            source_id=source_id,
            ingested_at=datetime(2026, 1, 3, tzinfo=UTC),
            source_revision="v3",
            entry_paths=["misc/note.md"],
            content_hash="cccc",
            memory_type=None,
        ))

        yield idx
        idx.close()

    def test_filter_semantic_returns_only_semantic(self, populated_index: ShadowIndex) -> None:
        results = populated_index.list_snapshots("src-filter", memory_type=MemoryType.SEMANTIC)
        assert len(results) == 1
        assert results[0].snapshot_id == "snap-semantic"
        assert results[0].memory_type is MemoryType.SEMANTIC

    def test_filter_episodic_returns_only_episodic(self, populated_index: ShadowIndex) -> None:
        results = populated_index.list_snapshots("src-filter", memory_type=MemoryType.EPISODIC)
        assert len(results) == 1
        assert results[0].snapshot_id == "snap-episodic"
        assert results[0].memory_type is MemoryType.EPISODIC

    def test_no_filter_returns_all_snapshots(self, populated_index: ShadowIndex) -> None:
        """Omitting memory_type must return all three snapshots (SEMANTIC, EPISODIC, None)."""
        results = populated_index.list_snapshots("src-filter")
        assert len(results) == 3

    def test_no_filter_returns_all_memory_types(self, populated_index: ShadowIndex) -> None:
        """Omitting memory_type must return snapshots of every distinct memory_type,
        including SEMANTIC, EPISODIC, and untyped (None) entries."""
        results = populated_index.list_snapshots("src-filter")
        returned_types = {r.memory_type for r in results}
        assert MemoryType.SEMANTIC in returned_types
        assert MemoryType.EPISODIC in returned_types
        assert None in returned_types

    def test_filter_excludes_null_memory_type_entries(self, populated_index: ShadowIndex) -> None:
        """Snapshots stored with memory_type=None must NOT appear when a filter is given."""
        semantic_ids = {r.snapshot_id for r in populated_index.list_snapshots("src-filter", memory_type=MemoryType.SEMANTIC)}
        episodic_ids = {r.snapshot_id for r in populated_index.list_snapshots("src-filter", memory_type=MemoryType.EPISODIC)}
        assert "snap-none" not in semantic_ids
        assert "snap-none" not in episodic_ids

    def test_null_memory_type_snapshots_included_without_filter(self, populated_index: ShadowIndex) -> None:
        """Snapshots stored with memory_type=None must appear when no filter is given."""
        all_ids = {r.snapshot_id for r in populated_index.list_snapshots("src-filter")}
        assert "snap-none" in all_ids

    def test_filter_with_no_matches_returns_empty_list(self, tmp_path: Path) -> None:
        """Filtering a source that has only SEMANTIC snapshots by EPISODIC yields []."""
        idx = ShadowIndex(tmp_path / "empty_filter.sqlite")
        source_id = "src-semantic-only"
        idx.add_source(SourceEntry(
            source_url="https://example.com/sem",
            source_id=source_id,
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-s1",
            source_id=source_id,
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_revision="v1",
            entry_paths=["README.md"],
            content_hash="1234",
            memory_type=MemoryType.SEMANTIC,
        ))

        results = idx.list_snapshots(source_id, memory_type=MemoryType.EPISODIC)
        assert results == []
        idx.close()

    def test_filter_on_source_with_no_snapshots_returns_empty_list(self, tmp_path: Path) -> None:
        """A source that exists but has zero snapshots must return [] for any filter."""
        idx = ShadowIndex(tmp_path / "no_snaps.sqlite")
        source_id = "src-no-snaps"
        idx.add_source(SourceEntry(
            source_url="https://example.com/empty",
            source_id=source_id,
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))

        assert idx.list_snapshots(source_id, memory_type=MemoryType.SEMANTIC) == []
        assert idx.list_snapshots(source_id, memory_type=MemoryType.EPISODIC) == []
        idx.close()

    def test_filter_on_all_null_memory_type_returns_empty_list(self, tmp_path: Path) -> None:
        """When every snapshot in a source has memory_type=None, filtering by any
        concrete type (SEMANTIC or EPISODIC) must return [].

        NULL is never equal to a string value in SQLite's WHERE clause, so the
        filter must not accidentally include un-typed entries.
        """
        idx = ShadowIndex(tmp_path / "null_types.sqlite")
        source_id = "src-null-types"
        idx.add_source(SourceEntry(
            source_url="https://example.com/null",
            source_id=source_id,
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        for i in range(3):
            idx.add_snapshot(SnapshotEntry(
                snapshot_id=f"snap-null-{i}",
                source_id=source_id,
                ingested_at=datetime(2026, 1, i + 1, tzinfo=UTC),
                source_revision=f"v{i}",
                entry_paths=[f"doc{i}.md"],
                content_hash=str(i),
                memory_type=None,
            ))

        assert idx.list_snapshots(source_id, memory_type=MemoryType.SEMANTIC) == []
        assert idx.list_snapshots(source_id, memory_type=MemoryType.EPISODIC) == []
        # Without filter all three are returned
        assert len(idx.list_snapshots(source_id)) == 3
        idx.close()

    def test_filter_unknown_source_id_returns_empty_list(self, tmp_path: Path) -> None:
        """Filtering by memory_type for a source_id that was never registered must
        return [] rather than raise.  This guards against callers passing stale IDs.
        """
        idx = ShadowIndex(tmp_path / "unknown_src.sqlite")

        assert idx.list_snapshots("does-not-exist", memory_type=MemoryType.SEMANTIC) == []
        assert idx.list_snapshots("does-not-exist", memory_type=MemoryType.EPISODIC) == []
        idx.close()

    def test_filter_preserves_ingested_at_desc_ordering(self, tmp_path: Path) -> None:
        """When multiple snapshots share the same memory_type they must be newest-first."""
        idx = ShadowIndex(tmp_path / "order.sqlite")
        source_id = "src-order"
        idx.add_source(SourceEntry(
            source_url="https://example.com/order",
            source_id=source_id,
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
        ))
        for day in [5, 3, 1]:
            idx.add_snapshot(SnapshotEntry(
                snapshot_id=f"snap-{day}",
                source_id=source_id,
                ingested_at=datetime(2026, 1, day, tzinfo=UTC),
                source_revision=f"v{day}",
                entry_paths=[],
                content_hash=str(day),
                memory_type=MemoryType.EPISODIC,
            ))

        results = idx.list_snapshots(source_id, memory_type=MemoryType.EPISODIC)
        assert [r.snapshot_id for r in results] == ["snap-5", "snap-3", "snap-1"]
        idx.close()


# --- list_snapshots multi-type filtering ---

class TestListSnapshotsMultipleMemoryTypes:
    """Verify that list_snapshots(memory_types=[...]) returns all requested types
    and excludes every other type, including NULL-typed entries.
    """

    SOURCE_ID = "src-multi"

    @pytest.fixture
    def populated_index(self, tmp_path: Path) -> ShadowIndex:
        """Index with one SEMANTIC, one EPISODIC, and one untyped snapshot."""
        idx = ShadowIndex(tmp_path / "multi_type.sqlite")
        idx.add_source(SourceEntry(
            source_url="https://example.com/multi",
            source_id=self.SOURCE_ID,
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        ))
        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-sem",
            source_id=self.SOURCE_ID,
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            source_revision="v1",
            entry_paths=["docs/overview.md"],
            content_hash="aaaa",
            memory_type=MemoryType.SEMANTIC,
        ))
        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-ep",
            source_id=self.SOURCE_ID,
            ingested_at=datetime(2026, 1, 2, tzinfo=UTC),
            source_revision="v2",
            entry_paths=["feeds/item.md"],
            content_hash="bbbb",
            memory_type=MemoryType.EPISODIC,
        ))
        idx.add_snapshot(SnapshotEntry(
            snapshot_id="snap-null",
            source_id=self.SOURCE_ID,
            ingested_at=datetime(2026, 1, 3, tzinfo=UTC),
            source_revision="v3",
            entry_paths=["misc/note.md"],
            content_hash="cccc",
            memory_type=None,
        ))
        yield idx
        idx.close()

    def test_both_types_returned_and_null_excluded(self, populated_index: ShadowIndex) -> None:
        """Requesting [SEMANTIC, EPISODIC] returns exactly those two; NULL-typed is excluded."""
        results = populated_index.list_snapshots(
            self.SOURCE_ID,
            memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        )
        ids = {r.snapshot_id for r in results}
        assert ids == {"snap-sem", "snap-ep"}
        assert "snap-null" not in ids

    def test_returned_entries_cover_all_requested_types(self, populated_index: ShadowIndex) -> None:
        """Result set contains at least one entry for every type in the filter list."""
        results = populated_index.list_snapshots(
            self.SOURCE_ID,
            memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        )
        returned_types = {r.memory_type for r in results}
        assert MemoryType.SEMANTIC in returned_types
        assert MemoryType.EPISODIC in returned_types

    def test_no_extra_types_in_results(self, populated_index: ShadowIndex) -> None:
        """Result set contains no types beyond those in the filter list."""
        requested = {MemoryType.SEMANTIC, MemoryType.EPISODIC}
        results = populated_index.list_snapshots(
            self.SOURCE_ID,
            memory_types=list(requested),
        )
        for entry in results:
            assert entry.memory_type in requested, (
                f"Unexpected memory_type {entry.memory_type!r} in results"
            )

    def test_single_type_list_behaves_like_scalar_filter(self, populated_index: ShadowIndex) -> None:
        """memory_types=[X] must return the same rows as memory_type=X."""
        via_list = populated_index.list_snapshots(
            self.SOURCE_ID, memory_types=[MemoryType.SEMANTIC]
        )
        via_scalar = populated_index.list_snapshots(
            self.SOURCE_ID, memory_type=MemoryType.SEMANTIC
        )
        assert {r.snapshot_id for r in via_list} == {r.snapshot_id for r in via_scalar}

    def test_empty_list_returns_no_results(self, populated_index: ShadowIndex) -> None:
        """An empty memory_types list matches nothing (no type is in the empty set)."""
        results = populated_index.list_snapshots(self.SOURCE_ID, memory_types=[])
        assert results == []

    def test_ordering_preserved_newest_first(self, tmp_path: Path) -> None:
        """Multi-type results are ordered newest-first across both memory types."""
        idx = ShadowIndex(tmp_path / "multi_order.sqlite")
        source_id = "src-order-multi"
        idx.add_source(SourceEntry(
            source_url="https://example.com/o",
            source_id=source_id,
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
        ))
        for day, mt in [(3, MemoryType.EPISODIC), (1, MemoryType.SEMANTIC), (2, MemoryType.EPISODIC)]:
            idx.add_snapshot(SnapshotEntry(
                snapshot_id=f"snap-{day}",
                source_id=source_id,
                ingested_at=datetime(2026, 1, day, tzinfo=UTC),
                source_revision=f"v{day}",
                entry_paths=[],
                content_hash=str(day),
                memory_type=mt,
            ))

        results = idx.list_snapshots(
            source_id,
            memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        )
        ingested_dates = [r.ingested_at for r in results]
        assert ingested_dates == sorted(ingested_dates, reverse=True)
        idx.close()


def test_index_file_sessions_source_is_episodic(shadow_index: ShadowIndex) -> None:
    """add_snapshot with memory_type derived from 'sessions' source_type stores EPISODIC.

    Session entries record time-bound interaction history, so they are classified
    as episodic memory.  This mirrors the SessionFileEntry path in the TypeScript
    codebase where indexFile with source='memory' produces episodic-typed entries.
    """
    source_id = str(uuid.uuid4())
    shadow_index.add_source(SourceEntry(
        source_url="local://sessions",
        source_id=source_id,
        source_type="sessions",
        update_policy=UpdatePolicy.MANUAL,
    ))

    derived_type = source_to_memory_type("sessions")
    assert derived_type is MemoryType.EPISODIC, "precondition: 'sessions' must map to EPISODIC"

    snapshot_id = str(uuid.uuid4())
    shadow_index.add_snapshot(SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id=source_id,
        ingested_at=datetime.now(UTC),
        source_revision="session-2026-03-31",
        entry_paths=["sessions/chat-abc.md"],
        content_hash="feedcafe",
        memory_type=derived_type,
    ))

    retrieved = shadow_index.get_snapshot(snapshot_id)
    assert retrieved is not None
    assert retrieved.memory_type is MemoryType.EPISODIC
    assert retrieved.memory_type.value == "episodic_memory"
