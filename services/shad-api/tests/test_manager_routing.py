"""Tests for MemoryStoreRegistry routing via manager.py.

Covers:
  1. source_to_memory_type — correct MemoryType for each known source type.
  2. MemoryStoreRegistry.backend_for — returns the right backend per tier.
  3. store / retrieve round-trips for all four concrete record subtypes.
  4. move — cross-tier record migration; source removed, destination stored.
  5. Unknown / None memory_type falls back to SEMANTIC backend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shad.vault.manager import (
    EpisodicMemoryBackend,
    MemoryStoreRegistry,
    ProceduralMemoryBackend,
    SemanticMemoryBackend,
    WorkingMemoryBackend,
)
from shad.vault.shadow_index import (
    BiTemporalRecord,
    EpisodicRecord,
    MemoryType,
    ProceduralRecord,
    SemanticRecord,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
    WorkingRecord,
    source_to_memory_type,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 31, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _btr(content: str = "test content") -> BiTemporalRecord[str]:
    return BiTemporalRecord(event_time=_NOW, ingested_at=_NOW, content=content)


def _rid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Record fixtures — one per concrete subtype
# ---------------------------------------------------------------------------


@pytest.fixture
def working_record() -> WorkingRecord:
    return WorkingRecord(
        record_id=_rid(),
        record=_btr("working memory payload"),
        ttl_seconds=300,
        context_window_id="ctx-abc",
    )


@pytest.fixture
def episodic_record() -> EpisodicRecord:
    return EpisodicRecord(
        record_id=_rid(),
        record=_btr("episodic memory payload"),
        session_id="session-001",
        decay_rate=0.9,
    )


@pytest.fixture
def semantic_record() -> SemanticRecord:
    return SemanticRecord(
        record_id=_rid(),
        record=_btr("semantic memory payload"),
        concept_name="dependency-injection",
        abstraction_level=2,
    )


@pytest.fixture
def procedural_record() -> ProceduralRecord:
    return ProceduralRecord(
        record_id=_rid(),
        record=_btr("procedural memory payload"),
        skill_name="auth-flow",
        invocation_count=5,
    )


# ---------------------------------------------------------------------------
# Registry fixture backed by a temp directory
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> MemoryStoreRegistry:
    return MemoryStoreRegistry(storage_path=tmp_path)


# ---------------------------------------------------------------------------
# ShadowIndex fixture (from conftest pattern)
# ---------------------------------------------------------------------------

_SOURCE_ID = "test-src-001"
_SOURCE_URL = "github://test-org/test-repo"


@pytest.fixture
def shadow_index(tmp_path: Path) -> ShadowIndex:
    index = ShadowIndex(tmp_path / "test.sqlite")
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="github",
            update_policy=UpdatePolicy.MANUAL,
        )
    )
    return index


@pytest.fixture
def working_snapshot(shadow_index: ShadowIndex) -> SnapshotEntry:
    snapshot = SnapshotEntry(
        snapshot_id=str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=_NOW,
        memory_type=MemoryType.WORKING,
    )
    shadow_index.add_snapshot(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# 1. source_to_memory_type
# ---------------------------------------------------------------------------


class TestSourceToMemoryType:
    def test_github_maps_to_semantic(self) -> None:
        assert source_to_memory_type("github") == MemoryType.SEMANTIC

    def test_url_maps_to_semantic(self) -> None:
        assert source_to_memory_type("url") == MemoryType.SEMANTIC

    def test_folder_maps_to_semantic(self) -> None:
        assert source_to_memory_type("folder") == MemoryType.SEMANTIC

    def test_feed_maps_to_episodic(self) -> None:
        assert source_to_memory_type("feed") == MemoryType.EPISODIC

    def test_sessions_maps_to_episodic(self) -> None:
        assert source_to_memory_type("sessions") == MemoryType.EPISODIC

    def test_unknown_falls_back_to_semantic(self) -> None:
        assert source_to_memory_type("unknown-type") == MemoryType.SEMANTIC

    def test_case_insensitive(self) -> None:
        assert source_to_memory_type("GITHUB") == MemoryType.SEMANTIC
        assert source_to_memory_type("FEED") == MemoryType.EPISODIC


# ---------------------------------------------------------------------------
# 2. MemoryStoreRegistry.backend_for
# ---------------------------------------------------------------------------


class TestRegistryBackendFor:
    def test_working_backend(self, registry: MemoryStoreRegistry) -> None:
        assert isinstance(registry.backend_for(MemoryType.WORKING), WorkingMemoryBackend)

    def test_episodic_backend(self, registry: MemoryStoreRegistry) -> None:
        assert isinstance(registry.backend_for(MemoryType.EPISODIC), EpisodicMemoryBackend)

    def test_semantic_backend(self, registry: MemoryStoreRegistry) -> None:
        assert isinstance(registry.backend_for(MemoryType.SEMANTIC), SemanticMemoryBackend)

    def test_procedural_backend(self, registry: MemoryStoreRegistry) -> None:
        assert isinstance(registry.backend_for(MemoryType.PROCEDURAL), ProceduralMemoryBackend)

    def test_none_falls_back_to_semantic(self, registry: MemoryStoreRegistry) -> None:
        assert isinstance(registry.backend_for(None), SemanticMemoryBackend)


# ---------------------------------------------------------------------------
# 3. store / retrieve round-trips
# ---------------------------------------------------------------------------


class TestRegistryStoreRetrieve:
    def test_working_round_trip(
        self, registry: MemoryStoreRegistry, working_record: WorkingRecord
    ) -> None:
        rid = registry.store(working_record)
        result = registry.retrieve(rid, MemoryType.WORKING)
        assert result is not None
        assert result.record_id == rid
        assert isinstance(result, WorkingRecord)

    def test_episodic_round_trip(
        self, registry: MemoryStoreRegistry, episodic_record: EpisodicRecord
    ) -> None:
        rid = registry.store(episodic_record)
        result = registry.retrieve(rid, MemoryType.EPISODIC)
        assert result is not None
        assert isinstance(result, EpisodicRecord)
        assert result.session_id == episodic_record.session_id

    def test_semantic_round_trip(
        self, registry: MemoryStoreRegistry, semantic_record: SemanticRecord
    ) -> None:
        rid = registry.store(semantic_record)
        result = registry.retrieve(rid, MemoryType.SEMANTIC)
        assert result is not None
        assert isinstance(result, SemanticRecord)
        assert result.concept_name == semantic_record.concept_name

    def test_procedural_round_trip(
        self, registry: MemoryStoreRegistry, procedural_record: ProceduralRecord
    ) -> None:
        rid = registry.store(procedural_record)
        result = registry.retrieve(rid, MemoryType.PROCEDURAL)
        assert result is not None
        assert isinstance(result, ProceduralRecord)
        assert result.skill_name == procedural_record.skill_name

    def test_retrieve_missing_returns_none(
        self, registry: MemoryStoreRegistry
    ) -> None:
        assert registry.retrieve("does-not-exist", MemoryType.SEMANTIC) is None

    def test_retrieve_from_wrong_tier_returns_none(
        self,
        registry: MemoryStoreRegistry,
        episodic_record: EpisodicRecord,
    ) -> None:
        registry.store(episodic_record)
        # stored in EPISODIC; SEMANTIC backend should not find it
        result = registry.retrieve(episodic_record.record_id, MemoryType.SEMANTIC)
        assert result is None


# ---------------------------------------------------------------------------
# 4. move — cross-tier migration
# ---------------------------------------------------------------------------


class TestRegistryMove:
    def test_working_to_episodic(
        self, registry: MemoryStoreRegistry, working_record: WorkingRecord
    ) -> None:
        registry.store(working_record)
        migrated = registry.move(
            working_record.record_id, MemoryType.WORKING, MemoryType.EPISODIC
        )
        assert isinstance(migrated, EpisodicRecord)
        # source removed
        assert registry.retrieve(working_record.record_id, MemoryType.WORKING) is None
        # destination stored
        assert registry.retrieve(working_record.record_id, MemoryType.EPISODIC) is not None

    def test_episodic_to_semantic(
        self, registry: MemoryStoreRegistry, episodic_record: EpisodicRecord
    ) -> None:
        registry.store(episodic_record)
        migrated = registry.move(
            episodic_record.record_id, MemoryType.EPISODIC, MemoryType.SEMANTIC
        )
        assert isinstance(migrated, SemanticRecord)
        assert registry.retrieve(episodic_record.record_id, MemoryType.EPISODIC) is None
        assert registry.retrieve(episodic_record.record_id, MemoryType.SEMANTIC) is not None

    def test_move_same_type_raises(
        self, registry: MemoryStoreRegistry, semantic_record: SemanticRecord
    ) -> None:
        registry.store(semantic_record)
        with pytest.raises(ValueError, match="must differ"):
            registry.move(
                semantic_record.record_id, MemoryType.SEMANTIC, MemoryType.SEMANTIC
            )

    def test_move_missing_record_raises(
        self, registry: MemoryStoreRegistry
    ) -> None:
        with pytest.raises(KeyError):
            registry.move("ghost-id", MemoryType.WORKING, MemoryType.EPISODIC)


# ---------------------------------------------------------------------------
# 5. list_all and delete
# ---------------------------------------------------------------------------


class TestRegistryListDelete:
    def test_list_all_returns_stored(
        self,
        registry: MemoryStoreRegistry,
        semantic_record: SemanticRecord,
    ) -> None:
        registry.store(semantic_record)
        items = registry.list_all(MemoryType.SEMANTIC)
        ids = [r.record_id for r in items]
        assert semantic_record.record_id in ids

    def test_delete_removes_record(
        self,
        registry: MemoryStoreRegistry,
        episodic_record: EpisodicRecord,
    ) -> None:
        registry.store(episodic_record)
        deleted = registry.delete(episodic_record.record_id, MemoryType.EPISODIC)
        assert deleted is True
        assert registry.retrieve(episodic_record.record_id, MemoryType.EPISODIC) is None

    def test_delete_missing_returns_false(
        self, registry: MemoryStoreRegistry
    ) -> None:
        assert registry.delete("nonexistent", MemoryType.PROCEDURAL) is False
