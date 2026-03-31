"""Vault memory store registry.

Maps each :class:`~shad.vault.shadow_index.MemoryType` variant to a
dedicated :class:`StoreBackend`, providing a single routing layer over the
four memory tiers (WORKING, EPISODIC, SEMANTIC, PROCEDURAL).

Usage::

    from shad.vault.manager import MemoryStoreRegistry
    from shad.vault.shadow_index import MemoryType

    registry = MemoryStoreRegistry()
    registry.store(some_working_record)
    record = registry.retrieve(record_id, MemoryType.WORKING)
    backend = registry.backend_for(MemoryType.SEMANTIC)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from shad.vault.contracts import MemoryRecord as ContractRecord
from shad.vault.shadow_index import (
    AnyMemoryRecord,
    BiTemporalRecord,
    EpisodicRecord,
    MemoryMetadata,
    MemoryType,
    ProceduralRecord,
    SemanticRecord,
    SnapshotEntry,
    WorkingRecord,
)

if TYPE_CHECKING:
    from shad.vault.consolidation import ConsolidationConfig
    from shad.vault.shadow_index import ShadowIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StoreBackend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StoreBackend(Protocol):
    """Minimum interface every memory store backend must implement.

    All four concrete backends (:class:`WorkingMemoryBackend`,
    :class:`EpisodicMemoryBackend`, :class:`SemanticMemoryBackend`,
    :class:`ProceduralMemoryBackend`) satisfy this protocol and can be used
    interchangeably wherever a ``StoreBackend`` is expected.
    """

    def store(self, record: AnyMemoryRecord) -> str:
        """Persist *record* and return its ``record_id``."""
        ...

    def retrieve(self, record_id: str) -> AnyMemoryRecord | None:
        """Return the record with *record_id*, or ``None`` if not found."""
        ...

    def delete(self, record_id: str) -> bool:
        """Remove the record; return ``True`` if it existed."""
        ...

    def list_all(self, limit: int = 100) -> list[AnyMemoryRecord]:
        """Return up to *limit* records, newest first."""
        ...


# ---------------------------------------------------------------------------
# Subclass-aware serialization helpers
# ---------------------------------------------------------------------------


def _record_to_dict(record: AnyMemoryRecord) -> dict:
    """Serialize an AnyMemoryRecord to a JSON-safe dict, preserving subclass fields."""
    meta = None
    if record.metadata is not None:
        m = record.metadata
        meta = {
            "source": m.source,
            "confidence": m.confidence,
            "tags": list(m.tags),
            "promoted_at": m.promoted_at,
            "promoted_from": m.promoted_from,
            "consolidated_at": m.consolidated_at,
            "consolidated_from": m.consolidated_from,
            "embedding": m.embedding,
        }
    base: dict = {
        "record_id": record.record_id,
        "memory_type": record.memory_type.value,
        "record": {
            "event_time": record.record.event_time.isoformat(),
            "ingested_at": record.record.ingested_at.isoformat(),
            "content": record.record.content,
        },
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "last_accessed_at": (
            record.last_accessed_at.isoformat()
            if record.last_accessed_at is not None
            else None
        ),
        "metadata": meta,
    }
    if isinstance(record, WorkingRecord):
        base["ttl_seconds"] = record.ttl_seconds
        base["context_window_id"] = record.context_window_id
    elif isinstance(record, EpisodicRecord):
        base["session_id"] = record.session_id
        base["decay_rate"] = record.decay_rate
    elif isinstance(record, SemanticRecord):
        base["concept_name"] = record.concept_name
        base["abstraction_level"] = record.abstraction_level
    elif isinstance(record, ProceduralRecord):
        base["skill_name"] = record.skill_name
        base["invocation_count"] = record.invocation_count
    return base


def _record_from_dict(data: dict) -> AnyMemoryRecord:
    """Reconstruct an AnyMemoryRecord from a dict produced by :func:`_record_to_dict`."""
    raw_meta = data.get("metadata")
    metadata: MemoryMetadata | None = None
    if raw_meta is not None:
        metadata = MemoryMetadata(
            source=raw_meta["source"],
            confidence=raw_meta["confidence"],
            tags=list(raw_meta.get("tags", [])),
            promoted_at=raw_meta.get("promoted_at"),
            promoted_from=raw_meta.get("promoted_from"),
            consolidated_at=raw_meta.get("consolidated_at"),
            consolidated_from=raw_meta.get("consolidated_from"),
            embedding=raw_meta.get("embedding"),
        )
    btr: BiTemporalRecord[str] = BiTemporalRecord(
        event_time=datetime.fromisoformat(data["record"]["event_time"]),
        ingested_at=datetime.fromisoformat(data["record"]["ingested_at"]),
        content=data["record"]["content"],
    )
    common: dict = dict(
        record_id=data["record_id"],
        record=btr,
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        last_accessed_at=(
            datetime.fromisoformat(data["last_accessed_at"])
            if data.get("last_accessed_at")
            else None
        ),
        metadata=metadata,
    )
    raw_mem_type = data.get("memory_type")
    try:
        mem_type = MemoryType(raw_mem_type)
    except (ValueError, KeyError):
        logger.warning(
            "Unknown or missing memory_type %r; routing to SEMANTIC store",
            raw_mem_type,
        )
        mem_type = MemoryType.SEMANTIC
    if mem_type == MemoryType.WORKING:
        return WorkingRecord(
            **common,
            ttl_seconds=data.get("ttl_seconds"),
            context_window_id=data.get("context_window_id"),
        )
    if mem_type == MemoryType.EPISODIC:
        return EpisodicRecord(
            **common,
            session_id=data.get("session_id"),
            decay_rate=data.get("decay_rate", 1.0),
        )
    if mem_type == MemoryType.SEMANTIC:
        return SemanticRecord(
            **common,
            concept_name=data.get("concept_name"),
            abstraction_level=data.get("abstraction_level", 0),
        )
    # PROCEDURAL
    return ProceduralRecord(
        **common,
        skill_name=data.get("skill_name"),
        invocation_count=data.get("invocation_count", 0),
    )


# ---------------------------------------------------------------------------
# Contract-record conversion helper
# ---------------------------------------------------------------------------


def _contract_to_any(record: ContractRecord) -> AnyMemoryRecord:
    """Convert a flat contracts.MemoryRecord to the matching AnyMemoryRecord subtype.

    Builds a BiTemporalRecord from the flat fields and maps the optional
    embedding + tags into a MemoryMetadata.  The memory_type discriminator
    selects the concrete subclass.
    """
    btr = BiTemporalRecord(
        event_time=record.created_at,
        ingested_at=record.updated_at,
        content=record.content,
    )
    meta = MemoryMetadata(
        source=record.source_refs[0] if record.source_refs else "",
        confidence=float(record.metadata.get("confidence", 1.0)),
        tags=list(record.tags),
        embedding=record.embedding,
    )
    common: dict = dict(
        record_id=record.record_id,
        record=btr,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=meta,
    )
    if record.memory_type == MemoryType.WORKING:
        return WorkingRecord(**common)
    if record.memory_type == MemoryType.EPISODIC:
        return EpisodicRecord(**common)
    if record.memory_type == MemoryType.PROCEDURAL:
        return ProceduralRecord(**common)
    return SemanticRecord(**common)


def _cast_record(source: AnyMemoryRecord, to_type: MemoryType) -> AnyMemoryRecord:
    """Recast *source* to the concrete subtype matching *to_type*.

    Common MemoryRecord fields (record_id, record, created_at, metadata,
    last_accessed_at) are preserved verbatim.  ``updated_at`` is stamped to
    now (UTC) to mark the migration event.  Type-specific fields from the
    source subtype (e.g. ``ttl_seconds`` on WorkingRecord, ``decay_rate`` on
    EpisodicRecord) are dropped; the target subtype receives its own defaults.

    Args:
        source:  The record to recast.
        to_type: The target memory type.

    Returns:
        A new concrete MemoryRecord subtype instance.

    Raises:
        ValueError: If *to_type* is not one of the four recognised MemoryType values.
    """
    common: dict = dict(
        record_id=source.record_id,
        record=source.record,
        created_at=source.created_at,
        updated_at=datetime.now(UTC),
        last_accessed_at=source.last_accessed_at,
        metadata=source.metadata,
    )
    if to_type == MemoryType.WORKING:
        return WorkingRecord(**common)
    if to_type == MemoryType.EPISODIC:
        return EpisodicRecord(**common)
    if to_type == MemoryType.SEMANTIC:
        return SemanticRecord(**common)
    if to_type == MemoryType.PROCEDURAL:
        return ProceduralRecord(**common)
    raise ValueError(f"Unsupported target memory type: {to_type!r}")


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


class WorkingMemoryBackend:
    """In-process volatile store for WORKING memory records.

    Records are held in a plain dict and evaporate at process exit.
    If a :class:`~shad.vault.shadow_index.WorkingRecord` sets
    ``ttl_seconds``, the record is silently evicted on access once that
    window has elapsed.
    """

    def __init__(self) -> None:
        # record_id → (record, monotonic store timestamp)
        self._store: dict[str, tuple[WorkingRecord, float]] = {}

    def store(self, record: AnyMemoryRecord) -> str:
        if not isinstance(record, WorkingRecord):
            raise TypeError(
                f"WorkingMemoryBackend expects WorkingRecord, got {type(record).__name__}"
            )
        self._store[record.record_id] = (record, time.monotonic())
        return record.record_id

    def retrieve(self, record_id: str) -> AnyMemoryRecord | None:
        entry = self._store.get(record_id)
        if entry is None:
            return None
        record, stored_at = entry
        if record.ttl_seconds is not None and time.monotonic() - stored_at > record.ttl_seconds:
            del self._store[record_id]
            return None
        return record

    def delete(self, record_id: str) -> bool:
        return self._store.pop(record_id, None) is not None

    def list_all(self, limit: int = 100) -> list[AnyMemoryRecord]:
        now = time.monotonic()
        expired = [
            rid
            for rid, (rec, stored_at) in self._store.items()
            if rec.ttl_seconds is not None and now - stored_at > rec.ttl_seconds
        ]
        for rid in expired:
            del self._store[rid]
        live = sorted(self._store.values(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in live[:limit]]


class _FileBackend:
    """Shared JSON-file persistence used by the three durable-tier backends."""

    def __init__(self, store_file: Path) -> None:
        self._store_file = store_file
        self._records: dict[str, AnyMemoryRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._store_file.exists():
            return
        try:
            with self._store_file.open() as fh:
                raw = json.load(fh)
            for item in raw.get("records", []):
                record = _record_from_dict(item)
                self._records[record.record_id] = record
        except Exception as exc:
            logger.warning("Failed to load %s: %s", self._store_file, exc)

    def _save(self) -> None:
        try:
            payload = {
                "records": [_record_to_dict(r) for r in self._records.values()],
                "updated_at": datetime.now(UTC).isoformat(),
            }
            self._store_file.parent.mkdir(parents=True, exist_ok=True)
            with self._store_file.open("w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            logger.error("Failed to save %s: %s", self._store_file, exc)

    def store(self, record: AnyMemoryRecord) -> str:
        self._records[record.record_id] = record
        self._save()
        return record.record_id

    def retrieve(self, record_id: str) -> AnyMemoryRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        if record_id not in self._records:
            return False
        del self._records[record_id]
        self._save()
        return True

    def list_all(self, limit: int = 100) -> list[AnyMemoryRecord]:
        return sorted(
            self._records.values(),
            key=lambda r: r.updated_at,
            reverse=True,
        )[:limit]


class EpisodicMemoryBackend(_FileBackend):
    """File-backed store for EPISODIC memory records (time-bound, session-scoped)."""

    def __init__(self, storage_path: Path) -> None:
        super().__init__(storage_path / "episodic.json")

    def store(self, record: AnyMemoryRecord) -> str:
        if not isinstance(record, EpisodicRecord):
            raise TypeError(
                f"EpisodicMemoryBackend expects EpisodicRecord, got {type(record).__name__}"
            )
        return super().store(record)

    def list_by_session(self, session_id: str) -> list[EpisodicRecord]:
        """Return all records belonging to *session_id*."""
        return [
            r
            for r in self._records.values()
            if isinstance(r, EpisodicRecord) and r.session_id == session_id
        ]


class SemanticMemoryBackend(_FileBackend):
    """File-backed store for SEMANTIC memory records (timeless, factual knowledge)."""

    def __init__(self, storage_path: Path) -> None:
        super().__init__(storage_path / "semantic.json")

    def store(self, record: AnyMemoryRecord) -> str:
        if not isinstance(record, SemanticRecord):
            raise TypeError(
                f"SemanticMemoryBackend expects SemanticRecord, got {type(record).__name__}"
            )
        return super().store(record)

    def find_by_concept(self, concept_name: str) -> list[SemanticRecord]:
        """Return all records with an exact ``concept_name`` match."""
        return [
            r
            for r in self._records.values()
            if isinstance(r, SemanticRecord) and r.concept_name == concept_name
        ]


class ProceduralMemoryBackend(_FileBackend):
    """File-backed store for PROCEDURAL memory records (executable how-to knowledge)."""

    def __init__(self, storage_path: Path) -> None:
        super().__init__(storage_path / "procedural.json")

    def store(self, record: AnyMemoryRecord) -> str:
        if not isinstance(record, ProceduralRecord):
            raise TypeError(
                f"ProceduralMemoryBackend expects ProceduralRecord, got {type(record).__name__}"
            )
        return super().store(record)

    def find_by_skill(self, skill_name: str) -> list[ProceduralRecord]:
        """Return all records with an exact ``skill_name`` match."""
        return [
            r
            for r in self._records.values()
            if isinstance(r, ProceduralRecord) and r.skill_name == skill_name
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MemoryStoreRegistry:
    """Routes memory records to their type-specific storage backend.

    Owns a ``dict[MemoryType, StoreBackend]`` that maps every
    :class:`~shad.vault.shadow_index.MemoryType` variant to exactly one
    backend.  All read/write operations are dispatched through
    :meth:`backend_for` so callers never reference backends directly.

    Backend mapping:

    +-----------------------+-------------------------------+
    | MemoryType            | Backend                       |
    +=======================+===============================+
    | WORKING               | WorkingMemoryBackend          |
    +-----------------------+-------------------------------+
    | EPISODIC              | EpisodicMemoryBackend         |
    +-----------------------+-------------------------------+
    | SEMANTIC              | SemanticMemoryBackend         |
    +-----------------------+-------------------------------+
    | PROCEDURAL            | ProceduralMemoryBackend       |
    +-----------------------+-------------------------------+

    Args:
        storage_path: Root directory for the three file-backed backends
                      (episodic.json, semantic.json, procedural.json are
                      written here).  The WORKING backend is always
                      in-process only and ignores this path.
                      Defaults to ``~/.shad/history/vault``.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        consolidation_config: ConsolidationConfig | None = None,
    ) -> None:
        if storage_path is None:
            from shad.utils.config import get_settings

            storage_path = get_settings().history_path / "vault"

        self._registry: dict[MemoryType, StoreBackend] = {
            MemoryType.WORKING: WorkingMemoryBackend(),
            MemoryType.EPISODIC: EpisodicMemoryBackend(storage_path),
            MemoryType.SEMANTIC: SemanticMemoryBackend(storage_path),
            MemoryType.PROCEDURAL: ProceduralMemoryBackend(storage_path),
        }
        self._consolidation_config = consolidation_config
        self._consolidation_timer: threading.Timer | None = None

    def backend_for(self, memory_type: MemoryType | None) -> StoreBackend:
        """Return the backend registered for *memory_type*.

        Unknown or ``None`` values fall back to the SEMANTIC backend and emit a
        warning, matching the :func:`source_to_memory_type` default.
        """
        if memory_type is None or memory_type not in self._registry:
            logger.warning(
                "Unknown or None memory_type %r; falling back to SEMANTIC store",
                memory_type,
            )
            return self._registry[MemoryType.SEMANTIC]
        return self._registry[memory_type]

    def store(self, record: AnyMemoryRecord) -> str:
        """Persist *record* in the backend that matches its ``memory_type``."""
        return self._registry[record.memory_type].store(record)

    def retrieve(self, record_id: str, memory_type: MemoryType) -> AnyMemoryRecord | None:
        """Retrieve *record_id* from the backend for *memory_type*."""
        return self._registry[memory_type].retrieve(record_id)

    def delete(self, record_id: str, memory_type: MemoryType) -> bool:
        """Delete *record_id* from the backend for *memory_type*."""
        return self._registry[memory_type].delete(record_id)

    def list_all(self, memory_type: MemoryType, limit: int = 100) -> list[AnyMemoryRecord]:
        """List up to *limit* records from the backend for *memory_type*."""
        return self._registry[memory_type].list_all(limit=limit)

    def dispatch_write(self, record: ContractRecord) -> str:
        """Write a flat ContractRecord to its type-matched backend.

        Reads ``record.memory_type``, looks up the correct backend via
        :meth:`backend_for`, converts the flat record to an
        :data:`AnyMemoryRecord`, and delegates to ``backend.store()``.

        Returns the ``record_id`` of the stored record.
        """
        bitemporal = _contract_to_any(record)
        return self.backend_for(record.memory_type).store(bitemporal)

    def dispatch_read(self, record: ContractRecord) -> AnyMemoryRecord | None:
        """Read from the backend matched by ``record.memory_type``.

        Uses ``record.record_id`` as the lookup key.  Returns the stored
        :data:`AnyMemoryRecord`, or ``None`` if not found.
        """
        return self.backend_for(record.memory_type).retrieve(record.record_id)

    def move(self, record_id: str, from_type: MemoryType, to_type: MemoryType) -> AnyMemoryRecord:
        """Migrate *record_id* from the *from_type* backend to the *to_type* backend.

        Retrieves the record from the source backend, recasts it to the target
        concrete subtype via :func:`_cast_record`, stores it in the destination
        backend, then removes it from the source backend.  The source record is
        only deleted after the destination write succeeds, so the system will
        never lose the record if the store step raises.

        Args:
            record_id: Identifies the record to migrate.
            from_type: The backend that currently holds the record.
            to_type:   The backend the record should be moved to.

        Returns:
            The newly stored :data:`AnyMemoryRecord` in the target backend.

        Raises:
            ValueError: If *from_type* == *to_type*.
            KeyError:   If the record is not found in the *from_type* backend.
        """
        if from_type == to_type:
            raise ValueError(
                f"Source and destination types must differ; both are {from_type!r}"
            )
        source = self._registry[from_type].retrieve(record_id)
        if source is None:
            raise KeyError(
                f"Record {record_id!r} not found in {from_type.value!r} backend"
            )
        target = _cast_record(source, to_type)
        self._registry[to_type].store(target)
        self._registry[from_type].delete(record_id)
        logger.debug(
            "Migrated record %s: %s → %s",
            record_id,
            from_type.value,
            to_type.value,
        )
        return target

    def run_update(self, record: AnyMemoryRecord) -> str:
        """Persist *record* and run consolidation threshold checks.

        Stores the record via :meth:`store`, then checks whether the EPISODIC
        backend has accumulated enough entries to trigger a consolidation pass
        (per :attr:`_consolidation_config`).  If the threshold is met and no
        consolidation is already pending, it is scheduled on a daemon thread.

        Args:
            record: The memory record to persist.

        Returns:
            The ``record_id`` of the stored record.
        """
        record_id = self.store(record)
        self._check_consolidation_threshold()
        return record_id

    def _check_consolidation_threshold(self) -> None:
        """Check episodic count against the consolidation threshold.

        Skipped when no :class:`~shad.vault.consolidation.ConsolidationConfig`
        was supplied or when consolidation is disabled.  If
        :meth:`~shad.vault.consolidation.ConsolidationConfig.should_consolidate`
        returns ``True`` and no timer is already running, schedules
        ``_run_consolidation`` on a daemon :class:`threading.Timer` with a
        zero-second delay so it fires on a background thread without blocking
        the caller.
        """
        if self._consolidation_config is None or not self._consolidation_config.enabled:
            return
        if self._consolidation_timer is not None:
            # consolidation already pending — skip duplicate scheduling
            return
        episodic_count = len(self._registry[MemoryType.EPISODIC].list_all(limit=100_000))
        if self._consolidation_config.should_consolidate(episodic_count):
            logger.debug(
                "Consolidation threshold met (%d >= %d); scheduling consolidation pass",
                episodic_count,
                self._consolidation_config.threshold,
            )
            self._consolidation_timer = threading.Timer(0.0, self._run_consolidation)
            self._consolidation_timer.daemon = True
            self._consolidation_timer.start()

    def _run_consolidation(self) -> None:
        """Execute the consolidation pipeline over the current EPISODIC records.

        Called on the background timer thread.  Resets ``_consolidation_timer``
        to ``None`` after completion so the next threshold check can schedule a
        fresh pass when needed.
        """
        try:
            from shad.vault.pipeline import run_pipeline

            records = list(self._registry[MemoryType.EPISODIC].list_all(limit=100_000))
            if records:
                result = run_pipeline(records, consolidation_config=self._consolidation_config)
                logger.debug(
                    "Consolidation pass complete (pipeline run %s): %d EPISODIC records processed",
                    result.run_id,
                    len(records),
                )
        except Exception:
            logger.exception("Consolidation pass failed")
        finally:
            self._consolidation_timer = None

    def dispose(self) -> None:
        """Cancel any pending consolidation timer and release resources.

        Safe to call more than once; subsequent calls are no-ops.  After
        ``dispose()`` returns no background consolidation work will be
        scheduled or running via this registry instance.
        """
        if self._consolidation_timer is not None:
            self._consolidation_timer.cancel()
            self._consolidation_timer = None
            logger.debug("MemoryStoreRegistry: consolidation timer cancelled")

    @property
    def registry(self) -> dict[MemoryType, StoreBackend]:
        """Snapshot of the internal ``{MemoryType: StoreBackend}`` mapping."""
        return dict(self._registry)


# ---------------------------------------------------------------------------
# Promotion orchestrators
# ---------------------------------------------------------------------------


def promote_working_to_episodic(
    record_id: str,
    registry: MemoryStoreRegistry,
    shadow_index: ShadowIndex,
    *,
    persist_time: datetime | None = None,
) -> tuple[AnyMemoryRecord, SnapshotEntry]:
    """Promote a WORKING record to EPISODIC in both the store and the shadow index.

    This is the single entry point for WORKING → EPISODIC transitions.  It:

    1. Moves the record from :class:`WorkingMemoryBackend` to
       :class:`EpisodicMemoryBackend` via :meth:`MemoryStoreRegistry.move`.
    2. Updates the shadow-index database entry to reflect the new
       ``memory_type`` and appends a :class:`~shad.vault.shadow_index.PromotionHistoryEntry`.

    The record migration happens first.  If the shadow-index update raises,
    the record will have been migrated but the index will be stale; callers
    should treat any exception as a signal to retry or roll back.

    Args:
        record_id:    The identifier shared by the record and its snapshot.
        registry:     The :class:`MemoryStoreRegistry` that owns the backends.
        shadow_index: The :class:`~shad.vault.shadow_index.ShadowIndex` tracking
                      snapshot metadata.
        persist_time: When the persist event occurred.  Defaults to now (UTC).

    Returns:
        ``(migrated_record, updated_snapshot)`` — the newly stored
        :data:`AnyMemoryRecord` and the updated
        :class:`~shad.vault.shadow_index.SnapshotEntry`.

    Raises:
        KeyError:   If the record is absent from the WORKING backend or the
                    snapshot is absent from the shadow index.
        ValueError: If the snapshot is not in WORKING state.
    """
    migrated = registry.move(record_id, MemoryType.WORKING, MemoryType.EPISODIC)
    snapshot = shadow_index.promote_to_episodic(record_id, persist_time=persist_time)
    return migrated, snapshot


def promote_episodic_to_semantic(
    record_id: str,
    registry: MemoryStoreRegistry,
    shadow_index: ShadowIndex,
    *,
    consolidation_time: datetime | None = None,
) -> tuple[AnyMemoryRecord, SnapshotEntry]:
    """Promote an EPISODIC record to SEMANTIC in both the store and the shadow index.

    This is the single entry point for EPISODIC → SEMANTIC transitions.  It:

    1. Moves the record from :class:`EpisodicMemoryBackend` to
       :class:`SemanticMemoryBackend` via :meth:`MemoryStoreRegistry.move`.
    2. Updates the shadow-index database entry to reflect the new
       ``memory_type`` and appends a :class:`~shad.vault.shadow_index.PromotionHistoryEntry`.

    The record migration happens first.  If the shadow-index update raises,
    the record will have been migrated but the index will be stale; callers
    should treat any exception as a signal to retry or roll back.

    Args:
        record_id:          The identifier shared by the record and its snapshot.
        registry:           The :class:`MemoryStoreRegistry` that owns the backends.
        shadow_index:       The :class:`~shad.vault.shadow_index.ShadowIndex` tracking
                            snapshot metadata.
        consolidation_time: When the consolidation event occurred.
                            Defaults to now (UTC).

    Returns:
        ``(migrated_record, updated_snapshot)`` — the newly stored
        :data:`AnyMemoryRecord` and the updated
        :class:`~shad.vault.shadow_index.SnapshotEntry`.

    Raises:
        KeyError:   If the record is absent from the EPISODIC backend or the
                    snapshot is absent from the shadow index.
        ValueError: If the snapshot is not in EPISODIC state.
    """
    migrated = registry.move(record_id, MemoryType.EPISODIC, MemoryType.SEMANTIC)
    snapshot = shadow_index.promote_to_semantic(record_id, consolidation_time=consolidation_time)
    return migrated, snapshot
