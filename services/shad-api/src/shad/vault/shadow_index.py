"""Shadow index for tracking collection sources and snapshots.

Per SPEC.md Section 2.12:
- Maps source_url → latest_snapshot
- Supports pinning specific versions
- Tracks update policies (manual, auto)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from shad.vault.contracts import GroupedNotes, PruneCandidate

T = TypeVar("T")

logger = logging.getLogger(__name__)


class UpdatePolicy(StrEnum):
    """Update policy for sources.

    - MANUAL: Only update when explicitly requested
    - AUTO: Automatically check for updates
    """

    MANUAL = "manual"
    AUTO = "auto"


class MemoryType(StrEnum):
    """Memory classification for indexed entries.

    - WORKING:    Transient / in-progress knowledge (not yet persisted)
    - SEMANTIC:   Factual / concept knowledge (timeless)
    - EPISODIC:   Event / time-bound knowledge (contextual)
    - PROCEDURAL: How-to / skill knowledge (executable patterns)
    """

    WORKING = "working_memory"
    SEMANTIC = "semantic_memory"
    EPISODIC = "episodic_memory"
    PROCEDURAL = "procedural_memory"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> MemoryType:
        """Parse a string into the matching MemoryType variant.

        Accepts the enum value (``"semantic_memory"``) or member name
        (``"SEMANTIC"``), case-insensitively.

        Raises:
            ValueError: if *value* is ``None``, empty, whitespace-only, or
                does not match any variant.
        """
        if not value or not value.strip():
            valid = ", ".join(f'"{m.value}"' for m in cls)
            raise ValueError(f"Invalid MemoryType {value!r}. Valid values: {valid}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        valid = ", ".join(f'"{m.value}"' for m in cls)
        raise ValueError(f"Invalid MemoryType {value!r}. Valid values: {valid}")

    @classmethod
    def all_types(cls) -> list[MemoryType]:
        """Return all MemoryType variants in definition order."""
        return list(cls)


@dataclass
class SourceEntry:
    """Entry for a tracked source.

    Attributes:
        source_url:       Canonical URL of the upstream source (e.g. a GitHub
                          repo URL or web page URL).
        source_id:        Stable, unique identifier assigned at registration time.
        source_type:      Classification string (``"github"``, ``"url"``,
                          ``"folder"``, ``"feed"``).  Determines the default
                          :class:`MemoryType` via :data:`_SOURCE_TYPE_TO_MEMORY_TYPE`.
        update_policy:    Controls whether the source is updated automatically
                          (:attr:`UpdatePolicy.AUTO`) or only on explicit request
                          (:attr:`UpdatePolicy.MANUAL`).
        created_at:       UTC timestamp of when this entry was first registered.
        pinned_snapshot:  If set, the ``snapshot_id`` of the snapshot that should
                          always be served; newer ingestions are stored but ignored
                          for retrieval until the pin is released.
        metadata:         Free-form key/value store for implementation-defined
                          extension data (e.g. branch name, auth tokens, labels).
    """

    source_url: str
    source_id: str
    source_type: str
    update_policy: UpdatePolicy
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    pinned_snapshot: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotEntry:
    """Entry for a snapshot of an ingested source revision.

    Attributes:
        snapshot_id:        Unique identifier for this snapshot (typically a
                            content-addressed hash or UUID).
        source_id:          Foreign key back to the parent :class:`SourceEntry`.
        ingested_at:        UTC timestamp of when the snapshot was ingested.
        source_revision:    Upstream revision identifier (e.g. git SHA, page
                            ETag, or feed ``lastModified`` header).
        entry_paths:        Ordered list of collection-relative paths that belong
                            to this snapshot.
        content_hash:       Hash of the combined snapshot content, used for
                            change detection and deduplication.
        metadata:           Free-form key/value store for implementation-defined
                            extension data.
        memory_type:        Current memory classification (:class:`MemoryType`).
                            ``None`` means unclassified (inherits from source type
                            on first access).
        promotion_history:  Ordered log of :class:`PromotionHistoryEntry` records
                            describing every memory-type transition this snapshot
                            has undergone.
        consolidated_into:  ``snapshot_id`` of the canonical entry that supersedes
                            this snapshot, or ``None`` if the snapshot is not
                            redundant.  Set by
                            :meth:`~shad.vault.shadow_index.ShadowIndex.mark_snapshot_consolidated`;
                            the original row is preserved in the database so that
                            audit trails remain intact.
    """

    snapshot_id: str
    source_id: str
    ingested_at: datetime
    source_revision: str
    entry_paths: list[str]
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType | None = None
    promotion_history: list[PromotionHistoryEntry] = field(default_factory=list)
    consolidated_into: str | None = None


@dataclass
class PromotionHistoryEntry:
    """A single state-transition record for a snapshot's memory lifecycle.

    Fields:
        from_type:  The memory type before the transition, or ``None`` if the
                    snapshot had no prior classification.
        to_type:    The memory type after the transition.
        timestamp:  When the transition occurred (UTC).
        reason:     Human-readable description of why the transition happened
                    (e.g. ``"persist event"`` or ``"session consolidation"``).
    """

    from_type: MemoryType | None
    to_type: MemoryType
    timestamp: datetime
    reason: str


_SOURCE_TYPE_TO_MEMORY_TYPE: dict[str, MemoryType] = {
    "github": MemoryType.SEMANTIC,
    "url": MemoryType.SEMANTIC,
    "folder": MemoryType.SEMANTIC,
    "feed": MemoryType.EPISODIC,
    "sessions": MemoryType.EPISODIC,
}


def source_to_memory_type(source_type: str) -> MemoryType:
    """Map a source type string to its MemoryType classification.

    - github / url / folder → SEMANTIC (timeless reference content)
    - feed → EPISODIC (time-bound events / news)
    - unknown values fall back to SEMANTIC
    """
    return _SOURCE_TYPE_TO_MEMORY_TYPE.get(source_type.lower(), MemoryType.SEMANTIC)


#: Canonical tag values drawn from the project taxonomy.
#: Tags on a :class:`MemoryMetadata` instance should come from this set,
#: though the field accepts any ``str`` to remain forward-compatible.
MEMORY_TAG_TAXONOMY: frozenset[str] = frozenset({
    # Memory lifecycle labels
    "working_memory",
    "episodic_memory",
    "semantic_memory",
    "procedural_memory",
    # Source type labels (mirrors _SOURCE_TYPE_TO_MEMORY_TYPE keys)
    "github",
    "url",
    "folder",
    "feed",
    "sessions",
    # Feature-area labels (mirrors FeatureAudit keys in contracts.py)
    "retrieval",
    "qmd_retriever",
    "filesystem_retriever",
    "gap_detection",
    "sources",
    "shadow_index",
    "cli",
    "sandbox",
    "engine",
    "verification",
})


@dataclass
class MemoryMetadata:
    """Typed metadata attached to a :class:`MemoryRecord`.

    Replaces the untyped ``dict[str, Any]`` previously used for per-record
    annotations so that callers get static-analysis coverage and IDE
    auto-complete on the common fields.

    Fields:
        source:            URL or identifier of the originating source
                           (e.g. a GitHub repo URL or a local folder path).
        confidence:        Confidence score in [0.0, 1.0].  Higher values
                           indicate stronger retrieval or LLM evidence.
                           Defaults to ``1.0`` (fully trusted).
        tags:              Labels from :data:`MEMORY_TAG_TAXONOMY`.  Arbitrary
                           strings are accepted for forward-compatibility.
        promoted_at:       ISO-8601 timestamp recorded when the record was
                           promoted WORKING → EPISODIC, or ``None``.
        promoted_from:     ``MemoryType.value`` string before episodic
                           promotion, or ``None`` if not yet promoted.
        consolidated_at:   ISO-8601 timestamp recorded when the record was
                           consolidated EPISODIC → SEMANTIC, or ``None``.
        consolidated_from: ``MemoryType.value`` string before semantic
                           consolidation, or ``None`` if not yet consolidated.
    """

    source: str
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    promoted_at: str | None = None
    promoted_from: str | None = None
    consolidated_at: str | None = None
    consolidated_from: str | None = None
    embedding: list[float] | None = None


@dataclass
class BiTemporalRecord(Generic[T]):
    """A record with two independent time axes and a generic content payload.

    Bi-temporal modelling tracks *when something happened* separately from
    *when the system learned about it*, enabling point-in-time queries on
    either axis without losing history.

    Type parameter:
        T:           Type of the content payload (e.g. ``str``, ``dict``,
                     or any domain-specific record type).

    Fields:
        event_time:  When the event occurred in the real world (valid time /
                     "business time").  For ingested sources this is the
                     publication or commit timestamp extracted from the source.
        ingested_at: When the record was written into this system (transaction
                     time / "system time").  Always set by the ingestion layer;
                     never back-dated.
        content:     The payload associated with this record.
    """

    event_time: datetime
    ingested_at: datetime
    content: T


def make_bitemporal_record(
    event_time: datetime,
    content: T,
    *,
    ingestion_time: datetime | None = None,
) -> BiTemporalRecord[T]:
    """Factory that creates a BiTemporalRecord with ingested_at set to now (UTC).

    Args:
        event_time:     When the event occurred in the real world.
        content:        The payload for this record (any type).
        ingestion_time: Override for ingested_at.  When provided the value is
                        used verbatim; when omitted ingested_at is stamped to
                        the current wall-clock time in UTC.

    Returns:
        A new ``BiTemporalRecord[T]`` with ingested_at set as described above.
        event_time and content are preserved verbatim.
    """
    return BiTemporalRecord(
        event_time=event_time,
        ingested_at=ingestion_time if ingestion_time is not None else datetime.now(UTC),
        content=content,
    )


@dataclass
class ConflictGroup:
    """A set of snapshot records that share the same event_time and conflict.

    Fields:
        event_time:         The shared timestamp that caused the conflict.
        conflicting_records: All snapshots competing for this event slot.
        resolved_record:    The winning snapshot after resolution, or ``None``
                            if the conflict has not yet been resolved.
    """

    event_time: datetime
    conflicting_records: list[SnapshotEntry]
    resolved_record: SnapshotEntry | None = None


def resolve_latest_ingestion_wins(group: ConflictGroup) -> ConflictGroup:
    """Resolve a ConflictGroup by selecting the most recently ingested record.

    Picks the :class:`SnapshotEntry` with the greatest ``ingested_at``
    timestamp and stores it in ``group.resolved_record``.  When multiple
    records share the same maximum ``ingested_at`` the first one encountered
    is chosen (stable, deterministic for equal timestamps).

    Args:
        group: The conflict group to resolve.  Modified in place *and*
               returned for convenient chaining.

    Returns:
        The same ``group`` with ``resolved_record`` set.
    """
    group.resolved_record = max(group.conflicting_records, key=lambda e: e.ingested_at)
    return group


def filter_groups_with_content_diff(
    records: list[BiTemporalRecord[str]],
) -> list[list[BiTemporalRecord[str]]]:
    """Return groups of records that share an event_time with differing content.

    Groups records by ``event_time``, then discards any group where every
    record carries identical content (i.e. the group contains no real
    divergence).  Only groups with at least two distinct content values are
    returned — these are the true bi-temporal conflicts worth inspecting.

    Args:
        records: Flat list of :class:`BiTemporalRecord` objects to analyse.

    Returns:
        List of groups (each group is a list of records with the same
        ``event_time``), filtered to only those where content differs across
        at least two records in the group.  Groups are returned in ascending
        ``event_time`` order.
    """
    groups: dict[datetime, list[BiTemporalRecord[str]]] = {}
    for record in records:
        groups.setdefault(record.event_time, []).append(record)

    return [
        group
        for _, group in sorted(groups.items())
        if len({r.content for r in group}) > 1
    ]


def serialize_promotion_history(history: list[PromotionHistoryEntry]) -> str:
    """Serialize a promotion history list to a JSON string."""
    return json.dumps([
        {
            "from_type": e.from_type.value if e.from_type is not None else None,
            "to_type": e.to_type.value,
            "timestamp": e.timestamp.isoformat(),
            "reason": e.reason,
        }
        for e in history
    ])


def deserialize_promotion_history(raw: str | None) -> list[PromotionHistoryEntry]:
    """Deserialize a JSON string back to a list of PromotionHistoryEntry objects."""
    if not raw:
        return []
    entries = json.loads(raw)
    return [
        PromotionHistoryEntry(
            from_type=MemoryType(e["from_type"]) if e.get("from_type") is not None else None,
            to_type=MemoryType(e["to_type"]),
            timestamp=datetime.fromisoformat(e["timestamp"]),
            reason=e["reason"],
        )
        for e in entries
    ]


@dataclass
class MemoryRecord:
    """A vault record with identity and memory classification.

    Composes :class:`BiTemporalRecord` for temporal tracking and adds an
    explicit ``record_id`` (identity) and ``memory_type`` classification,
    staying consistent with the bi-temporal dataclass patterns in this module.

    Fields:
        record_id:        Stable unique identifier for this record.
        memory_type:      Classification (WORKING / EPISODIC / SEMANTIC).
        record:           Bi-temporal wrapper carrying event_time, ingested_at,
                          and the string content payload.
        created_at:       When this MemoryRecord was first created (system time, UTC).
        updated_at:       When this MemoryRecord was last modified (system time, UTC).
        last_accessed_at: When this record was last read, or ``None`` if never
                          explicitly accessed.
        metadata:         Typed annotations (source, confidence, tags, and
                          promotion provenance).  ``None`` when no metadata
                          has been attached.
    """

    record_id: str
    memory_type: MemoryType
    record: BiTemporalRecord[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None
    metadata: MemoryMetadata | None = None

    @property
    def event_time(self) -> datetime:
        """When the event occurred in the real world."""
        return self.record.event_time

    @property
    def ingested_at(self) -> datetime:
        """When the record was written into this system."""
        return self.record.ingested_at

    @property
    def content(self) -> str:
        """The payload associated with this record."""
        return self.record.content


@dataclass
class WorkingRecord(MemoryRecord):
    """MemoryRecord locked to WORKING memory type — transient, in-progress knowledge.

    The ``memory_type`` discriminator is fixed to :attr:`MemoryType.WORKING` and
    excluded from ``__init__``; callers cannot supply a different value.

    Type-specific fields:
        ttl_seconds:       Seconds until this record expires automatically; ``None``
                           means no expiry.
        context_window_id: Identifier of the LLM context window this record
                           originated from, or ``None`` if untracked.
    """

    memory_type: MemoryType = field(default=MemoryType.WORKING, init=False)
    ttl_seconds: int | None = None
    context_window_id: str | None = None


@dataclass
class EpisodicRecord(MemoryRecord):
    """MemoryRecord locked to EPISODIC memory type — time-bound, session-scoped events.

    The ``memory_type`` discriminator is fixed to :attr:`MemoryType.EPISODIC` and
    excluded from ``__init__``; callers cannot supply a different value.

    Type-specific fields:
        session_id:  Identifier of the session this event belongs to, or ``None``
                     if not session-scoped.
        decay_rate:  Multiplier applied when computing temporal decay.  ``1.0``
                     (default) uses the global curve unchanged; values < 1.0 slow
                     decay, > 1.0 accelerate it.
    """

    memory_type: MemoryType = field(default=MemoryType.EPISODIC, init=False)
    session_id: str | None = None
    decay_rate: float = 1.0


@dataclass
class SemanticRecord(MemoryRecord):
    """MemoryRecord locked to SEMANTIC memory type — timeless, factual knowledge.

    The ``memory_type`` discriminator is fixed to :attr:`MemoryType.SEMANTIC` and
    excluded from ``__init__``; callers cannot supply a different value.

    Type-specific fields:
        concept_name:      Canonical name of the concept this record describes, or
                           ``None`` if untitled.
        abstraction_level: How abstract this knowledge is: ``0`` = concrete /
                           specific; higher values = broader / cross-domain.
    """

    memory_type: MemoryType = field(default=MemoryType.SEMANTIC, init=False)
    concept_name: str | None = None
    abstraction_level: int = 0


@dataclass
class ProceduralRecord(MemoryRecord):
    """MemoryRecord locked to PROCEDURAL memory type — executable how-to knowledge.

    The ``memory_type`` discriminator is fixed to :attr:`MemoryType.PROCEDURAL` and
    excluded from ``__init__``; callers cannot supply a different value.

    Type-specific fields:
        skill_name:       Name of the skill or procedure encoded here, or ``None``
                          if untitled.
        invocation_count: Number of times this procedure has been applied.
                          Starts at ``0``.
    """

    memory_type: MemoryType = field(default=MemoryType.PROCEDURAL, init=False)
    skill_name: str | None = None
    invocation_count: int = 0


#: Union of all four concrete MemoryRecord subtypes.  Use this annotation for
#: values that can hold any typed memory-record variant.
AnyMemoryRecord = WorkingRecord | EpisodicRecord | SemanticRecord | ProceduralRecord


def memory_record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    """Serialize a MemoryRecord to a plain dict suitable for json.dumps.

    All ``datetime`` values are converted to ISO-8601 strings.
    ``MemoryType`` is stored as its ``.value`` string.
    Optional fields (``last_accessed_at``, ``metadata``) are preserved as
    ``None`` when absent.
    """
    meta: dict[str, Any] | None = None
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
        }
    return {
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


def memory_record_from_dict(data: dict[str, Any]) -> MemoryRecord:
    """Reconstruct a MemoryRecord from a dict produced by ``memory_record_to_dict``.

    Raises:
        KeyError:   if a required key is missing.
        ValueError: if a datetime string or MemoryType value is malformed.
    """
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
        )
    raw_rec = data["record"]
    return MemoryRecord(
        record_id=data["record_id"],
        memory_type=MemoryType(data["memory_type"]),
        record=BiTemporalRecord(
            event_time=datetime.fromisoformat(raw_rec["event_time"]),
            ingested_at=datetime.fromisoformat(raw_rec["ingested_at"]),
            content=raw_rec["content"],
        ),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        last_accessed_at=(
            datetime.fromisoformat(data["last_accessed_at"])
            if data.get("last_accessed_at") is not None
            else None
        ),
        metadata=metadata,
    )


def memory_record_to_json(record: MemoryRecord) -> str:
    """Serialize a MemoryRecord to a JSON string."""
    return json.dumps(memory_record_to_dict(record))


def memory_record_from_json(raw: str) -> MemoryRecord:
    """Deserialize a MemoryRecord from a JSON string."""
    return memory_record_from_dict(json.loads(raw))


def group_snapshots(
    snapshots: list[SnapshotEntry],
    source_type_map: dict[str, str] | None = None,
    min_group_size: int = 1,
) -> list[GroupedNotes]:
    """Cluster related snapshot entries by topic similarity.

    Groups snapshots by ``(source_id, memory_type, topic)``.  Topic is derived
    from the most common top-level path component found in ``entry_paths``.
    Snapshots that share an identical ``content_hash`` across otherwise separate
    groups are merged into a single group — they represent exact-duplicate
    content and are the strongest consolidation candidates.

    Args:
        snapshots:       Flat list of :class:`SnapshotEntry` objects to cluster.
        source_type_map: Optional ``{source_id: source_type}`` mapping used to
                         resolve the :class:`MemoryType` of unclassified snapshots
                         (``memory_type is None``).  Without it, unclassified
                         snapshots default to :attr:`MemoryType.SEMANTIC`.
        min_group_size:  Groups with fewer members than this are excluded.

    Returns:
        List of :class:`~shad.vault.contracts.GroupedNotes`, one per
        ``(source_id, memory_type, topic)`` cluster, sorted by ``oldest_at``
        ascending.
    """
    from collections import Counter, defaultdict
    from pathlib import PurePosixPath

    from shad.vault.contracts import GroupedNotes

    def _resolve_memory_type(snap: SnapshotEntry) -> MemoryType:
        if snap.memory_type is not None:
            return snap.memory_type
        if source_type_map and snap.source_id in source_type_map:
            return source_to_memory_type(source_type_map[snap.source_id])
        return MemoryType.SEMANTIC

    def _topic_from_paths(paths: list[str]) -> str:
        if not paths:
            return "default"
        components: list[str] = []
        for p in paths:
            parts = PurePosixPath(p).parts
            for part in parts:
                if part and part != "/":
                    components.append(part)
                    break
        if not components:
            return "default"
        return Counter(components).most_common(1)[0][0]

    # Map content_hash → canonical group key so that identical-content snapshots
    # from different sources or paths collapse into a single group.
    hash_to_key: dict[str, tuple[str, str, str]] = {}
    buckets: dict[tuple[str, str, str], list[SnapshotEntry]] = defaultdict(list)

    for snap in snapshots:
        mem_type = _resolve_memory_type(snap)
        topic = _topic_from_paths(snap.entry_paths)
        key: tuple[str, str, str] = (snap.source_id, mem_type.value, topic)

        # Redirect to an existing group if another snapshot already claimed
        # this content_hash — exact duplicates must be co-located.
        if snap.content_hash in hash_to_key:
            key = hash_to_key[snap.content_hash]
        else:
            hash_to_key[snap.content_hash] = key

        buckets[key].append(snap)

    groups: list[GroupedNotes] = []
    for (source_id, mem_type_val, topic), snaps in buckets.items():
        if len(snaps) < min_group_size:
            continue
        snaps_sorted = sorted(snaps, key=lambda s: s.ingested_at)
        mem_type = MemoryType(mem_type_val)
        group_id = hashlib.sha256(
            f"{source_id}\x00{mem_type_val}\x00{topic}".encode()
        ).hexdigest()[:16]
        groups.append(
            GroupedNotes(
                group_id=group_id,
                memory_type=mem_type,
                record_ids=tuple(s.snapshot_id for s in snaps_sorted),
                topic=topic,
                oldest_at=snaps_sorted[0].ingested_at,
                newest_at=snaps_sorted[-1].ingested_at,
            )
        )

    return sorted(groups, key=lambda g: g.oldest_at)


class ShadowIndex:
    """SQLite-backed index for tracking sources and snapshots.

    Per SPEC.md Section 2.12:
    - Lightweight database at ~/.shad/index.sqlite
    - Maps source URLs to latest snapshots
    - Supports version pinning
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                source_url TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                update_policy TEXT NOT NULL,
                created_at TEXT NOT NULL,
                pinned_snapshot TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                entry_paths TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT,
                memory_type TEXT,
                promotion_history TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_source_id
            ON snapshots(source_id);

            CREATE INDEX IF NOT EXISTS idx_snapshots_ingested_at
            ON snapshots(ingested_at);
        """)
        self._conn.commit()

        # Migration: add memory_type column if it doesn't exist yet
        try:
            self._conn.execute("ALTER TABLE snapshots ADD COLUMN memory_type TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migration: add promotion_history column if it doesn't exist yet
        try:
            self._conn.execute("ALTER TABLE snapshots ADD COLUMN promotion_history TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migration: add consolidated_into column if it doesn't exist yet
        try:
            self._conn.execute("ALTER TABLE snapshots ADD COLUMN consolidated_into TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def _ensure_conn(self) -> sqlite3.Connection:
        """Ensure database connection is open."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def add_source(self, source: SourceEntry) -> None:
        """Add or update a source entry."""
        conn = self._ensure_conn()

        conn.execute("""
            INSERT OR REPLACE INTO sources
            (source_url, source_id, source_type, update_policy, created_at, pinned_snapshot, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            source.source_url,
            source.source_id,
            source.source_type,
            source.update_policy.value,
            source.created_at.isoformat(),
            source.pinned_snapshot,
            json.dumps(source.metadata),
        ))
        conn.commit()

    def get_source(self, source_url: str) -> SourceEntry | None:
        """Get a source by URL."""
        conn = self._ensure_conn()

        row = conn.execute(
            "SELECT * FROM sources WHERE source_url = ?",
            (source_url,)
        ).fetchone()

        if row is None:
            return None

        return SourceEntry(
            source_url=row["source_url"],
            source_id=row["source_id"],
            source_type=row["source_type"],
            update_policy=UpdatePolicy(row["update_policy"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            pinned_snapshot=row["pinned_snapshot"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def list_sources(self) -> list[SourceEntry]:
        """List all tracked sources."""
        conn = self._ensure_conn()

        rows = conn.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()

        return [
            SourceEntry(
                source_url=row["source_url"],
                source_id=row["source_id"],
                source_type=row["source_type"],
                update_policy=UpdatePolicy(row["update_policy"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                pinned_snapshot=row["pinned_snapshot"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    def add_snapshot(self, snapshot: SnapshotEntry) -> None:
        """Add a snapshot entry."""
        conn = self._ensure_conn()

        conn.execute("""
            INSERT OR REPLACE INTO snapshots
            (snapshot_id, source_id, ingested_at, source_revision, entry_paths, content_hash, metadata, memory_type, promotion_history, consolidated_into)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.snapshot_id,
            snapshot.source_id,
            snapshot.ingested_at.isoformat(),
            snapshot.source_revision,
            json.dumps(snapshot.entry_paths),
            snapshot.content_hash,
            json.dumps(snapshot.metadata),
            snapshot.memory_type.value if snapshot.memory_type is not None else None,
            serialize_promotion_history(snapshot.promotion_history),
            snapshot.consolidated_into,
        ))
        conn.commit()

    def get_snapshot(self, snapshot_id: str) -> SnapshotEntry | None:
        """Get a snapshot by ID."""
        conn = self._ensure_conn()

        row = conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        ).fetchone()

        if row is None:
            return None

        return SnapshotEntry(
            snapshot_id=row["snapshot_id"],
            source_id=row["source_id"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            source_revision=row["source_revision"],
            entry_paths=json.loads(row["entry_paths"]),
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            memory_type=MemoryType(row["memory_type"]) if row["memory_type"] else None,
            promotion_history=deserialize_promotion_history(row["promotion_history"]),
            consolidated_into=row["consolidated_into"],
        )

    def get_latest_snapshot(self, source_id: str) -> SnapshotEntry | None:
        """Get the latest snapshot for a source.

        If the source is pinned, returns the pinned version instead.
        """
        conn = self._ensure_conn()

        # Check if source is pinned
        source_row = conn.execute(
            "SELECT pinned_snapshot FROM sources WHERE source_id = ?",
            (source_id,)
        ).fetchone()

        if source_row and source_row["pinned_snapshot"]:
            return self.get_snapshot(source_row["pinned_snapshot"])

        # Get most recent snapshot
        row = conn.execute("""
            SELECT * FROM snapshots
            WHERE source_id = ?
            ORDER BY ingested_at DESC
            LIMIT 1
        """, (source_id,)).fetchone()

        if row is None:
            return None

        return SnapshotEntry(
            snapshot_id=row["snapshot_id"],
            source_id=row["source_id"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            source_revision=row["source_revision"],
            entry_paths=json.loads(row["entry_paths"]),
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            memory_type=MemoryType(row["memory_type"]) if row["memory_type"] else None,
            promotion_history=deserialize_promotion_history(row["promotion_history"]),
            consolidated_into=row["consolidated_into"],
        )

    def list_snapshots(
        self,
        source_id: str,
        memory_type: MemoryType | None = None,
        memory_types: list[MemoryType] | None = None,
    ) -> list[SnapshotEntry]:
        """List snapshots for a source, optionally filtered by memory_type(s).

        - *memory_types* (list): return only snapshots whose memory_type is in
          the list; the ``None``-typed entries are excluded.
        - *memory_type* (single): backward-compatible shorthand, equivalent to
          ``memory_types=[memory_type]``.
        - When both are omitted all snapshots for the source are returned
          regardless of their memory_type value.
        """
        conn = self._ensure_conn()

        # Merge both filter params into a canonical list.
        effective_types: list[MemoryType] | None = None
        if memory_types is not None:
            effective_types = list(memory_types)
        elif memory_type is not None:
            effective_types = [memory_type]

        if effective_types is not None:
            placeholders = ",".join("?" * len(effective_types))
            rows = conn.execute(
                f"SELECT * FROM snapshots WHERE source_id = ? AND memory_type IN ({placeholders}) ORDER BY ingested_at DESC",
                (source_id, *[t.value for t in effective_types]),
            ).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM snapshots
                WHERE source_id = ?
                ORDER BY ingested_at DESC
            """, (source_id,)).fetchall()

        return [
            SnapshotEntry(
                snapshot_id=row["snapshot_id"],
                source_id=row["source_id"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
                source_revision=row["source_revision"],
                entry_paths=json.loads(row["entry_paths"]),
                content_hash=row["content_hash"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                memory_type=MemoryType(row["memory_type"]) if row["memory_type"] else None,
                promotion_history=deserialize_promotion_history(row["promotion_history"]),
            )
            for row in rows
        ]

    def episodic_count(self, source_id: str, since: datetime) -> int:
        """Count episodic snapshots for a source ingested after *since*.

        Intended to measure how many new episodic events have accumulated
        since the last consolidation run.

        Args:
            source_id: The source to query.
            since:     Lower bound (exclusive) — typically the timestamp of
                       the last consolidation for this source.

        Returns:
            Number of snapshots with memory_type=EPISODIC whose ingested_at
            is strictly greater than *since*.
        """
        conn = self._ensure_conn()
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM snapshots
            WHERE source_id = ?
              AND memory_type = ?
              AND ingested_at > ?
            """,
            (source_id, MemoryType.EPISODIC.value, since.isoformat()),
        ).fetchone()
        return int(row["cnt"])

    def latest_episodic_ingested_at(self, source_id: str, since: datetime) -> datetime | None:
        """Return the latest ingested_at among episodic snapshots after *since*.

        Used to advance the consolidation watermark past all entries that
        triggered a consolidation run, even when those entries carry synthetic
        future timestamps (e.g. in tests).

        Args:
            source_id: The source to query.
            since:     Lower bound (exclusive), matching :meth:`episodic_count`.

        Returns:
            The maximum ``ingested_at`` as a timezone-aware datetime, or
            ``None`` if there are no matching snapshots.
        """
        conn = self._ensure_conn()
        row = conn.execute(
            """
            SELECT MAX(ingested_at) AS latest
            FROM snapshots
            WHERE source_id = ?
              AND memory_type = ?
              AND ingested_at > ?
            """,
            (source_id, MemoryType.EPISODIC.value, since.isoformat()),
        ).fetchone()
        raw = row["latest"] if row else None
        if raw is None:
            return None
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    def pin_snapshot(self, source_url: str, snapshot_id: str) -> None:
        """Pin a source to a specific snapshot version."""
        conn = self._ensure_conn()

        conn.execute("""
            UPDATE sources
            SET pinned_snapshot = ?
            WHERE source_url = ?
        """, (snapshot_id, source_url))
        conn.commit()

    def unpin_snapshot(self, source_url: str) -> None:
        """Remove pin from a source."""
        conn = self._ensure_conn()

        conn.execute("""
            UPDATE sources
            SET pinned_snapshot = NULL
            WHERE source_url = ?
        """, (source_url,))
        conn.commit()

    def promote_to_episodic(
        self,
        snapshot_id: str,
        *,
        persist_time: datetime | None = None,
    ) -> SnapshotEntry:
        """Promote a WORKING snapshot to EPISODIC on a persist event.

        The transition represents the moment a transient working-memory record is
        committed as a durable episodic event.  Two fields are updated in place:

        - ``memory_type`` → :attr:`MemoryType.EPISODIC`
        - ``ingested_at``  → *persist_time* (or ``datetime.now(UTC)`` when omitted)

        Provenance is appended to ``metadata``:

        - ``promoted_at``   – ISO-8601 timestamp of the persist event.
        - ``promoted_from`` – The previous memory type value (``"working_memory"``).

        Args:
            snapshot_id:  The snapshot to promote.
            persist_time: When the persist event occurred.  Defaults to now (UTC).

        Returns:
            The updated :class:`SnapshotEntry` re-read from the database.

        Raises:
            KeyError:   If no snapshot with *snapshot_id* exists.
            ValueError: If the snapshot is not in :attr:`MemoryType.WORKING` state.
        """
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise KeyError(f"Snapshot not found: {snapshot_id!r}")
        if snapshot.memory_type != MemoryType.WORKING:
            raise ValueError(
                f"Cannot promote snapshot {snapshot_id!r}: "
                f"expected {MemoryType.WORKING.value!r}, "
                f"got {snapshot.memory_type!r}"
            )

        promoted_at = persist_time if persist_time is not None else datetime.now(UTC)
        updated_metadata = {
            **snapshot.metadata,
            "promoted_at": promoted_at.isoformat(),
            "promoted_from": MemoryType.WORKING.value,
        }
        updated_history = snapshot.promotion_history + [
            PromotionHistoryEntry(
                from_type=MemoryType.WORKING,
                to_type=MemoryType.EPISODIC,
                timestamp=promoted_at,
                reason="persist event",
            )
        ]

        conn = self._ensure_conn()
        conn.execute(
            """
            UPDATE snapshots
               SET memory_type       = ?,
                   ingested_at       = ?,
                   metadata          = ?,
                   promotion_history = ?
             WHERE snapshot_id       = ?
            """,
            (
                MemoryType.EPISODIC.value,
                promoted_at.isoformat(),
                json.dumps(updated_metadata),
                serialize_promotion_history(updated_history),
                snapshot_id,
            ),
        )
        conn.commit()

        result = self.get_snapshot(snapshot_id)
        assert result is not None  # guaranteed: we just wrote it
        return result

    def promote_to_semantic(
        self,
        snapshot_id: str,
        *,
        consolidation_time: datetime | None = None,
    ) -> SnapshotEntry:
        """Promote an EPISODIC snapshot to SEMANTIC at a session boundary.

        The transition represents the moment time-bound episodic knowledge is
        consolidated into durable, timeless semantic memory.  Two fields are
        updated in place:

        - ``memory_type`` → :attr:`MemoryType.SEMANTIC`
        - ``ingested_at``  → *consolidation_time* (or ``datetime.now(UTC)`` when omitted)

        Provenance is appended to ``metadata``:

        - ``consolidated_at``   – ISO-8601 timestamp of the consolidation event.
        - ``consolidated_from`` – The previous memory type value (``"episodic_memory"``).

        All metadata accumulated during the episodic phase is preserved.

        Args:
            snapshot_id:        The snapshot to promote.
            consolidation_time: When the session boundary / consolidation event
                                occurred.  Defaults to now (UTC).

        Returns:
            The updated :class:`SnapshotEntry` re-read from the database.

        Raises:
            KeyError:   If no snapshot with *snapshot_id* exists.
            ValueError: If the snapshot is not in :attr:`MemoryType.EPISODIC` state.
        """
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise KeyError(f"Snapshot not found: {snapshot_id!r}")
        if snapshot.memory_type != MemoryType.EPISODIC:
            raise ValueError(
                f"Cannot promote snapshot {snapshot_id!r} to semantic: "
                f"expected {MemoryType.EPISODIC.value!r}, "
                f"got {snapshot.memory_type!r}"
            )

        consolidated_at = consolidation_time if consolidation_time is not None else datetime.now(UTC)
        updated_metadata = {
            **snapshot.metadata,
            "consolidated_at": consolidated_at.isoformat(),
            "consolidated_from": MemoryType.EPISODIC.value,
        }
        updated_history = snapshot.promotion_history + [
            PromotionHistoryEntry(
                from_type=MemoryType.EPISODIC,
                to_type=MemoryType.SEMANTIC,
                timestamp=consolidated_at,
                reason="session consolidation",
            )
        ]

        conn = self._ensure_conn()
        conn.execute(
            """
            UPDATE snapshots
               SET memory_type       = ?,
                   ingested_at       = ?,
                   metadata          = ?,
                   promotion_history = ?
             WHERE snapshot_id       = ?
            """,
            (
                MemoryType.SEMANTIC.value,
                consolidated_at.isoformat(),
                json.dumps(updated_metadata),
                serialize_promotion_history(updated_history),
                snapshot_id,
            ),
        )
        conn.commit()

        result = self.get_snapshot(snapshot_id)
        assert result is not None  # guaranteed: we just wrote it
        return result

    def mark_snapshot_consolidated(
        self,
        snapshot_id: str,
        canonical_id: str,
    ) -> bool:
        """Mark a snapshot as redundant by recording the canonical entry it was merged into.

        The row is **preserved** in the database; only the ``consolidated_into``
        column is updated.  This keeps the full audit trail intact while
        signalling to retrieval code that the snapshot should be skipped in
        favour of *canonical_id*.

        Args:
            snapshot_id:  The redundant snapshot to mark.
            canonical_id: ``snapshot_id`` of the canonical entry that supersedes it.

        Returns:
            ``True`` if the row was found and updated; ``False`` if no matching
            row existed.
        """
        conn = self._ensure_conn()
        cursor = conn.execute(
            "UPDATE snapshots SET consolidated_into = ? WHERE snapshot_id = ?",
            (canonical_id, snapshot_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def mark_redundant_snapshots(
        self,
        candidates: list[PruneCandidate],
        *,
        now: datetime | None = None,
    ) -> int:
        """Soft-prune redundant snapshots by marking them with ``consolidated_into``.

        Unlike :meth:`prune_snapshots`, this method **never deletes rows**.
        Instead it calls :meth:`mark_snapshot_consolidated` for each eligible
        candidate whose ``superseded_by`` field identifies the canonical entry.
        Candidates without a ``superseded_by`` value are skipped — there is no
        canonical target to point at.

        Eligibility rules mirror :meth:`prune_snapshots`:

        1. ``candidate.eligible_at <= now``
        2. ``candidate.snapshot_id`` is not pinned.
        3. ``candidate.superseded_by`` is not ``None``.

        Args:
            candidates: Prune candidates, typically produced by
                        :func:`~shad.vault.pruning.identify_snapshot_prune_candidates`
                        or :meth:`~shad.vault.pruning.RedundancyPruner.find_redundant`.
            now:        Reference time for eligibility.  Defaults to
                        ``datetime.now(UTC)``.

        Returns:
            Number of snapshots actually marked as consolidated.
        """
        if not candidates:
            return 0

        reference = now if now is not None else datetime.now(UTC)

        pinned: set[str] = {
            s.pinned_snapshot
            for s in self.list_sources()
            if s.pinned_snapshot is not None
        }

        marked = 0
        for candidate in candidates:
            if candidate.superseded_by is None:
                continue
            if candidate.snapshot_id in pinned:
                continue
            if candidate.eligible_at > reference:
                continue
            if self.mark_snapshot_consolidated(candidate.snapshot_id, candidate.superseded_by):
                marked += 1
        return marked

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot from the index by ID.

        Does not check for pins — callers are responsible for skipping pinned
        snapshots before calling this method.

        Args:
            snapshot_id: The snapshot to remove.

        Returns:
            ``True`` if a row was deleted; ``False`` if no matching row existed.
        """
        conn = self._ensure_conn()
        cursor = conn.execute(
            "DELETE FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def prune_snapshots(
        self,
        candidates: list[PruneCandidate],
        *,
        now: datetime | None = None,
    ) -> int:
        """Remove eligible prune candidates from the index.

        A candidate is deleted when **both** conditions hold:

        1. ``candidate.eligible_at <= now`` — the retention window has elapsed.
        2. ``candidate.snapshot_id`` is not the pinned snapshot for any source.

        Candidates that fail either condition are silently skipped.

        Args:
            candidates: List produced by
                        :func:`~shad.vault.pruning.identify_snapshot_prune_candidates`.
            now:        Reference time for eligibility.  Defaults to
                        ``datetime.now(UTC)``.

        Returns:
            Number of snapshots actually deleted.
        """
        if not candidates:
            return 0

        reference = now if now is not None else datetime.now(UTC)

        pinned: set[str] = {
            s.pinned_snapshot
            for s in self.list_sources()
            if s.pinned_snapshot is not None
        }

        deleted = 0
        for candidate in candidates:
            if candidate.snapshot_id in pinned:
                continue
            if candidate.eligible_at > reference:
                continue
            if self.delete_snapshot(candidate.snapshot_id):
                deleted += 1
        return deleted

    def group_snapshots(
        self,
        source_ids: list[str] | None = None,
        memory_type: MemoryType | None = None,
        min_group_size: int = 1,
    ) -> list[GroupedNotes]:
        """Group all tracked snapshots into topic-similarity clusters.

        Fetches snapshots from the index (optionally restricted by source or
        memory type), builds a source-type map for resolving unclassified
        snapshots, then delegates to the module-level :func:`group_snapshots`
        function for clustering.

        Args:
            source_ids:     Restrict to these source IDs.  ``None`` queries all
                            sources in the index.
            memory_type:    Pre-filter snapshots to this memory type before
                            grouping.  ``None`` includes all types.
            min_group_size: Passed through to :func:`group_snapshots`.

        Returns:
            List of :class:`~shad.vault.contracts.GroupedNotes` sorted by
            ``oldest_at`` ascending; see :func:`group_snapshots` for details.
        """
        conn = self._ensure_conn()
        sources = self.list_sources()
        source_type_map = {s.source_id: s.source_type for s in sources}

        if source_ids is not None:
            all_snaps: list[SnapshotEntry] = []
            for sid in source_ids:
                all_snaps.extend(self.list_snapshots(sid, memory_type=memory_type))
        else:
            if memory_type is not None:
                rows = conn.execute(
                    "SELECT * FROM snapshots WHERE memory_type = ? ORDER BY ingested_at DESC",
                    (memory_type.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM snapshots ORDER BY ingested_at DESC"
                ).fetchall()
            all_snaps = [
                SnapshotEntry(
                    snapshot_id=row["snapshot_id"],
                    source_id=row["source_id"],
                    ingested_at=datetime.fromisoformat(row["ingested_at"]),
                    source_revision=row["source_revision"],
                    entry_paths=json.loads(row["entry_paths"]),
                    content_hash=row["content_hash"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    memory_type=MemoryType(row["memory_type"]) if row["memory_type"] else None,
                    promotion_history=deserialize_promotion_history(row["promotion_history"]),
                )
                for row in rows
            ]

        return group_snapshots(all_snaps, source_type_map=source_type_map, min_group_size=min_group_size)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
