"""Shadow index for tracking vault sources and snapshots.

Per SPEC.md Section 2.12:
- Maps source_url → latest_snapshot
- Supports pinning specific versions
- Tracks update policies (manual, auto)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UpdatePolicy(str, Enum):
    """Update policy for sources.

    - MANUAL: Only update when explicitly requested
    - AUTO: Automatically check for updates
    """

    MANUAL = "manual"
    AUTO = "auto"


@dataclass
class SourceEntry:
    """Entry for a tracked source."""

    source_url: str
    source_id: str
    source_type: str
    update_policy: UpdatePolicy
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    pinned_snapshot: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotEntry:
    """Entry for a snapshot."""

    snapshot_id: str
    source_id: str
    ingested_at: datetime
    source_revision: str
    entry_paths: list[str]
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_source_id
            ON snapshots(source_id);

            CREATE INDEX IF NOT EXISTS idx_snapshots_ingested_at
            ON snapshots(ingested_at);
        """)
        self._conn.commit()

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
            (snapshot_id, source_id, ingested_at, source_revision, entry_paths, content_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.snapshot_id,
            snapshot.source_id,
            snapshot.ingested_at.isoformat(),
            snapshot.source_revision,
            json.dumps(snapshot.entry_paths),
            snapshot.content_hash,
            json.dumps(snapshot.metadata),
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
        )

    def list_snapshots(self, source_id: str) -> list[SnapshotEntry]:
        """List all snapshots for a source."""
        conn = self._ensure_conn()

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
            )
            for row in rows
        ]

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

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
