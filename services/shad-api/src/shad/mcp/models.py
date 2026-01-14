"""Data models for MCP client."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NoteType(str, Enum):
    """Types of notes in the vault (per OBSIDIAN_PIVOT.md Section 4.1)."""

    NOTE = "note"  # Derived insights, summaries
    SOURCE = "source"  # Raw text, transcripts, PDFs
    TASK = "task"  # Triggers n8n workflows
    REPORT = "report"  # Run output artifacts
    LEARNING = "learning"  # Extracted learnings
    UNTYPED = "untyped"  # Legacy notes without type


class NoteStatus(str, Enum):
    """Status values for notes."""

    # Task statuses
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"

    # Source/Note statuses
    RAW = "raw"
    PROCESSED = "processed"
    VERIFIED = "verified"

    # Learning statuses
    STAGING = "staging"

    # Fallback
    UNKNOWN = "unknown"


@dataclass
class NoteMetadata:
    """Metadata extracted from note frontmatter.

    Per OBSIDIAN_PIVOT.md Appendix A for frontmatter schemas.
    """

    note_type: NoteType = NoteType.UNTYPED
    status: NoteStatus = NoteStatus.UNKNOWN

    # Task-specific
    goal: str | None = None
    shad_run_id: str | None = None

    # Source-specific
    source_type: str | None = None

    # Learning-specific
    confidence: float | None = None
    cited_by: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)

    # Report-specific
    node_count: int | None = None
    max_depth: int | None = None
    token_usage: int | None = None
    wall_time_seconds: float | None = None

    # Common fields
    shad_processed: bool = False
    created: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_legacy(self) -> bool:
        """Check if this is a legacy untyped note."""
        return self.note_type == NoteType.UNTYPED

    @classmethod
    def from_frontmatter(cls, frontmatter: dict[str, Any]) -> NoteMetadata:
        """Create metadata from frontmatter dict."""
        # Parse type
        type_str = frontmatter.get("type", "")
        try:
            note_type = NoteType(type_str) if type_str else NoteType.UNTYPED
        except ValueError:
            note_type = NoteType.UNTYPED

        # Parse status
        status_str = frontmatter.get("status", "")
        try:
            status = NoteStatus(status_str) if status_str else NoteStatus.UNKNOWN
        except ValueError:
            status = NoteStatus.UNKNOWN

        return cls(
            note_type=note_type,
            status=status,
            goal=frontmatter.get("goal"),
            shad_run_id=frontmatter.get("shad_run_id"),
            source_type=frontmatter.get("source_type"),
            confidence=frontmatter.get("confidence"),
            cited_by=frontmatter.get("cited_by", []),
            contradicted_by=frontmatter.get("contradicted_by", []),
            node_count=frontmatter.get("node_count"),
            max_depth=frontmatter.get("max_depth"),
            token_usage=frontmatter.get("token_usage"),
            wall_time_seconds=frontmatter.get("wall_time_seconds"),
            shad_processed=frontmatter.get("shad_processed", False),
            created=frontmatter.get("created"),
            tags=frontmatter.get("tags", []),
        )

    def to_frontmatter(self) -> dict[str, Any]:
        """Convert to frontmatter dict."""
        result: dict[str, Any] = {}

        if self.note_type != NoteType.UNTYPED:
            result["type"] = self.note_type.value

        if self.status != NoteStatus.UNKNOWN:
            result["status"] = self.status.value

        # Add optional fields if set
        if self.goal is not None:
            result["goal"] = self.goal
        if self.shad_run_id is not None:
            result["shad_run_id"] = self.shad_run_id
        if self.source_type is not None:
            result["source_type"] = self.source_type
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.cited_by:
            result["cited_by"] = self.cited_by
        if self.contradicted_by:
            result["contradicted_by"] = self.contradicted_by
        if self.node_count is not None:
            result["node_count"] = self.node_count
        if self.max_depth is not None:
            result["max_depth"] = self.max_depth
        if self.token_usage is not None:
            result["token_usage"] = self.token_usage
        if self.wall_time_seconds is not None:
            result["wall_time_seconds"] = self.wall_time_seconds
        if self.shad_processed:
            result["shad_processed"] = self.shad_processed
        if self.created:
            result["created"] = self.created
        if self.tags:
            result["tags"] = self.tags

        return result


@dataclass
class VaultFile:
    """Represents a file in the vault."""

    relative_path: str
    exists: bool
    mtime: float = 0.0
    size: int = 0
    content_hash: str | None = None

    @classmethod
    def from_path(cls, path: Path, vault_root: Path) -> VaultFile:
        """Create from file path."""
        relative = path.relative_to(vault_root)
        stat = path.stat() if path.exists() else None

        return cls(
            relative_path=str(relative),
            exists=path.exists(),
            mtime=stat.st_mtime if stat else 0.0,
            size=stat.st_size if stat else 0,
        )

    def to_wikilink(self) -> str:
        """Generate full-path wikilink (per OBSIDIAN_PIVOT.md Section 4.3)."""
        # Remove .md extension
        link_path = self.relative_path
        if link_path.endswith(".md"):
            link_path = link_path[:-3]
        return f"[[{link_path}]]"


@dataclass
class VaultNote:
    """A note read from the vault."""

    path: str
    content: str
    metadata: NoteMetadata
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)
    mtime: float = 0.0
    content_hash: str | None = None

    def get_body(self) -> str:
        """Get note body without frontmatter."""
        # Remove frontmatter block if present
        if self.content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n?", self.content, re.DOTALL)
            if match:
                return self.content[match.end():]
        return self.content


@dataclass
class SearchResult:
    """A search result from the vault."""

    path: str
    content: str
    score: float = 0.0
    matched_line: str | None = None
    metadata: NoteMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "content": self.content,
            "score": self.score,
            "matched_line": self.matched_line,
        }


@dataclass
class DeleteResult:
    """Result of a delete operation (HITL gated)."""

    path: str
    requires_approval: bool = True
    hitl_queue_id: str | None = None
    approved: bool = False
    deleted: bool = False
    error: str | None = None
