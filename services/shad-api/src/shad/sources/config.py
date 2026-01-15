"""Source configuration models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of content source."""

    GITHUB = "github"
    URL = "url"
    FEED = "feed"
    FOLDER = "folder"


class SourceSchedule(str, Enum):
    """Sync schedule frequency."""

    MANUAL = "manual"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    def to_timedelta(self) -> timedelta | None:
        """Convert schedule to timedelta."""
        mapping = {
            SourceSchedule.MANUAL: None,
            SourceSchedule.HOURLY: timedelta(hours=1),
            SourceSchedule.DAILY: timedelta(days=1),
            SourceSchedule.WEEKLY: timedelta(weeks=1),
            SourceSchedule.MONTHLY: timedelta(days=30),
        }
        return mapping[self]


class Source(BaseModel):
    """A content source to sync."""

    id: str = Field(default="", description="Unique identifier (auto-generated)")
    type: SourceType = Field(..., description="Source type")
    url: str | None = Field(default=None, description="URL for github/url/feed sources")
    path: str | None = Field(default=None, description="Path for folder sources")
    schedule: SourceSchedule = Field(default=SourceSchedule.DAILY)
    preset: str = Field(default="docs", description="Ingestion preset for github")
    vault_path: str = Field(..., description="Target vault path")
    enabled: bool = Field(default=True)
    last_sync: datetime | None = Field(default=None)
    last_error: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Generate ID if not set."""
        if not self.id:
            import hashlib
            source_str = f"{self.type.value}:{self.url or self.path}"
            self.id = hashlib.sha256(source_str.encode()).hexdigest()[:12]

    def is_due(self) -> bool:
        """Check if this source is due for sync."""
        if self.schedule == SourceSchedule.MANUAL:
            return False
        if not self.enabled:
            return False
        if self.last_sync is None:
            return True

        interval = self.schedule.to_timedelta()
        if interval is None:
            return False

        # Ensure both datetimes are timezone-aware
        now = datetime.now(UTC)
        last = self.last_sync
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)

        return now - last >= interval

    def next_sync(self) -> datetime | None:
        """Calculate next sync time."""
        if self.schedule == SourceSchedule.MANUAL:
            return None

        interval = self.schedule.to_timedelta()
        if interval is None:
            return None

        if self.last_sync is None:
            return datetime.now(UTC)

        last = self.last_sync
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)

        return last + interval


class SourceConfig(BaseModel):
    """Configuration for all sources."""

    sources: list[Source] = Field(default_factory=list)
    default_vault: str | None = Field(default=None)

    @classmethod
    def load(cls, path: Path | None = None) -> SourceConfig:
        """Load config from YAML file."""
        if path is None:
            path = Path.home() / ".shad" / "sources.yaml"

        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Convert source dicts to Source objects
        sources = []
        for s in data.get("sources", []):
            sources.append(Source(**s))

        return cls(
            sources=sources,
            default_vault=data.get("default_vault"),
        )

    def save(self, path: Path | None = None) -> None:
        """Save config to YAML file."""
        if path is None:
            path = Path.home() / ".shad" / "sources.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert sources to plain dicts with string enum values
        sources_data = []
        for s in self.sources:
            source_dict = s.model_dump(exclude_none=True)
            # Convert enums to their string values
            if "type" in source_dict:
                source_dict["type"] = source_dict["type"].value if hasattr(source_dict["type"], "value") else source_dict["type"]
            if "schedule" in source_dict:
                source_dict["schedule"] = source_dict["schedule"].value if hasattr(source_dict["schedule"], "value") else source_dict["schedule"]
            # Convert datetime to ISO string
            if "last_sync" in source_dict and source_dict["last_sync"]:
                source_dict["last_sync"] = source_dict["last_sync"].isoformat() if hasattr(source_dict["last_sync"], "isoformat") else source_dict["last_sync"]
            sources_data.append(source_dict)

        data = {"sources": sources_data}
        if self.default_vault:
            data["default_vault"] = self.default_vault

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def add_source(self, source: Source) -> None:
        """Add a source, replacing if ID exists."""
        self.sources = [s for s in self.sources if s.id != source.id]
        self.sources.append(source)

    def remove_source(self, source_id: str) -> bool:
        """Remove a source by ID."""
        original_len = len(self.sources)
        self.sources = [s for s in self.sources if s.id != source_id]
        return len(self.sources) < original_len

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID."""
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def get_due_sources(self) -> list[Source]:
        """Get all sources that are due for sync."""
        return [s for s in self.sources if s.is_due()]
