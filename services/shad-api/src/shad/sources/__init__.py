"""Sources module for automated content ingestion."""

from shad.sources.config import Source, SourceConfig, SourceSchedule, SourceType
from shad.sources.manager import SourceManager
from shad.sources.scheduler import SourceScheduler

__all__ = [
    "Source",
    "SourceConfig",
    "SourceSchedule",
    "SourceType",
    "SourceManager",
    "SourceScheduler",
]
