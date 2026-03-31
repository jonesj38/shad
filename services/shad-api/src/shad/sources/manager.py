"""Source manager for syncing content."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from shad.sources.config import Source, SourceConfig, SourceSchedule, SourceType
from shad.sources.ingest import FeedIngester, FolderIngester, IngestResult, URLIngester
from shad.vault.consolidation import ConsolidationConfig
from shad.vault.ingestion import IngestPreset, VaultIngester
from shad.vault.shadow_index import ShadowIndex, source_to_memory_type

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass
class SyncResult:
    """Result of syncing sources."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: dict[str, IngestResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class SourceManager:
    """Manages source synchronization."""

    def __init__(
        self,
        config_path: Path | None = None,
        consolidation_config: ConsolidationConfig | None = None,
        shadow_index: ShadowIndex | None = None,
        on_consolidation: Callable[[Source, int], None] | None = None,
    ) -> None:
        self.config_path = config_path or Path.home() / ".shad" / "sources.yaml"
        self.consolidation_config = consolidation_config or ConsolidationConfig()
        self._shadow_index = shadow_index
        self._on_consolidation = on_consolidation
        self._config: SourceConfig | None = None

    @property
    def config(self) -> SourceConfig:
        """Load config lazily."""
        if self._config is None:
            self._config = SourceConfig.load(self.config_path)
        return self._config

    def reload_config(self) -> None:
        """Reload config from disk."""
        self._config = SourceConfig.load(self.config_path)

    def save_config(self) -> None:
        """Save config to disk."""
        self.config.save(self.config_path)

    def add_source(
        self,
        source_type: SourceType,
        url: str | None = None,
        path: str | None = None,
        collection_path: str | None = None,
        schedule: SourceSchedule = SourceSchedule.DAILY,
        preset: str = "docs",
    ) -> Source:
        """Add a new source."""
        if collection_path is None:
            collection_path = self.config.default_collection
            if collection_path is None:
                raise ValueError("No collection path specified and no default collection configured")

        source = Source(
            type=source_type,
            url=url,
            path=path,
            collection_path=collection_path,
            schedule=schedule,
            preset=preset,
        )

        self.config.add_source(source)
        self.save_config()
        return source

    def remove_source(self, source_id: str) -> bool:
        """Remove a source by ID."""
        removed = self.config.remove_source(source_id)
        if removed:
            self.save_config()
        return removed

    def list_sources(self) -> list[Source]:
        """List all sources."""
        return self.config.sources

    def get_due_sources(self) -> list[Source]:
        """Get sources that are due for sync."""
        return self.config.get_due_sources()

    async def sync_source(self, source: Source) -> IngestResult:
        """Sync a single source."""
        collection_path = Path(source.collection_path).expanduser()

        try:
            if source.type == SourceType.GITHUB:
                result = await self._sync_github(source, collection_path)
            elif source.type == SourceType.URL:
                result = await self._sync_url(source, collection_path)
            elif source.type == SourceType.FEED:
                result = await self._sync_feed(source, collection_path)
            elif source.type == SourceType.FOLDER:
                result = await self._sync_folder(source, collection_path)
            else:
                result = IngestResult(success=False, errors=[f"Unknown source type: {source.type}"])

            # Update source status
            source.last_sync = datetime.now(UTC)
            if result.success:
                source.last_error = None
                result.metadata["memory_type"] = source_to_memory_type(source.type.value).value
                self._check_consolidation(source)
            else:
                source.last_error = "; ".join(result.errors) if result.errors else "Unknown error"

            self.save_config()
            return result

        except Exception as e:
            logger.exception(f"Failed to sync source {source.id}: {e}")
            source.last_error = str(e)
            self.save_config()
            return IngestResult(success=False, errors=[str(e)])

    async def sync_all(self, force: bool = False) -> SyncResult:
        """Sync all due sources (or all if force=True)."""
        if force:
            sources = self.config.sources
        else:
            sources = self.get_due_sources()

        result = SyncResult(total=len(sources))

        for source in sources:
            if not source.enabled:
                result.skipped += 1
                continue

            logger.info(f"Syncing {source.type.value}: {source.url or source.path}")
            ingest_result = await self.sync_source(source)

            result.results[source.id] = ingest_result
            if ingest_result.success:
                result.successful += 1
            else:
                result.failed += 1
                result.errors[source.id] = "; ".join(ingest_result.errors)

        return result

    async def _sync_github(self, source: Source, collection_path: Path) -> IngestResult:
        """Sync a GitHub source."""
        if not source.url:
            return IngestResult(success=False, errors=["No URL specified for GitHub source"])

        ingester = VaultIngester(collection_path=collection_path)
        preset = IngestPreset(source.preset) if source.preset else IngestPreset.DOCS

        result = await ingester.ingest_github(source.url, preset=preset)

        # Convert VaultIngester result to our IngestResult
        return IngestResult(
            success=result.success,
            files_created=result.files_processed,
            errors=result.errors,
            metadata={"snapshot_id": result.snapshot_id} if result.snapshot_id else {},
        )

    async def _sync_url(self, source: Source, collection_path: Path) -> IngestResult:
        """Sync a URL source."""
        if not source.url:
            return IngestResult(success=False, errors=["No URL specified"])

        ingester = URLIngester(collection_path=collection_path)
        return await ingester.ingest(source.url)

    async def _sync_feed(self, source: Source, collection_path: Path) -> IngestResult:
        """Sync a feed source."""
        if not source.url:
            return IngestResult(success=False, errors=["No URL specified for feed"])

        ingester = FeedIngester(collection_path=collection_path)
        max_items = source.metadata.get("max_items", 10)
        return await ingester.ingest(source.url, max_items=max_items)

    async def _sync_folder(self, source: Source, collection_path: Path) -> IngestResult:
        """Sync a folder source."""
        if not source.path:
            return IngestResult(success=False, errors=["No path specified for folder"])

        ingester = FolderIngester(collection_path=collection_path)
        return await ingester.ingest(source.path)

    def _check_consolidation(self, source: Source) -> None:
        """Fire on_consolidation if episodic count exceeds the configured threshold.

        Uses source.metadata["last_consolidation"] as the lower bound so only
        entries accumulated since the previous consolidation run are counted.
        Stamps the timestamp on firing so the next call measures the next window.
        """
        if self._shadow_index is None or self._on_consolidation is None:
            return

        last_str = source.metadata.get("last_consolidation")
        since = datetime.fromisoformat(last_str) if last_str else _EPOCH
        count = self._shadow_index.episodic_count(source.id, since)

        if self.consolidation_config.should_consolidate(count):
            logger.info(
                "Consolidation triggered for source %s: %d episodic entries since %s",
                source.id,
                count,
                since.isoformat(),
            )
            self._on_consolidation(source, count)
            # Advance the watermark to the latest ingested_at of the entries
            # that triggered consolidation (max of now and the latest entry's
            # timestamp), so future-dated synthetic entries can't bleed into
            # the next consolidation window.
            latest = self._shadow_index.latest_episodic_ingested_at(source.id, since)
            watermark = max(datetime.now(UTC), latest) if latest is not None else datetime.now(UTC)
            source.metadata["last_consolidation"] = watermark.isoformat()

    def set_default_collection(self, collection_path: str) -> None:
        """Set the default collection path."""
        self.config.default_collection = collection_path
        self.save_config()
