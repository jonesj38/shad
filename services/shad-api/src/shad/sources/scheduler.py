"""Source scheduler for automated syncing."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable
from pathlib import Path

from shad.sources.manager import SourceManager, SyncResult

logger = logging.getLogger(__name__)


class SourceScheduler:
    """Scheduler for automated source syncing."""

    def __init__(
        self,
        config_path: Path | None = None,
        check_interval: int = 60,
        on_sync: Callable[[SyncResult], None] | None = None,
    ) -> None:
        """
        Initialize the scheduler.

        Args:
            config_path: Path to sources.yaml config
            check_interval: How often to check for due sources (seconds)
            on_sync: Callback function called after each sync
        """
        self.manager = SourceManager(config_path)
        self.check_interval = check_interval
        self.on_sync = on_sync
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        logger.info(f"Starting source scheduler (check interval: {self.check_interval}s)")

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop)

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the scheduler."""
        logger.info("Stopping scheduler...")
        self._running = False
        if self._task:
            self._task.cancel()

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Reload config to pick up changes
                self.manager.reload_config()

                # Get due sources
                due_sources = self.manager.get_due_sources()

                if due_sources:
                    logger.info(f"Found {len(due_sources)} sources due for sync")
                    result = await self.manager.sync_all(force=False)

                    # Log results
                    logger.info(
                        f"Sync complete: {result.successful} successful, "
                        f"{result.failed} failed, {result.skipped} skipped"
                    )

                    # Call callback if provided
                    if self.on_sync:
                        self.on_sync(result)

                    # Log any errors
                    for source_id, error in result.errors.items():
                        logger.error(f"Source {source_id} failed: {error}")

            except Exception as e:
                logger.exception(f"Error in scheduler loop: {e}")

            # Wait for next check
            if self._running:
                await asyncio.sleep(self.check_interval)

    async def run_once(self, force: bool = False) -> SyncResult:
        """Run a single sync cycle."""
        self.manager.reload_config()

        if force:
            logger.info("Running forced sync of all sources")
        else:
            logger.info("Running sync of due sources")

        result = await self.manager.sync_all(force=force)

        logger.info(
            f"Sync complete: {result.successful} successful, "
            f"{result.failed} failed, {result.skipped} skipped"
        )

        return result

    def get_status(self) -> dict:
        """Get scheduler status."""
        sources = self.manager.list_sources()
        due = self.manager.get_due_sources()

        return {
            "running": self._running,
            "check_interval": self.check_interval,
            "total_sources": len(sources),
            "due_sources": len(due),
            "sources": [
                {
                    "id": s.id,
                    "type": s.type.value,
                    "url": s.url or s.path,
                    "schedule": s.schedule.value,
                    "enabled": s.enabled,
                    "last_sync": s.last_sync.isoformat() if s.last_sync else None,
                    "next_sync": s.next_sync().isoformat() if s.next_sync() else None,
                    "last_error": s.last_error,
                    "is_due": s.is_due(),
                }
                for s in sources
            ],
        }


async def run_scheduler(
    config_path: Path | None = None,
    check_interval: int = 60,
) -> None:
    """Run the scheduler as a standalone daemon."""
    scheduler = SourceScheduler(config_path, check_interval)
    await scheduler.start()
