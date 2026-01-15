"""Human-in-the-Loop (HITL) review system.

Per SPEC.md Section 6.2:
- Latency tolerance: Batch review (async)
- Queue flagged items for periodic human review
- Runs complete with provisional results
- Provisional results get taint propagation + expiring cache
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """Status of a review item."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"


class ReviewPriority(str, Enum):
    """Priority levels for review items."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReviewItem:
    """An item queued for human review."""

    item_id: str
    run_id: str
    node_id: str
    status: ReviewStatus = ReviewStatus.PENDING
    priority: ReviewPriority = ReviewPriority.MEDIUM

    # Content to review
    task: str = ""
    result: str = ""
    context: str = ""

    # Validation info
    confidence: float = 0.5
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    # Metadata
    skill_name: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None
    reviewer: str | None = None
    review_notes: str = ""
    modified_result: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "item_id": self.item_id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "task": self.task,
            "result": self.result,
            "context": self.context[:500] if self.context else "",
            "confidence": self.confidence,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "skill_name": self.skill_name,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "modified_result": self.modified_result,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewItem:
        """Create from dict."""
        item = cls(
            item_id=data["item_id"],
            run_id=data["run_id"],
            node_id=data["node_id"],
            status=ReviewStatus(data.get("status", "pending")),
            priority=ReviewPriority(data.get("priority", "medium")),
            task=data.get("task", ""),
            result=data.get("result", ""),
            context=data.get("context", ""),
            confidence=data.get("confidence", 0.5),
            validation_errors=data.get("validation_errors", []),
            validation_warnings=data.get("validation_warnings", []),
            skill_name=data.get("skill_name"),
            review_notes=data.get("review_notes", ""),
            modified_result=data.get("modified_result"),
            tags=data.get("tags", []),
        )

        if data.get("created_at"):
            item.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("reviewed_at"):
            item.reviewed_at = datetime.fromisoformat(data["reviewed_at"])
        item.reviewer = data.get("reviewer")

        return item


class HITLQueue:
    """
    Queue for human-in-the-loop review items.

    Features:
    - File-based persistence (JSON)
    - Priority-based ordering
    - Expiration for stale items
    - Statistics tracking
    """

    DEFAULT_EXPIRY_HOURS = 24 * 7  # 1 week

    def __init__(self, storage_path: Path | None = None):
        settings = get_settings()
        self.storage_path = storage_path or (settings.history_path / "review_queue")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._queue_file = self.storage_path / "queue.json"
        self._items: dict[str, ReviewItem] = {}
        self._load()

    def _load(self) -> None:
        """Load queue from disk."""
        if self._queue_file.exists():
            try:
                with self._queue_file.open() as f:
                    data = json.load(f)
                    for item_data in data.get("items", []):
                        item = ReviewItem.from_dict(item_data)
                        self._items[item.item_id] = item
            except Exception as e:
                logger.warning(f"Failed to load review queue: {e}")

    def _save(self) -> None:
        """Save queue to disk."""
        try:
            data = {
                "items": [item.to_dict() for item in self._items.values()],
                "updated_at": datetime.now(UTC).isoformat(),
            }
            with self._queue_file.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save review queue: {e}")

    def add(
        self,
        run_id: str,
        node_id: str,
        task: str,
        result: str,
        context: str = "",
        confidence: float = 0.5,
        validation_errors: list[str] | None = None,
        validation_warnings: list[str] | None = None,
        skill_name: str | None = None,
        priority: ReviewPriority | None = None,
        tags: list[str] | None = None,
    ) -> ReviewItem:
        """
        Add an item to the review queue.

        Returns the created ReviewItem.
        """
        import uuid

        item_id = str(uuid.uuid4())[:8]

        # Auto-determine priority based on confidence
        if priority is None:
            if confidence < 0.3:
                priority = ReviewPriority.HIGH
            elif confidence < 0.5:
                priority = ReviewPriority.MEDIUM
            else:
                priority = ReviewPriority.LOW

        item = ReviewItem(
            item_id=item_id,
            run_id=run_id,
            node_id=node_id,
            priority=priority,
            task=task,
            result=result,
            context=context,
            confidence=confidence,
            validation_errors=validation_errors or [],
            validation_warnings=validation_warnings or [],
            skill_name=skill_name,
            tags=tags or [],
        )

        self._items[item_id] = item
        self._save()

        logger.info(f"Added review item {item_id} for run {run_id}")
        return item

    def get(self, item_id: str) -> ReviewItem | None:
        """Get a review item by ID."""
        return self._items.get(item_id)

    def list_pending(
        self,
        limit: int = 50,
        priority: ReviewPriority | None = None,
        skill_name: str | None = None,
    ) -> list[ReviewItem]:
        """List pending review items."""
        items = [
            item for item in self._items.values()
            if item.status == ReviewStatus.PENDING
        ]

        # Filter by priority
        if priority:
            items = [i for i in items if i.priority == priority]

        # Filter by skill
        if skill_name:
            items = [i for i in items if i.skill_name == skill_name]

        # Sort by priority (critical > high > medium > low), then by age
        priority_order = {
            ReviewPriority.CRITICAL: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3,
        }
        items.sort(key=lambda x: (priority_order[x.priority], x.created_at))

        return items[:limit]

    def approve(
        self,
        item_id: str,
        reviewer: str,
        notes: str = "",
    ) -> bool:
        """Approve a review item."""
        item = self._items.get(item_id)
        if not item:
            return False

        item.status = ReviewStatus.APPROVED
        item.reviewer = reviewer
        item.reviewed_at = datetime.now(UTC)
        item.review_notes = notes

        self._save()
        logger.info(f"Approved review item {item_id}")
        return True

    def reject(
        self,
        item_id: str,
        reviewer: str,
        notes: str = "",
    ) -> bool:
        """Reject a review item."""
        item = self._items.get(item_id)
        if not item:
            return False

        item.status = ReviewStatus.REJECTED
        item.reviewer = reviewer
        item.reviewed_at = datetime.now(UTC)
        item.review_notes = notes

        self._save()
        logger.info(f"Rejected review item {item_id}")
        return True

    def modify(
        self,
        item_id: str,
        reviewer: str,
        modified_result: str,
        notes: str = "",
    ) -> bool:
        """Approve with modifications."""
        item = self._items.get(item_id)
        if not item:
            return False

        item.status = ReviewStatus.MODIFIED
        item.reviewer = reviewer
        item.reviewed_at = datetime.now(UTC)
        item.review_notes = notes
        item.modified_result = modified_result

        self._save()
        logger.info(f"Modified review item {item_id}")
        return True

    def expire_old(self, hours: int | None = None) -> int:
        """Expire items older than specified hours."""
        hours = hours or self.DEFAULT_EXPIRY_HOURS
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        expired_count = 0
        for item in self._items.values():
            if item.status == ReviewStatus.PENDING and item.created_at < cutoff:
                item.status = ReviewStatus.EXPIRED
                expired_count += 1

        if expired_count > 0:
            self._save()
            logger.info(f"Expired {expired_count} review items")

        return expired_count

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        by_status: dict[str, int] = {}
        by_priority: dict[str, int] = {}

        for item in self._items.values():
            status = item.status.value
            priority = item.priority.value

            by_status[status] = by_status.get(status, 0) + 1
            if item.status == ReviewStatus.PENDING:
                by_priority[priority] = by_priority.get(priority, 0) + 1

        return {
            "total_items": len(self._items),
            "pending": by_status.get("pending", 0),
            "approved": by_status.get("approved", 0),
            "rejected": by_status.get("rejected", 0),
            "modified": by_status.get("modified", 0),
            "expired": by_status.get("expired", 0),
            "by_priority": by_priority,
        }

    def clear_completed(self, keep_days: int = 30) -> int:
        """Remove completed items older than specified days."""
        cutoff = datetime.now(UTC) - timedelta(days=keep_days)
        to_remove = []

        for item_id, item in self._items.items():
            if item.status != ReviewStatus.PENDING and item.reviewed_at:
                if item.reviewed_at < cutoff:
                    to_remove.append(item_id)

        for item_id in to_remove:
            del self._items[item_id]

        if to_remove:
            self._save()

        return len(to_remove)
