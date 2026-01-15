"""Learnings storage and promotion pipeline.

Per SPEC.md Section 15.1:
1. Capture everything as notes (default layer)
2. Propose patches/hints/negatives (automated suggestions)
3. Test via evals (comparative runs)
4. Promote via HITL review (human approval)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from shad.learnings.extractor import ExtractedLearning, LearningType
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class LearningStatus(str, Enum):
    """Status of a learning in the pipeline."""

    PROPOSED = "proposed"  # Initially extracted
    STAGED = "staged"  # Under evaluation
    APPROVED = "approved"  # HITL approved
    PROMOTED = "promoted"  # Applied to system
    REJECTED = "rejected"  # Rejected by review
    ARCHIVED = "archived"  # No longer relevant


@dataclass
class Learning:
    """A learning in the store with lifecycle tracking."""

    learning_id: str
    learning_type: LearningType
    content: str
    status: LearningStatus = LearningStatus.PROPOSED
    confidence: float = 0.5

    # Source tracking
    source_run_ids: list[str] = field(default_factory=list)
    skill_name: str | None = None
    tags: list[str] = field(default_factory=list)

    # Lifecycle timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    staged_at: datetime | None = None
    approved_at: datetime | None = None
    promoted_at: datetime | None = None

    # Evaluation
    eval_runs: list[str] = field(default_factory=list)
    eval_success_rate: float | None = None

    # Review
    reviewer: str | None = None
    review_notes: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "learning_id": self.learning_id,
            "learning_type": self.learning_type.value,
            "content": self.content,
            "status": self.status.value,
            "confidence": self.confidence,
            "source_run_ids": self.source_run_ids,
            "skill_name": self.skill_name,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "staged_at": self.staged_at.isoformat() if self.staged_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "eval_runs": self.eval_runs,
            "eval_success_rate": self.eval_success_rate,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Learning:
        """Create from dict."""
        learning = cls(
            learning_id=data["learning_id"],
            learning_type=LearningType(data["learning_type"]),
            content=data["content"],
            status=LearningStatus(data.get("status", "proposed")),
            confidence=data.get("confidence", 0.5),
            source_run_ids=data.get("source_run_ids", []),
            skill_name=data.get("skill_name"),
            tags=data.get("tags", []),
            eval_runs=data.get("eval_runs", []),
            eval_success_rate=data.get("eval_success_rate"),
            reviewer=data.get("reviewer"),
            review_notes=data.get("review_notes", ""),
            metadata=data.get("metadata", {}),
        )

        if data.get("created_at"):
            learning.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("staged_at"):
            learning.staged_at = datetime.fromisoformat(data["staged_at"])
        if data.get("approved_at"):
            learning.approved_at = datetime.fromisoformat(data["approved_at"])
        if data.get("promoted_at"):
            learning.promoted_at = datetime.fromisoformat(data["promoted_at"])

        return learning

    @classmethod
    def from_extracted(cls, extracted: ExtractedLearning) -> Learning:
        """Create from an ExtractedLearning."""
        import uuid

        return cls(
            learning_id=str(uuid.uuid4())[:8],
            learning_type=extracted.learning_type,
            content=extracted.content,
            confidence=extracted.confidence,
            source_run_ids=[extracted.source_run_id] if extracted.source_run_id else [],
            skill_name=extracted.skill_name,
            tags=extracted.tags,
            metadata=extracted.metadata,
        )


class LearningsStore:
    """
    Store for learnings with promotion pipeline.

    Features:
    - File-based persistence
    - Status tracking through pipeline stages
    - Tagging and search
    - Statistics
    """

    def __init__(self, storage_path: Path | None = None):
        settings = get_settings()
        self.storage_path = storage_path or (settings.history_path / "learnings")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._store_file = self.storage_path / "learnings.json"
        self._learnings: dict[str, Learning] = {}
        self._load()

    def _load(self) -> None:
        """Load store from disk."""
        if self._store_file.exists():
            try:
                with self._store_file.open() as f:
                    data = json.load(f)
                    for item_data in data.get("learnings", []):
                        learning = Learning.from_dict(item_data)
                        self._learnings[learning.learning_id] = learning
            except Exception as e:
                logger.warning(f"Failed to load learnings store: {e}")

    def _save(self) -> None:
        """Save store to disk."""
        try:
            data = {
                "learnings": [learning.to_dict() for learning in self._learnings.values()],
                "updated_at": datetime.now(UTC).isoformat(),
            }
            with self._store_file.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learnings store: {e}")

    def add(self, extracted: ExtractedLearning) -> Learning:
        """
        Add a new learning from an extraction.

        Returns the created Learning.
        """
        # Check for duplicates (simple content matching)
        for existing in self._learnings.values():
            if self._is_duplicate(extracted.content, existing.content):
                # Update existing instead of adding duplicate
                if extracted.source_run_id:
                    existing.source_run_ids.append(extracted.source_run_id)
                existing.confidence = max(existing.confidence, extracted.confidence)
                self._save()
                return existing

        learning = Learning.from_extracted(extracted)
        self._learnings[learning.learning_id] = learning
        self._save()

        logger.info(f"Added learning {learning.learning_id}: {learning.learning_type.value}")
        return learning

    def _is_duplicate(self, content1: str, content2: str) -> bool:
        """Check if two learnings are duplicates."""
        # Simple word overlap check
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / max(len(words1), len(words2))

        return similarity > 0.8

    def get(self, learning_id: str) -> Learning | None:
        """Get a learning by ID."""
        return self._learnings.get(learning_id)

    def list_by_status(
        self,
        status: LearningStatus,
        limit: int = 50,
        learning_type: LearningType | None = None,
    ) -> list[Learning]:
        """List learnings by status."""
        learnings = [
            item for item in self._learnings.values()
            if item.status == status
        ]

        if learning_type:
            learnings = [item for item in learnings if item.learning_type == learning_type]

        # Sort by confidence descending
        learnings.sort(key=lambda x: x.confidence, reverse=True)

        return learnings[:limit]

    def list_by_skill(self, skill_name: str) -> list[Learning]:
        """List learnings for a specific skill."""
        return [
            item for item in self._learnings.values()
            if item.skill_name == skill_name and item.status == LearningStatus.PROMOTED
        ]

    def search_by_tags(self, tags: list[str], match_all: bool = False) -> list[Learning]:
        """Search learnings by tags."""
        tag_set = {t.lower() for t in tags}
        results = []

        for learning in self._learnings.values():
            learning_tags = {t.lower() for t in learning.tags}
            if match_all:
                if tag_set <= learning_tags:
                    results.append(learning)
            else:
                if tag_set & learning_tags:
                    results.append(learning)

        return results

    # Pipeline operations

    def stage(self, learning_id: str) -> bool:
        """Move a learning to staged status for evaluation."""
        learning = self._learnings.get(learning_id)
        if not learning or learning.status != LearningStatus.PROPOSED:
            return False

        learning.status = LearningStatus.STAGED
        learning.staged_at = datetime.now(UTC)
        self._save()

        logger.info(f"Staged learning {learning_id}")
        return True

    def record_eval(
        self,
        learning_id: str,
        eval_run_id: str,
        success: bool,
    ) -> bool:
        """Record an evaluation run for a learning."""
        learning = self._learnings.get(learning_id)
        if not learning:
            return False

        learning.eval_runs.append(eval_run_id)

        # Update success rate
        if learning.eval_success_rate is None:
            learning.eval_success_rate = 1.0 if success else 0.0
        else:
            # Running average
            n = len(learning.eval_runs)
            old_rate = learning.eval_success_rate
            learning.eval_success_rate = (old_rate * (n - 1) + (1.0 if success else 0.0)) / n

        self._save()
        return True

    def approve(
        self,
        learning_id: str,
        reviewer: str,
        notes: str = "",
    ) -> bool:
        """Approve a learning after HITL review."""
        learning = self._learnings.get(learning_id)
        if not learning or learning.status not in (LearningStatus.PROPOSED, LearningStatus.STAGED):
            return False

        learning.status = LearningStatus.APPROVED
        learning.approved_at = datetime.now(UTC)
        learning.reviewer = reviewer
        learning.review_notes = notes
        self._save()

        logger.info(f"Approved learning {learning_id} by {reviewer}")
        return True

    def reject(
        self,
        learning_id: str,
        reviewer: str,
        notes: str = "",
    ) -> bool:
        """Reject a learning."""
        learning = self._learnings.get(learning_id)
        if not learning:
            return False

        learning.status = LearningStatus.REJECTED
        learning.reviewer = reviewer
        learning.review_notes = notes
        self._save()

        logger.info(f"Rejected learning {learning_id}")
        return True

    def promote(self, learning_id: str) -> bool:
        """
        Promote an approved learning to active use.

        This marks the learning as ready to be applied to the system
        (e.g., updating skill prompts, routing rules).
        """
        learning = self._learnings.get(learning_id)
        if not learning or learning.status != LearningStatus.APPROVED:
            return False

        learning.status = LearningStatus.PROMOTED
        learning.promoted_at = datetime.now(UTC)
        self._save()

        logger.info(f"Promoted learning {learning_id}")
        return True

    def archive(self, learning_id: str) -> bool:
        """Archive a learning."""
        learning = self._learnings.get(learning_id)
        if not learning:
            return False

        learning.status = LearningStatus.ARCHIVED
        self._save()
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for learning in self._learnings.values():
            status = learning.status.value
            ltype = learning.learning_type.value

            by_status[status] = by_status.get(status, 0) + 1
            by_type[ltype] = by_type.get(ltype, 0) + 1

        return {
            "total": len(self._learnings),
            "by_status": by_status,
            "by_type": by_type,
        }

    def get_promoted_for_skill(self, skill_name: str) -> list[str]:
        """
        Get promoted learnings applicable to a skill.

        Returns list of learning contents that should be applied.
        """
        learnings = self.list_by_skill(skill_name)
        return [item.content for item in learnings]

    def cleanup_old(self, days: int = 90) -> int:
        """Archive learnings older than specified days that weren't promoted."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        archived = 0

        for learning in self._learnings.values():
            if learning.status in (LearningStatus.PROPOSED, LearningStatus.REJECTED):
                if learning.created_at < cutoff:
                    learning.status = LearningStatus.ARCHIVED
                    archived += 1

        if archived:
            self._save()
            logger.info(f"Archived {archived} old learnings")

        return archived
