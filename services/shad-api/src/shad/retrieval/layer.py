"""Retrieval layer abstraction.

Defines the interface for retrieval backends (qmd, filesystem, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class RetrievalResult:
    """A search result from the retrieval layer."""

    path: str
    """Relative path to the document within the collection."""

    content: str
    """Full or partial content of the document."""

    score: float = 0.0
    """Relevance score (0.0 to 1.0, higher is better)."""

    snippet: str | None = None
    """Highlighted snippet containing the matched query."""

    collection: str = ""
    """Name of the collection/vault this result came from."""

    docid: str = ""
    """Unique document identifier (qmd hash or generated)."""

    matched_line: str | None = None
    """First line containing the query match."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (frontmatter, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "path": self.path,
            "content": self.content,
            "score": self.score,
            "snippet": self.snippet,
            "collection": self.collection,
            "docid": self.docid,
            "matched_line": self.matched_line,
            "metadata": self.metadata,
        }

    def full_path(self, base_path: Path | None = None) -> Path:
        """Get absolute path to the file.

        Args:
            base_path: Base path to prepend (if collection path not available)

        Returns:
            Absolute path to the document
        """
        if base_path:
            return base_path / self.path
        return Path(self.path)


@runtime_checkable
class RetrievalLayer(Protocol):
    """Protocol for retrieval backends.

    Implementations must provide async search() and get() methods.
    """

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        collections: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Search for documents matching the query.

        Args:
            query: Search query string
            mode: Search mode - "bm25" (keyword), "vector" (semantic), "hybrid" (both)
            collections: Optional list of collection names to search (None = all)
            limit: Maximum number of results to return
            min_score: Minimum relevance score threshold

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        ...

    async def get(
        self,
        path: str,
        collection: str | None = None,
    ) -> str | None:
        """Retrieve a single document by path.

        Args:
            path: Relative path to the document
            collection: Optional collection name (if multiple collections have same path)

        Returns:
            Document content or None if not found
        """
        ...

    async def status(self) -> dict[str, Any]:
        """Get status information about the retrieval backend.

        Returns:
            Dictionary with status info (collections, index health, etc.)
        """
        ...

    @property
    def available(self) -> bool:
        """Check if this retrieval backend is available."""
        ...
