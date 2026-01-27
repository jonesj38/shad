"""Filesystem retrieval backend.

Direct file search using filesystem traversal. Used as a fallback when qmd
is not available, or for simple deployments that don't need semantic search.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from shad.retrieval.layer import RetrievalResult

logger = logging.getLogger(__name__)


class FilesystemRetriever:
    """Retrieval backend using direct filesystem search.

    Provides BM25-style keyword matching via simple string search.
    Does not support semantic/vector search (mode="vector" falls back to keyword).

    This is the fallback when qmd is not available.
    """

    def __init__(
        self,
        paths: list[Path] | None = None,
        collection_names: dict[str, Path] | None = None,
    ) -> None:
        """Initialize with search paths.

        Args:
            paths: List of paths to search (treated as unnamed collections)
            collection_names: Mapping of collection name -> path
        """
        self._paths = paths or []
        self._collections = collection_names or {}

        # If paths provided but no names, create default names
        if self._paths and not self._collections:
            for path in self._paths:
                self._collections[path.name] = path

    @property
    def available(self) -> bool:
        """Filesystem search is always available."""
        return True

    def add_path(self, path: Path, name: str | None = None) -> None:
        """Add a path to search.

        Args:
            path: Directory path
            name: Collection name (defaults to directory name)
        """
        self._paths.append(path)
        collection_name = name or path.name
        self._collections[collection_name] = path

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
            mode: Search mode (all modes fall back to keyword search)
            collections: Optional list of collection names to search
            limit: Maximum number of results
            min_score: Minimum relevance score

        Returns:
            List of RetrievalResult objects
        """
        if mode == "vector":
            logger.warning("FilesystemRetriever does not support vector search, using keyword")

        results: list[RetrievalResult] = []
        query_lower = query.lower()

        # Determine which collections to search
        search_collections = {}
        if collections:
            for name in collections:
                if name in self._collections:
                    search_collections[name] = self._collections[name]
                else:
                    logger.warning(f"Collection '{name}' not found")
        else:
            search_collections = self._collections

        if not search_collections:
            logger.warning("No collections to search")
            return []

        # Extract query terms for multi-word scoring
        query_terms = self._extract_terms(query)

        for collection_name, base_path in search_collections.items():
            try:
                for md_file in base_path.rglob("*.md"):
                    # Skip hidden directories and artifacts
                    if any(part.startswith(".") for part in md_file.parts):
                        continue
                    if "artifacts" in md_file.parts:
                        continue

                    relative_path = str(md_file.relative_to(base_path))

                    try:
                        content = md_file.read_text(encoding="utf-8")
                        content_lower = content.lower()

                        # Check if query matches
                        if query_lower not in content_lower:
                            # Try individual terms
                            if not any(term in content_lower for term in query_terms):
                                continue

                        # Calculate relevance score
                        score = self._calculate_score(content, query, query_terms)

                        if score < min_score:
                            continue

                        # Find matched line
                        matched_line = self._find_matched_line(content, query_terms)

                        # Generate snippet
                        snippet = self._generate_snippet(content, query_terms)

                        # Generate stable docid
                        docid = hashlib.sha256(
                            f"{collection_name}:{relative_path}".encode()
                        ).hexdigest()[:6]

                        results.append(RetrievalResult(
                            path=relative_path,
                            content=content,
                            score=score,
                            snippet=snippet,
                            collection=collection_name,
                            docid=docid,
                            matched_line=matched_line,
                        ))

                    except Exception as e:
                        logger.debug(f"Error reading {md_file}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                continue

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def get(
        self,
        path: str,
        collection: str | None = None,
    ) -> str | None:
        """Retrieve a single document by path.

        Args:
            path: Relative path to the document
            collection: Collection name (required if multiple collections)

        Returns:
            Document content or None if not found
        """
        if collection and collection in self._collections:
            full_path = self._collections[collection] / path
            if full_path.exists():
                return full_path.read_text(encoding="utf-8")
            return None

        # Search all collections
        for base_path in self._collections.values():
            full_path = base_path / path
            if full_path.exists():
                return full_path.read_text(encoding="utf-8")

        return None

    async def status(self) -> dict[str, Any]:
        """Get status information.

        Returns:
            Dictionary with collections and file counts
        """
        collections_info = []
        for name, path in self._collections.items():
            file_count = sum(1 for _ in path.rglob("*.md"))
            collections_info.append({
                "name": name,
                "path": str(path),
                "files": file_count,
            })

        return {
            "available": True,
            "backend": "filesystem",
            "collections": collections_info,
            "total_collections": len(self._collections),
        }

    def _extract_terms(self, query: str) -> list[str]:
        """Extract search terms from query.

        Args:
            query: Search query

        Returns:
            List of lowercase terms (excluding stop words)
        """
        # Simple stop words
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        }

        # Split on non-word characters
        terms = re.split(r"\W+", query.lower())
        return [t for t in terms if t and len(t) > 2 and t not in stop_words]

    def _calculate_score(
        self,
        content: str,
        query: str,
        query_terms: list[str],
    ) -> float:
        """Calculate relevance score for a document.

        Uses a simplified BM25-style scoring:
        - Term frequency (TF): how often terms appear
        - Normalized by document length

        Args:
            content: Document content
            query: Original query
            query_terms: Extracted query terms

        Returns:
            Score between 0.0 and 1.0
        """
        content_lower = content.lower()
        content_len = len(content)

        if content_len == 0:
            return 0.0

        # Exact phrase match gets bonus
        phrase_bonus = 0.3 if query.lower() in content_lower else 0.0

        if not query_terms:
            return phrase_bonus

        # Count term matches
        matched_terms = 0
        total_frequency = 0

        for term in query_terms:
            count = content_lower.count(term)
            if count > 0:
                matched_terms += 1
                total_frequency += count

        if matched_terms == 0:
            return 0.0

        # Term match ratio (how many query terms matched)
        term_ratio = matched_terms / len(query_terms) * 0.5

        # Frequency score (normalized by length, capped)
        freq_score = min(total_frequency / content_len * 100, 0.3)

        # Combine scores
        score = phrase_bonus + term_ratio + freq_score

        return min(score, 1.0)

    def _find_matched_line(
        self,
        content: str,
        query_terms: list[str],
    ) -> str | None:
        """Find the first line containing query terms.

        Args:
            content: Document content
            query_terms: Query terms to find

        Returns:
            First matching line or None
        """
        for line in content.split("\n"):
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms):
                return line.strip()
        return None

    def _generate_snippet(
        self,
        content: str,
        query_terms: list[str],
        max_length: int = 200,
    ) -> str | None:
        """Generate a snippet around the first match.

        Args:
            content: Document content
            query_terms: Query terms
            max_length: Maximum snippet length

        Returns:
            Snippet with context around match
        """
        content_lower = content.lower()

        # Find first term position
        first_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos

        if first_pos == len(content):
            # No match found, return start of content
            return content[:max_length].strip() + "..." if len(content) > max_length else content

        # Extract context around match
        start = max(0, first_pos - 50)
        end = min(len(content), first_pos + max_length - 50)

        snippet = content[start:end].strip()

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet
