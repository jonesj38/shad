"""Vault Tools for sandboxed code execution.

These tools are importable by RLM-generated scripts for
interacting with the vault(s).

Supports:
- Direct filesystem operations (read, write, list)
- Search via qmd (hybrid BM25 + vectors) if available
- Fallback to filesystem search
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ObsidianTools:
    """Synchronous tools for vault operations in sandboxed scripts.

    Provides:
    - read_note: Read a note's content
    - write_note: Create/update a note
    - search: Search for notes (uses qmd if available)
    - list_notes: List notes in a directory
    - get_frontmatter: Extract YAML frontmatter
    - update_frontmatter: Update frontmatter properties
    """

    def __init__(
        self,
        vault_path: Path | str,
        collection_name: str | None = None,
    ):
        """Initialize with vault path.

        Args:
            vault_path: Path to the vault
            collection_name: Optional qmd collection name for this vault
        """
        self.vault_path = Path(vault_path)
        self.collection_name = collection_name or self.vault_path.name
        self._qmd_available: bool | None = None

    def read_note(self, path: str) -> str | None:
        """Read a note's content.

        Args:
            path: Relative path to the note

        Returns:
            Note content or None if not found
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            return None

        try:
            return full_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    def write_note(
        self,
        path: str,
        content: str,
        note_type: str = "note",
        frontmatter: dict[str, Any] | None = None,
    ) -> bool:
        """Write a new note.

        Args:
            path: Relative path for the note
            content: Note body content
            note_type: Type for frontmatter (note, source, task, etc.)
            frontmatter: Additional frontmatter fields

        Returns:
            True if successful
        """
        full_path = self.vault_path / path

        try:
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Build frontmatter
            fm = {"type": note_type}
            if frontmatter:
                fm.update(frontmatter)

            # Build content with frontmatter
            yaml_str = yaml.dump(fm, default_flow_style=False, sort_keys=False)
            final_content = f"---\n{yaml_str}---\n\n{content}"

            full_path.write_text(final_content, encoding="utf-8")
            return True

        except Exception as e:
            logger.error(f"Error writing {path}: {e}")
            return False

    @property
    def qmd_available(self) -> bool:
        """Check if qmd is available on the system."""
        if self._qmd_available is None:
            self._qmd_available = shutil.which("qmd") is not None
        return self._qmd_available

    def _search_via_qmd(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]] | None:
        """Search via qmd CLI.

        Uses qmd's hybrid search (BM25 + vectors + reranking).

        Args:
            query: Search query string
            limit: Maximum results
            mode: Search mode - "bm25", "vector", or "hybrid"

        Returns:
            List of search results, or None if qmd unavailable
        """
        if not self.qmd_available:
            return None

        # Map mode to qmd command
        cmd_map = {
            "bm25": "search",
            "vector": "vsearch",
            "hybrid": "query",
        }
        cmd = cmd_map.get(mode, "query")

        args = [
            "qmd", cmd, query,
            "-n", str(limit),
            "--json",
            "-c", self.collection_name,
        ]

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(f"qmd search failed: {result.stderr}")
                return None

            if not result.stdout:
                return []

            data = json.loads(result.stdout)
            results = []

            # Handle both list and dict formats
            items = data if isinstance(data, list) else data.get("results", [])

            for item in items:
                path = item.get("path", item.get("filepath", ""))
                results.append({
                    "path": path,
                    "content": item.get("content", ""),
                    "score": float(item.get("score", 0.0)),
                    "snippet": item.get("snippet", item.get("context", "")),
                    "matched_line": item.get("matched_line"),
                    "collection": item.get("collection", self.collection_name),
                    "docid": item.get("docid", ""),
                })

            return results

        except subprocess.TimeoutExpired:
            logger.warning("qmd search timed out")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse qmd output: {e}")
            return None
        except Exception as e:
            logger.warning(f"qmd search error: {e}")
            return None

    def _search_local(
        self,
        query: str,
        limit: int = 10,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search locally via filesystem traversal.

        Fallback when Obsidian API is unavailable.

        Args:
            query: Search query string
            limit: Maximum results
            path_filter: Optional path prefix filter

        Returns:
            List of search results
        """
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        try:
            for md_file in self.vault_path.rglob("*.md"):
                # Skip hidden and artifact directories
                if any(part.startswith(".") for part in md_file.parts):
                    continue
                if "artifacts" in md_file.parts:
                    continue

                relative_path = str(md_file.relative_to(self.vault_path))

                # Apply path filter
                if path_filter and not relative_path.startswith(path_filter):
                    continue

                try:
                    content = md_file.read_text(encoding="utf-8")

                    if query_lower in content.lower():
                        # Find matched line
                        matched_line = None
                        for line in content.split("\n"):
                            if query_lower in line.lower():
                                matched_line = line.strip()
                                break

                        # Simple relevance score
                        score = content.lower().count(query_lower) / len(content) * 100

                        results.append({
                            "path": relative_path,
                            "content": content,
                            "score": min(score, 1.0),
                            "matched_line": matched_line,
                        })

                        if len(results) >= limit:
                            break

                except Exception:
                    continue

            # Sort by score
            results.sort(key=lambda r: r.get("score", 0), reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Local search error: {e}")
            return []

    def search(
        self,
        query: str,
        limit: int = 10,
        path_filter: str | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search for notes matching query.

        Uses qmd for hybrid search (BM25 + vectors) if available,
        falling back to local filesystem search.

        Args:
            query: Search query string
            limit: Maximum results
            path_filter: Optional path prefix filter (only for filesystem search)
            mode: Search mode - "bm25" (fast), "vector" (semantic), "hybrid" (best)

        Returns:
            List of search results with keys:
            - path: Relative path to the note
            - content: Full note content
            - score: Relevance score (0.0 to 1.0)
            - snippet: Highlighted snippet (qmd only)
            - matched_line: First line containing match
        """
        # Try qmd search first (hybrid search with vectors)
        if self.qmd_available and not path_filter:
            # Note: qmd doesn't support path_filter, so skip for filtered searches
            qmd_results = self._search_via_qmd(query, limit, mode)
            if qmd_results is not None:
                logger.debug(f"Search via qmd returned {len(qmd_results)} results")
                return qmd_results

        # Fall back to local filesystem search
        if mode in ("vector", "hybrid"):
            logger.debug(f"Falling back to bm25 search (qmd unavailable for {mode})")
        return self._search_local(query, limit, path_filter)

    def list_notes(
        self,
        directory: str = "",
        recursive: bool = False,
    ) -> list[str]:
        """List notes in a directory.

        Args:
            directory: Directory relative to vault root
            recursive: Whether to recurse into subdirectories

        Returns:
            List of relative paths
        """
        dir_path = self.vault_path / directory

        if not dir_path.exists():
            return []

        pattern = "**/*.md" if recursive else "*.md"
        results = []

        for md_file in dir_path.glob(pattern):
            if md_file.is_file():
                results.append(str(md_file.relative_to(self.vault_path)))

        return results

    def get_frontmatter(self, path: str) -> dict[str, Any] | None:
        """Extract frontmatter from a note.

        Args:
            path: Relative path to the note

        Returns:
            Frontmatter dict or None
        """
        content = self.read_note(path)
        if not content:
            return None

        if not content.startswith("---"):
            return {}

        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return {}

        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return {}

    def update_frontmatter(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update frontmatter properties.

        Args:
            path: Relative path to the note
            updates: Frontmatter updates to apply

        Returns:
            True if successful
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            return False

        try:
            content = full_path.read_text(encoding="utf-8")

            # Parse existing frontmatter
            frontmatter: dict[str, Any] = {}
            body = content

            if content.startswith("---"):
                match = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
                if match:
                    frontmatter = yaml.safe_load(match.group(1)) or {}
                    body = match.group(2)

            # Apply updates
            frontmatter.update(updates)

            # Rebuild content
            yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            new_content = f"---\n{yaml_str}---\n{body}"

            full_path.write_text(new_content, encoding="utf-8")
            return True

        except Exception as e:
            logger.error(f"Error updating frontmatter: {e}")
            return False

    def get_hash(self, path: str) -> str | None:
        """Get content hash for a file.

        Args:
            path: Relative path to the file

        Returns:
            Content hash or None
        """
        content = self.read_note(path)
        if content is None:
            return None

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def create_wikilink(self, path: str) -> str:
        """Create a full-path wikilink.

        Per OBSIDIAN_PIVOT.md Section 4.3: Full Path Always.

        Args:
            path: Relative path to the note

        Returns:
            Full-path wikilink
        """
        # Remove .md extension
        if path.endswith(".md"):
            path = path[:-3]
        return f"[[{path}]]"


# Global instance for import in sandboxed code
# This will be set by the executor with the correct vault path
obsidian: ObsidianTools | None = None


def init_tools(
    vault_path: Path,
    collection_name: str | None = None,
) -> ObsidianTools:
    """Initialize the global obsidian tools instance.

    Args:
        vault_path: Path to the vault
        collection_name: Optional qmd collection name for this vault

    Returns:
        ObsidianTools instance
    """
    global obsidian
    obsidian = ObsidianTools(
        vault_path=vault_path,
        collection_name=collection_name,
    )
    return obsidian
