"""Obsidian Tools for sandboxed code execution.

Per OBSIDIAN_PIVOT.md Section 3.1:
These tools are importable by RLM-generated scripts for
interacting with the Obsidian vault.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)


class ObsidianTools:
    """Synchronous tools for vault operations in sandboxed scripts.

    These mirror the MCP tools but are synchronous for use in exec'd code:
    - read_note -> obsidian_read_note
    - write_note -> obsidian_create_note
    - search -> obsidian_global_search
    - list_notes -> obsidian_list_notes
    - update_frontmatter -> obsidian_manage_frontmatter
    """

    def __init__(
        self,
        vault_path: Path | str,
        api_url: str | None = None,
        api_key: str | None = None,
        verify_ssl: bool = False,
    ):
        """Initialize with vault path and optional API configuration.

        Args:
            vault_path: Path to the Obsidian vault
            api_url: Optional Obsidian Local REST API URL for indexed search
            api_key: Optional API key for authentication
            verify_ssl: Whether to verify SSL certificates (default False for local)
        """
        self.vault_path = Path(vault_path)
        self.api_url = api_url
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self._client: httpx.Client | None = None

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

    def _get_client(self) -> httpx.Client | None:
        """Get or create HTTP client for API calls.

        Returns:
            httpx.Client if API is configured, None otherwise
        """
        if not self.api_url:
            return None

        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.Client(
                base_url=self.api_url,
                headers=headers,
                verify=self.verify_ssl,
                timeout=30.0,
            )

        return self._client

    def _search_via_api(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]] | None:
        """Search via Obsidian Local REST API.

        Uses Obsidian's indexed search for faster queries on large vaults.

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of search results, or None if API unavailable
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            response = client.post(
                "/search/simple",
                params={"query": query},
            )

            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data[:limit]:
                    path = item.get("filename", "")
                    # Read full content for consistency with local search
                    content = self.read_note(path) or item.get("content", "")
                    results.append({
                        "path": path,
                        "content": content,
                        "score": item.get("score", 0.0),
                        "matched_line": item.get("matches", [None])[0] if item.get("matches") else None,
                    })
                return results
            else:
                logger.warning(f"Obsidian API returned status {response.status_code}")
                return None

        except httpx.ConnectError:
            logger.debug("Obsidian API not available, falling back to local search")
            return None
        except Exception as e:
            logger.warning(f"Obsidian API search error: {e}")
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
    ) -> list[dict[str, Any]]:
        """Search for notes matching query.

        Attempts to use Obsidian's indexed search API first for performance,
        falling back to local filesystem search if unavailable.

        Args:
            query: Search query string
            limit: Maximum results
            path_filter: Optional path prefix filter

        Returns:
            List of search results
        """
        # Try API search first (much faster for large vaults)
        if self.api_url and not path_filter:
            # Note: API doesn't support path_filter, so skip for filtered searches
            api_results = self._search_via_api(query, limit)
            if api_results is not None:
                logger.debug(f"Search via Obsidian API returned {len(api_results)} results")
                return api_results

        # Fall back to local search
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
    api_url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = False,
) -> ObsidianTools:
    """Initialize the global obsidian tools instance.

    Args:
        vault_path: Path to the vault
        api_url: Optional Obsidian Local REST API URL for indexed search
        api_key: Optional API key for authentication
        verify_ssl: Whether to verify SSL certificates

    Returns:
        ObsidianTools instance
    """
    global obsidian
    obsidian = ObsidianTools(
        vault_path=vault_path,
        api_url=api_url,
        api_key=api_key,
        verify_ssl=verify_ssl,
    )
    return obsidian
