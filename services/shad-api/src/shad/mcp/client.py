"""Obsidian MCP Client.

Connects to the Obsidian MCP Server (cyanheads/obsidian-mcp-server)
via the Local REST API plugin.

Per OBSIDIAN_PIVOT.md Section 2 and Section 5.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

from shad.mcp.models import (
    DeleteResult,
    NoteMetadata,
    NoteType,
    SearchResult,
    VaultFile,
    VaultNote,
)
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class ObsidianMCPClient:
    """Client for Obsidian MCP Server.

    Implements the MCP tools interface per OBSIDIAN_PIVOT.md Section 5.1:
    - obsidian_global_search: Vector or full-text search
    - obsidian_read_note: Retrieve file content and frontmatter
    - obsidian_list_notes: Scan directories
    - obsidian_manage_frontmatter: Update YAML properties
    - obsidian_create_note: Write new notes
    - obsidian_update_note: Append/overwrite existing notes
    - obsidian_delete_note: Delete note (HITL gated)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        vault_path: Path | str | None = None,
        verify_ssl: bool | None = None,
    ):
        """Initialize the client.

        Args:
            base_url: Obsidian Local REST API URL (default from settings)
            api_key: API key for authentication
            vault_path: Path to the Obsidian vault for direct file access
            verify_ssl: Whether to verify SSL certificates
        """
        settings = get_settings()

        self.base_url = base_url or settings.obsidian_base_url
        self.api_key = api_key or settings.obsidian_api_key
        self.vault_path = Path(vault_path) if vault_path else Path(settings.obsidian_vault_path)
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.obsidian_verify_ssl

        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._file_cache: dict[str, VaultFile] = {}

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def _create_client(self) -> httpx.AsyncClient:
        """Create HTTP client with authentication."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            verify=self.verify_ssl,
            timeout=30.0,
        )

    async def connect(self) -> bool:
        """Connect to the Obsidian API.

        Returns:
            True if connection successful.
        """
        if self._connected:
            return True

        try:
            self._client = self._create_client()
            response = await self._client.get("/")
            if response.status_code == 200:
                self._connected = True
                logger.info(f"Connected to Obsidian at {self.base_url}")
                return True
            else:
                logger.warning(f"Obsidian API returned status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Obsidian: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the API."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False

    async def _ensure_connected(self) -> bool:
        """Ensure client is connected."""
        if self._connected:
            return True
        return await self.connect()

    # ==================== Note Operations ====================

    async def read_note(self, path: str) -> VaultNote | None:
        """Read a note from the vault.

        Per OBSIDIAN_PIVOT.md Section 5.1: obsidian_read_note

        Args:
            path: Relative path to the note within the vault

        Returns:
            VaultNote or None if not found
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            logger.debug(f"Note not found: {path}")
            return None

        try:
            content = full_path.read_text(encoding="utf-8")
            stat = full_path.stat()

            # Parse frontmatter
            frontmatter, body = self._parse_frontmatter(content)
            metadata = NoteMetadata.from_frontmatter(frontmatter)

            # Calculate content hash for cache validation
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            return VaultNote(
                path=path,
                content=content,
                metadata=metadata,
                raw_frontmatter=frontmatter,
                mtime=stat.st_mtime,
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"Error reading note {path}: {e}")
            return None

    async def write_note(
        self,
        path: str,
        content: str,
        metadata: NoteMetadata | None = None,
        resolve_wikilinks: bool = False,
    ) -> bool:
        """Write a note to the vault.

        Per OBSIDIAN_PIVOT.md Section 5.1: obsidian_create_note

        Args:
            path: Relative path for the note
            content: Note content (without frontmatter)
            metadata: Optional metadata to include as frontmatter
            resolve_wikilinks: Whether to resolve wikilinks to full paths

        Returns:
            True if successful
        """
        full_path = self.vault_path / path

        try:
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Build full content with frontmatter
            final_content = content
            if metadata:
                frontmatter = metadata.to_frontmatter()
                if frontmatter:
                    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
                    final_content = f"---\n{yaml_str}---\n\n{content}"

            # Resolve wikilinks to full paths if requested
            if resolve_wikilinks:
                final_content = await self._resolve_wikilinks(final_content)

            full_path.write_text(final_content, encoding="utf-8")
            logger.info(f"Wrote note: {path}")
            return True

        except Exception as e:
            logger.error(f"Error writing note {path}: {e}")
            return False

    async def update_note(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> bool:
        """Update an existing note.

        Per OBSIDIAN_PIVOT.md Section 5.1: obsidian_update_note

        Args:
            path: Relative path to the note
            content: New content or content to append
            append: If True, append instead of replace

        Returns:
            True if successful
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            logger.warning(f"Note not found for update: {path}")
            return False

        try:
            if append:
                existing = full_path.read_text(encoding="utf-8")
                full_path.write_text(existing + "\n" + content, encoding="utf-8")
            else:
                full_path.write_text(content, encoding="utf-8")

            logger.info(f"Updated note: {path}")
            return True

        except Exception as e:
            logger.error(f"Error updating note {path}: {e}")
            return False

    async def delete_note(self, path: str) -> DeleteResult:
        """Request deletion of a note (HITL gated).

        Per OBSIDIAN_PIVOT.md Section 10.2 and 12.4:
        Delete operations are routed through HITL queue.

        Args:
            path: Relative path to the note

        Returns:
            DeleteResult with HITL queue info
        """
        # Generate HITL queue ID for approval workflow
        hitl_id = str(uuid.uuid4())[:8]

        logger.info(f"Delete request queued for HITL: {path} (queue_id: {hitl_id})")

        return DeleteResult(
            path=path,
            requires_approval=True,
            hitl_queue_id=hitl_id,
            approved=False,
            deleted=False,
        )

    async def execute_delete(self, path: str) -> DeleteResult:
        """Execute a pre-approved delete operation.

        Should only be called after HITL approval.

        Args:
            path: Relative path to the note

        Returns:
            DeleteResult with outcome
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            return DeleteResult(
                path=path,
                requires_approval=False,
                approved=True,
                deleted=False,
                error="File not found",
            )

        try:
            full_path.unlink()
            logger.info(f"Deleted note: {path}")
            return DeleteResult(
                path=path,
                requires_approval=False,
                approved=True,
                deleted=True,
            )
        except Exception as e:
            return DeleteResult(
                path=path,
                requires_approval=False,
                approved=True,
                deleted=False,
                error=str(e),
            )

    # ==================== Frontmatter Operations ====================

    async def update_frontmatter(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update frontmatter properties.

        Per OBSIDIAN_PIVOT.md Section 5.1: obsidian_manage_frontmatter

        Args:
            path: Relative path to the note
            updates: Dictionary of frontmatter updates

        Returns:
            True if successful
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            logger.warning(f"Note not found: {path}")
            return False

        try:
            content = full_path.read_text(encoding="utf-8")
            frontmatter, body = self._parse_frontmatter(content)

            # Apply updates
            frontmatter.update(updates)

            # Rebuild content
            yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            new_content = f"---\n{yaml_str}---\n{body}"

            full_path.write_text(new_content, encoding="utf-8")
            logger.info(f"Updated frontmatter for: {path}")
            return True

        except Exception as e:
            logger.error(f"Error updating frontmatter {path}: {e}")
            return False

    # ==================== Search Operations ====================

    async def search(
        self,
        query: str,
        limit: int = 10,
        path_filter: str | None = None,
        type_filter: NoteType | None = None,
    ) -> list[SearchResult]:
        """Search the vault.

        Per OBSIDIAN_PIVOT.md Section 5.1: obsidian_global_search

        This is a local implementation for direct vault access.
        For MCP server search, use search_via_api().

        Args:
            query: Search query string
            limit: Maximum results to return
            path_filter: Optional path prefix filter
            type_filter: Optional note type filter

        Returns:
            List of SearchResult objects
        """
        results: list[SearchResult] = []
        query_lower = query.lower()

        try:
            for md_file in self.vault_path.rglob("*.md"):
                # Skip hidden directories and artifacts
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
                    frontmatter, body = self._parse_frontmatter(content)
                    metadata = NoteMetadata.from_frontmatter(frontmatter)

                    # Apply type filter
                    if type_filter and metadata.note_type != type_filter:
                        continue

                    # Simple text matching
                    if query_lower in content.lower():
                        # Find matched line for context
                        matched_line = None
                        for line in content.split("\n"):
                            if query_lower in line.lower():
                                matched_line = line.strip()
                                break

                        # Calculate simple relevance score
                        score = content.lower().count(query_lower) / len(content) * 100

                        results.append(SearchResult(
                            path=relative_path,
                            content=content,
                            score=min(score, 1.0),
                            matched_line=matched_line,
                            metadata=metadata,
                        ))

                        if len(results) >= limit:
                            break

                except Exception as e:
                    logger.debug(f"Error reading file {md_file}: {e}")

            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def search_via_api(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search via Obsidian MCP Server API.

        Per OBSIDIAN_PIVOT.md Section 5.2: Uses MCP server's VaultCacheService.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of SearchResult objects
        """
        if not await self._ensure_connected():
            logger.warning("Not connected, falling back to local search")
            return await self.search(query, limit)

        try:
            response = await self._client.post(
                "/search/simple/",
                json={"query": query},
            )

            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data[:limit]:
                    results.append(SearchResult(
                        path=item.get("filename", ""),
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                    ))
                return results
            else:
                logger.warning(f"Search API returned {response.status_code}")
                return await self.search(query, limit)

        except Exception as e:
            logger.error(f"API search error: {e}")
            return await self.search(query, limit)

    # ==================== Directory Operations ====================

    async def list_notes(
        self,
        directory: str = "",
        recursive: bool = False,
    ) -> list[VaultFile]:
        """List notes in a directory.

        Per OBSIDIAN_PIVOT.md Section 5.1: obsidian_list_notes

        Args:
            directory: Directory path relative to vault root
            recursive: Whether to recurse into subdirectories

        Returns:
            List of VaultFile objects
        """
        dir_path = self.vault_path / directory

        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return []

        results: list[VaultFile] = []

        try:
            pattern = "**/*.md" if recursive else "*.md"
            for md_file in dir_path.glob(pattern):
                if md_file.is_file():
                    results.append(VaultFile.from_path(md_file, self.vault_path))

            return results

        except Exception as e:
            logger.error(f"Error listing notes: {e}")
            return []

    # ==================== Cache Operations ====================

    async def get_file_hash(self, path: str) -> str | None:
        """Get content hash for cache validation.

        Per OBSIDIAN_PIVOT.md Section 6.2: Hash Validation strategy.

        Args:
            path: Relative path to the file

        Returns:
            Content hash or None if file not found
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            return None

        try:
            content = full_path.read_text(encoding="utf-8")
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error hashing file {path}: {e}")
            return None

    async def get_file_mtime(self, path: str) -> float | None:
        """Get file modification time.

        Args:
            path: Relative path to the file

        Returns:
            Modification time or None if not found
        """
        full_path = self.vault_path / path

        if not full_path.exists():
            return None

        try:
            return full_path.stat().st_mtime
        except Exception:
            return None

    # ==================== Helper Methods ====================

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from content.

        Args:
            content: Full note content

        Returns:
            Tuple of (frontmatter dict, body content)
        """
        if not content.startswith("---"):
            return {}, content

        match = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
        if not match:
            return {}, content

        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
            body = match.group(2)
            return frontmatter, body
        except yaml.YAMLError:
            return {}, content

    async def _resolve_wikilinks(self, content: str) -> str:
        """Resolve bare wikilinks to full paths.

        Per OBSIDIAN_PIVOT.md Section 4.3: Full Path Always.

        Args:
            content: Content with wikilinks

        Returns:
            Content with resolved wikilinks
        """
        # Find all wikilinks
        wikilink_pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"

        def resolve_link(match: re.Match) -> str:
            link_text = match.group(1)

            # Skip if already has path separator
            if "/" in link_text:
                return match.group(0)

            # Search for matching file
            for md_file in self.vault_path.rglob(f"{link_text}.md"):
                relative = str(md_file.relative_to(self.vault_path))[:-3]  # Remove .md
                return f"[[{relative}]]"

            # Return original if not found
            return match.group(0)

        return re.sub(wikilink_pattern, resolve_link, content)

    async def standardize_legacy_note(self, path: str) -> bool:
        """Standardize a legacy note by inferring and adding type.

        Per OBSIDIAN_PIVOT.md Section 4.2: Progressive Standardization.

        Args:
            path: Path to the legacy note

        Returns:
            True if standardized successfully
        """
        note = await self.read_note(path)
        if not note or not note.metadata.is_legacy:
            return False

        # Infer type based on content/location
        inferred_type = NoteType.NOTE  # Default

        # Check path patterns
        path_lower = path.lower()
        if "source" in path_lower or "raw" in path_lower:
            inferred_type = NoteType.SOURCE
        elif "task" in path_lower:
            inferred_type = NoteType.TASK
        elif "learning" in path_lower:
            inferred_type = NoteType.LEARNING

        # Update frontmatter
        return await self.update_frontmatter(path, {
            "type": inferred_type.value,
            "shad_processed": True,
        })
