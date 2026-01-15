"""Vault ingestion pipeline.

Per SPEC.md Section 2.3 and 2.12:
- Ingestion pipeline: Clone, process repos with presets
- Snapshot metadata with frontmatter conversion
- Entry note generation for ingested content
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IngestPreset(str, Enum):
    """Ingestion presets defining what content to include.

    Per SPEC.md Section 2.12.1:
    - MIRROR: All files (complete copy)
    - DOCS: Documentation files only (default)
    - DEEP: Semantic index with code analysis
    """

    MIRROR = "mirror"
    DOCS = "docs"
    DEEP = "deep"


@dataclass
class SnapshotMetadata:
    """Metadata for an ingested snapshot.

    Tracks provenance information for vault content.
    """

    source_url: str
    source_type: str
    ingested_at: datetime
    source_revision: str
    content_hash: str
    snapshot_id: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_frontmatter(self) -> dict[str, Any]:
        """Convert metadata to YAML frontmatter format."""
        return {
            "source_url": self.source_url,
            "source_type": self.source_type,
            "ingested_at": self.ingested_at.isoformat(),
            "source_revision": self.source_revision,
            "content_hash": self.content_hash,
            "snapshot_id": self.snapshot_id,
            **self.extra,
        }

    def to_yaml(self) -> str:
        """Convert metadata to YAML string."""
        lines = ["---"]
        for key, value in self.to_frontmatter().items():
            lines.append(f"{key}: {value}")
        lines.append("---")
        return "\n".join(lines)


@dataclass
class IngestResult:
    """Result of an ingestion operation."""

    success: bool
    snapshot_id: str | None = None
    files_processed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: SnapshotMetadata | None = None


class VaultIngester:
    """Ingests external sources into the vault.

    Per SPEC.md Section 2.12:
    - Clones repositories
    - Processes files according to preset
    - Generates entry notes with metadata
    - Creates snapshot directory structure
    """

    # File patterns for DOCS preset
    DOCS_PATTERNS = [
        "*.md",
        "*.mdx",
        "*.txt",
        "*.rst",
        "docs/**/*",
        "documentation/**/*",
        "README*",
        "CHANGELOG*",
        "LICENSE*",
    ]

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path
        self._sources_dir = vault_path / "Sources"
        self._sources_dir.mkdir(exist_ok=True)

    def generate_snapshot_path(
        self,
        source_id: str,
        domain: str,
        date: datetime | None = None,
    ) -> Path:
        """Generate snapshot path following Sources/<domain>/<source_id>/<date>/ pattern."""
        if date is None:
            date = datetime.now(UTC)

        date_str = date.strftime("%Y-%m-%d")
        return self._sources_dir / domain / source_id / date_str

    def extract_source_id(self, url: str) -> str:
        """Extract source ID from URL.

        For GitHub URLs, extracts the repository name.
        """
        # Handle GitHub URLs
        match = re.search(r"github\.com/[^/]+/([^/]+?)(?:\.git)?$", url)
        if match:
            return match.group(1)

        # Fallback: use last path component
        parts = url.rstrip("/").split("/")
        return parts[-1].replace(".git", "")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r"https?://([^/]+)", url)
        if match:
            return match.group(1)
        return "unknown"

    async def _clone_repo(self, url: str, target: Path) -> Path:
        """Clone a git repository."""
        target.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", url, str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

        return target

    def _get_git_revision(self, repo_path: Path) -> str:
        """Get current git revision."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()[:7]
        except Exception:
            return "unknown"

    def _should_include_file(self, path: Path, preset: IngestPreset) -> bool:
        """Check if file should be included based on preset."""
        if preset == IngestPreset.MIRROR:
            return True

        if preset == IngestPreset.DOCS:
            # Check against docs patterns
            name = path.name.lower()
            suffix = path.suffix.lower()

            # Documentation file extensions
            if suffix in (".md", ".mdx", ".txt", ".rst"):
                return True

            # Common documentation files
            if name.startswith(("readme", "changelog", "license", "contributing")):
                return True

            # Files in docs directories
            parts = [p.lower() for p in path.parts]
            if "docs" in parts or "documentation" in parts:
                return True

            return False

        if preset == IngestPreset.DEEP:
            # Include docs plus source code
            suffix = path.suffix.lower()
            return suffix in (
                ".md", ".mdx", ".txt", ".rst",
                ".py", ".ts", ".tsx", ".js", ".jsx",
                ".java", ".go", ".rs", ".rb",
            )

        return False

    def _process_files(
        self,
        source_path: Path,
        preset: IngestPreset,
    ) -> list[str]:
        """Process files from source according to preset."""
        processed = []

        for file_path in source_path.rglob("*"):
            if file_path.is_file() and self._should_include_file(file_path, preset):
                rel_path = file_path.relative_to(source_path)
                processed.append(str(rel_path))

        return processed

    def _compute_content_hash(self, files: list[str], source_path: Path) -> str:
        """Compute content hash for files."""
        hasher = hashlib.sha256()

        for file_rel in sorted(files):
            file_path = source_path / file_rel
            if file_path.exists():
                hasher.update(file_path.read_bytes())

        return f"sha256:{hasher.hexdigest()[:16]}"

    def _create_entry_note(
        self,
        snapshot_path: Path,
        metadata: SnapshotMetadata,
        files: list[str],
    ) -> Path:
        """Create entry note for snapshot."""
        snapshot_path.mkdir(parents=True, exist_ok=True)
        entry_path = snapshot_path / "_entry.md"

        content = [
            metadata.to_yaml(),
            "",
            f"# {metadata.snapshot_id}",
            "",
            f"Ingested from: {metadata.source_url}",
            f"Revision: {metadata.source_revision}",
            f"Files: {len(files)}",
            "",
            "## Contents",
            "",
        ]

        for f in sorted(files)[:20]:  # Limit to first 20
            content.append(f"- [[{f}]]")

        if len(files) > 20:
            content.append(f"- ... and {len(files) - 20} more")

        entry_path.write_text("\n".join(content))
        return entry_path

    async def ingest_github(
        self,
        url: str,
        preset: IngestPreset = IngestPreset.DOCS,
    ) -> IngestResult:
        """Ingest a GitHub repository.

        Args:
            url: GitHub repository URL
            preset: Ingestion preset controlling what to include

        Returns:
            IngestResult with success status and metadata
        """
        source_id = self.extract_source_id(url)
        domain = self._extract_domain(url)
        now = datetime.now(UTC)

        snapshot_id = f"{source_id}@{now.strftime('%Y-%m-%d')}"
        snapshot_path = self.generate_snapshot_path(source_id, domain, now)

        try:
            # Clone repository
            clone_path = await self._clone_repo(url, snapshot_path / "_clone")

            # Get revision
            revision = self._get_git_revision(clone_path)

            # Process files
            files = self._process_files(clone_path, preset)

            if not files:
                return IngestResult(
                    success=False,
                    errors=["No files matched the preset criteria"],
                )

            # Compute content hash
            content_hash = self._compute_content_hash(files, clone_path)

            # Create metadata
            metadata = SnapshotMetadata(
                source_url=url,
                source_type="github_repo",
                ingested_at=now,
                source_revision=revision,
                content_hash=content_hash,
                snapshot_id=snapshot_id,
            )

            # Create entry note
            self._create_entry_note(snapshot_path, metadata, files)

            return IngestResult(
                success=True,
                snapshot_id=snapshot_id,
                files_processed=files,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestResult(
                success=False,
                errors=[str(e)],
            )
