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

    # Language mappings for code blocks
    LANGUAGE_MAP = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "fish",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
        ".md": "markdown",
        ".mdx": "mdx",
        ".rst": "rst",
        ".txt": "text",
    }

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

    def _is_markdown_file(self, path: Path) -> bool:
        """Check if file is already markdown."""
        return path.suffix.lower() in (".md", ".mdx")

    def _convert_file_to_note(
        self,
        source_file: Path,
        target_path: Path,
        rel_path: str,
        metadata: SnapshotMetadata,
    ) -> Path:
        """Convert a source file to a vault note.

        For markdown files: preserves content, adds/merges frontmatter.
        For code files: wraps content in code block with language hint.

        Args:
            source_file: Path to the source file
            target_path: Directory to write the note to
            rel_path: Relative path of file in source repo
            metadata: Snapshot metadata for frontmatter

        Returns:
            Path to the created note
        """
        try:
            content = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary file, skip
            logger.debug(f"Skipping binary file: {rel_path}")
            return None

        suffix = source_file.suffix.lower()
        is_markdown = self._is_markdown_file(source_file)

        # Build frontmatter
        frontmatter = {
            "type": "source",
            "source_url": metadata.source_url,
            "source_path": rel_path,
            "snapshot_id": metadata.snapshot_id,
            "ingested_at": metadata.ingested_at.isoformat(),
        }

        if not is_markdown:
            language = self.LANGUAGE_MAP.get(suffix, "")
            if language:
                frontmatter["language"] = language

        # Build note content
        lines = ["---"]
        for key, value in frontmatter.items():
            lines.append(f"{key}: {value}")
        lines.append("---")
        lines.append("")

        if is_markdown:
            # For markdown, strip existing frontmatter and use content directly
            if content.startswith("---"):
                # Find end of frontmatter
                end_match = re.search(r"\n---\n", content[3:])
                if end_match:
                    content = content[3 + end_match.end():]
            lines.append(content)
        else:
            # For code files, add header and wrap in code block
            lines.append(f"# {source_file.name}")
            lines.append("")
            lines.append(f"Source: `{rel_path}`")
            lines.append("")
            language = self.LANGUAGE_MAP.get(suffix, "")
            lines.append(f"```{language}")
            lines.append(content)
            if not content.endswith("\n"):
                lines.append("")
            lines.append("```")

        # Determine target file path
        # Convert source path to note path (preserve directory structure)
        note_rel_path = Path(rel_path)
        if not is_markdown:
            # Add .md extension for non-markdown files
            note_rel_path = note_rel_path.with_suffix(note_rel_path.suffix + ".md")

        note_path = target_path / note_rel_path
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("\n".join(lines), encoding="utf-8")

        return note_path

    def _convert_files_to_notes(
        self,
        source_path: Path,
        target_path: Path,
        files: list[str],
        metadata: SnapshotMetadata,
    ) -> list[str]:
        """Convert all matched files to vault notes.

        Args:
            source_path: Path to cloned source
            target_path: Snapshot directory for notes
            files: List of relative file paths to convert
            metadata: Snapshot metadata

        Returns:
            List of created note paths (relative to target_path)
        """
        created_notes = []

        for rel_path in files:
            source_file = source_path / rel_path
            if not source_file.exists():
                continue

            note_path = self._convert_file_to_note(
                source_file=source_file,
                target_path=target_path,
                rel_path=rel_path,
                metadata=metadata,
            )

            if note_path:
                # Store relative path for wikilinks
                note_rel = str(note_path.relative_to(target_path))
                created_notes.append(note_rel)
                logger.debug(f"Created note: {note_rel}")

        return created_notes

    def _create_entry_note(
        self,
        snapshot_path: Path,
        metadata: SnapshotMetadata,
        note_paths: list[str],
    ) -> Path:
        """Create entry note for snapshot.

        Args:
            snapshot_path: Directory for the snapshot
            metadata: Snapshot metadata
            note_paths: List of created note paths (relative to snapshot_path)

        Returns:
            Path to the entry note
        """
        snapshot_path.mkdir(parents=True, exist_ok=True)
        entry_path = snapshot_path / "_entry.md"

        # Get snapshot path relative to vault for full wikilinks
        snapshot_rel = snapshot_path.relative_to(self.vault_path)

        content = [
            metadata.to_yaml(),
            "",
            f"# {metadata.snapshot_id}",
            "",
            f"Ingested from: {metadata.source_url}",
            f"Revision: {metadata.source_revision}",
            f"Files: {len(note_paths)}",
            "",
            "## Contents",
            "",
        ]

        for note_path in sorted(note_paths)[:50]:  # Show more files
            # Create full path wikilink (remove .md for Obsidian)
            wikilink_path = str(snapshot_rel / note_path)
            if wikilink_path.endswith(".md"):
                wikilink_path = wikilink_path[:-3]
            content.append(f"- [[{wikilink_path}]]")

        if len(note_paths) > 50:
            content.append(f"- ... and {len(note_paths) - 50} more")

        entry_path.write_text("\n".join(content))
        return entry_path

    async def ingest_github(
        self,
        url: str,
        preset: IngestPreset = IngestPreset.DOCS,
        keep_clone: bool = False,
    ) -> IngestResult:
        """Ingest a GitHub repository.

        Clones the repository, converts matched files to vault notes with
        proper frontmatter, and creates an entry note linking to all content.

        Args:
            url: GitHub repository URL
            preset: Ingestion preset controlling what to include
            keep_clone: If True, keep the cloned repo (default: remove after processing)

        Returns:
            IngestResult with success status and metadata
        """
        import shutil

        source_id = self.extract_source_id(url)
        domain = self._extract_domain(url)
        now = datetime.now(UTC)

        snapshot_id = f"{source_id}@{now.strftime('%Y-%m-%d')}"
        snapshot_path = self.generate_snapshot_path(source_id, domain, now)
        clone_path = snapshot_path / "_clone"

        try:
            # Clone repository
            await self._clone_repo(url, clone_path)

            # Get revision
            revision = self._get_git_revision(clone_path)

            # Process files (get list of files matching preset)
            files = self._process_files(clone_path, preset)

            if not files:
                # Clean up empty clone
                if clone_path.exists():
                    shutil.rmtree(clone_path)
                return IngestResult(
                    success=False,
                    errors=["No files matched the preset criteria"],
                )

            # Compute content hash before conversion
            content_hash = self._compute_content_hash(files, clone_path)

            # Create metadata
            metadata = SnapshotMetadata(
                source_url=url,
                source_type="github_repo",
                ingested_at=now,
                source_revision=revision,
                content_hash=content_hash,
                snapshot_id=snapshot_id,
                extra={"preset": preset.value},
            )

            # Convert files to vault notes
            logger.info(f"Converting {len(files)} files to vault notes...")
            note_paths = self._convert_files_to_notes(
                source_path=clone_path,
                target_path=snapshot_path,
                files=files,
                metadata=metadata,
            )

            if not note_paths:
                return IngestResult(
                    success=False,
                    errors=["Failed to convert any files to notes"],
                )

            # Create entry note with links to all converted notes
            self._create_entry_note(snapshot_path, metadata, note_paths)

            # Clean up clone directory (keep notes only)
            if not keep_clone and clone_path.exists():
                shutil.rmtree(clone_path)
                logger.debug(f"Cleaned up clone directory: {clone_path}")

            logger.info(f"Ingested {len(note_paths)} notes from {url}")

            return IngestResult(
                success=True,
                snapshot_id=snapshot_id,
                files_processed=note_paths,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            # Clean up on failure
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            return IngestResult(
                success=False,
                errors=[str(e)],
            )
