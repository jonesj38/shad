"""File manifest for code generation output.

Per SPEC.md Section 2.7:
- Code generation produces file manifests (not raw text)
- Manifests contain paths, content, language, hash, source_nodes
- Writing to filesystem is always explicit (--write-files flag)
- Export index maps symbols to files for import resolution
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WriteMode(str, Enum):
    """Mode for writing files."""

    CREATE = "create"  # New file
    UPDATE = "update"  # Modify existing


@dataclass
class FileEntry:
    """A file entry in the manifest.

    Per SPEC.md Section 2.7.1:
    - path: File path relative to output root
    - content: File content
    - language: Programming language
    - hash: SHA256 hash for integrity verification
    - source_nodes: DAG nodes that contributed to this file
    """

    path: str
    content: str
    language: str
    mode: WriteMode = WriteMode.CREATE
    source_nodes: list[str] = field(default_factory=list)
    _content_hash: str | None = field(default=None, repr=False)

    @property
    def content_hash(self) -> str:
        """Get content hash, computing if needed."""
        if self._content_hash is None:
            self._content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return self._content_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "content": self.content,
            "language": self.language,
            "mode": self.mode.value,
            "hash": self.content_hash,
            "source_nodes": self.source_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileEntry:
        """Create from dictionary."""
        mode = WriteMode(data.get("mode", "create"))
        entry = cls(
            path=data["path"],
            content=data["content"],
            language=data.get("language", ""),
            mode=mode,
            source_nodes=data.get("source_nodes", []),
        )
        # Set hash if provided
        if "hash" in data:
            entry._content_hash = data["hash"]
        return entry


@dataclass
class ManifestNote:
    """A note or request in the manifest.

    Used for contract change requests, warnings, etc.
    """

    kind: str
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "detail": self.detail,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestNote:
        """Create from dictionary."""
        return cls(
            kind=data["kind"],
            detail=data["detail"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileManifest:
    """A manifest of files produced by code generation.

    Per SPEC.md Section 2.7.1:
    - Contains all files to be written
    - Tracks source nodes for each file
    - Includes notes/requests for contract changes
    """

    run_id: str
    files: list[FileEntry] = field(default_factory=list)
    notes: list[ManifestNote] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_file(self, entry: FileEntry) -> None:
        """Add a file entry to the manifest.

        Raises ValueError if path already exists.
        """
        for existing in self.files:
            if existing.path == entry.path:
                raise ValueError(f"File {entry.path} already exists in manifest")
        self.files.append(entry)

    def update_file(self, entry: FileEntry) -> None:
        """Update an existing file entry."""
        for i, existing in enumerate(self.files):
            if existing.path == entry.path:
                self.files[i] = entry
                return
        # If not found, add it
        self.files.append(entry)

    def get_file(self, path: str) -> FileEntry | None:
        """Get a file entry by path."""
        for entry in self.files:
            if entry.path == path:
                return entry
        return None

    def add_note(self, note: ManifestNote) -> None:
        """Add a note to the manifest."""
        self.notes.append(note)

    def validate(self) -> list[str]:
        """Validate manifest integrity.

        Returns list of validation errors.
        """
        errors: list[str] = []

        # Check for duplicate paths
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            seen = set()
            for path in paths:
                if path in seen:
                    errors.append(f"Duplicate path: {path}")
                seen.add(path)

        # Verify content hashes
        for entry in self.files:
            actual_hash = hashlib.sha256(entry.content.encode()).hexdigest()
            if entry._content_hash and entry._content_hash != actual_hash:
                errors.append(f"Hash mismatch for {entry.path}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "files": [f.to_dict() for f in self.files],
            "notes": [n.to_dict() for n in self.notes],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileManifest:
        """Create from dictionary."""
        manifest = cls(
            run_id=data["run_id"],
            metadata=data.get("metadata", {}),
        )
        for file_data in data.get("files", []):
            manifest.files.append(FileEntry.from_dict(file_data))
        for note_data in data.get("notes", []):
            manifest.notes.append(ManifestNote.from_dict(note_data))
        return manifest

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class WriteReport:
    """Report of manifest write operation.

    Includes written files, skipped conflicts, and errors.
    """

    success: bool
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)


class ManifestWriter:
    """Writes file manifests to disk.

    Per SPEC.md Section 2.7.4:
    - Only write under output_root (no ../ traversal)
    - --overwrite flag for conflicts (default: fail)
    - Emit write report with paths, skipped conflicts, hashes
    """

    def __init__(
        self,
        output_root: Path,
        overwrite: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.overwrite = overwrite

    def write(self, manifest: FileManifest) -> WriteReport:
        """Write manifest files to disk.

        Returns WriteReport with results.
        """
        report = WriteReport(success=True)

        for entry in manifest.files:
            try:
                result = self._write_file(entry)
                if result is None:
                    report.written.append(entry.path)
                    report.hashes[entry.path] = entry.content_hash
                elif result == "skipped":
                    report.skipped.append(entry.path)
                    report.success = False
            except Exception as e:
                report.errors.append(str(e))
                report.success = False

        return report

    def _write_file(self, entry: FileEntry) -> str | None:
        """Write a single file.

        Returns None on success, "skipped" if conflict, raises on error.
        """
        # Validate path (no traversal)
        if ".." in entry.path or entry.path.startswith("/"):
            raise ValueError(f"Path traversal not allowed: {entry.path}")

        target_path = self.output_root / entry.path

        # Ensure target is under output_root
        try:
            target_path.resolve().relative_to(self.output_root.resolve())
        except ValueError as e:
            raise ValueError(f"Path traversal not allowed: {entry.path}") from e

        # Check for conflicts
        if target_path.exists() and not self.overwrite:
            logger.warning(f"File exists, skipping: {entry.path}")
            return "skipped"

        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        target_path.write_text(entry.content)
        logger.info(f"Wrote: {entry.path}")

        return None


@dataclass
class ExportIndex:
    """Index of exports (symbol → file mapping).

    Per SPEC.md Section 2.7.2:
    - Built in first pass of code generation
    - Used by implementation nodes for correct imports
    """

    exports: dict[str, dict[str, Any]] = field(default_factory=dict)
    path_aliases: dict[str, str] = field(default_factory=dict)

    def add_export(
        self,
        symbol: str,
        from_file: str,
        export_type: str = "value",
    ) -> None:
        """Add an export to the index."""
        self.exports[symbol] = {
            "from": from_file,
            "type": export_type,
        }

    def get_import_path(self, symbol: str) -> str | None:
        """Get the import path for a symbol."""
        if symbol in self.exports:
            return self.exports[symbol]["from"]
        return None


def build_export_index(manifest: FileManifest) -> ExportIndex:
    """Build export index from manifest files.

    Parses TypeScript/JavaScript exports to build symbol → file mapping.
    """
    index = ExportIndex()

    for entry in manifest.files:
        if entry.language not in ("ts", "typescript", "js", "javascript"):
            continue

        exports = _extract_exports(entry.content)
        for symbol, export_type in exports:
            index.add_export(symbol, entry.path, export_type)

    return index


def _extract_exports(content: str) -> list[tuple[str, str]]:
    """Extract exported symbols from TypeScript/JavaScript content.

    Returns list of (symbol_name, export_type) tuples.
    """
    exports: list[tuple[str, str]] = []

    # Match export interface/type/class/enum
    type_exports = re.findall(
        r'export\s+(?:interface|type|class|enum)\s+(\w+)',
        content
    )
    for symbol in type_exports:
        exports.append((symbol, "type"))

    # Match export function
    func_exports = re.findall(
        r'export\s+(?:async\s+)?function\s+(\w+)',
        content
    )
    for symbol in func_exports:
        exports.append((symbol, "function"))

    # Match export const/let/var
    var_exports = re.findall(
        r'export\s+(?:const|let|var)\s+(\w+)',
        content
    )
    for symbol in var_exports:
        exports.append((symbol, "value"))

    # Match export default
    default_exports = re.findall(
        r'export\s+default\s+(?:class|function)?\s*(\w+)?',
        content
    )
    for symbol in default_exports:
        if symbol:
            exports.append((symbol, "default"))

    return exports
