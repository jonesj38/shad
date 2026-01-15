"""Tests for file manifest output.

Per SPEC.md Section 2.7:
- Code generation produces file manifests (not raw text)
- Manifests contain paths, content, language, hash, source_nodes
- Writing to filesystem is always explicit (--write-files flag)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shad.output.manifest import (
    FileEntry,
    FileManifest,
    ManifestNote,
    ManifestWriter,
    WriteMode,
    WriteReport,
)


class TestFileEntry:
    """Tests for file entries."""

    def test_create_file_entry(self) -> None:
        """Test creating a file entry."""
        content = "export interface User { id: string; }"
        entry = FileEntry(
            path="src/types.ts",
            content=content,
            language="ts",
        )

        assert entry.path == "src/types.ts"
        assert entry.content == content
        assert entry.language == "ts"

    def test_file_entry_auto_hash(self) -> None:
        """Test that file entry generates hash automatically."""
        content = "export interface User { id: string; }"
        entry = FileEntry(
            path="src/types.ts",
            content=content,
            language="ts",
        )

        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        assert entry.content_hash == expected_hash

    def test_file_entry_with_source_nodes(self) -> None:
        """Test file entry with source node tracking."""
        entry = FileEntry(
            path="src/api/users.ts",
            content="// User API",
            language="ts",
            source_nodes=["impl_api", "types_contracts"],
        )

        assert "impl_api" in entry.source_nodes
        assert "types_contracts" in entry.source_nodes

    def test_file_entry_write_mode(self) -> None:
        """Test file entry write modes."""
        # Create mode (new file)
        entry_create = FileEntry(
            path="new_file.ts",
            content="content",
            language="ts",
            mode=WriteMode.CREATE,
        )
        assert entry_create.mode == WriteMode.CREATE

        # Update mode (modify existing)
        entry_update = FileEntry(
            path="existing.ts",
            content="updated content",
            language="ts",
            mode=WriteMode.UPDATE,
        )
        assert entry_update.mode == WriteMode.UPDATE


class TestFileManifest:
    """Tests for file manifests."""

    def test_create_empty_manifest(self) -> None:
        """Test creating an empty manifest."""
        manifest = FileManifest(run_id="test-run-123")

        assert manifest.run_id == "test-run-123"
        assert len(manifest.files) == 0

    def test_add_file_to_manifest(self) -> None:
        """Test adding files to manifest."""
        manifest = FileManifest(run_id="test-run-123")

        entry = FileEntry(
            path="src/index.ts",
            content="export * from './types';",
            language="ts",
        )
        manifest.add_file(entry)

        assert len(manifest.files) == 1
        assert manifest.get_file("src/index.ts") == entry

    def test_manifest_prevents_duplicates(self) -> None:
        """Test that manifest prevents duplicate paths."""
        manifest = FileManifest(run_id="test-run-123")

        entry1 = FileEntry(path="src/index.ts", content="v1", language="ts")
        entry2 = FileEntry(path="src/index.ts", content="v2", language="ts")

        manifest.add_file(entry1)

        with pytest.raises(ValueError, match="already exists"):
            manifest.add_file(entry2)

    def test_manifest_update_existing(self) -> None:
        """Test updating existing file in manifest."""
        manifest = FileManifest(run_id="test-run-123")

        entry1 = FileEntry(path="src/index.ts", content="v1", language="ts")
        entry2 = FileEntry(path="src/index.ts", content="v2", language="ts")

        manifest.add_file(entry1)
        manifest.update_file(entry2)

        assert manifest.get_file("src/index.ts").content == "v2"

    def test_manifest_with_notes(self) -> None:
        """Test manifest with notes/requests."""
        manifest = FileManifest(run_id="test-run-123")

        note = ManifestNote(
            kind="contract_change_request",
            detail="Add UserRole enum to types",
        )
        manifest.add_note(note)

        assert len(manifest.notes) == 1
        assert manifest.notes[0].kind == "contract_change_request"

    def test_manifest_to_dict(self) -> None:
        """Test converting manifest to dictionary."""
        manifest = FileManifest(run_id="test-run-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User {}",
            language="ts",
        ))

        data = manifest.to_dict()

        assert data["run_id"] == "test-run-123"
        assert len(data["files"]) == 1
        assert data["files"][0]["path"] == "src/types.ts"

    def test_manifest_from_dict(self) -> None:
        """Test creating manifest from dictionary."""
        data = {
            "run_id": "test-run-123",
            "files": [
                {"path": "src/types.ts", "content": "content", "language": "ts"},
            ],
            "notes": [],
        }

        manifest = FileManifest.from_dict(data)

        assert manifest.run_id == "test-run-123"
        assert len(manifest.files) == 1


class TestManifestWriter:
    """Tests for writing manifests to disk."""

    @pytest.fixture
    def temp_output(self, tmp_path: Path) -> Path:
        """Create temp output directory."""
        return tmp_path / "output"

    @pytest.fixture
    def manifest(self) -> FileManifest:
        """Create test manifest."""
        manifest = FileManifest(run_id="test-run-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User { id: string; }",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content="import { User } from '../types';\nexport function getUser(): User {}",
            language="ts",
        ))
        return manifest

    def test_write_manifest_creates_files(
        self, manifest: FileManifest, temp_output: Path
    ) -> None:
        """Test writing manifest creates files on disk."""
        writer = ManifestWriter(output_root=temp_output)
        report = writer.write(manifest)

        assert report.success is True
        assert (temp_output / "src/types.ts").exists()
        assert (temp_output / "src/api/users.ts").exists()

    def test_write_manifest_respects_output_root(
        self, manifest: FileManifest, temp_output: Path
    ) -> None:
        """Test that files are written under output_root."""
        writer = ManifestWriter(output_root=temp_output)
        writer.write(manifest)

        # All files should be under temp_output
        for entry in manifest.files:
            file_path = temp_output / entry.path
            assert file_path.exists()

    def test_write_manifest_blocks_path_traversal(
        self, temp_output: Path
    ) -> None:
        """Test that path traversal is blocked."""
        manifest = FileManifest(run_id="test-run-123")
        manifest.add_file(FileEntry(
            path="../../../etc/passwd",  # Attempted traversal
            content="malicious content",
            language="txt",
        ))

        writer = ManifestWriter(output_root=temp_output)
        report = writer.write(manifest)

        assert report.success is False
        assert len(report.errors) > 0
        assert "path traversal" in report.errors[0].lower()

    def test_write_manifest_conflict_handling(
        self, manifest: FileManifest, temp_output: Path
    ) -> None:
        """Test handling of file conflicts."""
        # Create existing file
        temp_output.mkdir(parents=True)
        (temp_output / "src").mkdir()
        (temp_output / "src/types.ts").write_text("existing content")

        writer = ManifestWriter(output_root=temp_output, overwrite=False)
        report = writer.write(manifest)

        # Should fail due to conflict
        assert report.success is False
        assert len(report.skipped) > 0

    def test_write_manifest_overwrite_enabled(
        self, manifest: FileManifest, temp_output: Path
    ) -> None:
        """Test overwriting existing files when enabled."""
        # Create existing file
        temp_output.mkdir(parents=True)
        (temp_output / "src").mkdir()
        (temp_output / "src/types.ts").write_text("existing content")

        writer = ManifestWriter(output_root=temp_output, overwrite=True)
        report = writer.write(manifest)

        assert report.success is True
        new_content = (temp_output / "src/types.ts").read_text()
        assert "interface User" in new_content

    def test_write_report_includes_details(
        self, manifest: FileManifest, temp_output: Path
    ) -> None:
        """Test that write report includes file details."""
        writer = ManifestWriter(output_root=temp_output)
        report = writer.write(manifest)

        assert len(report.written) == 2
        assert "src/types.ts" in report.written
        assert "src/api/users.ts" in report.written


class TestManifestValidation:
    """Tests for manifest validation."""

    def test_validate_no_duplicate_paths(self) -> None:
        """Test validation catches duplicate paths."""
        manifest = FileManifest(run_id="test-123")
        manifest.files = [
            FileEntry(path="src/index.ts", content="1", language="ts"),
            FileEntry(path="src/index.ts", content="2", language="ts"),
        ]

        errors = manifest.validate()
        assert len(errors) > 0
        assert "duplicate" in errors[0].lower()

    def test_validate_path_format(self) -> None:
        """Test validation of path formats."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",  # Valid
            content="content",
            language="ts",
        ))

        errors = manifest.validate()
        assert len(errors) == 0

    def test_manifest_integrity_check(self) -> None:
        """Test manifest integrity check (hashes)."""
        manifest = FileManifest(run_id="test-123")
        entry = FileEntry(
            path="src/types.ts",
            content="original content",
            language="ts",
        )
        manifest.add_file(entry)

        # Manually corrupt content (simulating tampering)
        manifest.files[0].content = "modified content"
        # Hash should no longer match
        manifest.files[0]._content_hash = "fake_hash"

        errors = manifest.validate()
        assert len(errors) > 0


class TestExportIndex:
    """Tests for export index (symbol → file mapping)."""

    def test_build_export_index_from_manifest(self) -> None:
        """Test building export index from manifest files."""
        from shad.output.manifest import build_export_index

        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="""export interface User { id: string; }
export interface Post { title: string; }
export type UserId = string;""",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content="""export function getUser(id: string): User {}
export function createUser(data: UserData): User {}""",
            language="ts",
        ))

        index = build_export_index(manifest)

        assert "User" in index.exports
        assert index.exports["User"]["from"] == "src/types.ts"
        assert "getUser" in index.exports
        assert index.exports["getUser"]["from"] == "src/api/users.ts"
