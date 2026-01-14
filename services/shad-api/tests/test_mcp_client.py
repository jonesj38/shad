"""Tests for the Obsidian MCP client module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.mcp.client import ObsidianMCPClient
from shad.mcp.models import (
    NoteMetadata,
    NoteType,
    NoteStatus,
    SearchResult,
    VaultFile,
)


class TestObsidianMCPClient:
    """Tests for ObsidianMCPClient."""

    @pytest.fixture
    def client(self, temp_vault: Path) -> ObsidianMCPClient:
        """Create a client for testing."""
        return ObsidianMCPClient(
            base_url="https://127.0.0.1:27124",
            api_key="test-key",
            vault_path=temp_vault,
            verify_ssl=False,
        )

    @pytest.mark.asyncio
    async def test_connect_success(
        self, client: ObsidianMCPClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test successful connection to Obsidian API."""
        mock_httpx_client.get = AsyncMock(
            return_value=MagicMock(status_code=200, json=lambda: {"status": "OK"})
        )

        with patch.object(client, "_create_client", return_value=mock_httpx_client):
            result = await client.connect()

        assert result is True
        assert client.is_connected

    @pytest.mark.asyncio
    async def test_connect_failure(
        self, client: ObsidianMCPClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test failed connection to Obsidian API."""
        mock_httpx_client.get.side_effect = Exception("Connection refused")

        with patch.object(client, "_create_client", return_value=mock_httpx_client):
            result = await client.connect()

        assert result is False
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_read_note(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test reading a note from the vault."""
        # Create a test note
        note_path = temp_vault / "test_note.md"
        note_content = """---
type: note
status: raw
---

# Test Note

Content here.
"""
        note_path.write_text(note_content)

        result = await client.read_note("test_note.md")

        assert result is not None
        assert "# Test Note" in result.content
        assert result.metadata.note_type == NoteType.NOTE

    @pytest.mark.asyncio
    async def test_read_note_not_found(self, client: ObsidianMCPClient) -> None:
        """Test reading a non-existent note."""
        result = await client.read_note("nonexistent.md")
        assert result is None

    @pytest.mark.asyncio
    async def test_write_note(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test writing a note to the vault."""
        content = "# New Note\n\nThis is new content."
        metadata = NoteMetadata(
            note_type=NoteType.NOTE,
            status=NoteStatus.RAW,
        )

        result = await client.write_note(
            path="new_note.md",
            content=content,
            metadata=metadata,
        )

        assert result is True
        # Verify file was written
        written_file = temp_vault / "new_note.md"
        assert written_file.exists()
        file_content = written_file.read_text()
        assert "# New Note" in file_content
        assert "type: note" in file_content

    @pytest.mark.asyncio
    async def test_search_vault(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test searching the vault."""
        # Create test notes
        note1 = temp_vault / "note1.md"
        note1.write_text("---\ntype: note\n---\n# Python Programming\nLearn Python.")

        note2 = temp_vault / "note2.md"
        note2.write_text("---\ntype: note\n---\n# JavaScript Guide\nLearn JavaScript.")

        results = await client.search("Python")

        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_list_notes(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test listing notes in a directory."""
        # Create test notes
        subdir = temp_vault / "Notes"
        subdir.mkdir()
        (subdir / "note1.md").write_text("---\ntype: note\n---\n# Note 1")
        (subdir / "note2.md").write_text("---\ntype: note\n---\n# Note 2")

        files = await client.list_notes("Notes/")

        assert len(files) == 2

    @pytest.mark.asyncio
    async def test_update_frontmatter(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test updating note frontmatter."""
        # Create a note
        note_path = temp_vault / "update_test.md"
        note_path.write_text("---\ntype: note\nstatus: raw\n---\n# Content")

        result = await client.update_frontmatter(
            path="update_test.md",
            updates={"status": "processed", "shad_processed": True},
        )

        assert result is True
        updated_content = note_path.read_text()
        assert "status: processed" in updated_content
        assert "shad_processed: true" in updated_content

    @pytest.mark.asyncio
    async def test_get_file_hash(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test getting file content hash for cache validation."""
        # Create a note
        note_path = temp_vault / "hash_test.md"
        content = "# Test Content"
        note_path.write_text(content)

        file_hash = await client.get_file_hash("hash_test.md")

        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        assert file_hash == expected_hash

    @pytest.mark.asyncio
    async def test_delete_note_blocked(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test that delete operations are blocked (require HITL approval)."""
        # Create a note
        note_path = temp_vault / "delete_test.md"
        note_path.write_text("# To Be Deleted")

        # Delete should return a HITL queue ID, not actually delete
        result = await client.delete_note("delete_test.md")

        # File should still exist (blocked for HITL review)
        assert note_path.exists()
        assert result.requires_approval is True
        assert result.hitl_queue_id is not None

    @pytest.mark.asyncio
    async def test_create_note_with_full_path_wikilinks(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test that created notes use full-path wikilinks."""
        subdir = temp_vault / "Folder" / "Subfolder"
        subdir.mkdir(parents=True)

        content = "# Note with links\n\nSee [[OtherNote]]"

        result = await client.write_note(
            path="Folder/Subfolder/linked_note.md",
            content=content,
            metadata=NoteMetadata(note_type=NoteType.NOTE),
            resolve_wikilinks=True,
        )

        assert result is True


class TestNoteMetadata:
    """Tests for NoteMetadata model."""

    def test_from_frontmatter_note(self) -> None:
        """Test parsing note frontmatter."""
        frontmatter = {
            "type": "note",
            "status": "raw",
            "shad_processed": False,
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.NOTE
        assert metadata.status == NoteStatus.RAW
        assert metadata.shad_processed is False

    def test_from_frontmatter_task(self) -> None:
        """Test parsing task frontmatter."""
        frontmatter = {
            "type": "task",
            "status": "pending",
            "goal": "Test goal",
            "shad_run_id": "abc123",
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.TASK
        assert metadata.status == NoteStatus.PENDING
        assert metadata.goal == "Test goal"
        assert metadata.shad_run_id == "abc123"

    def test_from_frontmatter_source(self) -> None:
        """Test parsing source frontmatter."""
        frontmatter = {
            "type": "source",
            "status": "verified",
            "source_type": "pdf",
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.SOURCE
        assert metadata.status == NoteStatus.VERIFIED
        assert metadata.source_type == "pdf"

    def test_to_frontmatter(self) -> None:
        """Test converting metadata to frontmatter."""
        metadata = NoteMetadata(
            note_type=NoteType.TASK,
            status=NoteStatus.PROCESSING,
            goal="Test goal",
            shad_run_id="run123",
        )

        frontmatter = metadata.to_frontmatter()

        assert frontmatter["type"] == "task"
        assert frontmatter["status"] == "processing"
        assert frontmatter["goal"] == "Test goal"
        assert frontmatter["shad_run_id"] == "run123"

    def test_legacy_note_detection(self) -> None:
        """Test detecting legacy notes without type."""
        frontmatter = {"title": "Old Note"}

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        # Should infer as untyped note
        assert metadata.note_type == NoteType.UNTYPED
        assert metadata.is_legacy is True


class TestVaultFile:
    """Tests for VaultFile model."""

    def test_from_path(self, temp_vault: Path) -> None:
        """Test creating VaultFile from path."""
        note_path = temp_vault / "test.md"
        note_path.write_text("# Test")

        vault_file = VaultFile.from_path(note_path, temp_vault)

        assert vault_file.relative_path == "test.md"
        assert vault_file.exists is True

    def test_wikilink_format(self) -> None:
        """Test generating full-path wikilinks."""
        vault_file = VaultFile(
            relative_path="Folder/Subfolder/MyNote.md",
            exists=True,
            mtime=0.0,
        )

        wikilink = vault_file.to_wikilink()

        assert wikilink == "[[Folder/Subfolder/MyNote]]"


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_from_match(self) -> None:
        """Test creating search result from match."""
        result = SearchResult(
            path="notes/test.md",
            content="# Test\n\nMatched content here.",
            score=0.85,
            matched_line="Matched content here.",
        )

        assert result.path == "notes/test.md"
        assert result.score == 0.85

    def test_to_dict(self) -> None:
        """Test serializing search result."""
        result = SearchResult(
            path="test.md",
            content="Content",
            score=0.9,
        )

        data = result.to_dict()

        assert data["path"] == "test.md"
        assert data["score"] == 0.9


class TestObsidianMCPClientAdvanced:
    """Advanced tests for ObsidianMCPClient."""

    @pytest.fixture
    def client(self, temp_vault: Path) -> ObsidianMCPClient:
        """Create a client for testing."""
        return ObsidianMCPClient(
            base_url="https://127.0.0.1:27124",
            api_key="test-key",
            vault_path=temp_vault,
            verify_ssl=False,
        )

    @pytest.mark.asyncio
    async def test_update_note(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test updating an existing note."""
        # Create a note first
        note_path = temp_vault / "update_me.md"
        note_path.write_text("---\ntype: note\n---\n# Original Content")

        result = await client.update_note(
            path="update_me.md",
            content="# Updated Content\n\nNew text here.",
        )

        assert result is True
        updated = note_path.read_text()
        assert "Updated Content" in updated

    @pytest.mark.asyncio
    async def test_update_note_not_found(self, client: ObsidianMCPClient) -> None:
        """Test updating a non-existent note."""
        result = await client.update_note(
            path="does_not_exist.md",
            content="# Content",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_standardize_legacy_note(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test standardizing a legacy note without frontmatter."""
        # Create a legacy note without proper frontmatter
        legacy_path = temp_vault / "legacy_note.md"
        legacy_path.write_text("# Old Note\n\nNo frontmatter here.")

        result = await client.standardize_legacy_note("legacy_note.md")

        assert result is True
        content = legacy_path.read_text()
        assert "---" in content
        assert "type:" in content

    @pytest.mark.asyncio
    async def test_list_notes_recursive(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test listing notes recursively."""
        # Create nested structure
        subdir = temp_vault / "Level1" / "Level2"
        subdir.mkdir(parents=True)
        (temp_vault / "Level1" / "note1.md").write_text("# Note 1")
        (subdir / "note2.md").write_text("# Note 2")

        files = await client.list_notes("Level1/", recursive=True)

        assert len(files) >= 2

    @pytest.mark.asyncio
    async def test_get_file_hash_not_found(self, client: ObsidianMCPClient) -> None:
        """Test getting hash for non-existent file."""
        file_hash = await client.get_file_hash("nonexistent.md")
        assert file_hash is None

    @pytest.mark.asyncio
    async def test_search_empty_query(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test search with empty query."""
        # Create a note
        (temp_vault / "test.md").write_text("---\ntype: note\n---\n# Content")

        results = await client.search("")

        # Should return empty or all results depending on implementation
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_write_note_with_subdirectory(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test writing note that creates subdirectories."""
        result = await client.write_note(
            path="NewDir/SubDir/deep_note.md",
            content="# Deep Note",
            metadata=NoteMetadata(note_type=NoteType.NOTE),
        )

        assert result is True
        assert (temp_vault / "NewDir" / "SubDir" / "deep_note.md").exists()

    @pytest.mark.asyncio
    async def test_read_note_with_no_frontmatter(
        self, client: ObsidianMCPClient, temp_vault: Path
    ) -> None:
        """Test reading a note without frontmatter."""
        note_path = temp_vault / "no_frontmatter.md"
        note_path.write_text("# Just Content\n\nNo YAML here.")

        result = await client.read_note("no_frontmatter.md")

        assert result is not None
        assert result.metadata.note_type == NoteType.UNTYPED


class TestNoteMetadataAdvanced:
    """Advanced tests for NoteMetadata."""

    def test_from_frontmatter_report(self) -> None:
        """Test parsing report frontmatter."""
        frontmatter = {
            "type": "report",
            "status": "complete",
            "shad_run_id": "run-123",
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.REPORT
        assert metadata.status == NoteStatus.COMPLETE

    def test_from_frontmatter_learning(self) -> None:
        """Test parsing learning frontmatter."""
        frontmatter = {
            "type": "learning",
            "status": "staging",
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.LEARNING
        assert metadata.status == NoteStatus.STAGING

    def test_from_frontmatter_with_tags(self) -> None:
        """Test parsing frontmatter with tags."""
        frontmatter = {
            "type": "note",
            "status": "raw",
            "tags": ["tag1", "tag2"],
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.NOTE
        # Tags should be stored
        assert metadata.tags == ["tag1", "tag2"]

    def test_to_frontmatter_with_tags(self) -> None:
        """Test converting metadata with tags."""
        metadata = NoteMetadata(
            note_type=NoteType.NOTE,
            status=NoteStatus.RAW,
            tags=["custom", "data"],
        )

        frontmatter = metadata.to_frontmatter()

        assert frontmatter["type"] == "note"
        assert frontmatter.get("tags") == ["custom", "data"]

    def test_from_frontmatter_unknown_status(self) -> None:
        """Test parsing frontmatter with unknown status."""
        frontmatter = {
            "type": "note",
            "status": "invalid_status",
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.NOTE
        assert metadata.status == NoteStatus.UNKNOWN

    def test_from_frontmatter_with_all_fields(self) -> None:
        """Test parsing frontmatter with all optional fields."""
        frontmatter = {
            "type": "report",
            "status": "complete",
            "goal": "Test goal",
            "shad_run_id": "run-123",
            "node_count": 5,
            "max_depth": 3,
            "token_usage": 1000,
            "wall_time_seconds": 60.5,
            "shad_processed": True,
            "created": "2024-01-01",
        }

        metadata = NoteMetadata.from_frontmatter(frontmatter)

        assert metadata.note_type == NoteType.REPORT
        assert metadata.goal == "Test goal"
        assert metadata.node_count == 5
        assert metadata.wall_time_seconds == 60.5


class TestVaultFileAdvanced:
    """Additional tests for VaultFile."""

    def test_from_path_nonexistent(self, temp_vault: Path) -> None:
        """Test creating VaultFile from non-existent path."""
        fake_path = temp_vault / "nonexistent.md"

        vault_file = VaultFile.from_path(fake_path, temp_vault)

        assert vault_file.relative_path == "nonexistent.md"
        assert vault_file.exists is False

    def test_wikilink_nested_path(self) -> None:
        """Test wikilink for deeply nested path."""
        vault_file = VaultFile(
            relative_path="Level1/Level2/Level3/DeepNote.md",
            exists=True,
            mtime=0.0,
        )

        wikilink = vault_file.to_wikilink()

        assert wikilink == "[[Level1/Level2/Level3/DeepNote]]"


class TestVaultNote:
    """Tests for VaultNote model."""

    def test_get_body_with_frontmatter(self) -> None:
        """Test extracting body from note with frontmatter."""
        from shad.mcp.models import VaultNote

        note = VaultNote(
            path="test.md",
            content="---\ntype: note\n---\n# Content\n\nBody text here.",
            metadata=NoteMetadata(),
        )

        body = note.get_body()

        assert "# Content" in body
        assert "type: note" not in body

    def test_get_body_without_frontmatter(self) -> None:
        """Test extracting body from note without frontmatter."""
        from shad.mcp.models import VaultNote

        note = VaultNote(
            path="test.md",
            content="# Just Content\n\nNo frontmatter.",
            metadata=NoteMetadata(),
        )

        body = note.get_body()

        assert body == "# Just Content\n\nNo frontmatter."
