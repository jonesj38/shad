"""MCP (Model Context Protocol) client for Obsidian integration."""

from shad.mcp.client import ObsidianMCPClient
from shad.mcp.models import (
    DeleteResult,
    NoteMetadata,
    NoteStatus,
    NoteType,
    SearchResult,
    VaultFile,
    VaultNote,
)

__all__ = [
    "ObsidianMCPClient",
    "DeleteResult",
    "NoteMetadata",
    "NoteStatus",
    "NoteType",
    "SearchResult",
    "VaultFile",
    "VaultNote",
]
