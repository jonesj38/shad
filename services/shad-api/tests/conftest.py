"""Pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_vault() -> Generator[Path, None, None]:
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir) / "TestVault"
        vault_path.mkdir()
        # Create Shad directories
        (vault_path / "Shad" / "History").mkdir(parents=True)
        (vault_path / "Shad" / "Skills").mkdir(parents=True)
        (vault_path / "Shad" / "Staging").mkdir(parents=True)
        yield vault_path


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Create a mock httpx client."""
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    client.patch = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def sample_frontmatter() -> dict:
    """Sample frontmatter for testing."""
    return {
        "type": "note",
        "status": "raw",
        "shad_processed": False,
        "created": "2026-01-14",
    }


@pytest.fixture
def sample_task_frontmatter() -> dict:
    """Sample task frontmatter for testing."""
    return {
        "type": "task",
        "status": "pending",
        "goal": "Test goal",
        "shad_run_id": None,
        "created": "2026-01-14",
    }


@pytest.fixture
def sample_note_content() -> str:
    """Sample note content with frontmatter."""
    return """---
type: note
status: raw
---

# Test Note

This is a test note with some content.

## Section 1

Some important information here.
"""


@pytest.fixture
def sample_source_content() -> str:
    """Sample source content with frontmatter."""
    return """---
type: source
status: raw
source_type: manual
shad_processed: false
---

# Source Document

Raw source content that needs processing.
"""
