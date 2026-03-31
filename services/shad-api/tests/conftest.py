"""Pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from shad.retrieval.layer import RetrievalResult
from shad.vault.decay import DecayConfig, DecayCurve
from shad.vault.shadow_index import MemoryType


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_vault() -> Generator[Path, None, None]:
    """Create a temporary collection directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collection_path = Path(tmpdir) / "TestVault"
        collection_path.mkdir()
        # Create Shad directories
        (collection_path / "Shad" / "History").mkdir(parents=True)
        (collection_path / "Shad" / "Skills").mkdir(parents=True)
        (collection_path / "Shad" / "Staging").mkdir(parents=True)
        yield collection_path


@pytest.fixture
def temp_collection(temp_vault: Path) -> Path:
    """Alias for temp_vault — used by sandbox and engine tests."""
    return temp_vault


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


# ---------------------------------------------------------------------------
# RetrievalResult fixtures — episodic, semantic, and mixed
# Timestamps are relative to 2026-03-31 (current date) to keep tests stable.
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 31, tzinfo=UTC)


@pytest.fixture
def episodic_results() -> list[RetrievalResult]:
    """Four episodic RetrievalResults with varied ages and base scores.

    Ordered newest-first, scores decrease with age to support decay tests.
    """
    return [
        RetrievalResult(
            path="feeds/tech-news/ai-breakthrough.md",
            content="Major AI breakthrough announced today.",
            score=0.92,
            collection="my-vault",
            docid="ep-001",
            metadata={
                "memory_type": MemoryType.EPISODIC.value,
                "ingested_at": (_NOW - timedelta(hours=2)).isoformat(),
                "source_type": "feed",
            },
        ),
        RetrievalResult(
            path="feeds/tech-news/model-release.md",
            content="New model released last week.",
            score=0.74,
            collection="my-vault",
            docid="ep-002",
            metadata={
                "memory_type": MemoryType.EPISODIC.value,
                "ingested_at": (_NOW - timedelta(days=7)).isoformat(),
                "source_type": "feed",
            },
        ),
        RetrievalResult(
            path="feeds/events/conference-recap.md",
            content="Conference held last month.",
            score=0.55,
            collection="my-vault",
            docid="ep-003",
            metadata={
                "memory_type": MemoryType.EPISODIC.value,
                "ingested_at": (_NOW - timedelta(days=30)).isoformat(),
                "source_type": "feed",
            },
        ),
        RetrievalResult(
            path="feeds/events/old-announcement.md",
            content="Announcement from six months ago.",
            score=0.38,
            collection="my-vault",
            docid="ep-004",
            metadata={
                "memory_type": MemoryType.EPISODIC.value,
                "ingested_at": (_NOW - timedelta(days=180)).isoformat(),
                "source_type": "feed",
            },
        ),
    ]


@pytest.fixture
def semantic_results() -> list[RetrievalResult]:
    """Four semantic RetrievalResults with varied ages and base scores.

    Semantic entries are timeless references (github/url); scores and ages
    are varied so ranking tests can verify decay does not apply to them.
    """
    return [
        RetrievalResult(
            path="sources/github-com/openai/tiktoken/README.md",
            content="Tiktoken is a fast BPE tokenizer for use with OpenAI models.",
            score=0.88,
            collection="my-vault",
            docid="sm-001",
            metadata={
                "memory_type": MemoryType.SEMANTIC.value,
                "ingested_at": (_NOW - timedelta(days=3)).isoformat(),
                "source_type": "github",
            },
        ),
        RetrievalResult(
            path="sources/github-com/anthropics/sdk-python/docs/usage.md",
            content="The Anthropic Python SDK provides access to the Anthropic API.",
            score=0.81,
            collection="my-vault",
            docid="sm-002",
            metadata={
                "memory_type": MemoryType.SEMANTIC.value,
                "ingested_at": (_NOW - timedelta(days=14)).isoformat(),
                "source_type": "github",
            },
        ),
        RetrievalResult(
            path="sources/url/docs-site/architecture-overview.md",
            content="The system architecture consists of three layers.",
            score=0.63,
            collection="my-vault",
            docid="sm-003",
            metadata={
                "memory_type": MemoryType.SEMANTIC.value,
                "ingested_at": (_NOW - timedelta(days=60)).isoformat(),
                "source_type": "url",
            },
        ),
        RetrievalResult(
            path="sources/github-com/legacy/patterns/patterns.md",
            content="Patterns established in the early design phase.",
            score=0.45,
            collection="my-vault",
            docid="sm-004",
            metadata={
                "memory_type": MemoryType.SEMANTIC.value,
                "ingested_at": (_NOW - timedelta(days=365)).isoformat(),
                "source_type": "github",
            },
        ),
    ]


@pytest.fixture
def mixed_memory_results(
    episodic_results: list[RetrievalResult],
    semantic_results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Combined episodic + semantic results, interleaved by score descending."""
    combined = episodic_results + semantic_results
    return sorted(combined, key=lambda r: r.score, reverse=True)


# ---------------------------------------------------------------------------
# Shared DecayConfig fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exp_decay_config() -> DecayConfig:
    """Exponential decay config with 90-day halflife (= 7 776 000 s)."""
    return DecayConfig(halflife_seconds=90.0 * 86_400.0, curve=DecayCurve.EXPONENTIAL)


@pytest.fixture
def lin_decay_config() -> DecayConfig:
    """Linear decay config with 90-day halflife (reaches zero at 180 d)."""
    return DecayConfig(halflife_seconds=90.0 * 86_400.0, curve=DecayCurve.LINEAR)


# ---------------------------------------------------------------------------
# Temporal-decay baseline: results at exact halflife multiples
#
# These fixtures give every test the same well-known inputs so that expected
# scores are derivable by formula rather than hard-coded magic numbers.
#
# Halflife = 90 days (_DECAY_HL_DAYS).  Reference time = _NOW (2026-03-31).
#
# Exponential adjusted scores   score * 0.5^(age_days / 90):
#
#   path            base   age_days   expected (exp)
#   ─────────────────────────────────────────────────
#   fresh.md        0.90   0          0.9000
#   half-hl.md      0.80   45 (½ HL)  0.80 * 0.5^0.5  ≈ 0.5657
#   one-hl.md       0.95   90 (1 HL)  0.95 * 0.5       = 0.4750
#   two-hl.md       0.70   180 (2 HL) 0.70 * 0.25      = 0.1750
#   four-hl.md      0.85   360 (4 HL) 0.85 * 0.0625   ≈ 0.0531
#   no-ts.md        0.60   None       0.6000 (preserved, no ingested_at)
#   future.md       0.40   -1 d       0.4000 (negative age → no decay)
#   zero-score.md   0.00   45 (½ HL)  0.0000
# ---------------------------------------------------------------------------

_DECAY_HL_DAYS = 90.0
_DECAY_HL_SECONDS = _DECAY_HL_DAYS * 86_400.0


@pytest.fixture
def decay_baseline_now() -> datetime:
    """Fixed reference 'now' for all decay_baseline_results calculations."""
    return _NOW


@pytest.fixture
def decay_baseline_config() -> DecayConfig:
    """Exponential DecayConfig whose halflife matches decay_baseline_results."""
    return DecayConfig(halflife_seconds=_DECAY_HL_SECONDS, curve=DecayCurve.EXPONENTIAL)


@pytest.fixture
def decay_baseline_results() -> list[RetrievalResult]:
    """Eight RetrievalResults spanning a wide age range for decay math tests.

    Results are intentionally *not* sorted so that tests verify the reranker
    produces the correct order rather than relying on input ordering.
    """
    hl = timedelta(days=_DECAY_HL_DAYS)
    return [
        RetrievalResult(
            path="fresh.md",
            content="A brand-new document ingested right now.",
            score=0.90,
            collection="test-vault",
            docid="bl-001",
            metadata={
                "ingested_at": _NOW.isoformat(),
                "source_type": "feed",
            },
        ),
        RetrievalResult(
            path="four-hl.md",
            content="Very stale document from four halflives ago.",
            score=0.85,
            collection="test-vault",
            docid="bl-002",
            metadata={
                "ingested_at": (_NOW - 4 * hl).isoformat(),
                "source_type": "url",
            },
        ),
        RetrievalResult(
            path="one-hl.md",
            content="Document from exactly one halflife ago.",
            score=0.95,
            collection="test-vault",
            docid="bl-003",
            metadata={
                "ingested_at": (_NOW - hl).isoformat(),
                "source_type": "github",
            },
        ),
        RetrievalResult(
            path="no-ts.md",
            content="Document without an ingested_at timestamp.",
            score=0.60,
            collection="test-vault",
            docid="bl-004",
            metadata={
                "source_type": "manual",
            },
        ),
        RetrievalResult(
            path="half-hl.md",
            content="Document from half a halflife ago.",
            score=0.80,
            collection="test-vault",
            docid="bl-005",
            metadata={
                "ingested_at": (_NOW - hl / 2).isoformat(),
                "source_type": "feed",
            },
        ),
        RetrievalResult(
            path="two-hl.md",
            content="Document from two halflives ago.",
            score=0.70,
            collection="test-vault",
            docid="bl-006",
            metadata={
                "ingested_at": (_NOW - 2 * hl).isoformat(),
                "source_type": "github",
            },
        ),
        RetrievalResult(
            path="future.md",
            content="Document with a future timestamp (should not be penalised).",
            score=0.40,
            collection="test-vault",
            docid="bl-007",
            metadata={
                "ingested_at": (_NOW + timedelta(days=1)).isoformat(),
                "source_type": "manual",
            },
        ),
        RetrievalResult(
            path="zero-score.md",
            content="Document with a zero base relevance score.",
            score=0.00,
            collection="test-vault",
            docid="bl-008",
            metadata={
                "ingested_at": (_NOW - hl / 2).isoformat(),
                "source_type": "feed",
            },
        ),
    ]
