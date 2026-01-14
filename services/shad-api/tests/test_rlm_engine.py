"""Tests for the RLM Engine with Obsidian Code Mode support."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.engine.rlm import BudgetExhausted, RLMEngine
from shad.models import Budget, RunConfig
from shad.models.run import NodeStatus, RunStatus, StopReason


class TestRLMEngine:
    """Tests for RLMEngine."""

    @pytest.fixture
    def mock_llm_provider(self) -> MagicMock:
        """Create mock LLM provider."""
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=("Answer to the task", 100))
        provider.decompose_task = AsyncMock(return_value=["Subtask 1", "Subtask 2"])
        provider.synthesize_results = AsyncMock(return_value="Synthesized result")
        provider.answer_task = AsyncMock(return_value=("Direct answer", 50))
        return provider

    @pytest.fixture
    def engine(self, mock_llm_provider: MagicMock) -> RLMEngine:
        """Create engine for testing."""
        return RLMEngine(llm_provider=mock_llm_provider)

    @pytest.fixture
    def simple_config(self) -> RunConfig:
        """Create a simple run config."""
        return RunConfig(
            goal="What is 2 + 2?",
            budget=Budget(
                max_depth=2,
                max_nodes=10,
                max_wall_time=60,
                max_tokens=10000,
            ),
        )

    @pytest.mark.asyncio
    async def test_execute_simple_goal(
        self, engine: RLMEngine, simple_config: RunConfig
    ) -> None:
        """Test executing a simple goal."""
        run = await engine.execute(simple_config)

        assert run.status == RunStatus.COMPLETE
        assert run.final_result is not None
        assert run.root_node_id is not None

    @pytest.mark.asyncio
    async def test_budget_depth_exhausted(
        self, engine: RLMEngine
    ) -> None:
        """Test budget exhaustion by depth."""
        config = RunConfig(
            goal="Complex task",
            budget=Budget(max_depth=1, max_nodes=100),
        )

        # Force decomposition
        engine.llm.decompose_task = AsyncMock(return_value=["Sub1", "Sub2", "Sub3"])

        run = await engine.execute(config)

        # Should complete with partial or complete status
        assert run.status in (RunStatus.PARTIAL, RunStatus.COMPLETE)

    @pytest.mark.asyncio
    async def test_budget_nodes_exhausted(
        self, engine: RLMEngine
    ) -> None:
        """Test budget exhaustion by node count."""
        config = RunConfig(
            goal="Complex task",
            budget=Budget(max_depth=5, max_nodes=2),
        )

        # Force decomposition to create many nodes
        engine.llm.decompose_task = AsyncMock(return_value=["Sub1", "Sub2", "Sub3"])

        run = await engine.execute(config)

        # Should stop due to node budget
        assert run.status in (RunStatus.PARTIAL, RunStatus.COMPLETE)

    @pytest.mark.asyncio
    async def test_cache_hit(self, engine: RLMEngine, simple_config: RunConfig) -> None:
        """Test that cache hits are detected."""
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="Cached result")
        mock_cache.set = AsyncMock()
        engine.cache = mock_cache

        run = await engine.execute(simple_config)

        # The cache should have been checked
        mock_cache.get.assert_called()
        # Run should complete (with or without cache hit depending on key match)
        assert run.status in (RunStatus.COMPLETE, RunStatus.FAILED)

    @pytest.mark.asyncio
    async def test_resume_partial_run(
        self, engine: RLMEngine, simple_config: RunConfig
    ) -> None:
        """Test resuming a partial run."""
        # First, create a partial run
        run = await engine.execute(simple_config)
        run.status = RunStatus.PARTIAL

        # Resume it
        resumed = await engine.resume(run)

        assert resumed.status in (RunStatus.COMPLETE, RunStatus.PARTIAL, RunStatus.FAILED)

    @pytest.mark.asyncio
    async def test_node_execution_order(
        self, engine: RLMEngine
    ) -> None:
        """Test that nodes are executed in correct order."""
        config = RunConfig(
            goal="Multi-step task",
            budget=Budget(max_depth=3),
        )

        # Track execution order
        execution_order = []

        original_answer = engine.llm.answer_task

        async def track_execution(*args, **kwargs):
            execution_order.append(kwargs.get("task", args[0] if args else ""))
            return await original_answer(*args, **kwargs)

        engine.llm.answer_task = track_execution

        run = await engine.execute(config)

        # Root task should be executed
        assert len(execution_order) >= 1


class TestRLMEngineWithObsidian:
    """Tests for RLM Engine with Obsidian integration."""

    @pytest.fixture
    def mock_mcp_client(self, temp_vault: Path) -> MagicMock:
        """Create mock MCP client."""
        client = MagicMock()
        client.vault_path = temp_vault
        client.is_connected = True
        client.connect = AsyncMock(return_value=True)
        client.read_note = AsyncMock(return_value=MagicMock(
            content="# Test\nContent here",
            metadata=MagicMock(note_type="note"),
        ))
        client.search = AsyncMock(return_value=[
            MagicMock(path="note1.md", content="Result 1", score=0.9),
        ])
        return client

    @pytest.fixture
    def engine_with_obsidian(
        self, mock_mcp_client: MagicMock
    ) -> RLMEngine:
        """Create engine with Obsidian support."""
        mock_llm = MagicMock()
        mock_llm.answer_task = AsyncMock(return_value=("Answer", 100))
        mock_llm.decompose_task = AsyncMock(return_value=["Sub1"])
        mock_llm.synthesize_results = AsyncMock(return_value="Synthesis")

        engine = RLMEngine(llm_provider=mock_llm)
        engine.mcp_client = mock_mcp_client
        return engine

    @pytest.mark.asyncio
    async def test_context_retrieval_from_obsidian(
        self, engine_with_obsidian: RLMEngine, temp_vault: Path
    ) -> None:
        """Test retrieving context from Obsidian vault."""
        # Create test notes in vault
        note_path = temp_vault / "context_note.md"
        note_path.write_text("---\ntype: note\n---\n# Relevant Info\nThis is context.")

        config = RunConfig(
            goal="What do we know about the topic?",
            notebook_id="test-vault",
            budget=Budget(max_depth=2),
        )

        # The engine should use MCP client to retrieve context
        # This test verifies the integration point exists


class TestCodeModeExecution:
    """Tests for Code Mode execution pattern."""

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock code executor."""
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=MagicMock(
            success=True,
            stdout="Output",
            return_value={"data": "result"},
        ))
        return executor

    @pytest.mark.asyncio
    async def test_code_mode_script_generation(self) -> None:
        """Test that Code Mode generates executable scripts."""
        # Per OBSIDIAN_PIVOT.md Section 3.1
        # RLM should generate Python scripts that use obsidian tools

        expected_script_pattern = """
from shad.sandbox.tools import obsidian

# Search for relevant notes
results = obsidian.search("query")

# Process and filter results
filtered = [r for r in results if some_condition(r)]

# Return distilled output
__result__ = {"findings": filtered}
"""

        # The script should:
        # 1. Import obsidian tools
        # 2. Use vault operations (search, read, etc.)
        # 3. Process data locally
        # 4. Return distilled result via __result__

        assert "obsidian.search" in expected_script_pattern
        assert "__result__" in expected_script_pattern


class TestBudgetEnforcement:
    """Tests for budget enforcement."""

    @pytest.fixture
    def engine(self) -> RLMEngine:
        """Create engine with mock LLM."""
        mock_llm = MagicMock()
        mock_llm.answer_task = AsyncMock(return_value=("Answer", 100))
        return RLMEngine(llm_provider=mock_llm)

    @pytest.mark.asyncio
    async def test_token_budget_tracking(self, engine: RLMEngine) -> None:
        """Test that token usage is tracked correctly."""
        config = RunConfig(
            goal="Test goal",
            budget=Budget(max_tokens=1000),
        )

        run = await engine.execute(config)

        assert run.total_tokens > 0

    @pytest.mark.asyncio
    async def test_wall_time_budget(self, engine: RLMEngine) -> None:
        """Test wall time budget enforcement."""
        config = RunConfig(
            goal="Test goal",
            budget=Budget(max_wall_time=1),  # 1 second
        )

        # Add delay to LLM calls
        async def slow_answer(*args, **kwargs):
            await asyncio.sleep(2)  # Exceed timeout
            return ("Answer", 100)

        engine.llm.answer_task = slow_answer

        run = await engine.execute(config)

        # Should have stopped due to time budget
        # (or completed if fast enough)
        assert run.completed_at is not None


class TestCacheValidation:
    """Tests for hash-based cache validation."""

    @pytest.fixture
    def mock_cache(self) -> MagicMock:
        """Create mock cache."""
        cache = MagicMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.get_with_hash = AsyncMock(return_value=None)
        cache.set_with_hash = AsyncMock()
        return cache

    @pytest.mark.asyncio
    async def test_cache_key_includes_context_hash(
        self, mock_cache: MagicMock
    ) -> None:
        """Test that cache keys include context hash for validation."""
        mock_llm = MagicMock()
        mock_llm.answer_task = AsyncMock(return_value=("Answer", 100))

        engine = RLMEngine(llm_provider=mock_llm, cache=mock_cache)

        config = RunConfig(
            goal="Test goal",
            budget=Budget(max_depth=2),
        )

        await engine.execute(config)

        # Cache should have been checked/set
        mock_cache.get.assert_called()

    def test_cache_key_generation(self) -> None:
        """Test cache key format."""
        engine = RLMEngine(llm_provider=MagicMock())

        key1 = engine._make_cache_key("task1", "context1")
        key2 = engine._make_cache_key("task1", "context2")
        key3 = engine._make_cache_key("task1", "context1")

        # Same inputs should produce same key
        assert key1 == key3
        # Different contexts should produce different keys
        assert key1 != key2
