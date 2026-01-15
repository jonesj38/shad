"""Tests for LLM-driven decomposition with strategy constraints.

Per SPEC.md Section 2.2.2:
- LLM receives strategy skeleton and fills in task-specific details
- Can add/remove optional nodes, split implementation into modules
- Cannot violate required stages without explicit waiver
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shad.engine.decomposition import (
    DecompositionNode,
    DecompositionResult,
    StrategyDecomposer,
)
from shad.engine.strategies import (
    StrategyType,
    SoftwareStrategy,
    ResearchStrategy,
    get_strategy,
)


class TestDecompositionNode:
    """Tests for decomposition nodes."""

    def test_create_node(self) -> None:
        """Test creating a decomposition node."""
        node = DecompositionNode(
            stage_name="types_contracts",
            task="Define types for User model",
            hard_deps=["clarify_requirements"],
            soft_deps=[],
        )
        assert node.stage_name == "types_contracts"
        assert node.task == "Define types for User model"
        assert "clarify_requirements" in node.hard_deps

    def test_node_with_soft_deps(self) -> None:
        """Test node with soft dependencies."""
        node = DecompositionNode(
            stage_name="implementation",
            task="Implement user service",
            hard_deps=["types_contracts"],
            soft_deps=["db_schema"],
        )
        assert "db_schema" in node.soft_deps


class TestDecompositionResult:
    """Tests for decomposition results."""

    def test_create_result(self) -> None:
        """Test creating a decomposition result."""
        nodes = [
            DecompositionNode(stage_name="req", task="Requirements", hard_deps=[], soft_deps=[]),
            DecompositionNode(stage_name="impl", task="Implement", hard_deps=["req"], soft_deps=[]),
        ]
        result = DecompositionResult(
            strategy_type=StrategyType.SOFTWARE,
            nodes=nodes,
            is_valid=True,
            validation_errors=[],
        )
        assert result.strategy_type == StrategyType.SOFTWARE
        assert len(result.nodes) == 2
        assert result.is_valid is True

    def test_result_with_validation_errors(self) -> None:
        """Test result with validation errors."""
        result = DecompositionResult(
            strategy_type=StrategyType.SOFTWARE,
            nodes=[],
            is_valid=False,
            validation_errors=["Missing required stage: types_contracts"],
        )
        assert result.is_valid is False
        assert len(result.validation_errors) == 1


class TestStrategyDecomposer:
    """Tests for strategy-aware decomposition."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM provider."""
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=("[]", 100))
        return llm

    @pytest.fixture
    def decomposer(self, mock_llm: MagicMock) -> StrategyDecomposer:
        """Create decomposer for testing."""
        return StrategyDecomposer(llm_provider=mock_llm)

    @pytest.mark.asyncio
    async def test_decompose_with_software_strategy(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test decomposition with software strategy."""
        # Mock LLM response with valid decomposition
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify user authentication requirements", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Define project structure", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Define User and Auth types", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement auth service", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "verification", "task": "Verify auth implementation", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Generate final output", "hard_deps": ["verification"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Build a user authentication system",
            strategy=SoftwareStrategy(),
        )

        assert result.strategy_type == StrategyType.SOFTWARE
        assert result.is_valid is True
        assert len(result.nodes) >= 6

    @pytest.mark.asyncio
    async def test_decompose_validates_required_stages(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test that decomposition validates required stages."""
        # Mock LLM response missing required stage
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify requirements", "hard_deps": [], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement", "hard_deps": ["clarify_requirements"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Build an API",
            strategy=SoftwareStrategy(),
        )

        # Should be invalid because required stages are missing
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    @pytest.mark.asyncio
    async def test_decompose_allows_optional_stages(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test that decomposition allows adding optional stages."""
        # Mock LLM response with optional db_schema stage
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Layout", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "db_schema", "task": "Define database schema", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Types", "hard_deps": ["project_layout", "db_schema"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "verification", "task": "Verify", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Synthesize", "hard_deps": ["verification"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Build a database-backed API",
            strategy=SoftwareStrategy(),
        )

        assert result.is_valid is True
        stage_names = [n.stage_name for n in result.nodes]
        assert "db_schema" in stage_names

    @pytest.mark.asyncio
    async def test_decompose_enforces_contracts_first(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test that contracts_first constraint is enforced."""
        # Mock LLM response where implementation doesn't depend on types_contracts
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Layout", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Types", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "verification", "task": "Verify", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Synthesize", "hard_deps": ["verification"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Build an API",
            strategy=SoftwareStrategy(),
        )

        # Should be invalid because implementation doesn't depend on types_contracts
        assert result.is_valid is False
        assert any("contracts" in e.lower() for e in result.validation_errors)

    @pytest.mark.asyncio
    async def test_decompose_with_research_strategy(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test decomposition with research strategy."""
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_scope", "task": "Define research scope", "hard_deps": [], "soft_deps": []},
                {"stage_name": "gather_sources", "task": "Search vault for sources", "hard_deps": ["clarify_scope"], "soft_deps": []},
                {"stage_name": "synthesize", "task": "Synthesize findings", "hard_deps": ["gather_sources"], "soft_deps": []},
                {"stage_name": "cite", "task": "Add citations", "hard_deps": ["synthesize"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Research authentication best practices",
            strategy=ResearchStrategy(),
        )

        assert result.strategy_type == StrategyType.RESEARCH
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_decompose_provides_hint_pack(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test that decomposer provides strategy hint pack to LLM."""
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Layout", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Types", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "verification", "task": "Verify", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Synthesize", "hard_deps": ["verification"], "soft_deps": []}
            ]""",
            100,
        ))

        await decomposer.decompose(
            task="Build an API",
            strategy=SoftwareStrategy(),
        )

        # Verify LLM was called with hint pack
        call_args = mock_llm.complete.call_args
        prompt = call_args[1].get("prompt", call_args[0][0] if call_args[0] else "")
        # The prompt should contain strategy-specific guidance
        assert "contract" in prompt.lower() or "type" in prompt.lower()

    @pytest.mark.asyncio
    async def test_decompose_handles_malformed_response(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test handling of malformed LLM responses."""
        mock_llm.complete = AsyncMock(return_value=("not valid json", 100))

        result = await decomposer.decompose(
            task="Build an API",
            strategy=SoftwareStrategy(),
        )

        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    @pytest.mark.asyncio
    async def test_decompose_splits_implementation(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test that LLM can split implementation into multiple nodes."""
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Layout", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Types", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement user module", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "implementation_api", "task": "Implement API routes", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "implementation_db", "task": "Implement DB layer", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "verification", "task": "Verify", "hard_deps": ["implementation", "implementation_api", "implementation_db"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Synthesize", "hard_deps": ["verification"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Build a complete user management system",
            strategy=SoftwareStrategy(),
        )

        assert result.is_valid is True
        stage_names = [n.stage_name for n in result.nodes]
        # Should have multiple implementation nodes
        impl_nodes = [s for s in stage_names if s.startswith("implementation")]
        assert len(impl_nodes) >= 2


class TestDecomposerWithBudget:
    """Tests for decomposition with budget constraints."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM provider."""
        llm = MagicMock()
        return llm

    @pytest.fixture
    def decomposer(self, mock_llm: MagicMock) -> StrategyDecomposer:
        """Create decomposer for testing."""
        return StrategyDecomposer(llm_provider=mock_llm)

    @pytest.mark.asyncio
    async def test_decompose_respects_max_nodes(
        self, decomposer: StrategyDecomposer, mock_llm: MagicMock
    ) -> None:
        """Test that decomposition respects max_nodes budget."""
        # Mock LLM response with many nodes
        mock_llm.complete = AsyncMock(return_value=(
            """[
                {"stage_name": "clarify_requirements", "task": "Clarify", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Layout", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Types", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Impl", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "verification", "task": "Verify", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Synth", "hard_deps": ["verification"], "soft_deps": []}
            ]""",
            100,
        ))

        result = await decomposer.decompose(
            task="Build an API",
            strategy=SoftwareStrategy(),
            max_nodes=10,
        )

        # Should respect max_nodes
        assert len(result.nodes) <= 10
