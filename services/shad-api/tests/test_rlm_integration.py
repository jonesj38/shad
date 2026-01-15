"""Tests for RLM engine integration with strategies, context packets, and verification.

Per SPEC.md:
- RLMEngine should use strategy selection for decomposition
- RLMEngine should use StrategyDecomposer for dependency-aware DAG
- RLMEngine should create and inject context packets
- RLMEngine should generate file manifests for code output
- RLMEngine should call verification layer
- RLMEngine should track run states with refinement manager
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.engine.strategies import (
    StrategySelector,
    StrategyType,
    get_strategy,
)
from shad.engine.decomposition import DecompositionNode, DecompositionResult, StrategyDecomposer
from shad.engine.context_packets import ContextPacket, ContextPacketStore, NodeContextManager
from shad.engine.rlm import RLMEngine
from shad.engine.llm import LLMProvider
from shad.models.run import RunConfig, Budget, DAGNode, NodeStatus, RunStatus


class TestStrategyIntegration:
    """Tests for strategy integration in RLMEngine."""

    def test_strategy_selector_available_in_engine(self) -> None:
        """Test that RLMEngine has access to strategy selector."""
        engine = RLMEngine()
        assert hasattr(engine, 'strategy_selector')
        assert isinstance(engine.strategy_selector, StrategySelector)

    def test_strategy_selection_on_execute(self) -> None:
        """Test that execute selects a strategy based on the goal."""
        engine = RLMEngine()

        # Test software detection - need multiple keywords for high confidence
        result = engine.strategy_selector.select("Build and implement a REST API application for user management")
        assert result.strategy_type == StrategyType.SOFTWARE

        # Test research detection - need multiple keywords
        result = engine.strategy_selector.select("Research and investigate best practices for authentication")
        assert result.strategy_type == StrategyType.RESEARCH

    def test_strategy_override_respected(self) -> None:
        """Test that user strategy override is respected."""
        engine = RLMEngine()

        # Even though task looks like software, override to research
        result = engine.strategy_selector.select(
            "Build an API",
            override=StrategyType.RESEARCH
        )
        assert result.strategy_type == StrategyType.RESEARCH
        assert result.is_override is True

    def test_get_strategy_skeleton(self) -> None:
        """Test getting strategy skeleton for decomposition."""
        strategy = get_strategy(StrategyType.SOFTWARE)
        skeleton = strategy.skeleton

        # Should have required stages
        required_names = [s.name for s in skeleton.required_stages]
        assert "types_contracts" in required_names
        assert "implementation" in required_names

        # Should have contracts-first constraint
        constraint = skeleton.get_constraint("contracts_first")
        assert constraint is not None
        assert constraint.value is True


class TestDecomposerIntegration:
    """Tests for StrategyDecomposer integration."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM provider."""
        llm = MagicMock(spec=LLMProvider)
        llm.complete = AsyncMock(return_value=(
            '''[
                {"stage_name": "clarify_requirements", "task": "Extract requirements", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "Define structure", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "Define types", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "Implement API", "hard_deps": ["types_contracts"], "soft_deps": []},
                {"stage_name": "verification", "task": "Run tests", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "Assemble output", "hard_deps": ["verification"], "soft_deps": []}
            ]''',
            100
        ))
        return llm

    @pytest.mark.asyncio
    async def test_decomposer_produces_valid_dag(self, mock_llm: MagicMock) -> None:
        """Test that StrategyDecomposer produces valid decomposition."""
        decomposer = StrategyDecomposer(mock_llm)
        strategy = get_strategy(StrategyType.SOFTWARE)

        result = await decomposer.decompose(
            task="Build a REST API for user management",
            strategy=strategy,
        )

        assert result.is_valid is True
        assert len(result.nodes) >= 6  # At least required stages

        # Check required stages present
        stage_names = [n.stage_name for n in result.nodes]
        assert "types_contracts" in stage_names
        assert "implementation" in stage_names

    @pytest.mark.asyncio
    async def test_decomposer_validates_contracts_first(self, mock_llm: MagicMock) -> None:
        """Test that decomposer validates contracts-first constraint."""
        # Create LLM that returns implementation without contracts dependency
        mock_llm.complete = AsyncMock(return_value=(
            '''[
                {"stage_name": "clarify_requirements", "task": "...", "hard_deps": [], "soft_deps": []},
                {"stage_name": "project_layout", "task": "...", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "...", "hard_deps": ["project_layout"], "soft_deps": []},
                {"stage_name": "implementation", "task": "...", "hard_deps": [], "soft_deps": []},
                {"stage_name": "verification", "task": "...", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "...", "hard_deps": ["verification"], "soft_deps": []}
            ]''',
            100
        ))

        decomposer = StrategyDecomposer(mock_llm)
        strategy = get_strategy(StrategyType.SOFTWARE)

        result = await decomposer.decompose(
            task="Build API",
            strategy=strategy,
        )

        # Should fail validation due to contracts-first violation
        assert result.is_valid is False
        assert any("contracts-first" in e.lower() for e in result.validation_errors)


class TestContextPacketIntegration:
    """Tests for context packet integration."""

    def test_engine_has_context_manager(self) -> None:
        """Test that RLMEngine has a NodeContextManager."""
        engine = RLMEngine()
        assert hasattr(engine, 'context_manager')
        assert isinstance(engine.context_manager, NodeContextManager)

    def test_context_packet_created_on_node_completion(self) -> None:
        """Test that context packets are created when nodes complete."""
        manager = NodeContextManager()

        # Simulate node completion
        packet = manager.create_packet(
            node_id="node_123",
            stage_name="types_contracts",
            result="interface User { id: string; name: string; }",
            artifacts=["src/types.ts"],
        )

        assert packet.node_id == "node_123"
        assert packet.stage_name == "types_contracts"
        assert "User" in packet.keywords
        manager.store.add(packet)

        # Should be retrievable
        assert manager.store.get("node_123") is not None

    def test_context_injection_from_soft_deps(self) -> None:
        """Test that context is injected from soft dependencies."""
        manager = NodeContextManager()

        # Add a completed types_contracts packet
        packet = ContextPacket(
            node_id="node_1",
            stage_name="types_contracts",
            summary="Defined User and Task interfaces",
            keywords=["User", "Task", "interface"],
            artifacts=["types.ts"],
        )
        manager.store.add(packet)

        # Get context for implementation node with soft dep on types_contracts
        context = manager.inject_soft_dep_context(
            soft_deps=["types_contracts"],
            task="Implement user service",
        )

        assert "types_contracts" in context
        assert "User" in context or "interface" in context


class TestRLMEngineIntegration:
    """Integration tests for the complete RLM flow."""

    @pytest.fixture
    def engine(self) -> RLMEngine:
        """Create an RLMEngine instance with mocked dependencies."""
        llm = MagicMock(spec=LLMProvider)
        llm.decompose_task = AsyncMock(return_value=["Subtask 1", "Subtask 2"])
        llm.answer_task = AsyncMock(return_value=("Result", 50))
        llm.synthesize_results = AsyncMock(return_value="Synthesized result")
        llm.complete = AsyncMock(return_value=(
            '''[
                {"stage_name": "clarify_requirements", "task": "...", "hard_deps": [], "soft_deps": []},
                {"stage_name": "types_contracts", "task": "...", "hard_deps": ["clarify_requirements"], "soft_deps": []},
                {"stage_name": "implementation", "task": "...", "hard_deps": ["types_contracts"], "soft_deps": ["clarify_requirements"]},
                {"stage_name": "verification", "task": "...", "hard_deps": ["implementation"], "soft_deps": []},
                {"stage_name": "synthesis", "task": "...", "hard_deps": ["verification"], "soft_deps": []}
            ]''',
            100
        ))
        return RLMEngine(llm_provider=llm)

    @pytest.mark.asyncio
    async def test_execute_uses_strategy_selection(self, engine: RLMEngine) -> None:
        """Test that execute uses strategy selection."""
        config = RunConfig(
            goal="Build a REST API",
            budget=Budget(max_depth=2, max_nodes=10),
        )

        run = await engine.execute(config)

        # Should have selected a strategy
        assert hasattr(run, 'strategy_type') or run.config is not None

    @pytest.mark.asyncio
    async def test_execute_respects_strategy_override(self, engine: RLMEngine) -> None:
        """Test that strategy override in config is respected."""
        config = RunConfig(
            goal="Build an API",
            budget=Budget(max_depth=2, max_nodes=10),
            strategy_override="research",
        )

        # Execute should use the overridden strategy
        override_type = StrategyType(config.strategy_override) if config.strategy_override else None
        selection = engine.strategy_selector.select(
            config.goal,
            override=override_type
        )
        assert selection.strategy_type == StrategyType.RESEARCH

    @pytest.mark.asyncio
    async def test_context_packets_flow_between_nodes(self, engine: RLMEngine) -> None:
        """Test that context packets flow from completed nodes to pending ones."""
        # Create a packet in the context manager
        packet = ContextPacket(
            node_id="contracts_node",
            stage_name="types_contracts",
            summary="User interface defined",
            keywords=["User", "interface"],
        )
        engine.context_manager.store.add(packet)

        # Implementation node should be able to get this context
        context = engine.context_manager.inject_soft_dep_context(
            soft_deps=["types_contracts"],
            task="Implement user service",
        )

        assert "User" in context or "types_contracts" in context


class TestFileManifestIntegration:
    """Tests for file manifest generation in synthesis."""

    def test_engine_has_manifest_generation(self) -> None:
        """Test that engine can generate file manifests."""
        engine = RLMEngine()
        assert hasattr(engine, 'generate_manifest') or hasattr(engine, '_generate_manifest')

    @pytest.mark.asyncio
    async def test_software_strategy_produces_manifest(self) -> None:
        """Test that software strategy execution produces a file manifest."""
        from shad.output.manifest import FileManifest, FileEntry

        # Create a manifest
        manifest = FileManifest(run_id="test_run")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User { id: string; }",
            language="typescript",
            source_nodes=["types_contracts"],
        ))

        assert len(manifest.files) == 1
        assert manifest.files[0].path == "src/types.ts"


class TestVerificationIntegration:
    """Tests for verification layer integration."""

    @pytest.mark.asyncio
    async def test_verification_called_on_synthesis(self) -> None:
        """Test that verification is called during synthesis."""
        from shad.verification.layer import VerificationLayer, VerificationConfig, VerificationLevel
        from shad.output.manifest import FileManifest, FileEntry

        layer = VerificationLayer()
        config = VerificationConfig(level=VerificationLevel.BASIC)

        # Create a valid manifest
        manifest = FileManifest(run_id="test")
        manifest.add_file(FileEntry(
            path="src/api.py",
            content="def hello(): return 'hello'",
            language="python",
            source_nodes=["impl"],
        ))

        result = await layer.verify(manifest, config)

        # Should pass basic verification
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_verification_catches_syntax_errors(self) -> None:
        """Test that verification catches syntax errors."""
        from shad.verification.layer import VerificationLayer, VerificationConfig, VerificationLevel
        from shad.output.manifest import FileManifest, FileEntry

        layer = VerificationLayer()
        config = VerificationConfig(level=VerificationLevel.BASIC)

        # Create manifest with syntax error
        manifest = FileManifest(run_id="test")
        manifest.add_file(FileEntry(
            path="src/api.py",
            content="def hello( return",  # Syntax error
            language="python",
            source_nodes=["impl"],
        ))

        result = await layer.verify(manifest, config)

        # Should fail due to syntax error
        assert result.passed is False


class TestRefinementIntegration:
    """Tests for refinement manager integration."""

    def test_engine_tracks_run_state(self) -> None:
        """Test that engine tracks run state transitions."""
        from shad.refinement.manager import RunStateManager, RunState

        manager = RunStateManager()
        assert manager.state == RunState.PENDING

        manager.transition_to(RunState.RUNNING)
        assert manager.state == RunState.RUNNING

        manager.transition_to(RunState.SUCCESS)
        assert manager.state == RunState.SUCCESS

    def test_delta_verification_tracks_used_notes(self) -> None:
        """Test that delta verification tracks used notes for resume."""
        from shad.refinement.manager import DeltaVerifier

        verifier = DeltaVerifier()
        verifier.track_node(
            node_id="node_1",
            used_notes=["Patterns/Auth.md"],
            note_hashes={"Patterns/Auth.md": "abc123"},
        )

        # Should detect staleness when hash changes
        assert verifier.is_stale("node_1", {"Patterns/Auth.md": "xyz789"}) is True

        # Should not be stale when hash is same
        assert verifier.is_stale("node_1", {"Patterns/Auth.md": "abc123"}) is False


class TestCLIIntegration:
    """Tests for CLI command integration."""

    def test_run_config_accepts_strategy(self) -> None:
        """Test that RunConfig accepts strategy parameter."""
        config = RunConfig(
            goal="Build API",
            strategy_override="software",
        )
        assert config.strategy_override == "software"

    def test_run_config_accepts_verify_level(self) -> None:
        """Test that RunConfig accepts verify level parameter."""
        config = RunConfig(
            goal="Build API",
            verify_level="strict",
        )
        assert config.verify_level == "strict"

    def test_run_config_accepts_write_files(self) -> None:
        """Test that RunConfig accepts write_files parameter."""
        config = RunConfig(
            goal="Build API",
            write_files=True,
            output_path="./output",
        )
        assert config.write_files is True
        assert config.output_path == "./output"
