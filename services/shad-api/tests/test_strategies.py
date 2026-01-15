"""Tests for strategy skeletons and strategy selection.

Per SPEC.md Section 2.2:
- Each strategy defines required stages, optional stages, constraints
- Heuristic classifier runs first (keyword matching)
- LLM can request strategy switch mid-execution
"""

from __future__ import annotations

import pytest

from shad.engine.strategies import (
    Strategy,
    StrategySkeleton,
    StrategyStage,
    StrategyConstraint,
    StrategySelector,
    StrategyType,
    SoftwareStrategy,
    ResearchStrategy,
    AnalysisStrategy,
    PlanningStrategy,
)


class TestStrategyStage:
    """Tests for strategy stages."""

    def test_create_required_stage(self) -> None:
        """Test creating a required stage."""
        stage = StrategyStage(
            name="types_contracts",
            description="Define types and contracts",
            required=True,
        )
        assert stage.name == "types_contracts"
        assert stage.required is True

    def test_create_optional_stage(self) -> None:
        """Test creating an optional stage."""
        stage = StrategyStage(
            name="db_schema",
            description="Define database schema",
            required=False,
        )
        assert stage.required is False


class TestStrategyConstraint:
    """Tests for strategy constraints."""

    def test_create_constraint(self) -> None:
        """Test creating a constraint."""
        constraint = StrategyConstraint(
            name="contracts_first",
            description="Types & contracts must be defined before implementation",
            value=True,
        )
        assert constraint.name == "contracts_first"
        assert constraint.value is True


class TestStrategySkeleton:
    """Tests for strategy skeletons."""

    def test_create_skeleton_with_required_stages(self) -> None:
        """Test creating a skeleton with required stages."""
        skeleton = StrategySkeleton(
            strategy_type=StrategyType.SOFTWARE,
            required_stages=[
                StrategyStage(name="clarify_requirements", description="Clarify requirements", required=True),
                StrategyStage(name="types_contracts", description="Define types", required=True),
            ],
            optional_stages=[
                StrategyStage(name="db_schema", description="DB schema", required=False),
            ],
            constraints=[
                StrategyConstraint(name="contracts_first", description="Contracts first", value=True),
            ],
        )
        assert len(skeleton.required_stages) == 2
        assert len(skeleton.optional_stages) == 1
        assert len(skeleton.constraints) == 1

    def test_skeleton_get_stage_by_name(self) -> None:
        """Test getting a stage by name."""
        skeleton = StrategySkeleton(
            strategy_type=StrategyType.SOFTWARE,
            required_stages=[
                StrategyStage(name="types_contracts", description="Types", required=True),
            ],
            optional_stages=[],
            constraints=[],
        )
        stage = skeleton.get_stage("types_contracts")
        assert stage is not None
        assert stage.name == "types_contracts"

    def test_skeleton_all_stages(self) -> None:
        """Test getting all stages (required + optional)."""
        skeleton = StrategySkeleton(
            strategy_type=StrategyType.SOFTWARE,
            required_stages=[
                StrategyStage(name="req1", description="R1", required=True),
            ],
            optional_stages=[
                StrategyStage(name="opt1", description="O1", required=False),
            ],
            constraints=[],
        )
        all_stages = skeleton.all_stages
        assert len(all_stages) == 2

    def test_skeleton_validate_dag_missing_required(self) -> None:
        """Test that validation fails when required stages are missing."""
        skeleton = StrategySkeleton(
            strategy_type=StrategyType.SOFTWARE,
            required_stages=[
                StrategyStage(name="types_contracts", description="Types", required=True),
                StrategyStage(name="implementation", description="Impl", required=True),
            ],
            optional_stages=[],
            constraints=[],
        )
        dag_stages = ["types_contracts"]  # Missing implementation
        is_valid, missing = skeleton.validate_dag(dag_stages)
        assert is_valid is False
        assert "implementation" in missing


class TestSoftwareStrategy:
    """Tests for the software strategy."""

    def test_software_strategy_has_required_stages(self) -> None:
        """Test that software strategy has the required stages per SPEC.md."""
        strategy = SoftwareStrategy()
        skeleton = strategy.skeleton

        required_names = [s.name for s in skeleton.required_stages]

        # Per SPEC.md Section 2.2.1
        assert "clarify_requirements" in required_names
        assert "project_layout" in required_names
        assert "types_contracts" in required_names
        assert "implementation" in required_names
        assert "verification" in required_names
        assert "synthesis" in required_names

    def test_software_strategy_has_optional_stages(self) -> None:
        """Test that software strategy has optional stages."""
        strategy = SoftwareStrategy()
        skeleton = strategy.skeleton

        optional_names = [s.name for s in skeleton.optional_stages]

        # Per SPEC.md Section 2.2.1
        assert "db_schema" in optional_names
        assert "auth" in optional_names
        assert "openapi" in optional_names

    def test_software_strategy_has_contracts_first_constraint(self) -> None:
        """Test that software strategy enforces contracts_first."""
        strategy = SoftwareStrategy()
        skeleton = strategy.skeleton

        constraint_names = [c.name for c in skeleton.constraints]
        assert "contracts_first" in constraint_names

    def test_software_strategy_default_dependencies(self) -> None:
        """Test that software strategy has proper default dependencies."""
        strategy = SoftwareStrategy()
        deps = strategy.default_dependencies

        # types_contracts must complete before implementation
        assert ("types_contracts", "implementation") in deps


class TestResearchStrategy:
    """Tests for the research strategy."""

    def test_research_strategy_has_required_stages(self) -> None:
        """Test that research strategy has the required stages per SPEC.md."""
        strategy = ResearchStrategy()
        skeleton = strategy.skeleton

        required_names = [s.name for s in skeleton.required_stages]

        # Per SPEC.md Section 2.2.1
        assert "clarify_scope" in required_names
        assert "gather_sources" in required_names
        assert "synthesize" in required_names
        assert "cite" in required_names

    def test_research_strategy_must_cite_vault_constraint(self) -> None:
        """Test that research strategy has must_cite_vault constraint."""
        strategy = ResearchStrategy()
        skeleton = strategy.skeleton

        constraint_names = [c.name for c in skeleton.constraints]
        assert "must_cite_vault" in constraint_names


class TestAnalysisStrategy:
    """Tests for the analysis strategy."""

    def test_analysis_strategy_has_required_stages(self) -> None:
        """Test that analysis strategy has required stages."""
        strategy = AnalysisStrategy()
        skeleton = strategy.skeleton

        required_names = [s.name for s in skeleton.required_stages]
        assert len(required_names) >= 3


class TestPlanningStrategy:
    """Tests for the planning strategy."""

    def test_planning_strategy_has_required_stages(self) -> None:
        """Test that planning strategy has required stages."""
        strategy = PlanningStrategy()
        skeleton = strategy.skeleton

        required_names = [s.name for s in skeleton.required_stages]
        assert len(required_names) >= 2


class TestStrategySelector:
    """Tests for heuristic strategy selection."""

    @pytest.fixture
    def selector(self) -> StrategySelector:
        """Create a strategy selector."""
        return StrategySelector()

    def test_select_software_strategy_by_keywords(
        self, selector: StrategySelector
    ) -> None:
        """Test selecting software strategy by keywords."""
        task = "Build a REST API with authentication"
        result = selector.select(task)

        assert result.strategy_type == StrategyType.SOFTWARE
        assert result.confidence >= 0.7

    def test_select_research_strategy_by_keywords(
        self, selector: StrategySelector
    ) -> None:
        """Test selecting research strategy by keywords."""
        task = "Research the best practices for API design"
        result = selector.select(task)

        assert result.strategy_type == StrategyType.RESEARCH

    def test_select_analysis_strategy_by_keywords(
        self, selector: StrategySelector
    ) -> None:
        """Test selecting analysis strategy by keywords."""
        task = "Analyze the performance of the current system"
        result = selector.select(task)

        assert result.strategy_type == StrategyType.ANALYSIS

    def test_select_planning_strategy_by_keywords(
        self, selector: StrategySelector
    ) -> None:
        """Test selecting planning strategy by keywords."""
        task = "Create a project plan with timeline and milestones"
        result = selector.select(task)

        assert result.strategy_type == StrategyType.PLANNING

    def test_default_to_analysis_when_uncertain(
        self, selector: StrategySelector
    ) -> None:
        """Test defaulting to analysis when confidence is low."""
        task = "Something vague"
        result = selector.select(task)

        # Per SPEC.md: confidence < 0.7 → default to analysis
        if result.confidence < 0.7:
            assert result.strategy_type == StrategyType.ANALYSIS

    def test_user_override_strategy(self, selector: StrategySelector) -> None:
        """Test that user can override strategy selection."""
        task = "Build an API"  # Would normally select software
        result = selector.select(task, override=StrategyType.RESEARCH)

        assert result.strategy_type == StrategyType.RESEARCH
        assert result.is_override is True

    def test_confidence_threshold(self, selector: StrategySelector) -> None:
        """Test confidence threshold behavior."""
        # High confidence task
        task = "Build a web application with database and API"
        result = selector.select(task)

        # Should have high confidence for software strategy
        assert result.confidence >= 0.5


class TestStrategySelectionResult:
    """Tests for strategy selection results."""

    def test_selection_result_structure(self) -> None:
        """Test that selection result has expected structure."""
        from shad.engine.strategies import StrategySelectionResult

        result = StrategySelectionResult(
            strategy_type=StrategyType.SOFTWARE,
            confidence=0.85,
            is_override=False,
            matched_keywords=["build", "api"],
        )

        assert result.strategy_type == StrategyType.SOFTWARE
        assert result.confidence == 0.85
        assert result.is_override is False
        assert "build" in result.matched_keywords


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_get_strategy_by_type(self) -> None:
        """Test getting a strategy by type."""
        from shad.engine.strategies import get_strategy

        strategy = get_strategy(StrategyType.SOFTWARE)
        assert isinstance(strategy, SoftwareStrategy)

    def test_get_all_strategies(self) -> None:
        """Test getting all strategies."""
        from shad.engine.strategies import get_all_strategies

        strategies = get_all_strategies()
        assert len(strategies) >= 4  # software, research, analysis, planning


class TestStrategyIntegration:
    """Integration tests for strategy system."""

    def test_software_strategy_dag_generation_hint(self) -> None:
        """Test that software strategy can provide DAG generation hints."""
        strategy = SoftwareStrategy()
        hints = strategy.get_decomposition_hints(
            task="Build a user authentication system"
        )

        # Should provide relevant hints
        assert hints is not None
        assert "contracts" in hints.lower() or "types" in hints.lower()

    def test_strategy_provides_hint_pack(self) -> None:
        """Test that strategy provides hint pack for LLM."""
        strategy = SoftwareStrategy()
        hint_pack = strategy.get_hint_pack()

        assert hint_pack is not None
        assert len(hint_pack) > 0
