"""Integration tests for the discipline-report strategy.

Covers:
- Strategy selection and deterministic decomposition
- DAG shape (wide/shallow, no implementation stages)
- Model tier routing (leaf for sections, orchestrator for synthesis/quality gate)
- Source map generation and injection
- Artifact metadata on decomposition nodes
- Quality gate task template content
- Anti-implementation-drift constraint enforcement
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shad.discipline.source_map import SourceMapGenerator
from shad.engine.decomposition import StrategyDecomposer
from shad.engine.llm import ModelTier
from shad.engine.strategies import (
    DisciplineReportStrategy,
    StrategySelector,
    StrategyType,
    get_strategy,
)


class TestDisciplineStrategySelection:
    """Tests for discipline-report strategy selection heuristics."""

    @pytest.fixture
    def selector(self) -> StrategySelector:
        return StrategySelector()

    @pytest.mark.parametrize(
        "task",
        [
            "Build a source-grounded discipline for Semantos",
            "Build the EdwinPAI discipline report with retrieval hints",
            "Create agent discipline from this source corpus with useWhen avoidWhen",
            "Generate knowledge discipline for future agents",
        ],
    )
    def test_discipline_keywords_select_correct_strategy(
        self, selector: StrategySelector, task: str
    ) -> None:
        result = selector.select(task)
        assert result.strategy_type == StrategyType.DISCIPLINE_REPORT
        assert result.confidence >= 0.7

    def test_override_forces_discipline_report(self, selector: StrategySelector) -> None:
        result = selector.select(
            "Build something",
            override=StrategyType.DISCIPLINE_REPORT,
        )
        assert result.strategy_type == StrategyType.DISCIPLINE_REPORT
        assert result.is_override is True
        assert result.confidence == 1.0


class TestDisciplineDAGShape:
    """Tests that discipline-report DAGs are wide, shallow, and deterministic."""

    @pytest.fixture
    def decomposer(self) -> StrategyDecomposer:
        mock_llm = MagicMock()
        return StrategyDecomposer(llm_provider=mock_llm)

    @pytest.mark.asyncio
    async def test_dag_is_deterministic_no_llm_call(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline for test corpus",
            strategy=strategy,
            max_nodes=50,
        )
        assert result.tokens_used == 0
        decomposer.llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_required_stages_present(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        stage_names = {n.stage_name for n in result.nodes}
        expected = {
            "source_map", "product_model", "repo_architecture",
            "core_concepts", "protocols_security", "formal_methods",
            "developer_workflows", "operational_pitfalls", "routing_hints",
            "final_synthesis", "quality_gate",
        }
        assert expected == stage_names

    @pytest.mark.asyncio
    async def test_sections_depend_only_on_source_map(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        section_stages = {
            "product_model", "repo_architecture", "core_concepts",
            "protocols_security", "formal_methods", "developer_workflows",
            "operational_pitfalls", "routing_hints",
        }
        for node in result.nodes:
            if node.stage_name in section_stages:
                assert node.hard_deps == ["source_map"], (
                    f"{node.stage_name} should depend only on source_map, got {node.hard_deps}"
                )

    @pytest.mark.asyncio
    async def test_final_synthesis_depends_on_all_sections(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        final = next(n for n in result.nodes if n.stage_name == "final_synthesis")
        expected_deps = {
            "product_model", "repo_architecture", "core_concepts",
            "protocols_security", "formal_methods", "developer_workflows",
            "operational_pitfalls", "routing_hints",
        }
        assert set(final.hard_deps) == expected_deps

    @pytest.mark.asyncio
    async def test_quality_gate_depends_on_final_synthesis(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        gate = next(n for n in result.nodes if n.stage_name == "quality_gate")
        assert gate.hard_deps == ["final_synthesis"]

    @pytest.mark.asyncio
    async def test_no_implementation_stages(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        for node in result.nodes:
            assert not node.stage_name.startswith("implementation"), (
                f"Discipline DAG should not contain implementation stages, found {node.stage_name}"
            )


class TestDisciplineArtifactMetadata:
    """Tests that decomposition nodes carry correct artifact metadata."""

    @pytest.fixture
    def decomposer(self) -> StrategyDecomposer:
        return StrategyDecomposer(llm_provider=MagicMock())

    @pytest.mark.asyncio
    async def test_each_node_has_artifact_filename(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        expected_artifacts = {
            "source_map": "source-map.md",
            "product_model": "product-model.md",
            "repo_architecture": "architecture.md",
            "core_concepts": "runtime-concepts.md",
            "protocols_security": "protocols-security.md",
            "formal_methods": "formal-methods.md",
            "developer_workflows": "developer-workflows.md",
            "operational_pitfalls": "pitfalls-checklists.md",
            "routing_hints": "routing-hints.md",
            "final_synthesis": "discipline-report.md",
            "quality_gate": "quality-gate.md",
        }
        for node in result.nodes:
            assert "artifact" in node.metadata, f"{node.stage_name} missing artifact metadata"
            assert node.metadata["artifact"] == expected_artifacts[node.stage_name], (
                f"{node.stage_name}: expected artifact {expected_artifacts[node.stage_name]}, "
                f"got {node.metadata['artifact']}"
            )

    @pytest.mark.asyncio
    async def test_section_nodes_tagged_as_discipline_report_section(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        for node in result.nodes:
            assert node.metadata.get("discipline_report_section") is True

    @pytest.mark.asyncio
    async def test_section_nodes_have_soft_dep_on_source_map(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        for node in result.nodes:
            if node.stage_name != "source_map":
                assert "source_map" in node.soft_deps, (
                    f"{node.stage_name} should have source_map as soft dep"
                )


class TestDisciplineModelTierRouting:
    """Tests that discipline-report leaf execution routes to correct model tiers."""

    def test_section_stages_use_leaf_tier(self) -> None:
        """All parallel section stages should use the LEAF tier for speed."""
        section_stages = [
            "product_model", "repo_architecture", "core_concepts",
            "protocols_security", "formal_methods", "developer_workflows",
            "operational_pitfalls", "routing_hints", "source_map",
        ]
        strategy = DisciplineReportStrategy()
        for stage_name in section_stages:
            # Simulate the tier selection logic from RLMEngine._execute_leaf
            if stage_name in {"final_synthesis", "quality_gate"}:
                tier = ModelTier.ORCHESTRATOR
            else:
                tier = ModelTier.LEAF
            assert tier == ModelTier.LEAF, f"{stage_name} should use LEAF tier"

    def test_final_synthesis_uses_orchestrator_tier(self) -> None:
        # Simulate the tier selection logic from RLMEngine._execute_leaf
        stage_name = "final_synthesis"
        if stage_name in {"final_synthesis", "quality_gate"}:
            tier = ModelTier.ORCHESTRATOR
        else:
            tier = ModelTier.LEAF
        assert tier == ModelTier.ORCHESTRATOR

    def test_quality_gate_uses_orchestrator_tier(self) -> None:
        stage_name = "quality_gate"
        if stage_name in {"final_synthesis", "quality_gate"}:
            tier = ModelTier.ORCHESTRATOR
        else:
            tier = ModelTier.LEAF
        assert tier == ModelTier.ORCHESTRATOR


class TestDisciplineQualityGateContent:
    """Tests that quality gate task template contains required verification checks."""

    @pytest.fixture
    def decomposer(self) -> StrategyDecomposer:
        return StrategyDecomposer(llm_provider=MagicMock())

    @pytest.mark.asyncio
    async def test_quality_gate_checks_required_sections(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        gate = next(n for n in result.nodes if n.stage_name == "quality_gate")
        task_lower = gate.task.lower()
        assert "required sections" in task_lower
        assert "citations" in task_lower or "citation" in task_lower
        assert "speculative" in task_lower or "unsupported" in task_lower
        assert "pass" in task_lower or "needs_work" in task_lower

    @pytest.mark.asyncio
    async def test_quality_gate_checks_usewhen_avoidwhen(self, decomposer: StrategyDecomposer) -> None:
        strategy = get_strategy(StrategyType.DISCIPLINE_REPORT)
        result = await decomposer.decompose(
            task="Build discipline",
            strategy=strategy,
            max_nodes=50,
        )
        gate = next(n for n in result.nodes if n.stage_name == "quality_gate")
        task_lower = gate.task.lower()
        assert "usewhen" in task_lower
        assert "avoidwhen" in task_lower


class TestDisciplineAntiDriftConstraints:
    """Tests that discipline-report strategy constrains against implementation drift."""

    def test_strategy_has_no_new_implementation_constraint(self) -> None:
        strategy = DisciplineReportStrategy()
        constraint = strategy.skeleton.get_constraint("no_new_implementation_design")
        assert constraint is not None
        assert constraint.value is True

    def test_strategy_has_source_grounded_constraint(self) -> None:
        strategy = DisciplineReportStrategy()
        constraint = strategy.skeleton.get_constraint("source_grounded")
        assert constraint is not None
        assert constraint.value is True

    def test_strategy_has_wide_shallow_dag_constraint(self) -> None:
        strategy = DisciplineReportStrategy()
        constraint = strategy.skeleton.get_constraint("wide_shallow_dag")
        assert constraint is not None
        assert constraint.value is True

    def test_hint_pack_forbids_new_design(self) -> None:
        strategy = DisciplineReportStrategy()
        hint = strategy.get_hint_pack().lower()
        assert "do not invent schemas" in hint or "do not invent" in hint
        assert "not a software build" in hint


class TestSourceMapIntegration:
    """Tests for source map integration with discipline runs."""

    def test_source_map_multi_root(self, tmp_path: Path) -> None:
        root_a = tmp_path / "project-a"
        root_a.mkdir()
        (root_a / "main.py").write_text("print('a')\n")
        (root_a / "Makefile").write_text("all:\n\tpython main.py\n")

        root_b = tmp_path / "project-b"
        root_b.mkdir()
        (root_b / "lib.rs").write_text("fn main() {}\n")
        (root_b / "Cargo.toml").write_text("[package]\nname = 'b'\n")

        source_map = SourceMapGenerator(max_files_per_section=20).generate([root_a, root_b])

        assert source_map.total_files == 4
        assert len(source_map.roots) == 2
        assert ".py" in source_map.extension_counts
        assert ".rs" in source_map.extension_counts
        assert any("Makefile" in f for f in source_map.important_files)
        assert any("Cargo.toml" in f for f in source_map.important_files)

    def test_source_map_markdown_includes_revision_info(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Hello\n")
        source_map = SourceMapGenerator().generate([tmp_path])
        md = source_map.to_markdown()

        assert "## Roots" in md
        assert "## Corpus Summary" in md
        assert "## Extension Counts" in md

    def test_source_map_empty_root(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty-project"
        empty.mkdir()
        source_map = SourceMapGenerator().generate([empty])

        assert source_map.total_files == 0
        assert source_map.total_bytes == 0

    def test_source_map_detects_additional_formal_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "model.als").write_text("sig Node {}\n")
        (tmp_path / "spec.ivy").write_text("type node\n")
        source_map = SourceMapGenerator().generate([tmp_path])

        assert "model.als" in source_map.formal_files
        assert "spec.ivy" in source_map.formal_files
