"""LLM-driven decomposition with strategy constraints.

Per SPEC.md Section 2.2.2:
- LLM receives strategy name, skeleton, and hint pack
- LLM fills in task-specific details within skeleton constraints
- Can add/remove optional nodes, split implementation into modules
- Cannot violate required stages without explicit waiver
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shad.engine.strategies import (
    Strategy,
    StrategyType,
)

if TYPE_CHECKING:
    from shad.engine.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class DecompositionNode:
    """A node in the decomposition DAG.

    Each node represents a subtask with dependencies.
    """

    stage_name: str
    task: str
    hard_deps: list[str] = field(default_factory=list)
    soft_deps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    """Result of strategy-aware decomposition.

    Contains the decomposed nodes and validation status.
    """

    strategy_type: StrategyType
    nodes: list[DecompositionNode]
    is_valid: bool
    validation_errors: list[str]
    tokens_used: int = 0


class StrategyDecomposer:
    """Decomposes tasks using strategy skeletons and LLM refinement.

    Per SPEC.md Section 2.2:
    1. Receives task and strategy
    2. Provides skeleton + hint pack to LLM
    3. LLM generates decomposition within constraints
    4. Validates against required stages and constraints
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    async def decompose(
        self,
        task: str,
        strategy: Strategy,
        context: str = "",
        max_nodes: int = 50,
    ) -> DecompositionResult:
        """Decompose a task using the given strategy.

        Args:
            task: The task to decompose
            strategy: The strategy to use
            context: Optional collection context
            max_nodes: Maximum number of nodes to generate

        Returns:
            DecompositionResult with nodes and validation status
        """
        logger.info(f"[DECOMPOSE] Starting decomposition for: {task[:100]}...")
        logger.info(f"[DECOMPOSE] Strategy: {strategy.strategy_type.value}")

        if strategy.strategy_type == StrategyType.DISCIPLINE_REPORT:
            return self._decompose_discipline_report(task, strategy, max_nodes)

        # Build prompt with strategy constraints
        prompt = self._build_decomposition_prompt(task, strategy, context, max_nodes)

        try:
            # Call LLM for decomposition
            response, tokens = await self.llm.complete(
                prompt=prompt,
                system=self._build_system_prompt(strategy),
                temperature=0.3,
            )

            # Parse LLM response
            nodes = self._parse_response(response)

            # Validate against strategy
            validation_errors = self._validate_decomposition(nodes, strategy)

            is_valid = len(validation_errors) == 0

            return DecompositionResult(
                strategy_type=strategy.strategy_type,
                nodes=nodes,
                is_valid=is_valid,
                validation_errors=validation_errors,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(f"[DECOMPOSE] Error: {e}")
            return DecompositionResult(
                strategy_type=strategy.strategy_type,
                nodes=[],
                is_valid=False,
                validation_errors=[f"Decomposition failed: {e!s}"],
            )

    def _decompose_discipline_report(
        self,
        task: str,
        strategy: Strategy,
        max_nodes: int,
    ) -> DecompositionResult:
        """Build a deterministic wide/shallow DAG for discipline reports.

        Generic LLM decomposition tends to treat "build a discipline" as a
        software implementation task and creates serial contract/implementation
        loops. Discipline reports have a known artifact shape, so generate the
        section DAG directly and spend LLM calls on source-grounded section
        writing instead of planning the plan.
        """
        skeleton = strategy.skeleton
        nodes: list[DecompositionNode] = []
        stage_by_name = {stage.name: stage for stage in skeleton.required_stages}

        section_artifacts = {
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

        task_templates = {
            "source_map": (
                "Summarize the deterministic source-map and source revision context already provided. "
                "Identify the source corpus boundaries, important file groups, formal/spec assets, "
                "build/test/deploy surfaces, and any obvious coverage gaps. Do not invent files."
            ),
            "product_model": (
                "Write the product/system mental model section for the discipline report. Explain what "
                "the system is, who/what it serves, major capabilities, and how a future agent should "
                "think about it. Ground claims in retrieved source context."
            ),
            "repo_architecture": (
                "Write the repository architecture section. Map major packages, modules, languages, "
                "entrypoints, generated/vendor areas, and source-authority rules. Cite paths and symbols."
            ),
            "core_concepts": (
                "Write the core concepts section. Cover runtime concepts, data models, state machines, "
                "protocol flows, APIs, and important invariants visible in source. Cite files."
            ),
            "protocols_security": (
                "Write the protocols/security/trust section. Cover cryptographic or trust assumptions, "
                "authorization boundaries, threat model hints, policies, services, and security-sensitive "
                "runtime behavior. Mark uncertainty explicitly."
            ),
            "formal_methods": (
                "Write the formal methods section. Connect TLA+, Lean, fuzzing, specs, proofs, configs, "
                "and invariants to implementation behavior. Explain what has evidence and what remains "
                "unverified. Cite formal/spec files."
            ),
            "developer_workflows": (
                "Write the developer workflows section. Cover build, test, release, deployment, service, "
                "mobile/runtime, and operational commands/files. Prefer exact commands from source."
            ),
            "operational_pitfalls": (
                "Write the pitfalls/checklists section. Include common mistakes, source conflict rules, "
                "safe update workflow, validation checklist, and escalation points for future agents."
            ),
            "routing_hints": (
                "Write recommended runtime retrieval hints for the discipline. Include useWhen, avoidWhen, "
                "preferred source/artifact collections, query patterns, and when to fall back to raw sources."
            ),
            "final_synthesis": (
                "Assemble the canonical discipline report from section results. Include a layered artifact "
                "manifest for source-map.md, architecture.md, formal-methods.md, runtime-concepts.md, "
                "developer-workflows.md, pitfalls-checklists.md, routing-hints.md, and the canonical final "
                "report. Keep it source-grounded and do not design new code."
            ),
            "quality_gate": (
                "Verify the discipline report against quality criteria: required sections present, major "
                "claims grounded with citations/paths/quotes, formal/runtime/deploy surfaces covered, "
                "useWhen/avoidWhen hints actionable, and unsupported/speculative claims clearly marked. "
                "Return PASS/NEEDS_WORK with concrete fixes."
            ),
        }

        for stage in skeleton.required_stages:
            if len(nodes) >= max_nodes:
                break
            nodes.append(
                DecompositionNode(
                    stage_name=stage.name,
                    task=(
                        f"{task}\n\n"
                        f"Discipline-report stage: {stage.name}. {task_templates[stage.name]}\n"
                        f"Expected layered artifact: {section_artifacts[stage.name]}"
                    ),
                    hard_deps=list(stage.depends_on),
                    soft_deps=[] if stage.name == "source_map" else ["source_map"],
                    metadata={
                        "artifact": section_artifacts[stage.name],
                        "discipline_report_section": True,
                        "tags": list(stage_by_name[stage.name].tags),
                    },
                )
            )

        validation_errors = self._validate_decomposition(nodes, strategy)
        return DecompositionResult(
            strategy_type=strategy.strategy_type,
            nodes=nodes,
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            tokens_used=0,
        )

    def _build_system_prompt(self, strategy: Strategy) -> str:
        """Build system prompt with strategy hint pack."""
        return f"""You are a task decomposition expert. Decompose tasks into subtasks
following the given strategy and constraints.

{strategy.get_hint_pack()}

Return ONLY a JSON array of decomposition nodes. Each node must have:
- stage_name: Name of the stage (must match skeleton stages or be a valid extension)
- task: Specific task description
- hard_deps: Array of stage names that MUST complete before this node
- soft_deps: Array of stage names that are useful but not required

Example output:
[
    {{"stage_name": "clarify_requirements", "task": "...", "hard_deps": [], "soft_deps": []}},
    {{"stage_name": "types_contracts", "task": "...", "hard_deps": ["clarify_requirements"], "soft_deps": []}}
]"""

    def _build_decomposition_prompt(
        self,
        task: str,
        strategy: Strategy,
        context: str,
        max_nodes: int,
    ) -> str:
        """Build the decomposition prompt."""
        skeleton = strategy.skeleton

        # Build stage descriptions
        required_stages = "\n".join(
            f"  - {s.name}: {s.description} (REQUIRED)"
            for s in skeleton.required_stages
        )
        optional_stages = "\n".join(
            f"  - {s.name}: {s.description} (optional)"
            for s in skeleton.optional_stages
        )

        # Build constraint descriptions
        constraints = "\n".join(
            f"  - {c.name}: {c.description}"
            for c in skeleton.constraints
        )

        # Build default dependencies
        default_deps = "\n".join(
            f"  - {from_stage} -> {to_stage}"
            for from_stage, to_stage in skeleton.default_dependencies
        )

        context_section = ""
        if context:
            context_section = f"""
<vault_context>
{context[:5000]}
</vault_context>
"""

        return f"""Decompose this task following the {strategy.strategy_type.value} strategy.

Task: {task}
{context_section}
STRATEGY: {strategy.strategy_type.value}

REQUIRED STAGES (must include all):
{required_stages}

OPTIONAL STAGES (include if relevant):
{optional_stages}

CONSTRAINTS (must satisfy all):
{constraints}

DEFAULT DEPENDENCIES:
{default_deps}

RULES:
1. Include ALL required stages
2. Add optional stages only if relevant to the task
3. You may split implementation into multiple nodes (e.g., implementation_auth, implementation_api)
4. All implementation nodes must depend on types_contracts (for software strategy)
5. Maximum {max_nodes} nodes total

Return ONLY a JSON array of decomposition nodes."""

    def _parse_response(self, response: str) -> list[DecompositionNode]:
        """Parse LLM response into decomposition nodes."""
        try:
            # Clean response
            json_str = response.strip()

            # Handle markdown code blocks
            if "```" in json_str:
                parts = json_str.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        json_str = part[4:].strip()
                        break
                    elif part.startswith("["):
                        json_str = part
                        break

            # Parse JSON
            data = json.loads(json_str)

            if not isinstance(data, list):
                logger.warning(f"[DECOMPOSE] Response is not a list: {type(data)}")
                return []

            nodes = []
            for item in data:
                if isinstance(item, dict):
                    node = DecompositionNode(
                        stage_name=item.get("stage_name", "unknown"),
                        task=item.get("task", ""),
                        hard_deps=item.get("hard_deps", []),
                        soft_deps=item.get("soft_deps", []),
                        metadata=item.get("metadata", {}),
                    )
                    nodes.append(node)

            return nodes

        except json.JSONDecodeError as e:
            logger.error(f"[DECOMPOSE] Failed to parse JSON: {e}")
            return []

    def _validate_decomposition(
        self,
        nodes: list[DecompositionNode],
        strategy: Strategy,
    ) -> list[str]:
        """Validate decomposition against strategy constraints.

        Returns list of validation errors.
        """
        errors: list[str] = []
        skeleton = strategy.skeleton

        # Check required stages
        node_stages = {n.stage_name for n in nodes}

        # Allow stage name extensions (e.g., implementation_auth counts as implementation)
        def matches_stage(node_stage: str, required_stage: str) -> bool:
            return node_stage == required_stage or node_stage.startswith(f"{required_stage}_")

        for required in skeleton.required_stages:
            if not any(matches_stage(ns, required.name) for ns in node_stages):
                errors.append(f"Missing required stage: {required.name}")

        # Check constraints
        for constraint in skeleton.constraints:
            if constraint.name == "contracts_first" and constraint.value:
                errors.extend(self._check_contracts_first(nodes))
            elif constraint.name == "imports_must_resolve" and constraint.value:
                # This is a runtime constraint, not checked at decomposition time
                pass

        return errors

    def _check_contracts_first(self, nodes: list[DecompositionNode]) -> list[str]:
        """Check that implementation nodes depend on types_contracts."""
        errors: list[str] = []

        # Find types_contracts node(s)
        contract_stages = {
            n.stage_name for n in nodes
            if n.stage_name == "types_contracts" or n.stage_name.startswith("types_contracts_")
        }

        if not contract_stages:
            # No contracts stage, already reported as missing required stage
            return errors

        # Check implementation nodes depend on contracts
        for node in nodes:
            if node.stage_name == "implementation" or node.stage_name.startswith("implementation_"):
                # Must have types_contracts (or variant) in hard_deps
                has_contract_dep = any(
                    dep in contract_stages or dep.startswith("types_contracts")
                    for dep in node.hard_deps
                )
                if not has_contract_dep:
                    errors.append(
                        f"Contracts-first violation: {node.stage_name} must depend on types_contracts"
                    )

        return errors
