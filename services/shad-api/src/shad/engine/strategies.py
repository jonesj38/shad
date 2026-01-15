"""Strategy skeletons for task-aware decomposition.

Per SPEC.md Section 2.2:
- Each strategy defines required stages, optional stages, constraints
- Strategy selection is hybrid (heuristic + LLM override)
- Skeletons enforce invariants (contracts-first, imports-must-resolve, etc.)

Strategies:
- software: Build software with contracts-first, type consistency
- research: Gather and cite sources from vault
- analysis: Analyze data with explicit criteria and tradeoffs
- planning: Create plans with milestones and dependencies
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrategyType(str, Enum):
    """Types of decomposition strategies."""

    SOFTWARE = "software"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"


@dataclass
class StrategyStage:
    """A stage in a strategy skeleton.

    Stages are the building blocks of a decomposition strategy.
    Required stages must exist in the DAG; optional stages may be added.
    """

    name: str
    description: str
    required: bool = True
    depends_on: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class StrategyConstraint:
    """A constraint that must be satisfied by the decomposition.

    Constraints enforce invariants like contracts-first ordering,
    import resolution requirements, etc.
    """

    name: str
    description: str
    value: Any = True


@dataclass
class StrategySkeleton:
    """A skeleton defining the structure of a decomposition strategy.

    Per SPEC.md Section 2.2.1:
    - required_stages: Must exist in DAG
    - optional_stages: Added when relevant
    - constraints: Invariants the LLM must respect
    """

    strategy_type: StrategyType
    required_stages: list[StrategyStage]
    optional_stages: list[StrategyStage]
    constraints: list[StrategyConstraint]
    default_dependencies: list[tuple[str, str]] = field(default_factory=list)

    def get_stage(self, name: str) -> StrategyStage | None:
        """Get a stage by name."""
        for stage in self.required_stages + self.optional_stages:
            if stage.name == name:
                return stage
        return None

    @property
    def all_stages(self) -> list[StrategyStage]:
        """Get all stages (required + optional)."""
        return self.required_stages + self.optional_stages

    def validate_dag(self, stage_names: list[str]) -> tuple[bool, list[str]]:
        """Validate that a DAG contains all required stages.

        Returns:
            Tuple of (is_valid, missing_stages)
        """
        required_names = {s.name for s in self.required_stages}
        present_names = set(stage_names)
        missing = required_names - present_names
        return len(missing) == 0, list(missing)

    def get_constraint(self, name: str) -> StrategyConstraint | None:
        """Get a constraint by name."""
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None


@dataclass
class StrategySelectionResult:
    """Result of strategy selection.

    Per SPEC.md Section 2.2.3:
    - strategy_type: Selected strategy
    - confidence: Confidence in selection (0-1)
    - is_override: Whether user overrode selection
    - matched_keywords: Keywords that influenced selection
    """

    strategy_type: StrategyType
    confidence: float
    is_override: bool = False
    matched_keywords: list[str] = field(default_factory=list)


class Strategy(ABC):
    """Base class for decomposition strategies."""

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        ...

    @property
    @abstractmethod
    def skeleton(self) -> StrategySkeleton:
        """Get the strategy skeleton."""
        ...

    @property
    def default_dependencies(self) -> list[tuple[str, str]]:
        """Get default dependencies between stages.

        Returns list of (from_stage, to_stage) tuples indicating
        that from_stage must complete before to_stage.
        """
        return self.skeleton.default_dependencies

    def get_decomposition_hints(self, task: str) -> str:
        """Get hints for LLM decomposition based on task.

        These hints help the LLM understand how to decompose
        the task within the strategy's constraints.
        """
        return self.get_hint_pack()

    @abstractmethod
    def get_hint_pack(self) -> str:
        """Get the hint pack (system prompt additions) for this strategy.

        The hint pack provides additional instructions to the LLM
        for decomposing tasks using this strategy.
        """
        ...


class SoftwareStrategy(Strategy):
    """Software development strategy.

    Per SPEC.md Section 2.2.1:
    - Required: clarify_requirements, project_layout, types_contracts,
                implementation, verification, synthesis
    - Optional: db_schema, auth, openapi, migrations, docs
    - Constraints: contracts_first, imports_must_resolve, no_implicit_writes
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.SOFTWARE

    @property
    def skeleton(self) -> StrategySkeleton:
        return StrategySkeleton(
            strategy_type=StrategyType.SOFTWARE,
            required_stages=[
                StrategyStage(
                    name="clarify_requirements",
                    description="Clarify and document requirements",
                    required=True,
                    tags=["planning"],
                ),
                StrategyStage(
                    name="project_layout",
                    description="Define project structure and file layout",
                    required=True,
                    depends_on=["clarify_requirements"],
                    tags=["planning"],
                ),
                StrategyStage(
                    name="types_contracts",
                    description="Define types, interfaces, and contracts",
                    required=True,
                    depends_on=["project_layout"],
                    tags=["contracts", "types"],
                ),
                StrategyStage(
                    name="implementation",
                    description="Implement the solution based on contracts",
                    required=True,
                    depends_on=["types_contracts"],
                    tags=["code"],
                ),
                StrategyStage(
                    name="verification",
                    description="Verify implementation against requirements",
                    required=True,
                    depends_on=["implementation"],
                    tags=["testing"],
                ),
                StrategyStage(
                    name="synthesis",
                    description="Synthesize final output and documentation",
                    required=True,
                    depends_on=["verification"],
                    tags=["output"],
                ),
            ],
            optional_stages=[
                StrategyStage(
                    name="db_schema",
                    description="Define database schema",
                    required=False,
                    depends_on=["clarify_requirements"],
                    tags=["database"],
                ),
                StrategyStage(
                    name="auth",
                    description="Implement authentication/authorization",
                    required=False,
                    depends_on=["types_contracts"],
                    tags=["security"],
                ),
                StrategyStage(
                    name="openapi",
                    description="Define OpenAPI specification",
                    required=False,
                    depends_on=["types_contracts"],
                    tags=["api"],
                ),
                StrategyStage(
                    name="migrations",
                    description="Create database migrations",
                    required=False,
                    depends_on=["db_schema"],
                    tags=["database"],
                ),
                StrategyStage(
                    name="docs",
                    description="Generate documentation",
                    required=False,
                    depends_on=["implementation"],
                    tags=["documentation"],
                ),
            ],
            constraints=[
                StrategyConstraint(
                    name="contracts_first",
                    description="Types & contracts must be defined before implementation",
                    value=True,
                ),
                StrategyConstraint(
                    name="imports_must_resolve",
                    description="All imports must resolve to existing files/symbols",
                    value=True,
                ),
                StrategyConstraint(
                    name="no_implicit_writes",
                    description="No file writes without explicit approval",
                    value=True,
                ),
            ],
            default_dependencies=[
                ("clarify_requirements", "project_layout"),
                ("project_layout", "types_contracts"),
                ("types_contracts", "implementation"),
                ("implementation", "verification"),
                ("verification", "synthesis"),
            ],
        )

    def get_hint_pack(self) -> str:
        """Get hint pack for software strategy."""
        return """You are decomposing a software development task.

IMPORTANT CONSTRAINTS:
1. CONTRACTS FIRST: Define types and contracts before implementation
   - Create type definitions, interfaces, and contracts early
   - Implementation nodes must import from contract definitions
   - Never define types inline in implementation code

2. IMPORTS MUST RESOLVE: All imports must resolve to existing files
   - Build an export index in the types_contracts stage
   - Implementation imports from the export index
   - Validate all imports before completion

3. NO IMPLICIT WRITES: Do not write files without explicit approval
   - Output is a manifest of files to be written
   - Actual file writes require --write-files flag

DECOMPOSITION STRUCTURE:
1. Clarify requirements - Extract and document requirements
2. Project layout - Define file structure
3. Types & contracts - Define types, interfaces (CRITICAL: before implementation)
4. Implementation - Implement based on contracts
5. Verification - Check syntax, types, tests
6. Synthesis - Assemble final output

Add optional stages as needed:
- db_schema: If task involves database
- auth: If task involves authentication
- openapi: If task involves REST API
- migrations: If task involves database changes
- docs: If documentation is requested"""


class ResearchStrategy(Strategy):
    """Research strategy for gathering and citing sources.

    Per SPEC.md Section 2.2.1:
    - Required: clarify_scope, gather_sources, synthesize, cite
    - Optional: compare_perspectives, identify_gaps
    - Constraints: must_cite_vault, max_claims_per_source
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.RESEARCH

    @property
    def skeleton(self) -> StrategySkeleton:
        return StrategySkeleton(
            strategy_type=StrategyType.RESEARCH,
            required_stages=[
                StrategyStage(
                    name="clarify_scope",
                    description="Clarify research scope and questions",
                    required=True,
                    tags=["planning"],
                ),
                StrategyStage(
                    name="gather_sources",
                    description="Gather sources from vault",
                    required=True,
                    depends_on=["clarify_scope"],
                    tags=["retrieval"],
                ),
                StrategyStage(
                    name="synthesize",
                    description="Synthesize findings from sources",
                    required=True,
                    depends_on=["gather_sources"],
                    tags=["synthesis"],
                ),
                StrategyStage(
                    name="cite",
                    description="Add citations to sources",
                    required=True,
                    depends_on=["synthesize"],
                    tags=["citations"],
                ),
            ],
            optional_stages=[
                StrategyStage(
                    name="compare_perspectives",
                    description="Compare different perspectives from sources",
                    required=False,
                    depends_on=["gather_sources"],
                    tags=["analysis"],
                ),
                StrategyStage(
                    name="identify_gaps",
                    description="Identify gaps in available sources",
                    required=False,
                    depends_on=["gather_sources"],
                    tags=["analysis"],
                ),
            ],
            constraints=[
                StrategyConstraint(
                    name="must_cite_vault",
                    description="All claims must be supported by vault sources",
                    value=True,
                ),
                StrategyConstraint(
                    name="max_claims_per_source",
                    description="Maximum claims to make from a single source",
                    value=5,
                ),
            ],
            default_dependencies=[
                ("clarify_scope", "gather_sources"),
                ("gather_sources", "synthesize"),
                ("synthesize", "cite"),
            ],
        )

    def get_hint_pack(self) -> str:
        """Get hint pack for research strategy."""
        return """You are decomposing a research task.

IMPORTANT CONSTRAINTS:
1. MUST CITE VAULT: All claims must be supported by vault sources
   - Use [[wikilinks]] for citations
   - Do not make unsupported claims
   - Indicate confidence level for claims

2. MAX CLAIMS PER SOURCE: Limit claims from any single source to 5
   - Diversify sources
   - Cross-reference findings

DECOMPOSITION STRUCTURE:
1. Clarify scope - Define research questions
2. Gather sources - Search vault for relevant notes
3. Synthesize - Combine findings into coherent answer
4. Cite - Add proper citations

Add optional stages as needed:
- compare_perspectives: When sources have different viewpoints
- identify_gaps: When research may be incomplete"""


class AnalysisStrategy(Strategy):
    """Analysis strategy for data analysis tasks.

    - Required: define_criteria, gather_data, analyze, conclude
    - Optional: compare_alternatives, visualize
    - Constraints: explicit_criteria, document_tradeoffs
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.ANALYSIS

    @property
    def skeleton(self) -> StrategySkeleton:
        return StrategySkeleton(
            strategy_type=StrategyType.ANALYSIS,
            required_stages=[
                StrategyStage(
                    name="define_criteria",
                    description="Define analysis criteria and metrics",
                    required=True,
                    tags=["planning"],
                ),
                StrategyStage(
                    name="gather_data",
                    description="Gather data for analysis",
                    required=True,
                    depends_on=["define_criteria"],
                    tags=["retrieval"],
                ),
                StrategyStage(
                    name="analyze",
                    description="Perform analysis against criteria",
                    required=True,
                    depends_on=["gather_data"],
                    tags=["analysis"],
                ),
                StrategyStage(
                    name="conclude",
                    description="Draw conclusions from analysis",
                    required=True,
                    depends_on=["analyze"],
                    tags=["synthesis"],
                ),
            ],
            optional_stages=[
                StrategyStage(
                    name="compare_alternatives",
                    description="Compare alternative approaches",
                    required=False,
                    depends_on=["analyze"],
                    tags=["analysis"],
                ),
                StrategyStage(
                    name="visualize",
                    description="Create visualizations of findings",
                    required=False,
                    depends_on=["analyze"],
                    tags=["output"],
                ),
            ],
            constraints=[
                StrategyConstraint(
                    name="explicit_criteria",
                    description="Analysis criteria must be explicit",
                    value=True,
                ),
                StrategyConstraint(
                    name="document_tradeoffs",
                    description="Tradeoffs must be documented",
                    value=True,
                ),
            ],
            default_dependencies=[
                ("define_criteria", "gather_data"),
                ("gather_data", "analyze"),
                ("analyze", "conclude"),
            ],
        )

    def get_hint_pack(self) -> str:
        """Get hint pack for analysis strategy."""
        return """You are decomposing an analysis task.

IMPORTANT CONSTRAINTS:
1. EXPLICIT CRITERIA: Analysis criteria must be clearly defined
   - Define metrics and evaluation criteria upfront
   - Document how criteria will be measured

2. DOCUMENT TRADEOFFS: All tradeoffs must be explicitly documented
   - Identify pros/cons of each option
   - Make tradeoffs transparent

DECOMPOSITION STRUCTURE:
1. Define criteria - Establish analysis metrics
2. Gather data - Collect relevant data
3. Analyze - Evaluate against criteria
4. Conclude - Draw supported conclusions

Add optional stages as needed:
- compare_alternatives: When multiple options exist
- visualize: When visual representation helps"""


class PlanningStrategy(Strategy):
    """Planning strategy for creating project plans.

    - Required: define_goals, identify_milestones, sequence_tasks, document_plan
    - Optional: identify_risks, allocate_resources
    - Constraints: clear_dependencies, measurable_milestones
    """

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PLANNING

    @property
    def skeleton(self) -> StrategySkeleton:
        return StrategySkeleton(
            strategy_type=StrategyType.PLANNING,
            required_stages=[
                StrategyStage(
                    name="define_goals",
                    description="Define project goals and success criteria",
                    required=True,
                    tags=["planning"],
                ),
                StrategyStage(
                    name="identify_milestones",
                    description="Identify key milestones",
                    required=True,
                    depends_on=["define_goals"],
                    tags=["planning"],
                ),
                StrategyStage(
                    name="sequence_tasks",
                    description="Sequence tasks with dependencies",
                    required=True,
                    depends_on=["identify_milestones"],
                    tags=["planning"],
                ),
                StrategyStage(
                    name="document_plan",
                    description="Document the complete plan",
                    required=True,
                    depends_on=["sequence_tasks"],
                    tags=["output"],
                ),
            ],
            optional_stages=[
                StrategyStage(
                    name="identify_risks",
                    description="Identify and document risks",
                    required=False,
                    depends_on=["identify_milestones"],
                    tags=["planning"],
                ),
                StrategyStage(
                    name="allocate_resources",
                    description="Allocate resources to tasks",
                    required=False,
                    depends_on=["sequence_tasks"],
                    tags=["planning"],
                ),
            ],
            constraints=[
                StrategyConstraint(
                    name="clear_dependencies",
                    description="Task dependencies must be clear",
                    value=True,
                ),
                StrategyConstraint(
                    name="measurable_milestones",
                    description="Milestones must be measurable",
                    value=True,
                ),
            ],
            default_dependencies=[
                ("define_goals", "identify_milestones"),
                ("identify_milestones", "sequence_tasks"),
                ("sequence_tasks", "document_plan"),
            ],
        )

    def get_hint_pack(self) -> str:
        """Get hint pack for planning strategy."""
        return """You are decomposing a planning task.

IMPORTANT CONSTRAINTS:
1. CLEAR DEPENDENCIES: Task dependencies must be explicit
   - Show which tasks depend on others
   - No circular dependencies

2. MEASURABLE MILESTONES: All milestones must be measurable
   - Define success criteria for each milestone
   - Make progress trackable

DECOMPOSITION STRUCTURE:
1. Define goals - Establish project goals
2. Identify milestones - Set key milestones
3. Sequence tasks - Order tasks with dependencies
4. Document plan - Create final plan document

Add optional stages as needed:
- identify_risks: When risks need documentation
- allocate_resources: When resource planning is needed"""


# Strategy keyword patterns for heuristic selection
STRATEGY_KEYWORDS: dict[StrategyType, list[str]] = {
    StrategyType.SOFTWARE: [
        "build", "create", "implement", "develop", "code", "program",
        "api", "app", "application", "service", "system", "backend",
        "frontend", "database", "web", "mobile", "function", "class",
        "module", "package", "library", "framework", "deploy", "release",
        "feature", "component", "integration", "authentication", "auth",
    ],
    StrategyType.RESEARCH: [
        "research", "investigate", "study", "explore", "find", "discover",
        "learn", "understand", "survey", "review", "literature", "sources",
        "evidence", "data", "information", "facts", "history", "background",
        "state of the art", "best practices", "compare", "contrast",
    ],
    StrategyType.ANALYSIS: [
        "analyze", "analyse", "evaluate", "assess", "measure", "compare",
        "benchmark", "profile", "performance", "metrics", "statistics",
        "data analysis", "examine", "inspect", "audit", "review", "report",
        "tradeoffs", "pros and cons", "advantages", "disadvantages",
    ],
    StrategyType.PLANNING: [
        "plan", "schedule", "roadmap", "timeline", "milestone", "phase",
        "project", "strategy", "organize", "coordinate", "prioritize",
        "sequence", "budget", "resource", "allocate", "deadline",
        "kickoff", "launch", "rollout",
    ],
}


class StrategySelector:
    """Heuristic strategy selector.

    Per SPEC.md Section 2.2.3:
    1. Heuristic classifier (no LLM call): Pattern-match keywords
    2. confidence >= 0.7 → proceed with guess
    3. confidence < 0.7 → default to analysis
    4. User override always wins
    """

    def __init__(self) -> None:
        self.keywords = STRATEGY_KEYWORDS

    def select(
        self,
        task: str,
        override: StrategyType | None = None,
    ) -> StrategySelectionResult:
        """Select a strategy for a task.

        Args:
            task: The task description
            override: Optional user override

        Returns:
            StrategySelectionResult with type, confidence, and metadata
        """
        # User override always wins
        if override is not None:
            return StrategySelectionResult(
                strategy_type=override,
                confidence=1.0,
                is_override=True,
                matched_keywords=[],
            )

        # Normalize task for matching
        task_lower = task.lower()
        task_words = set(re.findall(r'\b\w+\b', task_lower))

        # Score each strategy
        scores: dict[StrategyType, tuple[float, list[str]]] = {}

        for strategy_type, keywords in self.keywords.items():
            matched = []
            for keyword in keywords:
                # Check for keyword or keyword phrase in task
                if keyword.lower() in task_lower:
                    matched.append(keyword)
                # Check for word overlap
                keyword_words = set(keyword.lower().split())
                if keyword_words & task_words:
                    if keyword not in matched:
                        matched.append(keyword)

            # Calculate confidence based on matched keywords
            if matched:
                # More matches = higher confidence, capped at 1.0
                confidence = min(1.0, len(matched) * 0.15 + 0.3)
                scores[strategy_type] = (confidence, matched)

        # Find best match
        if scores:
            best_type = max(scores.keys(), key=lambda t: scores[t][0])
            best_confidence, best_keywords = scores[best_type]

            # Per SPEC.md: confidence < 0.7 → default to analysis
            if best_confidence < 0.7:
                return StrategySelectionResult(
                    strategy_type=StrategyType.ANALYSIS,
                    confidence=best_confidence,
                    is_override=False,
                    matched_keywords=best_keywords,
                )

            return StrategySelectionResult(
                strategy_type=best_type,
                confidence=best_confidence,
                is_override=False,
                matched_keywords=best_keywords,
            )

        # Default to analysis when no matches
        return StrategySelectionResult(
            strategy_type=StrategyType.ANALYSIS,
            confidence=0.3,
            is_override=False,
            matched_keywords=[],
        )


# Strategy registry
_STRATEGIES: dict[StrategyType, Strategy] = {
    StrategyType.SOFTWARE: SoftwareStrategy(),
    StrategyType.RESEARCH: ResearchStrategy(),
    StrategyType.ANALYSIS: AnalysisStrategy(),
    StrategyType.PLANNING: PlanningStrategy(),
}


def get_strategy(strategy_type: StrategyType) -> Strategy:
    """Get a strategy by type."""
    return _STRATEGIES[strategy_type]


def get_all_strategies() -> list[Strategy]:
    """Get all available strategies."""
    return list(_STRATEGIES.values())
