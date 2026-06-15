"""Deterministic staged discipline build planning.

This module intentionally does no LLM work.  It captures the discipline build
shape learned from large real corpora: generate a source map first, then run
focused stage analyses that can be scheduled in parallel, then verify and
synthesize.  The generated package is suitable for review, committing, or for a
separate workflow/runner to execute.
"""

from __future__ import annotations

import fnmatch
import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shad.discipline.source_map import DEFAULT_EXCLUDES, SourceMapGenerator

DEFAULT_STAGE_ORDER = [
    "architecture",
    "runtime",
    "storage-data",
    "protocols-security",
    "formal-methods",
    "developer-workflows",
    "pitfalls",
    "routing-hints",
    "verification",
    "final-synthesis",
]

PARALLEL_STAGE_GROUPS = [
    ["architecture", "runtime", "storage-data"],
    ["protocols-security", "formal-methods", "developer-workflows"],
    ["pitfalls", "routing-hints"],
    ["verification"],
    ["final-synthesis"],
]

STAGE_PROMPTS: dict[str, str] = {
    "architecture": """Produce architecture.md for this discipline. Focus on source-backed modules, boundaries, data/control flow, extension points, and key invariants. Prefer code/proofs/tests/config over prose docs. Include Mermaid diagrams when they clarify structure. Cite concrete paths.""",
    "runtime": """Produce runtime-concepts.md. Explain how the system behaves when running: processes, services, CLIs/APIs, lifecycle, queues, background workers, state transitions, and operational failure modes. Cite concrete paths.""",
    "storage-data": """Produce storage-data-model.md. Map durable state, schemas, database/filesystem layout, caches, migrations, indexes, serialization formats, and cleanup rules. Cite concrete paths.""",
    "protocols-security": """Produce protocols-security.md. Explain protocols, authn/authz, signing, trust boundaries, network/API contracts, sensitive operations, and known security pitfalls. Cite concrete paths.""",
    "formal-methods": """Produce formal-methods.md. Identify specs, tests, properties, invariants, model-checking assets, verification assumptions, and gaps. If no formal assets exist, distinguish explicit tests from inferred invariants. Cite concrete paths.""",
    "developer-workflows": """Produce developer-workflows.md. Document local setup, build/test/lint commands, release/deploy workflows, debugging procedures, and safe modification patterns. Cite concrete paths.""",
    "pitfalls": """Produce pitfalls-checklists.md. Extract common failure modes, misleading docs, sharp edges, regression checks, and operational checklists. Cite concrete paths and include concrete symptoms/remedies.""",
    "routing-hints": """Produce routing-hints.md. Define when an agent should use this discipline, when to avoid it, which artifacts to consult for common task types, and source-authority rules. Cite artifacts/source paths.""",
    "verification": """Produce verification-report.md. Check every staged artifact for source-backed claims, missing major subsystems, docs-over-code drift, citation quality, contradictions, and required follow-up. Use the source map and all stage outputs.""",
    "final-synthesis": """Produce discipline-report.md. Synthesize the verified staged artifacts into a durable expert context pack with source-authority notes, diagrams, artifact index, routing guidance, open questions, and provenance summary.""",
}


@dataclass(frozen=True)
class DisciplineStage:
    """A deterministic discipline build stage."""

    id: str
    output: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    parallel_group: int = 0


@dataclass(frozen=True)
class DisciplinePlan:
    """Serializable staged discipline build plan."""

    name: str
    package_dir: str
    source_roots: list[str]
    excludes: list[str]
    stages: list[DisciplineStage]
    parallel_groups: list[list[str]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name} Discipline Build Plan",
            "",
            "## Source Roots",
        ]
        for root in self.source_roots:
            lines.append(f"- `{root}`")
        lines.extend(["", "## Excludes"])
        if self.excludes:
            for pattern in self.excludes:
                lines.append(f"- `{pattern}`")
        else:
            lines.append("- None beyond Shad defaults")
        lines.extend(["", "## Parallel Stage Groups"])
        for idx, group in enumerate(self.parallel_groups, start=1):
            lines.append(f"{idx}. " + ", ".join(f"`{stage}`" for stage in group))
        lines.extend(["", "## Stages"])
        for stage in self.stages:
            deps = ", ".join(stage.depends_on) if stage.depends_on else "source-map"
            lines.extend([
                f"### {stage.id}",
                f"- Output: `{stage.output}`",
                f"- Depends on: {deps}",
                f"- Parallel group: {stage.parallel_group}",
                "",
                stage.prompt,
                "",
            ])
        return "\n".join(lines).rstrip() + "\n"


class DisciplinePackageScaffolder:
    """Create a deterministic staged discipline package on disk."""

    def __init__(self, *, max_files_per_section: int = 120) -> None:
        self.max_files_per_section = max_files_per_section

    def build_plan(
        self,
        *,
        name: str,
        package_dir: Path | str,
        source_roots: list[Path | str],
        excludes: list[str] | None = None,
    ) -> DisciplinePlan:
        roots = [str(Path(root).expanduser().resolve()) for root in source_roots]
        exclude_list = list(dict.fromkeys(excludes or []))
        stages: list[DisciplineStage] = []
        dependency_outputs: dict[str, list[str]] = {}
        for group_idx, group in enumerate(PARALLEL_STAGE_GROUPS, start=1):
            for stage_id in group:
                if stage_id == "verification":
                    deps = [stage for stage in DEFAULT_STAGE_ORDER if stage not in {"verification", "final-synthesis"}]
                    output = "03-Analysis/verification-report.md"
                elif stage_id == "final-synthesis":
                    deps = ["verification"]
                    output = "00-Final-Reports/discipline-report.md"
                else:
                    deps = dependency_outputs.get(stage_id, [])
                    output = f"03-Analysis/{self._output_name(stage_id)}.md"
                stages.append(
                    DisciplineStage(
                        id=stage_id,
                        output=output,
                        prompt=STAGE_PROMPTS[stage_id],
                        depends_on=deps,
                        parallel_group=group_idx,
                    )
                )
        return DisciplinePlan(
            name=name,
            package_dir=str(Path(package_dir).expanduser().resolve()),
            source_roots=roots,
            excludes=exclude_list,
            stages=stages,
            parallel_groups=PARALLEL_STAGE_GROUPS,
        )

    def scaffold(
        self,
        *,
        name: str,
        package_dir: Path | str,
        source_roots: list[Path | str],
        excludes: list[str] | None = None,
        overwrite: bool = False,
    ) -> DisciplinePlan:
        package_path = Path(package_dir).expanduser().resolve()
        roots = [Path(root).expanduser().resolve() for root in source_roots]
        exclude_list = list(dict.fromkeys(excludes or []))
        if package_path.exists() and not overwrite and any(package_path.iterdir()):
            raise FileExistsError(f"Package directory is not empty: {package_path}")

        for rel in [
            "00-Final-Reports",
            "01-Methodology",
            "02-Source-Map",
            "03-Analysis",
            "04-Data",
            "05-Scripts/stage-prompts",
            "06-Visualizations",
            "07-Out-Reports",
            "Sources",
        ]:
            (package_path / rel).mkdir(parents=True, exist_ok=True)

        plan = self.build_plan(name=name, package_dir=package_path, source_roots=roots, excludes=exclude_list)
        (package_path / "04-Data" / "discipline-plan.json").write_text(plan.to_json(), encoding="utf-8")
        (package_path / "01-Methodology" / "discipline-plan.md").write_text(plan.to_markdown(), encoding="utf-8")

        source_map = SourceMapGenerator(
            excludes=DEFAULT_EXCLUDES | set(exclude_list),
            max_files_per_section=self.max_files_per_section,
        ).generate([root for root in roots if self._is_included(root, exclude_list)])
        (package_path / "02-Source-Map" / "source-map.md").write_text(source_map.to_markdown(), encoding="utf-8")
        (package_path / "04-Data" / "source-roots.json").write_text(
            json.dumps({"sources": plan.source_roots, "excludes": exclude_list}, indent=2) + "\n",
            encoding="utf-8",
        )

        for stage in plan.stages:
            prompt_path = package_path / "05-Scripts" / "stage-prompts" / f"{stage.id}.md"
            prompt_path.write_text(self._render_stage_prompt(plan, stage), encoding="utf-8")

        (package_path / "05-Scripts" / "run-discipline-stage.sh").write_text(
            self._render_stage_runner(plan), encoding="utf-8"
        )
        (package_path / "README.md").write_text(self._render_readme(plan), encoding="utf-8")
        return plan

    @staticmethod
    def _output_name(stage_id: str) -> str:
        if stage_id == "runtime":
            return "runtime-concepts"
        if stage_id == "storage-data":
            return "storage-data-model"
        if stage_id == "pitfalls":
            return "pitfalls-checklists"
        return stage_id

    @staticmethod
    def _is_included(path: Path, excludes: list[str]) -> bool:
        text = str(path)
        return not any(fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in excludes)

    @staticmethod
    def _render_stage_prompt(plan: DisciplinePlan, stage: DisciplineStage) -> str:
        return (
            f"# Stage: {stage.id}\n\n"
            f"Output path: `{stage.output}`\n\n"
            "Use `02-Source-Map/source-map.md` and the configured source roots as source authority. "
            "Prefer code, tests, specs, config, and runtime files over stale planning prose. "
            "Every substantial claim should name concrete source paths.\n\n"
            f"{stage.prompt}\n"
        )

    @staticmethod
    def _render_stage_runner(plan: DisciplinePlan) -> str:
        quoted_sources = " ".join(f"--sources {shlex.quote(root)}" for root in plan.source_roots)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "STAGE=${1:?stage id required}",
            "ROOT=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)",
            "PROMPT=\"$ROOT/05-Scripts/stage-prompts/$STAGE.md\"",
            "OUT=$(python3 - <<'PY' \"$ROOT/04-Data/discipline-plan.json\" \"$STAGE\"",
            "import json, sys",
            "plan=json.load(open(sys.argv[1]))",
            "stage=next(s for s in plan['stages'] if s['id']==sys.argv[2])",
            "print(stage['output'])",
            "PY",
            ")",
            "mkdir -p \"$ROOT/$(dirname \"$OUT\")\" \"$ROOT/07-Out-Reports/logs\"",
            "shad run \"$(cat \"$PROMPT\")\" --strategy discipline-report --profile deep "
            f"{quoted_sources} > \"$ROOT/$OUT\" 2> \"$ROOT/07-Out-Reports/logs/$STAGE.log\"",
            "echo \"wrote $ROOT/$OUT\"",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_readme(plan: DisciplinePlan) -> str:
        return (
            f"# {plan.name} Discipline Package\n\n"
            "Generated by `shad discipline scaffold`. This is a deterministic package scaffold; "
            "it has not run expensive RLM stages yet.\n\n"
            "Start with `01-Methodology/discipline-plan.md`, inspect `02-Source-Map/source-map.md`, "
            "then run focused stages from `05-Scripts/stage-prompts/` as needed.\n"
        )
