"""Tests for deterministic staged discipline package planning."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from shad.cli.main import cli
from shad.discipline.planner import DisciplinePackageScaffolder


def test_build_plan_has_expected_parallel_stages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    plan = DisciplinePackageScaffolder().build_plan(
        name="Example",
        package_dir=tmp_path / "pkg",
        source_roots=[source],
        excludes=["/private/docs"],
    )

    stage_ids = [stage.id for stage in plan.stages]
    assert stage_ids == [
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
    assert plan.parallel_groups[0] == ["architecture", "runtime", "storage-data"]
    assert plan.stages[-2].depends_on == stage_ids[:-2]
    assert plan.stages[-1].depends_on == ["verification"]
    assert plan.excludes == ["/private/docs"]


def test_scaffold_writes_package_layout_source_map_and_prompts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "service.py").write_text("print('hello')\n", encoding="utf-8")
    excluded = tmp_path / "edwin-docs-private"
    excluded.mkdir()
    (excluded / "secret.md").write_text("private docs\n", encoding="utf-8")
    package = tmp_path / "package"

    plan = DisciplinePackageScaffolder().scaffold(
        name="Example",
        package_dir=package,
        source_roots=[source, excluded],
        excludes=[str(excluded)],
    )

    assert (package / "00-Final-Reports").is_dir()
    assert (package / "02-Source-Map" / "source-map.md").is_file()
    assert (package / "04-Data" / "discipline-plan.json").is_file()
    assert (package / "05-Scripts" / "stage-prompts" / "architecture.md").is_file()
    assert (package / "05-Scripts" / "run-discipline-stage.sh").is_file()
    assert "edwin-docs-private" not in (package / "02-Source-Map" / "source-map.md").read_text()

    saved = json.loads((package / "04-Data" / "discipline-plan.json").read_text())
    assert saved["name"] == plan.name
    assert saved["stages"][0]["id"] == "architecture"


def test_cli_scaffold_generates_package(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Source\n", encoding="utf-8")
    package = tmp_path / "package"

    result = CliRunner().invoke(
        cli,
        [
            "discipline",
            "scaffold",
            "Example",
            str(package),
            "--source",
            str(source),
            "--exclude",
            "edwin-docs-private",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Scaffolded discipline package" in result.output
    assert (package / "01-Methodology" / "discipline-plan.md").is_file()
    assert (package / "04-Data" / "source-roots.json").is_file()
