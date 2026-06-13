"""Tests for discipline source-map generation."""

from __future__ import annotations

from pathlib import Path

from shad.discipline.source_map import SourceMapGenerator


def test_source_map_detects_formal_docs_tests_and_deploy_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n")
    (tmp_path / "spec").mkdir()
    (tmp_path / "spec" / "Protocol.tla").write_text("---- MODULE Protocol ----\n")
    (tmp_path / "proofs").mkdir()
    (tmp_path / "proofs" / "Invariant.lean").write_text("theorem x : True := by trivial\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "protocol_test.py").write_text("def test_x(): pass\n")
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / "agent.service").write_text("[Service]\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")

    source_map = SourceMapGenerator(max_files_per_section=20).generate([tmp_path])
    rendered = source_map.to_markdown()

    assert source_map.total_files == 6
    assert source_map.extension_counts[".md"] == 1
    assert source_map.extension_counts[".tla"] == 1
    assert source_map.extension_counts[".lean"] == 1
    assert "README.md" in source_map.important_files
    assert "spec/Protocol.tla" in source_map.formal_files
    assert "proofs/Invariant.lean" in source_map.formal_files
    assert "tests/protocol_test.py" in source_map.test_files
    assert "deploy/agent.service" in source_map.deploy_files
    assert "## Formal / Spec Assets" in rendered
    assert "## Deploy / Runtime Files" in rendered
