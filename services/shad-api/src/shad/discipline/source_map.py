"""Deterministic source-map generation for discipline builds.

The source map is intentionally cheap and model-free. It gives a discipline
RLM run a stable overview of the corpus before the model starts retrieval and
section synthesis, reducing exploratory LLM calls while improving coverage.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    ".next",
    ".turbo",
}

IMPORTANT_NAMES = {
    "readme",
    "license",
    "changelog",
    "contributing",
    "makefile",
    "dockerfile",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "pyproject.toml",
    "poetry.lock",
    "requirements.txt",
    "cargo.toml",
    "cargo.lock",
    "go.mod",
    "go.sum",
    "mix.exs",
    "pubspec.yaml",
    "build.gradle",
    "settings.gradle",
    "gradle.properties",
    "compose.yaml",
    "docker-compose.yml",
}

FORMAL_EXTENSIONS = {".tla", ".cfg", ".lean", ".als", ".ivy"}
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".adoc"}
TEST_MARKERS = {"test", "tests", "spec", "specs", "fuzz", "fuzzer", "property"}
DEPLOY_MARKERS = {"deploy", "deployment", "docker", "compose", "service", "plist", "policy", "systemd", "k8s", "kubernetes", "helm"}


@dataclass
class FileEntry:
    """A single source-map file entry."""

    path: str
    size: int
    extension: str


@dataclass
class SourceMap:
    """Deterministic summary of a source corpus."""

    roots: list[str]
    revisions: dict[str, str | None]
    total_files: int
    total_bytes: int
    extension_counts: dict[str, int]
    top_directories: dict[str, int]
    important_files: list[str] = field(default_factory=list)
    formal_files: list[str] = field(default_factory=list)
    docs_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    deploy_files: list[str] = field(default_factory=list)
    sample_files: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render as stable Markdown for direct RLM context or artifact output."""
        lines = [
            "# Source Map",
            "",
            "## Roots",
        ]
        for root in self.roots:
            revision = self.revisions.get(root)
            suffix = f" @ `{revision}`" if revision else ""
            lines.append(f"- `{root}`{suffix}")

        lines.extend([
            "",
            "## Corpus Summary",
            f"- Files: {self.total_files}",
            f"- Bytes: {self.total_bytes}",
            "",
            "## Extension Counts",
        ])
        for ext, count in self.extension_counts.items():
            lines.append(f"- `{ext}`: {count}")

        lines.extend(["", "## Top Directories"])
        for directory, count in self.top_directories.items():
            lines.append(f"- `{directory}`: {count}")

        self._append_file_section(lines, "Important Files", self.important_files)
        self._append_file_section(lines, "Formal / Spec Assets", self.formal_files)
        self._append_file_section(lines, "Documentation Files", self.docs_files)
        self._append_file_section(lines, "Test / Fuzz Files", self.test_files)
        self._append_file_section(lines, "Deploy / Runtime Files", self.deploy_files)
        self._append_file_section(lines, "Sample Files", self.sample_files)
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _append_file_section(lines: list[str], title: str, paths: list[str]) -> None:
        lines.extend(["", f"## {title}"])
        if not paths:
            lines.append("- None detected")
            return
        for path in paths:
            lines.append(f"- `{path}`")


class SourceMapGenerator:
    """Generate deterministic source maps from one or more filesystem roots."""

    def __init__(self, excludes: set[str] | None = None, max_files_per_section: int = 80) -> None:
        self.excludes = excludes or DEFAULT_EXCLUDES
        self.max_files_per_section = max_files_per_section

    def generate(self, roots: list[Path | str]) -> SourceMap:
        normalized_roots = [Path(root).expanduser().resolve() for root in roots]
        files: list[FileEntry] = []
        extension_counts: Counter[str] = Counter()
        top_directories: Counter[str] = Counter()
        important: list[str] = []
        formal: list[str] = []
        docs: list[str] = []
        tests: list[str] = []
        deploy: list[str] = []
        revisions: dict[str, str | None] = {}

        for root in normalized_roots:
            revisions[str(root)] = self._git_revision(root)
            if not root.exists():
                continue
            for path in self._iter_files(root):
                rel = self._relative_label(path, normalized_roots)
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                ext = path.suffix.lower() or "[none]"
                files.append(FileEntry(path=rel, size=size, extension=ext))
                extension_counts[ext] += 1
                top_directories[self._top_directory(rel)] += 1

                lower_parts = {part.lower() for part in path.parts}
                lower_name = path.name.lower()
                lower_rel = rel.lower()
                stem = path.stem.lower()

                if lower_name in IMPORTANT_NAMES or stem in IMPORTANT_NAMES:
                    important.append(rel)
                if ext in FORMAL_EXTENSIONS:
                    formal.append(rel)
                if ext in DOC_EXTENSIONS:
                    docs.append(rel)
                if lower_parts & TEST_MARKERS or any(marker in lower_rel for marker in ("/test", "/spec", "/fuzz")):
                    tests.append(rel)
                if ext in {".service", ".plist", ".policy"} or lower_parts & DEPLOY_MARKERS or any(marker in lower_rel for marker in DEPLOY_MARKERS):
                    deploy.append(rel)

        files.sort(key=lambda f: f.path)
        return SourceMap(
            roots=[str(root) for root in normalized_roots],
            revisions=revisions,
            total_files=len(files),
            total_bytes=sum(f.size for f in files),
            extension_counts=dict(extension_counts.most_common()),
            top_directories=dict(top_directories.most_common(40)),
            important_files=self._limit(important),
            formal_files=self._limit(formal),
            docs_files=self._limit(docs),
            test_files=self._limit(tests),
            deploy_files=self._limit(deploy),
            sample_files=[f.path for f in files[: self.max_files_per_section]],
        )

    def _iter_files(self, root: Path):  # type: ignore[no-untyped-def]
        if root.is_file():
            yield root
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in self.excludes for part in path.parts):
                continue
            yield path

    def _limit(self, paths: list[str]) -> list[str]:
        return sorted(dict.fromkeys(paths))[: self.max_files_per_section]

    @staticmethod
    def _relative_label(path: Path, roots: list[Path]) -> str:
        for root in roots:
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        return str(path)

    @staticmethod
    def _top_directory(rel_path: str) -> str:
        parts = Path(rel_path).parts
        if len(parts) <= 1:
            return "."
        return parts[0]

    @staticmethod
    def _git_revision(root: Path) -> str | None:
        candidate = root if root.is_dir() else root.parent
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
