"""Skill definitions and metadata parsing.

Skills are modular, composable units of domain expertise per SPEC.md.

Structure:
    Skills/<SkillName>/
    ├── SKILL.md        # Routing rules + domain knowledge (YAML frontmatter)
    ├── workflows/      # Step-by-step procedures
    ├── tools/          # Deterministic helpers
    └── tests/          # Evals and regressions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillMetadata:
    """Metadata from SKILL.md frontmatter."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    use_when: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    inputs_schema: dict[str, str] = field(default_factory=dict)
    outputs_schema: dict[str, str] = field(default_factory=dict)
    tools_allowed: list[str] = field(default_factory=list)
    priority: int = 0
    cost_profile: str = "medium"  # cheap, medium, expensive
    composes_with: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    default_voice: str | None = None
    entry_workflows: list[str] = field(default_factory=lambda: ["default"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillMetadata:
        """Create metadata from parsed YAML dict."""
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            use_when=data.get("use_when", []),
            intents=data.get("intents", []),
            entities=data.get("entities", []),
            inputs_schema=data.get("inputs_schema", {}),
            outputs_schema=data.get("outputs_schema", {}),
            tools_allowed=data.get("tools_allowed", []),
            priority=data.get("priority", 0),
            cost_profile=data.get("cost_profile", "medium"),
            composes_with=data.get("composes_with", []),
            exclusions=data.get("exclusions", []),
            default_voice=data.get("default_voice"),
            entry_workflows=data.get("entry_workflows", ["default"]),
        )


@dataclass
class Skill:
    """A loaded skill with metadata and content."""

    path: Path
    metadata: SkillMetadata
    content: str = ""  # Markdown body after frontmatter
    workflows: dict[str, str] = field(default_factory=dict)
    tools: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def load(cls, skill_path: Path) -> Skill:
        """Load a skill from its directory."""
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_path}")

        # Parse SKILL.md
        raw_content = skill_md.read_text()
        metadata, content = cls._parse_frontmatter(raw_content)

        skill = cls(
            path=skill_path,
            metadata=SkillMetadata.from_dict(metadata),
            content=content,
        )

        # Load workflows
        workflows_dir = skill_path / "workflows"
        if workflows_dir.exists():
            for wf_file in workflows_dir.glob("*.md"):
                skill.workflows[wf_file.stem] = wf_file.read_text()

        # Index tools
        tools_dir = skill_path / "tools"
        if tools_dir.exists():
            for tool_file in tools_dir.glob("*.py"):
                skill.tools[tool_file.stem] = tool_file

        return skill

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return {}, content

        frontmatter_raw = match.group(1)
        body = match.group(2)

        try:
            frontmatter = yaml.safe_load(frontmatter_raw)
            return frontmatter if isinstance(frontmatter, dict) else {}, body
        except yaml.YAMLError:
            return {}, content

    def matches_pattern(self, goal: str) -> bool:
        """Check if goal matches any use_when patterns."""
        goal_lower = goal.lower()

        for pattern in self.metadata.use_when:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*")
            if re.search(regex_pattern, goal_lower, re.IGNORECASE):
                return True

        return False

    def matches_intent(self, intent: str) -> bool:
        """Check if intent matches skill intents."""
        return intent.lower() in [i.lower() for i in self.metadata.intents]

    def matches_entities(self, entities: list[str]) -> float:
        """Calculate entity match score (0-1)."""
        if not entities or not self.metadata.entities:
            return 0.0

        skill_entities = {e.lower() for e in self.metadata.entities}
        goal_entities = {e.lower() for e in entities}
        overlap = skill_entities & goal_entities

        return len(overlap) / max(len(skill_entities), 1)

    def is_excluded(self, goal: str) -> bool:
        """Check if goal matches any exclusion patterns."""
        goal_lower = goal.lower()

        for exclusion in self.metadata.exclusions:
            if exclusion.lower() in goal_lower:
                return True

        return False

    def get_workflow(self, name: str = "default") -> str | None:
        """Get a workflow by name."""
        return self.workflows.get(name)
