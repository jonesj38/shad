"""Skill Router - Routes goals to appropriate skills.

Per SPEC.md Section 4.3:
1. Normalize goal → GoalSpec
2. Candidate generation (fast recall, 5-12 skills)
3. Ranking (deterministic scoring)
4. Selection (one primary, support if needed)
5. Composition: Primary orchestrates support skills
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shad.models.goal import GoalSpec
from shad.skills.skill import Skill
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


# Scoring weights per SPEC.md
WEIGHTS = {
    "intent": 0.3,
    "triggers": 0.25,
    "context": 0.1,
    "embed": 0.1,
    "history": 0.1,
    "risk": 0.05,
    "exclusion": 0.2,
    "priority": 0.1,
}


@dataclass
class SkillScore:
    """Scoring breakdown for a skill candidate."""

    skill: Skill
    intent_match: float = 0.0
    trigger_match: float = 0.0
    context_match: float = 0.0
    embedding_sim: float = 0.0
    history_success: float = 0.5  # Default neutral
    risk_mismatch: float = 0.0
    exclusion_hit: float = 0.0
    priority_bonus: float = 0.0

    @property
    def total(self) -> float:
        """Calculate total weighted score."""
        return (
            WEIGHTS["intent"] * self.intent_match
            + WEIGHTS["triggers"] * self.trigger_match
            + WEIGHTS["context"] * self.context_match
            + WEIGHTS["embed"] * self.embedding_sim
            + WEIGHTS["history"] * self.history_success
            - WEIGHTS["risk"] * self.risk_mismatch
            - WEIGHTS["exclusion"] * self.exclusion_hit
            + WEIGHTS["priority"] * self.priority_bonus
        )


@dataclass
class RoutingDecision:
    """Result of skill routing."""

    goal_spec: GoalSpec
    primary_skill: Skill | None
    support_skills: list[Skill] = field(default_factory=list)
    scores: list[SkillScore] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "goal": self.goal_spec.raw_goal,
            "intent": self.goal_spec.intent,
            "entities": self.goal_spec.entities,
            "primary_skill": self.primary_skill.metadata.name if self.primary_skill else None,
            "support_skills": [s.metadata.name for s in self.support_skills],
            "scores": [
                {
                    "skill": score.skill.metadata.name,
                    "total": round(score.total, 3),
                    "breakdown": {
                        "intent_match": score.intent_match,
                        "trigger_match": score.trigger_match,
                        "context_match": score.context_match,
                        "exclusion_hit": score.exclusion_hit,
                        "priority_bonus": score.priority_bonus,
                    },
                }
                for score in self.scores
            ],
            "reasoning": self.reasoning,
        }


class SkillRouter:
    """
    Routes goals to appropriate skills using deterministic scoring.

    Features:
    - Skill loading from filesystem
    - Pattern matching and intent detection
    - Weighted scoring with configurable weights
    - Loop guard tracking to prevent skill cycles
    """

    def __init__(self, skills_path: Path | None = None):
        settings = get_settings()
        self.skills_path = skills_path or settings.skills_path
        self.skills: dict[str, Skill] = {}
        self._call_stack: list[str] = []  # For cycle detection

    def load_skills(self) -> int:
        """
        Load all skills from the skills directory.

        Returns number of skills loaded.
        """
        self.skills.clear()

        if not self.skills_path.exists():
            logger.warning(f"Skills path does not exist: {self.skills_path}")
            return 0

        count = 0
        for skill_dir in self.skills_path.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                skill = Skill.load(skill_dir)
                self.skills[skill.metadata.name] = skill
                count += 1
                logger.debug(f"Loaded skill: {skill.metadata.name}")
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_dir}: {e}")

        logger.info(f"Loaded {count} skills from {self.skills_path}")
        return count

    def route(self, goal: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """
        Route a goal to the appropriate skill(s).

        Args:
            goal: Raw goal string
            context: Optional context (vault_path, prior skills, etc.)

        Returns:
            RoutingDecision with selected skills and scoring breakdown
        """
        # Step 1: Normalize goal
        goal_spec = GoalSpec.from_goal(goal)

        # Step 2: Generate candidates
        candidates = self._generate_candidates(goal_spec)

        if not candidates:
            return RoutingDecision(
                goal_spec=goal_spec,
                primary_skill=None,
                reasoning="No matching skills found",
            )

        # Step 3: Score candidates
        scores = [self._score_candidate(skill, goal_spec, context) for skill in candidates]

        # Sort by score descending
        scores.sort(key=lambda s: s.total, reverse=True)

        # Step 4: Select primary and support skills
        primary = scores[0].skill if scores else None
        support: list[Skill] = []

        # Add support skills if they compose well with primary
        if primary:
            for score in scores[1:4]:  # Check top 3 runners-up
                if (
                    score.total > 0.3  # Minimum threshold
                    and score.skill.metadata.name in primary.metadata.composes_with
                ):
                    support.append(score.skill)

        reasoning = self._generate_reasoning(goal_spec, scores, primary)

        return RoutingDecision(
            goal_spec=goal_spec,
            primary_skill=primary,
            support_skills=support,
            scores=scores,
            reasoning=reasoning,
        )

    def _generate_candidates(self, goal_spec: GoalSpec) -> list[Skill]:
        """
        Generate candidate skills using fast recall methods.

        Uses:
        - Pattern matching (use_when globs)
        - Intent matching
        - Entity overlap
        """
        candidates: list[Skill] = []

        for skill in self.skills.values():
            # Skip if excluded
            if skill.is_excluded(goal_spec.raw_goal):
                continue

            # Check pattern match
            if skill.matches_pattern(goal_spec.raw_goal):
                candidates.append(skill)
                continue

            # Check intent match
            if skill.matches_intent(goal_spec.intent):
                candidates.append(skill)
                continue

            # Check entity overlap
            if skill.matches_entities(goal_spec.entities) > 0.3:
                candidates.append(skill)
                continue

        # Limit to 12 candidates
        return candidates[:12]

    def _score_candidate(
        self,
        skill: Skill,
        goal_spec: GoalSpec,
        context: dict[str, Any] | None,
    ) -> SkillScore:
        """Score a skill candidate against the goal."""
        score = SkillScore(skill=skill)

        # Intent match (exact or partial)
        if skill.matches_intent(goal_spec.intent):
            score.intent_match = 1.0
        elif any(i in goal_spec.intent for i in skill.metadata.intents):
            score.intent_match = 0.5

        # Trigger/pattern match
        if skill.matches_pattern(goal_spec.raw_goal):
            score.trigger_match = 1.0

        # Entity/context match
        score.context_match = skill.matches_entities(goal_spec.entities)

        # Exclusion penalty
        if skill.is_excluded(goal_spec.raw_goal):
            score.exclusion_hit = 1.0

        # Priority bonus (normalized 0-1 assuming max priority is 20)
        score.priority_bonus = min(skill.metadata.priority / 20.0, 1.0)

        # TODO: Add embedding similarity when embeddings are available
        # TODO: Add history success rate when learning system is in place

        return score

    def _generate_reasoning(
        self,
        goal_spec: GoalSpec,
        scores: list[SkillScore],
        primary: Skill | None,
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        if not primary:
            return f"No skill matched goal intent '{goal_spec.intent}'"

        parts = [f"Selected '{primary.metadata.name}' for intent '{goal_spec.intent}'"]

        if scores and scores[0].trigger_match > 0:
            parts.append("pattern matched use_when rules")

        if scores and scores[0].context_match > 0:
            parts.append(f"entity overlap {scores[0].context_match:.0%}")

        return "; ".join(parts)

    # Loop guards
    def push_skill(self, skill_name: str) -> bool:
        """
        Push a skill onto the call stack.

        Returns False if this would create a cycle.
        """
        if skill_name in self._call_stack:
            logger.warning(f"Skill cycle detected: {skill_name} already in {self._call_stack}")
            return False

        skill = self.skills.get(skill_name)
        if skill:
            # Check depth limit
            max_depth = skill.metadata.priority  # Reuse priority as max depth
            if len(self._call_stack) >= max_depth and max_depth > 0:
                logger.warning(f"Skill depth limit reached for {skill_name}")
                return False

        self._call_stack.append(skill_name)
        return True

    def pop_skill(self) -> str | None:
        """Pop the most recent skill from the call stack."""
        return self._call_stack.pop() if self._call_stack else None

    def clear_stack(self) -> None:
        """Clear the skill call stack."""
        self._call_stack.clear()

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self.skills.get(name)

    def list_skills(self) -> list[dict[str, Any]]:
        """List all loaded skills with basic metadata."""
        return [
            {
                "name": s.metadata.name,
                "version": s.metadata.version,
                "description": s.metadata.description,
                "priority": s.metadata.priority,
                "intents": s.metadata.intents,
            }
            for s in self.skills.values()
        ]
