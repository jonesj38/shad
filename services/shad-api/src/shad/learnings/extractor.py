"""Learning extraction from run results.

Per SPEC.md Section 15:
1. Capture everything as notes (default layer)
2. Propose patches/hints/negatives (automated suggestions)
3. Test via evals (comparative runs)
4. Promote via HITL review (human approval)

Learning Types:
- Prompt patches: Amendments to skill prompts
- Routing hints: "Goals containing X → skill Y"
- Negative examples: Failure cases to avoid
- Notes: OpenNotebookLM entries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from shad.engine.llm import LLMProvider, ModelTier

logger = logging.getLogger(__name__)


class LearningType(str, Enum):
    """Types of learnings that can be extracted."""

    PROMPT_PATCH = "prompt_patch"  # Amendments to skill prompts
    ROUTING_HINT = "routing_hint"  # Goal → skill mapping hints
    NEGATIVE_EXAMPLE = "negative_example"  # Failure cases to avoid
    NOTE = "note"  # General knowledge note
    FACT = "fact"  # Extracted factual claim
    PATTERN = "pattern"  # Recognized pattern in queries


@dataclass
class ExtractedLearning:
    """A learning extracted from a run."""

    learning_type: LearningType
    content: str
    confidence: float = 0.5
    source_run_id: str = ""
    source_node_id: str = ""
    skill_name: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "learning_type": self.learning_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "source_run_id": self.source_run_id,
            "source_node_id": self.source_node_id,
            "skill_name": self.skill_name,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class LearningExtractor:
    """
    Extracts learnings from completed runs.

    Analyzes run results to identify:
    - Prompt improvements
    - Routing optimizations
    - Failure patterns
    - Knowledge to persist
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider or LLMProvider()

    async def extract_from_run(
        self,
        run: Any,
        include_failures: bool = True,
    ) -> list[ExtractedLearning]:
        """
        Extract learnings from a completed run.

        Args:
            run: A completed Run object
            include_failures: Whether to extract from failed nodes

        Returns:
            List of extracted learnings
        """
        learnings: list[ExtractedLearning] = []

        # Extract from successful nodes
        for node in run.nodes.values():
            if node.status.value == "succeeded" and node.result:
                # Extract facts from result
                facts = await self._extract_facts(node.result, run.run_id, node.node_id)
                learnings.extend(facts)

            elif include_failures and node.status.value == "failed":
                # Learn from failures
                negative = await self._extract_negative(
                    node.task,
                    node.error or "Unknown error",
                    run.run_id,
                    node.node_id,
                )
                if negative:
                    learnings.append(negative)

        # Extract routing hints if we have skill information
        routing_hint = self._extract_routing_hint(run)
        if routing_hint:
            learnings.append(routing_hint)

        return learnings

    async def _extract_facts(
        self,
        result: str,
        run_id: str,
        node_id: str,
    ) -> list[ExtractedLearning]:
        """Extract factual learnings from a result."""
        prompt = f"""Extract key factual learnings from this text that would be useful
to remember for future queries. Focus on:
1. Concrete facts and findings
2. Useful patterns or heuristics
3. Important definitions or concepts

Text:
{result[:2000]}

Return a JSON array of learnings:
[
  {{"fact": "...", "confidence": 0.0-1.0, "tags": ["tag1"]}},
  ...
]

Only include high-quality, reusable learnings."""

        try:
            response, _ = await self.llm.complete(
                prompt=prompt,
                tier=ModelTier.LEAF,
                temperature=0.1,
            )

            import json

            # Parse response
            response = response.strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            facts_data = json.loads(response.strip())
            if not isinstance(facts_data, list):
                return []

            learnings = []
            for fact in facts_data[:5]:  # Limit to 5 facts per result
                if fact.get("fact"):
                    learnings.append(ExtractedLearning(
                        learning_type=LearningType.FACT,
                        content=fact["fact"],
                        confidence=fact.get("confidence", 0.5),
                        source_run_id=run_id,
                        source_node_id=node_id,
                        tags=fact.get("tags", []),
                    ))

            return learnings

        except Exception as e:
            logger.debug(f"Fact extraction failed: {e}")
            return []

    async def _extract_negative(
        self,
        task: str,
        error: str,
        run_id: str,
        node_id: str,
    ) -> ExtractedLearning | None:
        """Extract a negative example from a failure."""
        # Create a learning about what went wrong
        content = f"Task '{task[:100]}...' failed with: {error[:200]}"

        return ExtractedLearning(
            learning_type=LearningType.NEGATIVE_EXAMPLE,
            content=content,
            confidence=0.8,
            source_run_id=run_id,
            source_node_id=node_id,
            tags=["failure", "avoid"],
            metadata={"original_task": task, "error": error},
        )

    def _extract_routing_hint(self, run: Any) -> ExtractedLearning | None:
        """Extract routing hints from successful runs."""
        # Only create hints for successful runs
        if run.status.value != "complete":
            return None

        # Extract keywords from goal
        goal = run.config.goal.lower()
        keywords = []

        # Simple keyword extraction
        for word in goal.split():
            if len(word) > 4 and word.isalpha():
                keywords.append(word)

        if not keywords:
            return None

        # Get skill name if available
        skill_name = getattr(run.config, "skill_name", None)
        if not skill_name:
            return None

        content = f"Goals containing {', '.join(keywords[:3])} → skill {skill_name}"

        return ExtractedLearning(
            learning_type=LearningType.ROUTING_HINT,
            content=content,
            confidence=0.6,
            source_run_id=run.run_id,
            skill_name=skill_name,
            tags=["routing", *keywords[:3]],
        )

    async def propose_prompt_patch(
        self,
        skill_name: str,
        original_prompt: str,
        failure_examples: list[dict[str, str]],
    ) -> ExtractedLearning | None:
        """
        Propose a prompt patch based on failure patterns.

        Args:
            skill_name: Name of the skill to patch
            original_prompt: Current skill prompt
            failure_examples: List of {task, error} dicts

        Returns:
            Proposed prompt patch or None
        """
        if not failure_examples:
            return None

        failures_text = "\n".join(
            f"- Task: {f['task'][:100]}... Error: {f['error'][:100]}"
            for f in failure_examples[:5]
        )

        prompt = f"""Given these failure cases for the '{skill_name}' skill:

{failures_text}

And this current prompt excerpt:
{original_prompt[:500]}...

Propose a SHORT amendment to the prompt that would help avoid these failures.
Return ONLY the amendment text, nothing else.
Keep it under 100 words."""

        try:
            response, _ = await self.llm.complete(
                prompt=prompt,
                tier=ModelTier.WORKER,
                temperature=0.3,
            )

            return ExtractedLearning(
                learning_type=LearningType.PROMPT_PATCH,
                content=response.strip(),
                confidence=0.5,
                skill_name=skill_name,
                tags=["prompt", "patch", skill_name],
                metadata={"failure_count": len(failure_examples)},
            )

        except Exception as e:
            logger.warning(f"Prompt patch proposal failed: {e}")
            return None
