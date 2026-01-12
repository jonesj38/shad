"""Novelty detection for pruning diminishing returns.

Per SPEC.md Section 5.3, pruning uses diminishing returns (novelty metric):

Detection layers:
1. Cheap prefilter: Embedding distance from existing results
2. Backbone: Fact extraction diff (new facts vs. rephrased existing)
3. Tie-breaker: LLM judge for marginal value scoring
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from shad.engine.llm import LLMProvider, ModelTier

logger = logging.getLogger(__name__)


@dataclass
class NoveltyScore:
    """Result of novelty detection."""

    score: float  # 0.0 = duplicate, 1.0 = completely novel
    is_novel: bool
    new_facts: list[str] = field(default_factory=list)
    duplicate_facts: list[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_prune(self) -> bool:
        """Check if result should be pruned due to low novelty."""
        return self.score < 0.2


class NoveltyDetector:
    """
    Detects novelty of results to enable pruning.

    Used to avoid wasting compute on redundant information.
    """

    # Thresholds
    NOVELTY_THRESHOLD = 0.3  # Below this, consider pruning
    HIGH_NOVELTY_THRESHOLD = 0.7  # Above this, definitely keep

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        use_embeddings: bool = True,
    ):
        self.llm = llm_provider or LLMProvider()
        self.use_embeddings = use_embeddings
        self._seen_hashes: set[str] = set()
        self._existing_facts: list[str] = []

    def reset(self) -> None:
        """Reset detector state for new run."""
        self._seen_hashes.clear()
        self._existing_facts.clear()

    async def check_novelty(
        self,
        result: str,
        existing_results: list[str] | None = None,
    ) -> NoveltyScore:
        """
        Check novelty of a result against existing results.

        Args:
            result: New result to check
            existing_results: List of existing results to compare against

        Returns:
            NoveltyScore with score and reasoning
        """
        existing = existing_results or []

        # Layer 1: Cheap hash-based duplicate detection
        result_hash = self._hash_content(result)
        if result_hash in self._seen_hashes:
            return NoveltyScore(
                score=0.0,
                is_novel=False,
                reasoning="Exact duplicate detected",
            )
        self._seen_hashes.add(result_hash)

        # Layer 2: Fact extraction and comparison
        if existing:
            try:
                fact_score = await self._check_fact_novelty(result, existing)
                if fact_score.score < self.NOVELTY_THRESHOLD:
                    return fact_score
            except Exception as e:
                logger.warning(f"Fact novelty check failed: {e}")

        # Layer 3: LLM judge for marginal cases
        if existing and len(existing) <= 5:
            try:
                judge_score = await self._judge_novelty(result, existing)
                return judge_score
            except Exception as e:
                logger.warning(f"LLM novelty judge failed: {e}")

        # Default: assume novel if no strong signal otherwise
        return NoveltyScore(
            score=0.8,
            is_novel=True,
            reasoning="No significant overlap detected",
        )

    def _hash_content(self, content: str) -> str:
        """Create content hash for duplicate detection."""
        # Normalize: lowercase, remove extra whitespace
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def _check_fact_novelty(
        self,
        result: str,
        existing: list[str],
    ) -> NoveltyScore:
        """Extract and compare facts to check novelty."""
        # Extract facts from new result
        new_facts = await self._extract_facts(result)

        # Compare against existing facts
        if not self._existing_facts:
            for ex in existing:
                facts = await self._extract_facts(ex)
                self._existing_facts.extend(facts)

        if not new_facts:
            return NoveltyScore(
                score=0.5,
                is_novel=True,
                reasoning="Could not extract facts",
            )

        # Check overlap
        existing_set = {f.lower() for f in self._existing_facts}
        novel_facts = []
        duplicate_facts = []

        for fact in new_facts:
            if any(self._facts_similar(fact, ef) for ef in existing_set):
                duplicate_facts.append(fact)
            else:
                novel_facts.append(fact)

        if not novel_facts:
            return NoveltyScore(
                score=0.1,
                is_novel=False,
                new_facts=[],
                duplicate_facts=duplicate_facts,
                reasoning="All facts already covered",
            )

        score = len(novel_facts) / len(new_facts)

        return NoveltyScore(
            score=score,
            is_novel=score >= self.NOVELTY_THRESHOLD,
            new_facts=novel_facts,
            duplicate_facts=duplicate_facts,
            reasoning=f"{len(novel_facts)}/{len(new_facts)} facts are novel",
        )

    async def _extract_facts(self, text: str) -> list[str]:
        """Extract key facts from text using LLM."""
        prompt = f"""Extract the key factual claims from this text.
Return ONLY a JSON array of short fact statements.

Text:
{text[:2000]}

Return format: ["fact 1", "fact 2", "fact 3"]"""

        try:
            response, _ = await self.llm.complete(
                prompt=prompt,
                tier=ModelTier.LEAF,
                temperature=0.1,
            )

            import json

            # Parse JSON array
            response = response.strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            facts = json.loads(response.strip())
            return facts if isinstance(facts, list) else []

        except Exception as e:
            logger.debug(f"Fact extraction failed: {e}")
            # Fallback: simple sentence extraction
            sentences = text.replace("\n", " ").split(". ")
            return [s.strip() for s in sentences[:10] if len(s) > 20]

    def _facts_similar(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are semantically similar."""
        # Simple word overlap check
        words1 = set(fact1.lower().split())
        words2 = set(fact2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / min(len(words1), len(words2))

        return similarity > 0.6

    async def _judge_novelty(
        self,
        result: str,
        existing: list[str],
    ) -> NoveltyScore:
        """Use LLM judge to score novelty."""
        existing_text = "\n---\n".join(existing[-3:])  # Last 3 results

        prompt = f"""You are a novelty judge. Score how much NEW information the
New Result provides compared to Existing Results.

Existing Results:
{existing_text[:3000]}

New Result:
{result[:1500]}

Respond with JSON:
{{
  "novelty_score": 0.0-1.0,
  "new_information": ["item1", "item2"],
  "redundant_information": ["item1"],
  "reasoning": "explanation"
}}"""

        try:
            response, _ = await self.llm.complete(
                prompt=prompt,
                tier=ModelTier.JUDGE,
                temperature=0.1,
            )

            import json

            # Parse response
            response = response.strip()
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            judgment = json.loads(response.strip())

            return NoveltyScore(
                score=judgment.get("novelty_score", 0.5),
                is_novel=judgment.get("novelty_score", 0.5) >= self.NOVELTY_THRESHOLD,
                new_facts=judgment.get("new_information", []),
                duplicate_facts=judgment.get("redundant_information", []),
                reasoning=judgment.get("reasoning", ""),
                metadata={"judge_response": response},
            )

        except Exception as e:
            logger.warning(f"Novelty judge failed: {e}")
            return NoveltyScore(
                score=0.5,
                is_novel=True,
                reasoning=f"Judge error: {str(e)}",
            )
