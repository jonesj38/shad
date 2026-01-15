"""Gap detection for vault curation.

Per SPEC.md Section 2.12:
- Combined scoring: history_pain, coverage_miss, llm_score
- Query history analysis for identifying weak spots
- Gap reports with suggestions
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GapScore:
    """Score for a detected knowledge gap."""

    topic: str
    score: float
    evidence: list[str]
    suggestions: list[str]
    priority: str  # "high", "medium", "low"

    def __post_init__(self) -> None:
        """Validate priority."""
        if self.priority not in ("high", "medium", "low"):
            raise ValueError(f"Invalid priority: {self.priority}")


@dataclass
class GapReport:
    """Report of detected knowledge gaps.

    Per SPEC.md Section 2.12.1:
    - Lists gaps ranked by combined score
    - Includes evidence and suggestions for each
    - Tracks overall coverage metrics
    """

    gaps: list[GapScore]
    total_queries_analyzed: int
    coverage_percentage: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert report to markdown format."""
        lines = [
            "# Vault Gap Report",
            "",
            f"**Queries Analyzed:** {self.total_queries_analyzed}",
            f"**Coverage:** {self.coverage_percentage:.0%}",
            "",
            "## Identified Gaps",
            "",
        ]

        for gap in sorted(self.gaps, key=lambda g: g.score, reverse=True):
            lines.append(f"### {gap.topic}")
            lines.append(f"**Priority:** {gap.priority} | **Score:** {gap.score:.2f}")
            lines.append("")

            if gap.evidence:
                lines.append("**Evidence:**")
                for ev in gap.evidence:
                    lines.append(f"- {ev}")
                lines.append("")

            if gap.suggestions:
                lines.append("**Suggestions:**")
                for sug in gap.suggestions:
                    lines.append(f"- {sug}")
                lines.append("")

        return "\n".join(lines)


@dataclass
class QueryResult:
    """Result of a query for history tracking."""

    query: str
    retrieval_score: float
    fallback_used: bool
    timestamp: float = 0.0


class QueryHistoryAnalyzer:
    """Analyzes query history to identify problematic topics.

    Tracks query frequency, retrieval scores, and fallback usage
    to identify areas where vault content is lacking.
    """

    def __init__(self) -> None:
        self._queries: list[QueryResult] = []
        self._topic_queries: dict[str, list[QueryResult]] = {}

    def add_query(
        self,
        query: str,
        retrieval_score: float,
        fallback_used: bool = False,
    ) -> None:
        """Add a query result to history."""
        result = QueryResult(
            query=query,
            retrieval_score=retrieval_score,
            fallback_used=fallback_used,
        )
        self._queries.append(result)

        # Index by topic (simple: use query as topic)
        topic = query.lower().strip()
        if topic not in self._topic_queries:
            self._topic_queries[topic] = []
        self._topic_queries[topic].append(result)

    def get_query_frequency(self) -> dict[str, int]:
        """Get frequency count for each topic."""
        return {
            topic: len(queries)
            for topic, queries in self._topic_queries.items()
        }

    def get_median_score(self, topic: str) -> float:
        """Get median retrieval score for a topic."""
        topic = topic.lower().strip()
        queries = self._topic_queries.get(topic, [])

        if not queries:
            return 0.0

        scores = [q.retrieval_score for q in queries]
        return statistics.median(scores)

    def get_fallback_rate(self, topic: str) -> float:
        """Get fallback usage rate for a topic."""
        topic = topic.lower().strip()
        queries = self._topic_queries.get(topic, [])

        if not queries:
            return 0.0

        fallbacks = sum(1 for q in queries if q.fallback_used)
        return fallbacks / len(queries)

    def get_problematic_topics(self, threshold: float = 0.5) -> list[str]:
        """Get topics with poor retrieval performance.

        Args:
            threshold: Median score threshold below which topics are problematic

        Returns:
            List of problematic topic names
        """
        problematic = []

        for topic in self._topic_queries:
            median = self.get_median_score(topic)
            if median < threshold:
                problematic.append(topic)

        return problematic


class GapDetector:
    """Detects knowledge gaps in the vault.

    Per SPEC.md Section 2.12.1:
    gap_score = 0.55 * history_pain + 0.25 * coverage_miss + 0.20 * llm_score
    """

    # Scoring weights per SPEC.md
    HISTORY_WEIGHT = 0.55
    COVERAGE_WEIGHT = 0.25
    LLM_WEIGHT = 0.20

    def __init__(self) -> None:
        self._query_results: list[QueryResult] = []
        self._topic_results: dict[str, list[QueryResult]] = {}

    def calculate_gap_score(
        self,
        history_pain: float,
        coverage_miss: float,
        llm_score: float,
    ) -> float:
        """Calculate combined gap score.

        Per SPEC.md Section 2.12.1:
        gap_score = 0.55 * history_pain + 0.25 * coverage_miss + 0.20 * llm_score
        """
        return (
            self.HISTORY_WEIGHT * history_pain
            + self.COVERAGE_WEIGHT * coverage_miss
            + self.LLM_WEIGHT * llm_score
        )

    def add_query_result(
        self,
        query: str,
        retrieval_score: float,
        fallback_used: bool = False,
    ) -> None:
        """Add a query result for history-based gap detection."""
        result = QueryResult(
            query=query,
            retrieval_score=retrieval_score,
            fallback_used=fallback_used,
        )
        self._query_results.append(result)

        # Index by topic
        topic = query.lower().strip()
        if topic not in self._topic_results:
            self._topic_results[topic] = []
        self._topic_results[topic].append(result)

    def calculate_history_pain(self, topic: str) -> float:
        """Calculate history pain score for a topic.

        History pain is high when:
        - Retrieval scores are consistently low
        - Fallback is frequently used
        """
        topic = topic.lower().strip()
        results = self._topic_results.get(topic, [])

        if not results:
            return 0.0

        # Low scores increase pain
        avg_score = sum(r.retrieval_score for r in results) / len(results)
        score_pain = 1.0 - avg_score

        # Fallback usage increases pain
        fallback_rate = sum(1 for r in results if r.fallback_used) / len(results)

        # Combine: weight score_pain higher
        return 0.6 * score_pain + 0.4 * fallback_rate

    def find_coverage_misses(
        self,
        common_topics: list[str],
        vault_topics: list[str],
    ) -> list[str]:
        """Find topics that should have anchor notes but don't."""
        vault_set = {t.lower() for t in vault_topics}
        misses = []

        for topic in common_topics:
            if topic.lower() not in vault_set:
                misses.append(topic)

        return misses

    def _score_to_priority(self, score: float) -> str:
        """Convert score to priority level."""
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def generate_report(
        self,
        vault_topics: list[str],
        llm_scores: dict[str, float] | None = None,
    ) -> GapReport:
        """Generate a gap report.

        Args:
            vault_topics: List of topics currently in the vault
            llm_scores: Optional LLM-generated relevance scores per topic

        Returns:
            GapReport with identified gaps
        """
        llm_scores = llm_scores or {}
        gaps: list[GapScore] = []

        # Analyze each topic from query history
        for topic, results in self._topic_results.items():
            history_pain = self.calculate_history_pain(topic)

            # Check coverage
            is_covered = any(
                topic.lower() in vt.lower() or vt.lower() in topic.lower()
                for vt in vault_topics
            )
            coverage_miss = 0.0 if is_covered else 1.0

            # Get LLM score if available
            llm_score = llm_scores.get(topic, 0.5)

            # Calculate combined score
            score = self.calculate_gap_score(
                history_pain=history_pain,
                coverage_miss=coverage_miss,
                llm_score=llm_score,
            )

            # Only report significant gaps
            if score >= 0.3:
                # Build evidence
                evidence = []
                if len(results) > 1:
                    evidence.append(f"{len(results)} queries")

                avg_score = sum(r.retrieval_score for r in results) / len(results)
                if avg_score < 0.5:
                    evidence.append(f"median score {avg_score:.2f}")

                fallback_rate = sum(1 for r in results if r.fallback_used) / len(results)
                if fallback_rate > 0.3:
                    evidence.append(f"{fallback_rate:.0%} fallback")

                # Generate suggestions
                suggestions = []
                if coverage_miss > 0:
                    # Suggest file path based on topic
                    topic_path = topic.replace(" ", "").title()
                    suggestions.append(f"Patterns/{topic_path}.md")

                gaps.append(GapScore(
                    topic=topic,
                    score=score,
                    evidence=evidence,
                    suggestions=suggestions,
                    priority=self._score_to_priority(score),
                ))

        # Calculate coverage
        covered_count = sum(
            1 for vt in vault_topics
            if any(vt.lower() in t or t in vt.lower() for t in self._topic_results)
        )
        coverage_pct = covered_count / len(vault_topics) if vault_topics else 0.0

        return GapReport(
            gaps=gaps,
            total_queries_analyzed=len(self._query_results),
            coverage_percentage=coverage_pct,
        )
