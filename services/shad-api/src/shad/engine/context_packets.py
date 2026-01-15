"""Context packets for cross-subtask context sharing.

Per SPEC.md Section 1.6:
- Soft dependencies enable cross-subtask context sharing
- Decomposition emits hard_deps (must complete) and soft_deps (useful if available)
- Completed nodes produce context packets (summary, artifacts, keywords)
- Scheduler injects packets into pending nodes' retrieval
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextPacket:
    """A context packet produced by a completed node.

    Contains distilled information that can be injected into
    other nodes' context for cross-subtask sharing.
    """

    node_id: str
    stage_name: str
    summary: str
    artifacts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Convert packet to a string for context injection."""
        parts = [
            f"[{self.stage_name}] {self.summary}",
        ]

        if self.artifacts:
            artifacts_str = ", ".join(self.artifacts[:5])  # Limit to 5
            parts.append(f"Artifacts: {artifacts_str}")

        if self.keywords:
            keywords_str = ", ".join(self.keywords[:10])  # Limit to 10
            parts.append(f"Keywords: {keywords_str}")

        return "\n".join(parts)

    def relevance_score(self, query: str) -> float:
        """Calculate relevance score for a query.

        Returns a score between 0 and 1 indicating how relevant
        this packet is to the given query.
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'implement', 'create', 'build', 'make', 'add', 'and', 'or',
        }
        query_words = query_words - stop_words

        if not query_words:
            return 0.5  # Neutral score if no meaningful words

        # Check keyword matches
        keyword_lower = [k.lower() for k in self.keywords]
        keyword_matches = sum(1 for kw in keyword_lower if kw in query_words)

        # Check summary matches
        summary_lower = self.summary.lower()
        summary_matches = sum(1 for w in query_words if w in summary_lower)

        # Calculate score
        total_matches = keyword_matches + summary_matches
        max_possible = len(query_words) * 2  # Keywords + summary

        if max_possible == 0:
            return 0.5

        return min(1.0, total_matches / max_possible + 0.1)  # Base score of 0.1


class ContextPacketStore:
    """Storage for context packets.

    Provides efficient access by node_id and stage_name,
    as well as relevance-based retrieval.
    """

    def __init__(self) -> None:
        self._packets: dict[str, ContextPacket] = {}
        self._by_stage: dict[str, list[ContextPacket]] = {}

    def add(self, packet: ContextPacket) -> None:
        """Add a packet to the store."""
        self._packets[packet.node_id] = packet

        if packet.stage_name not in self._by_stage:
            self._by_stage[packet.stage_name] = []
        self._by_stage[packet.stage_name].append(packet)

    def get(self, node_id: str) -> ContextPacket | None:
        """Get a packet by node ID."""
        return self._packets.get(node_id)

    def get_by_stage(self, stage_name: str) -> list[ContextPacket]:
        """Get all packets for a stage name."""
        # Also match stage name prefixes (e.g., types_contracts matches types_contracts_auth)
        packets = []
        for stage, stage_packets in self._by_stage.items():
            if stage == stage_name or stage.startswith(f"{stage_name}_"):
                packets.extend(stage_packets)
        return packets

    def get_relevant(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
    ) -> list[ContextPacket]:
        """Get packets relevant to a query, sorted by relevance."""
        scored = []
        for packet in self._packets.values():
            score = packet.relevance_score(query)
            if score >= min_score:
                scored.append((score, packet))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [packet for _, packet in scored[:limit]]

    def all_packets(self) -> list[ContextPacket]:
        """Get all packets."""
        return list(self._packets.values())

    def clear(self) -> None:
        """Clear all packets."""
        self._packets.clear()
        self._by_stage.clear()


class NodeContextManager:
    """Manages context packet creation and injection.

    Responsible for:
    - Creating packets from completed node results
    - Extracting keywords from results
    - Injecting relevant context into pending nodes
    """

    def __init__(self) -> None:
        self.store = ContextPacketStore()

    def create_packet(
        self,
        node_id: str,
        stage_name: str,
        result: str,
        artifacts: list[str] | None = None,
    ) -> ContextPacket:
        """Create a context packet from a node's result.

        Extracts keywords and creates a summary from the result.
        """
        # Extract keywords from result
        keywords = self._extract_keywords(result)

        # Create summary (first significant lines)
        summary = self._create_summary(result)

        return ContextPacket(
            node_id=node_id,
            stage_name=stage_name,
            summary=summary,
            artifacts=artifacts or [],
            keywords=keywords,
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text.

        Looks for:
        - TypeScript/JavaScript identifiers (PascalCase, camelCase)
        - Interface/type/class names
        - Function names
        - Variable names
        """
        keywords: set[str] = set()

        # Find PascalCase words (likely type/class names)
        pascal_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
        keywords.update(pascal_case)

        # Find camelCase words (likely variables/functions)
        camel_case = re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', text)
        keywords.update(camel_case)

        # Find common code patterns
        interface_names = re.findall(r'(?:interface|type|class|enum)\s+(\w+)', text)
        keywords.update(interface_names)

        function_names = re.findall(r'(?:function|const|let|var)\s+(\w+)\s*[=(]', text)
        keywords.update(function_names)

        # Filter out very short or common words
        filtered = [kw for kw in keywords if len(kw) > 2]

        return list(filtered)[:20]  # Limit to 20 keywords

    def _create_summary(self, text: str, max_length: int = 200) -> str:
        """Create a summary from text.

        Takes the first meaningful lines up to max_length.
        """
        lines = text.strip().split('\n')

        # Filter out empty lines and comments
        meaningful_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                meaningful_lines.append(stripped)
                if len('\n'.join(meaningful_lines)) > max_length:
                    break

        summary = '\n'.join(meaningful_lines)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    def get_context_for_node(
        self,
        node_id: str,
        stage_name: str,
        task: str,
        soft_deps: list[str] | None = None,
    ) -> str:
        """Get context for a node based on soft dependencies and task.

        Combines context from:
        1. Completed soft dependency nodes
        2. Other relevant completed nodes (based on task similarity)
        """
        context_parts: list[str] = []

        # Get context from soft dependencies
        if soft_deps:
            for dep_stage in soft_deps:
                packets = self.store.get_by_stage(dep_stage)
                for packet in packets:
                    context_parts.append(packet.to_context_string())

        # Get additional relevant context
        relevant_packets = self.store.get_relevant(task, limit=3)
        for packet in relevant_packets:
            # Avoid duplicates
            if packet.node_id not in [p.split(']')[0].strip('[')
                                       for p in context_parts]:
                context_str = packet.to_context_string()
                if context_str not in context_parts:
                    context_parts.append(context_str)

        if not context_parts:
            return ""

        return "\n---\n".join(context_parts)

    def inject_soft_dep_context(
        self,
        soft_deps: list[str],
        task: str,
    ) -> str:
        """Inject context from soft dependencies.

        Per SPEC.md: When a node starts, inject relevant context
        from completed soft dependency nodes.
        """
        context_parts: list[str] = []

        # Collect context from soft deps - always include if from soft dep stage
        for dep_stage in soft_deps:
            packets = self.store.get_by_stage(dep_stage)
            for packet in packets:
                # Always include context from explicit soft deps
                # The soft dep itself implies relevance
                context_parts.append(packet.to_context_string())

        if not context_parts:
            return ""

        header = "--- Context from completed dependencies ---"
        return f"{header}\n\n" + "\n\n".join(context_parts)

    def get_relevant_context(
        self,
        task: str,
        limit: int = 5,
    ) -> str:
        """Get relevant context for a task from all packets."""
        packets = self.store.get_relevant(task, limit=limit)

        if not packets:
            return ""

        context_parts = [p.to_context_string() for p in packets]
        return "\n---\n".join(context_parts)
