"""GoalSpec model for task normalization and routing."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level for a goal."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GoalSpec(BaseModel):
    """Normalized goal specification for routing and execution."""

    raw_goal: str = Field(..., description="Original goal text")
    normalized_goal: str = Field(..., description="Cleaned/normalized goal")
    intent: str = Field(default="general", description="Detected intent")
    entities: list[str] = Field(default_factory=list, description="Extracted entities")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Parsed constraints")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Assessed risk level")

    @classmethod
    def from_goal(cls, goal: str) -> GoalSpec:
        """Create a GoalSpec from a raw goal string."""
        # Simple normalization - MVP doesn't need full NLP
        normalized = goal.strip().lower()

        # Simple intent detection
        intent = "general"
        if any(w in normalized for w in ["research", "investigate", "find"]):
            intent = "research"
        elif any(w in normalized for w in ["summarize", "summary", "tldr"]):
            intent = "summarize"
        elif any(w in normalized for w in ["compare", "versus", "vs"]):
            intent = "compare"
        elif any(w in normalized for w in ["explain", "what is", "how does"]):
            intent = "explain"
        elif any(w in normalized for w in ["analyze", "analysis"]):
            intent = "analyze"

        return cls(
            raw_goal=goal,
            normalized_goal=normalized,
            intent=intent,
        )
