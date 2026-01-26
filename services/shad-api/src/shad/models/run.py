"""Run and DAG models for Shad execution."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Status of a run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"


class NodeStatus(str, Enum):
    """Status of a DAG node."""

    CREATED = "created"
    READY = "ready"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PRUNED = "pruned"
    CACHE_HIT = "cache_hit"


class StopReason(str, Enum):
    """Reason why a run or node stopped."""

    COMPLETE = "complete"
    BUDGET_DEPTH = "budget_depth"
    BUDGET_NODES = "budget_nodes"
    BUDGET_TIME = "budget_time"
    BUDGET_TOKENS = "budget_tokens"
    NOVELTY_PRUNED = "novelty_pruned"
    ERROR = "error"
    ABORTED = "aborted"


class Budget(BaseModel):
    """Budget constraints for a run."""

    max_wall_time: int = Field(default=300, description="Maximum wall time in seconds")
    max_tokens: int = Field(default=100000, description="Maximum total tokens")
    max_nodes: int = Field(default=50, description="Maximum DAG nodes")
    max_depth: int = Field(default=3, description="Maximum recursion depth")
    max_branching_factor: int = Field(default=7, description="Maximum children per node")


class ModelConfig(BaseModel):
    """Model configuration for different execution tiers."""

    orchestrator_model: str | None = Field(
        default=None, description="Model for planning/synthesis (top level)"
    )
    worker_model: str | None = Field(
        default=None, description="Model for mid-depth execution"
    )
    leaf_model: str | None = Field(
        default=None, description="Model for fast parallel execution (leaves)"
    )


class RunConfig(BaseModel):
    """Configuration for a run."""

    goal: str = Field(..., description="The goal/task to accomplish")
    vault_path: str | None = Field(default=None, description="Obsidian vault path for context")
    budget: Budget = Field(default_factory=Budget)
    voice: str | None = Field(default=None, description="Voice to use for output rendering")

    # Strategy configuration (Phase 3)
    strategy_override: str | None = Field(default=None, description="Override automatic strategy selection")

    # Verification configuration (Phase 5)
    verify_level: str | None = Field(default=None, description="Verification level: off, basic, build, strict")

    # Output configuration (Phase 4)
    write_files: bool = Field(default=False, description="Write output files to disk")
    output_path: str | None = Field(default=None, description="Output directory for files")

    # Model configuration
    model_config_override: ModelConfig | None = Field(
        default=None, description="Override default model selection per tier"
    )


class DAGNode(BaseModel):
    """A node in the execution DAG."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    depth: int = 0
    task: str = Field(..., description="The sub-task for this node")
    status: NodeStatus = NodeStatus.CREATED
    result: str | None = None
    children: list[str] = Field(default_factory=list)
    cache_key: str | None = None
    cache_hit: bool = False
    tokens_used: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    stop_reason: StopReason | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def duration_ms(self) -> int | None:
        """Calculate duration in milliseconds."""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None


class Run(BaseModel):
    """A complete run execution."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: RunConfig
    status: RunStatus = RunStatus.PENDING
    root_node_id: str | None = None
    nodes: dict[str, DAGNode] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_tokens: int = 0
    stop_reason: StopReason | None = None
    error: str | None = None
    final_result: str | None = None
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional run metadata")

    def get_node(self, node_id: str) -> DAGNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the run."""
        self.nodes[node.node_id] = node

    def completed_nodes(self) -> list[DAGNode]:
        """Get all completed nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.SUCCEEDED]

    def failed_nodes(self) -> list[DAGNode]:
        """Get all failed nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

    def pending_nodes(self) -> list[DAGNode]:
        """Get all pending nodes."""
        return [
            n
            for n in self.nodes.values()
            if n.status in (NodeStatus.CREATED, NodeStatus.READY, NodeStatus.STARTED)
        ]
