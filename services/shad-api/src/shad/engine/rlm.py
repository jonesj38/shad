"""RLM Engine - Recursive Language Model execution engine.

The core reasoning engine that:
1. Treats prompts as external environments
2. Recursively decomposes problems
3. Caches/verifies/recomposes results
4. Enforces budget constraints
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from shad.engine.llm import LLMProvider, ModelTier
from shad.models.goal import GoalSpec
from shad.models.notebook import Note, RetrievalResult, Source
from shad.models.run import (
    DAGNode,
    NodeStatus,
    Run,
    RunConfig,
    RunStatus,
    StopReason,
)
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """Raised when a budget limit is reached."""

    def __init__(self, reason: StopReason, message: str):
        self.reason = reason
        super().__init__(message)


class RLMEngine:
    """
    Recursive Language Model Engine.

    Executes reasoning tasks by:
    - Decomposing goals into sub-tasks
    - Running sub-tasks recursively
    - Synthesizing results
    - Enforcing budget constraints
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        notebook_store: Any | None = None,
        cache: Any | None = None,
    ):
        self.llm = llm_provider or LLMProvider()
        self.notebook_store = notebook_store
        self.cache = cache
        self.settings = get_settings()

    async def execute(self, config: RunConfig) -> Run:
        """
        Execute a run with the given configuration.

        This is the main entry point for reasoning tasks.
        """
        run = Run(config=config)
        run.status = RunStatus.RUNNING
        run.started_at = datetime.utcnow()

        logger.info(f"Starting run {run.run_id} with goal: {config.goal}")

        try:
            # Create goal spec
            goal_spec = GoalSpec.from_goal(config.goal)

            # Retrieve context if notebook specified
            context = ""
            if config.notebook_id and self.notebook_store:
                retrieval = await self._retrieve_context(config.notebook_id, goal_spec)
                context = self._format_retrieval_context(retrieval)

            # Create root node
            root_node = DAGNode(
                task=config.goal,
                depth=0,
            )
            run.root_node_id = root_node.node_id
            run.add_node(root_node)

            # Execute recursively
            await self._execute_node(run, root_node, context)

            # Determine final status
            if run.stop_reason in (StopReason.BUDGET_DEPTH, StopReason.BUDGET_NODES,
                                   StopReason.BUDGET_TIME, StopReason.BUDGET_TOKENS):
                run.status = RunStatus.PARTIAL
            elif root_node.status == NodeStatus.SUCCEEDED:
                run.status = RunStatus.COMPLETE
                run.final_result = root_node.result
            else:
                run.status = RunStatus.FAILED
                run.error = root_node.error

        except BudgetExhausted as e:
            run.status = RunStatus.PARTIAL
            run.stop_reason = e.reason
            logger.warning(f"Run {run.run_id} stopped: {e}")

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            logger.exception(f"Run {run.run_id} failed: {e}")

        finally:
            run.completed_at = datetime.utcnow()

        return run

    async def resume(self, run: Run) -> Run:
        """
        Resume a partial or failed run.

        Continues from the last checkpoint, reusing completed nodes.
        """
        if run.status not in (RunStatus.PARTIAL, RunStatus.FAILED):
            raise ValueError(f"Cannot resume run with status {run.status}")

        run.status = RunStatus.RUNNING
        run.stop_reason = None
        run.error = None

        logger.info(f"Resuming run {run.run_id}")

        try:
            # Find pending nodes to continue
            pending = run.pending_nodes()
            if not pending and run.root_node_id:
                # Re-execute from root if no pending nodes
                root = run.get_node(run.root_node_id)
                if root and root.status != NodeStatus.SUCCEEDED:
                    pending = [root]

            # Execute pending nodes
            for node in pending:
                context = ""  # TODO: restore context from checkpoint
                await self._execute_node(run, node, context)

            # Update status
            root = run.get_node(run.root_node_id) if run.root_node_id else None
            if root and root.status == NodeStatus.SUCCEEDED:
                run.status = RunStatus.COMPLETE
                run.final_result = root.result
            elif run.stop_reason:
                run.status = RunStatus.PARTIAL
            else:
                run.status = RunStatus.FAILED

        except BudgetExhausted as e:
            run.status = RunStatus.PARTIAL
            run.stop_reason = e.reason

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            logger.exception(f"Resume failed: {e}")

        finally:
            run.completed_at = datetime.utcnow()

        return run

    async def _execute_node(
        self,
        run: Run,
        node: DAGNode,
        context: str,
    ) -> None:
        """Execute a single node in the DAG."""
        node.status = NodeStatus.STARTED
        node.start_time = datetime.utcnow()

        # Check budgets before execution
        self._check_budgets(run, node)

        try:
            # Check cache first
            cache_key = self._make_cache_key(node.task, context)
            node.cache_key = cache_key

            if self.cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    node.result = cached
                    node.cache_hit = True
                    node.status = NodeStatus.CACHE_HIT
                    logger.debug(f"Cache hit for node {node.node_id}")
                    return

            # Decide whether to decompose or answer directly
            should_decompose = (
                node.depth < run.config.budget.max_depth - 1
                and len(node.task) > 100  # Simple heuristic: longer tasks may need decomposition
            )

            if should_decompose:
                await self._decompose_and_execute(run, node, context)
            else:
                await self._execute_leaf(run, node, context)

            # Cache successful result
            if node.result and self.cache:
                await self.cache.set(cache_key, node.result)

        except BudgetExhausted:
            raise
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            logger.error(f"Node {node.node_id} failed: {e}")

        finally:
            node.end_time = datetime.utcnow()

    async def _decompose_and_execute(
        self,
        run: Run,
        node: DAGNode,
        context: str,
    ) -> None:
        """Decompose a node into subtasks and execute them."""
        # Decompose task
        subtasks = await self.llm.decompose_task(
            task=node.task,
            context=context,
            max_subtasks=run.config.budget.max_branching_factor,
        )

        if len(subtasks) <= 1:
            # No meaningful decomposition, execute as leaf
            await self._execute_leaf(run, node, context)
            return

        # Create child nodes
        child_results: list[tuple[str, str]] = []

        for subtask in subtasks:
            # Check budgets
            self._check_budgets(run, node)

            child = DAGNode(
                task=subtask,
                parent_id=node.node_id,
                depth=node.depth + 1,
            )
            run.add_node(child)
            node.children.append(child.node_id)

            # Execute child
            await self._execute_node(run, child, context)

            if child.status == NodeStatus.SUCCEEDED and child.result:
                child_results.append((subtask, child.result))

            # Update token count
            run.total_tokens += child.tokens_used

        # Synthesize results
        if child_results:
            synthesis = await self.llm.synthesize_results(
                task=node.task,
                subtask_results=child_results,
                context=context,
            )
            node.result = synthesis
            node.status = NodeStatus.SUCCEEDED
        else:
            node.status = NodeStatus.FAILED
            node.error = "No successful subtasks to synthesize"

    async def _execute_leaf(
        self,
        run: Run,
        node: DAGNode,
        context: str,
    ) -> None:
        """Execute a leaf node directly without decomposition."""
        # Select model tier based on depth
        tier = ModelTier.LEAF if node.depth >= 2 else ModelTier.WORKER

        answer, tokens = await self.llm.answer_task(
            task=node.task,
            context=context,
            tier=tier,
        )

        node.result = answer
        node.tokens_used = tokens
        node.status = NodeStatus.SUCCEEDED
        run.total_tokens += tokens

    def _check_budgets(self, run: Run, node: DAGNode) -> None:
        """Check all budget constraints and raise if exhausted."""
        budget = run.config.budget

        # Check depth
        if node.depth >= budget.max_depth:
            run.stop_reason = StopReason.BUDGET_DEPTH
            raise BudgetExhausted(
                StopReason.BUDGET_DEPTH,
                f"Max depth {budget.max_depth} reached"
            )

        # Check node count
        if len(run.nodes) >= budget.max_nodes:
            run.stop_reason = StopReason.BUDGET_NODES
            raise BudgetExhausted(
                StopReason.BUDGET_NODES,
                f"Max nodes {budget.max_nodes} reached"
            )

        # Check wall time
        if run.started_at:
            elapsed = (datetime.utcnow() - run.started_at).total_seconds()
            if elapsed >= budget.max_wall_time:
                run.stop_reason = StopReason.BUDGET_TIME
                raise BudgetExhausted(
                    StopReason.BUDGET_TIME,
                    f"Max wall time {budget.max_wall_time}s exceeded"
                )

        # Check tokens
        if run.total_tokens >= budget.max_tokens:
            run.stop_reason = StopReason.BUDGET_TOKENS
            raise BudgetExhausted(
                StopReason.BUDGET_TOKENS,
                f"Max tokens {budget.max_tokens} exceeded"
            )

    def _make_cache_key(self, task: str, context: str) -> str:
        """Generate a cache key for a task."""
        # Simple hash-based key for MVP
        import hashlib

        key_data = f"{task}::{context[:500] if context else ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def _retrieve_context(
        self,
        notebook_id: str,
        goal_spec: GoalSpec,
    ) -> RetrievalResult:
        """Retrieve relevant context from a notebook."""
        if not self.notebook_store:
            return RetrievalResult()

        # Query notebook store
        return await self.notebook_store.retrieve(
            notebook_id=notebook_id,
            query=goal_spec.normalized_goal,
            limit=10,
        )

    def _format_retrieval_context(self, retrieval: RetrievalResult) -> str:
        """Format retrieval results as context string."""
        if not retrieval.nodes:
            return ""

        parts = []
        for node in retrieval.nodes:
            if isinstance(node, Source):
                parts.append(f"[Source: {node.title}]\n{node.extracted_text or node.content}")
            elif isinstance(node, Note):
                parts.append(f"[Note: {node.title}]\n{node.content}")
            else:
                parts.append(f"[{node.title}]\n{node.content}")

        return "\n\n---\n\n".join(parts)
