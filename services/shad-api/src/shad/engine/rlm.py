"""RLM Engine - Recursive Language Model execution engine.

The core reasoning engine that:
1. Treats prompts as external environments
2. Recursively decomposes problems
3. Caches/verifies/recomposes results
4. Enforces budget constraints

Per OBSIDIAN_PIVOT.md Section 3: Code Execution (The RLM Pattern)
- RLM writes Python scripts that import MCP tools
- Scripts execute in sandboxed container with vault access
- Scripts filter, aggregate, process vault data before returning
- Only final distilled output enters context window
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from shad.engine.llm import LLMProvider, ModelTier
from shad.models.goal import GoalSpec
from shad.models.run import (
    DAGNode,
    NodeStatus,
    Run,
    RunConfig,
    RunStatus,
    StopReason,
)
from shad.utils.config import get_settings

if TYPE_CHECKING:
    from shad.cache.redis_cache import RedisCache
    from shad.mcp.client import ObsidianMCPClient
    from shad.sandbox.executor import CodeExecutor

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

    Per OBSIDIAN_PIVOT.md:
    - Uses Code Mode for vault operations
    - Supports Obsidian MCP client for vault access
    - Implements hash-based cache validation
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        cache: RedisCache | None = None,
        mcp_client: ObsidianMCPClient | None = None,
        code_executor: CodeExecutor | None = None,
    ):
        self.llm = llm_provider or LLMProvider()
        self.cache: RedisCache | None = cache
        self.mcp_client: ObsidianMCPClient | None = mcp_client
        self.code_executor: CodeExecutor | None = code_executor
        self.settings = get_settings()
        self._use_code_mode = mcp_client is not None

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

            # Retrieve context from Obsidian vault if specified
            context = ""
            if config.vault_path:
                logger.debug(f"[CONTEXT] Vault path: {config.vault_path}")
                if self.mcp_client:
                    context = await self._retrieve_vault_context(goal_spec.normalized_goal)
                    if context:
                        logger.debug(f"[CONTEXT] Context length: {len(context)} chars")
                    else:
                        logger.debug("[CONTEXT] No context retrieved from vault")
                else:
                    logger.debug("[CONTEXT] MCP client NOT available - cannot retrieve context")
            else:
                logger.debug("[CONTEXT] No vault_path specified")

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

    def _make_cache_key(
        self,
        task: str,
        context: str,
        context_hash: str | None = None,
    ) -> str:
        """Generate a cache key for a task.

        Per OBSIDIAN_PIVOT.md Section 6.2: Hash Validation
        Cache keys include context_hash derived from file content/mtime.
        """
        # Include context hash for cache coherence
        if context_hash:
            key_data = f"{task}::{context_hash}"
        else:
            key_data = f"{task}::{context[:500] if context else ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def _get_context_hash(self, query: str) -> str | None:
        """Get hash of context sources for cache validation.

        Per OBSIDIAN_PIVOT.md Section 6.2: Before cache lookup,
        query MCP server for current file hash.
        """
        if not self.mcp_client:
            return None

        try:
            # Get hashes of relevant files
            results = await self.mcp_client.search(query, limit=5)
            if not results:
                return None

            # Combine hashes
            hashes = []
            for result in results:
                file_hash = await self.mcp_client.get_file_hash(result.path)
                if file_hash:
                    hashes.append(file_hash)

            if hashes:
                combined = ":".join(sorted(hashes))
                return hashlib.sha256(combined.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to get context hash: {e}")

        return None

    async def _retrieve_vault_context(self, query: str, limit: int = 10) -> str:
        """Retrieve relevant context from the Obsidian vault.

        Uses the MCP client to search the vault and format results.
        """
        if not self.mcp_client:
            logger.warning("[RETRIEVAL] No MCP client configured")
            return ""

        logger.info(f"[RETRIEVAL] Searching vault for: {query[:100]}...")

        try:
            results = await self.mcp_client.search(query, limit=limit)
            if not results:
                logger.info("[RETRIEVAL] No results found")
                return ""

            logger.info(f"[RETRIEVAL] Found {len(results)} results")

            # Format results as context
            parts = []
            for result in results:
                # Create wikilink for citation per Section 4.3
                path = result.path
                if path.endswith(".md"):
                    path = path[:-3]
                link = f"[[{path}]]"

                title = result.title or result.path
                content = result.content[:2000] if result.content else ""

                parts.append(f"[{title}] ({link})\n{content}")

            return "\n\n---\n\n".join(parts)

        except Exception as e:
            logger.error(f"[RETRIEVAL] Failed: {e}")
            return ""

    async def _execute_code_mode(
        self,
        run: Run,
        node: DAGNode,
        script: str,
    ) -> str | None:
        """Execute a Code Mode script.

        Per OBSIDIAN_PIVOT.md Section 3.1:
        1. RLM writes Python script importing MCP tools
        2. Script executes in sandboxed container
        3. Script filters/aggregates data before returning
        4. Only final output enters context window
        """
        if not self.code_executor:
            logger.warning("Code executor not configured, falling back to direct execution")
            return None

        try:
            result = await self.code_executor.execute(script)

            if result.success:
                if result.return_value:
                    return str(result.return_value)
                return result.stdout
            else:
                logger.error(f"Code execution failed: {result.error_message}")
                node.error = f"Code execution failed: {result.error_message}"
                return None

        except Exception as e:
            logger.error(f"Code Mode execution error: {e}")
            return None

    def _generate_retrieval_script(self, query: str, vault_path: str) -> str:
        """Generate a Code Mode script for context retrieval.

        Per OBSIDIAN_PIVOT.md Section 3.2: Context Initialization
        - Runs start with empty context (goal text only)
        - Code Mode scripts fetch data dynamically
        """
        return f'''
from shad.sandbox.tools import obsidian

# Search for relevant notes in the vault
results = obsidian.search("{query}", limit=10)

# Extract and format relevant content
context_parts = []
for r in results:
    path = r.get("path", "")
    content = r.get("content", "")
    score = r.get("score", 0)

    if score > 0.1:  # Filter by relevance
        # Get note title from first heading or filename
        lines = content.split("\\n")
        title = path
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        context_parts.append({{
            "title": title,
            "path": path,
            "content": content[:2000],  # Limit content size
            "score": score,
        }})

# Return distilled context
__result__ = {{
    "vault_path": "{vault_path}",
    "results": context_parts,
    "total": len(context_parts),
}}
'''

    async def retrieve_context_code_mode(
        self,
        vault_path: str,
        query: str,
    ) -> str:
        """Retrieve context using Code Mode execution.

        Per OBSIDIAN_PIVOT.md Section 3.1 and 3.2:
        Uses code execution to fetch and filter vault data.
        """
        if not self.code_executor:
            return ""

        script = self._generate_retrieval_script(query, vault_path)

        try:
            result = await self.code_executor.execute(script)

            if result.success and result.return_value:
                data = result.return_value
                if isinstance(data, dict) and "results" in data:
                    parts = []
                    for item in data["results"]:
                        title = item.get("title", "Untitled")
                        content = item.get("content", "")
                        path = item.get("path", "")
                        # Use full-path wikilink per Section 4.3
                        link = f"[[{path[:-3] if path.endswith('.md') else path}]]"
                        parts.append(f"[{title}] ({link})\n{content}")
                    return "\n\n---\n\n".join(parts)

        except Exception as e:
            logger.error(f"Code Mode retrieval failed: {e}")

        return ""
