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

import asyncio
import hashlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from shad.engine.context_packets import NodeContextManager
from shad.engine.decomposition import StrategyDecomposer
from shad.engine.llm import LLMProvider, ModelTier
from shad.engine.strategies import StrategySelector, StrategyType, get_strategy
from shad.models.goal import GoalSpec
from shad.models.run import (
    DAGNode,
    NodeStatus,
    Run,
    RunConfig,
    RunStatus,
    StopReason,
)
from shad.output.manifest import FileEntry, FileManifest
from shad.refinement.manager import (
    DeltaVerifier,
    MaxIterationsPolicy,
    RunState,
    RunStateManager,
)
from shad.retrieval import RetrievalLayer, get_retriever
from shad.sandbox.executor import CodeExecutor, SandboxConfig
from shad.utils.config import get_settings
from shad.verification.layer import (
    VerificationConfig,
    VerificationLayer,
    VerificationLevel,
    VerificationResult,
)

if TYPE_CHECKING:
    from shad.cache.redis_cache import RedisCache

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
        retriever: RetrievalLayer | None = None,
        code_executor: CodeExecutor | None = None,
        vault_path: Path | str | None = None,
        collections: list[str] | None = None,
        use_code_mode: bool = True,
    ):
        self.llm = llm_provider or LLMProvider()
        self.cache: RedisCache | None = cache
        self.settings = get_settings()

        # Store vault path
        self.vault_path = Path(vault_path) if vault_path else None

        # Store collection names for multi-vault search
        self.collections = collections or []

        # Initialize retrieval layer
        if retriever:
            self.retriever = retriever
        else:
            # Auto-detect best available retriever
            collection_paths = {}
            if self.vault_path:
                collection_paths[self.vault_path.name] = self.vault_path
            self.retriever = get_retriever(
                paths=[self.vault_path] if self.vault_path else None,
                collection_names=collection_paths if collection_paths else None,
            )
        logger.info(f"[RETRIEVAL] Using {type(self.retriever).__name__}")

        # Initialize CodeExecutor for Code Mode
        self._use_code_mode = use_code_mode and self.vault_path is not None
        if code_executor:
            self.code_executor = code_executor
        elif self._use_code_mode and self.vault_path:
            sandbox_config = SandboxConfig(
                vault_path=self.vault_path,
            )
            self.code_executor = CodeExecutor(sandbox_config)
            logger.info(f"[CODE_MODE] Initialized CodeExecutor with vault: {self.vault_path}")
        else:
            self.code_executor = None

        # Initialize strategy selection and decomposition (Phase 3)
        self.strategy_selector = StrategySelector()
        self.decomposer = StrategyDecomposer(self.llm)

        # Initialize context packet management (Phase 3)
        self.context_manager = NodeContextManager()

        # Initialize verification layer (Phase 5)
        self.verification_layer = VerificationLayer()

        # Initialize refinement manager (Phase 6)
        self.delta_verifier = DeltaVerifier()
        self.state_manager = RunStateManager()
        self.iterations_policy = MaxIterationsPolicy()

    async def execute(self, config: RunConfig) -> Run:
        """
        Execute a run with the given configuration.

        This is the main entry point for reasoning tasks.
        """
        run = Run(config=config)
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(UTC)

        # Initialize fresh state manager for this run (Phase 6)
        self.state_manager = RunStateManager()
        self.state_manager.transition_to(RunState.RUNNING)

        logger.info(f"Starting run {run.run_id} with goal: {config.goal}")

        try:
            # Create goal spec
            goal_spec = GoalSpec.from_goal(config.goal)

            # Select strategy (Phase 3 integration)
            override_type = StrategyType(config.strategy_override) if config.strategy_override else None
            strategy_result = self.strategy_selector.select(config.goal, override=override_type)
            self._selected_strategy = get_strategy(strategy_result.strategy_type)
            logger.info(f"[STRATEGY] Selected: {strategy_result.strategy_type.value} "
                       f"(confidence: {strategy_result.confidence:.2f}, override: {strategy_result.is_override})")

            # Retrieve context from vault(s) if specified
            context = ""
            if config.vault_path or self.collections:
                logger.debug(f"[CONTEXT] Vault path: {config.vault_path}, collections: {self.collections}")
                if self.retriever.available:
                    context = await self._retrieve_vault_context(goal_spec.normalized_goal)
                    if context:
                        logger.debug(f"[CONTEXT] Context length: {len(context)} chars")
                    else:
                        logger.debug("[CONTEXT] No context retrieved from vault")
                else:
                    logger.debug("[CONTEXT] Retriever NOT available - cannot retrieve context")
            else:
                logger.debug("[CONTEXT] No vault_path or collections specified")

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
            elif root_node.status in (NodeStatus.SUCCEEDED, NodeStatus.CACHE_HIT):
                run.status = RunStatus.COMPLETE
                run.final_result = root_node.result

                # Generate file manifest for software strategy (Phase 4)
                if (hasattr(self, '_selected_strategy') and
                    self._selected_strategy and
                    self._selected_strategy.strategy_type == StrategyType.SOFTWARE):

                    # Collect results from all nodes
                    results = {n.node_id: n.result for n in run.nodes.values()
                              if n.result and n.status == NodeStatus.SUCCEEDED}
                    manifest = self.generate_manifest(run, results)

                    if manifest.files:
                        logger.info(f"[MANIFEST] Generated manifest with {len(manifest.files)} files")

                        # Verify manifest (Phase 5)
                        verify_level_str = config.verify_level or "basic"
                        verify_level = VerificationLevel(verify_level_str)
                        verification_result = await self.verify_manifest(manifest, verify_level)

                        if verification_result.passed:
                            logger.info("[VERIFICATION] All checks passed")
                            run.metadata["manifest"] = manifest.to_dict()
                            try:
                                self.state_manager.transition_to(RunState.SUCCESS)
                            except ValueError:
                                pass
                        else:
                            logger.warning(f"[VERIFICATION] Failed: {len(verification_result.failed_checks)} check(s)")
                            for check in verification_result.failed_checks:
                                logger.warning(f"  - {check.check_name}: {check.errors}")

                            # Attempt repair loop (Phase 6)
                            blocking_failures = verification_result.blocking_failures
                            if blocking_failures:
                                repaired_manifest = await self._attempt_repair(
                                    run=run,
                                    manifest=manifest,
                                    failures=blocking_failures,
                                    verify_level=verify_level,
                                )
                                if repaired_manifest:
                                    manifest = repaired_manifest
                                    # Re-verify after repair
                                    verification_result = await self.verify_manifest(manifest, verify_level)
                                    if verification_result.passed:
                                        logger.info("[REPAIR] Repair successful, all checks now pass")
                                        run.metadata["manifest"] = manifest.to_dict()
                                        run.metadata["repaired"] = True
                                    else:
                                        blocking_failures = verification_result.blocking_failures

                            # Store manifest with verification status
                            run.metadata["manifest"] = manifest.to_dict()
                            run.metadata["verification"] = {
                                "passed": verification_result.passed,
                                "blocking_failure_count": len(blocking_failures),
                                "blocking_failures": [c.to_dict() for c in blocking_failures],
                                "checks": [c.to_dict() for c in verification_result.checks],
                            }

                            # Use MaxIterationsPolicy to determine final state (Phase 6)
                            if blocking_failures:
                                final_state = self.iterations_policy.determine_final_state(
                                    is_high_impact=len(manifest.files) > 3,
                                    has_artifacts=len(manifest.files) > 0,
                                    verification_passed=False,
                                    verification_advisory=False,
                                )
                                # Map refinement state to run status
                                if final_state == RunState.NEEDS_HUMAN:
                                    run.status = RunStatus.PARTIAL
                                    run.metadata["needs_human"] = True
                                    run.error = f"Verification failed: {len(blocking_failures)} blocking failure(s). Human review needed."
                                elif final_state == RunState.PARTIAL:
                                    run.status = RunStatus.PARTIAL
                                    run.error = f"Verification failed: {len(blocking_failures)} blocking failure(s)"
                                else:
                                    run.status = RunStatus.FAILED
                                    run.error = f"Verification failed: {len(blocking_failures)} blocking failure(s)"

                                # Track state transition
                                try:
                                    self.state_manager.transition_to(final_state)
                                except ValueError:
                                    pass  # State transition may fail if already in final state
            else:
                run.status = RunStatus.FAILED
                run.error = root_node.error
                try:
                    self.state_manager.transition_to(RunState.FAILED)
                except ValueError:
                    pass

        except BudgetExhausted as e:
            run.status = RunStatus.PARTIAL
            run.stop_reason = e.reason
            logger.warning(f"Run {run.run_id} stopped: {e}")
            try:
                self.state_manager.transition_to(RunState.PARTIAL)
            except ValueError:
                pass

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            logger.exception(f"Run {run.run_id} failed: {e}")
            try:
                self.state_manager.transition_to(RunState.FAILED)
            except ValueError:
                pass

        finally:
            run.completed_at = datetime.now(UTC)
            # Store final refinement state in metadata
            run.metadata["refinement_state"] = self.state_manager.state.value

        return run

    async def resume(self, run: Run, replay_mode: str | None = None) -> Run:
        """
        Resume a partial or failed run.

        Per SPEC.md Section 2.8.2: Uses delta verification.
        - Checks vault manifest against stored hashes
        - Stale nodes undergo re-verification or re-execution
        - Unchanged nodes are trusted

        Args:
            run: The run to resume
            replay_mode: Optional replay mode ('stale', 'node_id', 'subtree:node_id')
        """
        if run.status not in (RunStatus.PARTIAL, RunStatus.FAILED):
            raise ValueError(f"Cannot resume run with status {run.status}")

        run.status = RunStatus.RUNNING
        run.stop_reason = None
        run.error = None

        # Re-initialize state manager for resume
        self.state_manager = RunStateManager()
        self.state_manager.transition_to(RunState.RUNNING)

        logger.info(f"Resuming run {run.run_id}")

        try:
            # Determine which nodes to replay based on delta verification
            nodes_to_replay: list[str] = []

            if replay_mode == "stale":
                # Get current vault hashes and find stale nodes
                current_hashes = await self._get_current_vault_hashes(run)
                nodes_to_replay = self.delta_verifier.get_stale_nodes(current_hashes)
                logger.info(f"[DELTA] Found {len(nodes_to_replay)} stale nodes")
            elif replay_mode and replay_mode.startswith("subtree:"):
                # Replay specific subtree
                root_id = replay_mode.split(":")[1]
                self.delta_verifier.mark_subtree_for_replay(root_id)
                nodes_to_replay = self.delta_verifier.get_nodes_to_replay()
            elif replay_mode:
                # Replay specific node
                nodes_to_replay = [replay_mode]
            else:
                # Default: replay pending nodes
                nodes_to_replay = [n.node_id for n in run.pending_nodes()]

            # If no nodes to replay, check root
            if not nodes_to_replay and run.root_node_id:
                root = run.get_node(run.root_node_id)
                if root and root.status != NodeStatus.SUCCEEDED:
                    nodes_to_replay = [run.root_node_id]

            # Execute nodes to replay
            for node_id in nodes_to_replay:
                node = run.get_node(node_id)
                if not node:
                    continue

                # Reset node status for re-execution
                node.status = NodeStatus.CREATED
                node.result = None
                node.error = None

                context = ""  # Retrieve fresh context
                if self.retriever.available:
                    context = await self._retrieve_vault_context(node.task, limit=5)

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
            run.completed_at = datetime.now(UTC)

        return run

    async def _execute_node(
        self,
        run: Run,
        node: DAGNode,
        context: str,
    ) -> None:
        """Execute a single node in the DAG."""
        node.status = NodeStatus.STARTED
        node.start_time = datetime.now(UTC)

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

            # Track context for delta verification (Phase 6)
            if node.status == NodeStatus.SUCCEEDED and context:
                self._track_node_context(node.node_id, context)

        except BudgetExhausted:
            raise
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error = str(e)
            logger.error(f"Node {node.node_id} failed: {e}")

        finally:
            node.end_time = datetime.now(UTC)

    async def _decompose_and_execute(
        self,
        run: Run,
        node: DAGNode,
        context: str,
    ) -> None:
        """Decompose a node into subtasks and execute them.

        Per SPEC.md: Uses StrategyDecomposer for dependency-aware DAG.
        """
        # Use strategy-aware decomposition if strategy is selected (Phase 3)
        if hasattr(self, '_selected_strategy') and self._selected_strategy:
            decomp_result = await self.decomposer.decompose(
                task=node.task,
                strategy=self._selected_strategy,
                context=context,
                max_nodes=run.config.budget.max_branching_factor,
            )

            if not decomp_result.is_valid:
                logger.warning(f"[DECOMPOSE] Validation errors: {decomp_result.validation_errors}")

            # Convert DecompositionNodes to subtasks
            subtasks = [(dn.stage_name, dn.task, dn.hard_deps, dn.soft_deps)
                       for dn in decomp_result.nodes]
            run.total_tokens += decomp_result.tokens_used
        else:
            # Fallback to simple decomposition
            simple_subtasks = await self.llm.decompose_task(
                task=node.task,
                context=context,
                max_subtasks=run.config.budget.max_branching_factor,
            )
            subtasks = [(f"subtask_{i}", st, [], []) for i, st in enumerate(simple_subtasks)]

        if len(subtasks) <= 1:
            # No meaningful decomposition, execute as leaf
            await self._execute_leaf(run, node, context)
            return

        # Phase 1: Create all child nodes and build dependency graph
        stage_to_node: dict[str, str] = {}  # stage_name -> node_id
        node_to_stage: dict[str, str] = {}  # node_id -> stage_name
        pending_nodes: dict[str, tuple[str, list[str], list[str]]] = {}  # node_id -> (task, hard_deps, soft_deps)

        for stage_name, task, hard_deps, soft_deps in subtasks:
            self._check_budgets(run, node)

            child = DAGNode(
                task=task,
                parent_id=node.node_id,
                depth=node.depth + 1,
                metadata={
                    "stage_name": stage_name,
                    "hard_deps": hard_deps,
                    "soft_deps": soft_deps,
                }
            )
            run.add_node(child)
            node.children.append(child.node_id)
            stage_to_node[stage_name] = child.node_id
            node_to_stage[child.node_id] = stage_name
            pending_nodes[child.node_id] = (task, hard_deps, soft_deps)

        # Phase 2: Execute nodes in parallel waves based on dependency satisfaction
        completed_stages: set[str] = set()
        child_results: list[tuple[str, str]] = []

        while pending_nodes:
            # Find nodes whose hard dependencies are all satisfied
            ready_nodes: list[str] = []
            for node_id, (_task, hard_deps, _soft_deps) in pending_nodes.items():
                if all(dep in completed_stages for dep in hard_deps):
                    ready_nodes.append(node_id)

            if not ready_nodes:
                # No nodes ready - circular dependency or all remaining have unmet deps
                logger.warning("[PARALLEL] No ready nodes - possible circular dependency")
                break

            logger.info(f"[PARALLEL] Executing {len(ready_nodes)} nodes in parallel")

            # Execute ready nodes in parallel
            async def execute_single_node(nid: str) -> tuple[str, DAGNode]:
                """Execute a single node and return (node_id, node)."""
                child = run.get_node(nid)
                if not child:
                    return (nid, child)  # type: ignore

                task, hard_deps, soft_deps = pending_nodes[nid]

                # Inject context from soft dependencies
                subtask_context = context
                if soft_deps:
                    soft_dep_context = self.context_manager.inject_soft_dep_context(
                        soft_deps=soft_deps,
                        task=task,
                    )
                    if soft_dep_context:
                        subtask_context = f"{soft_dep_context}\n\n{context}"
                        logger.info(f"[CONTEXT] Injected {len(soft_dep_context)} chars from soft deps: {soft_deps}")

                # Retrieve subtask-specific context from vault
                if self.retriever.available:
                    fresh_context = await self._retrieve_vault_context(task, limit=5)
                    if fresh_context:
                        logger.info(f"[CONTEXT] Retrieved {len(fresh_context)} chars for subtask: {task[:50]}...")
                        subtask_context = f"{subtask_context}\n\n{fresh_context}"

                # Execute the node
                await self._execute_node(run, child, subtask_context)
                return (nid, child)

            # Run all ready nodes in parallel
            results = await asyncio.gather(*[execute_single_node(nid) for nid in ready_nodes])

            # Process results
            for node_id, child in results:
                if child is None:
                    continue

                stage_name = node_to_stage[node_id]

                # Create context packet for completed node
                if child.status == NodeStatus.SUCCEEDED and child.result:
                    packet = self.context_manager.create_packet(
                        node_id=child.node_id,
                        stage_name=stage_name,
                        result=child.result,
                        artifacts=child.metadata.get("artifacts", []),
                    )
                    self.context_manager.store.add(packet)
                    logger.info(f"[CONTEXT_PACKET] Created packet for {stage_name}: {len(packet.keywords)} keywords")

                    child_results.append((child.task, child.result))

                # Update token count
                run.total_tokens += child.tokens_used

                # Mark stage as completed
                completed_stages.add(stage_name)

                # Remove from pending
                del pending_nodes[node_id]

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
            elapsed = (datetime.now(UTC) - run.started_at).total_seconds()
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

        Searches for relevant documents and hashes their content to detect changes.
        """
        if not self.retriever.available:
            return None

        try:
            # Get relevant results
            results = await self.retriever.search(
                query,
                mode="bm25",  # Fast keyword search for hash computation
                collections=self.collections if self.collections else None,
                limit=5,
            )
            if not results:
                return None

            # Combine content hashes
            hashes = []
            for result in results:
                # Hash the content (or docid if available)
                content_hash = hashlib.sha256(
                    result.content.encode() if result.content else result.docid.encode()
                ).hexdigest()[:8]
                hashes.append(content_hash)

            if hashes:
                combined = ":".join(sorted(hashes))
                return hashlib.sha256(combined.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to get context hash: {e}")

        return None

    async def _retrieve_vault_context(self, query: str, limit: int = 10) -> str:
        """Retrieve relevant context from the vault(s).

        Uses the configured retrieval backend (qmd or filesystem).
        If Code Mode is enabled, tries LLM-generated scripts first.
        """
        if not self.retriever.available:
            logger.warning("[RETRIEVAL] No retriever available")
            return ""

        # Try Code Mode first if enabled
        if self._use_code_mode and self.code_executor:
            context = await self._retrieve_via_code_mode(query)
            if context:
                return context
            logger.warning("[RETRIEVAL] Code Mode failed, falling back to direct search")

        # Fallback: Direct retriever search
        return await self._retrieve_direct(query, limit)

    async def _retrieve_via_code_mode(self, query: str) -> str:
        """Retrieve context using LLM-generated Code Mode script.

        Per OBSIDIAN_PIVOT.md Section 3.1:
        1. LLM generates Python script for this specific query
        2. Script executes in sandboxed environment with vault access
        3. Script filters/aggregates data before returning
        4. Only final distilled output enters context window
        """
        if not self.code_executor:
            return ""

        logger.info(f"[CODE_MODE] Generating retrieval script for: {query[:80]}...")

        try:
            # Have LLM generate a custom retrieval script
            script, tokens = await self.llm.generate_retrieval_script(
                task=query,
                vault_info=f"vault_path: {self.vault_path}" if self.vault_path else "",
            )

            if not script:
                logger.warning("[CODE_MODE] LLM generated empty script")
                return ""

            logger.info(f"[CODE_MODE] Executing generated script ({len(script)} chars)...")
            logger.debug(f"[CODE_MODE] Script:\n{script}")

            # Execute the script in sandbox
            result = await self.code_executor.execute(script)

            if result.success:
                if result.return_value:
                    context = str(result.return_value)
                    logger.info(f"[CODE_MODE] Script returned {len(context)} chars via __result__")
                    return context
                elif result.stdout:
                    logger.info(f"[CODE_MODE] Script returned {len(result.stdout)} chars via stdout")
                    return result.stdout
                else:
                    logger.warning("[CODE_MODE] Script succeeded but returned no output")
                    return ""
            else:
                logger.error(f"[CODE_MODE] Script execution failed: {result.error_message}")
                logger.error(f"[CODE_MODE] stderr: {result.stderr}")
                return ""

        except Exception as e:
            logger.error(f"[CODE_MODE] Error: {e}")
            return ""

    async def _retrieve_direct(self, query: str, limit: int = 10) -> str:
        """Direct retriever search fallback (non-Code Mode).

        Uses the configured retriever (qmd or filesystem) to search and format results.
        Extracts content around matched keywords rather than from file start.
        """
        if not self.retriever.available:
            return ""

        # Extract keywords from full query (removes stop words, markdown, etc.)
        search_query = self._extract_search_keywords(query)
        if not search_query:
            logger.warning("[RETRIEVAL] No keywords extracted from query")
            return ""

        retriever_name = type(self.retriever).__name__
        logger.info(f"[RETRIEVAL] Direct search via {retriever_name} for: {search_query}")

        try:
            # Search using the retrieval layer with extracted keywords
            # Note: hybrid mode requires GPU for LLM reranking
            results = await self.retriever.search(
                search_query,
                mode="bm25",  # Keyword search (no GPU required)
                collections=self.collections if self.collections else None,
                limit=limit,
            )

            if not results:
                logger.info("[RETRIEVAL] No results found")
                return ""

            logger.info(f"[RETRIEVAL] Found {len(results)} results")

            # Format results as context
            parts = []
            for result in results:
                # Create wikilink for citation
                path = result.path
                if path.endswith(".md"):
                    path = path[:-3]

                # Include collection name if multi-vault
                if result.collection:
                    link = f"[[{result.collection}:{path}]]"
                    title = f"{result.collection}/{result.path}"
                else:
                    link = f"[[{path}]]"
                    title = result.path

                # Extract content around matched keywords instead of file start
                content = self._extract_relevant_sections(
                    result.content, query, max_chars=4000
                ) if result.content else ""

                parts.append(f"[{title}] ({link})\n{content}")

            return "\n\n---\n\n".join(parts)

        except Exception as e:
            logger.error(f"[RETRIEVAL] Failed: {e}")
            return ""

    # Common stop words for keyword extraction
    STOP_WORDS = frozenset({
        # Standard stop words
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'do', 'does', 'did', 'have', 'has', 'had', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'we', 'how', 'what',
        'which', 'who', 'when', 'where', 'why', 'and', 'or', 'not',
        'it', 'its', 'this', 'that', 'these', 'those', 'can', 'could',
        'would', 'should', 'will', 'may', 'might', 'must', 'shall',
        'into', 'onto', 'upon', 'about', 'after', 'before', 'between',
        'through', 'during', 'under', 'over', 'above', 'below',
        'all', 'any', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'than', 'too', 'very', 'just', 'also',
        # Task/goal meta-words (describe what to do, not what to search for)
        'goal', 'goals', 'task', 'tasks', 'run', 'runs', 'statement',
        'build', 'create', 'make', 'implement', 'add', 'update', 'fix',
        'write', 'develop', 'design', 'setup', 'configure', 'enable',
        'use', 'using', 'need', 'needs', 'want', 'wants', 'please',
        'help', 'ensure', 'provide', 'include', 'support', 'allow',
        'following', 'based', 'like', 'new', 'existing', 'current',
    })

    def _extract_search_keywords(self, query: str, max_keywords: int = 8) -> str:
        """Extract search keywords from a goal/task description.

        Removes stop words, short words, and markdown formatting to create
        a clean search query for qmd.

        Args:
            query: Full goal or task description (may be multi-line)
            max_keywords: Maximum number of keywords to return

        Returns:
            Space-separated keywords suitable for search
        """
        # Remove markdown formatting
        clean = re.sub(r'#+ ', '', query)  # Headers
        clean = re.sub(r'\*\*|__|\*|_', '', clean)  # Bold/italic
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)  # Links
        clean = re.sub(r'```[\s\S]*?```', '', clean)  # Code blocks
        clean = re.sub(r'`[^`]+`', '', clean)  # Inline code

        # Extract words
        words = re.split(r'\W+', clean.lower())

        # Filter: remove stop words, short words, and duplicates while preserving order
        seen: set[str] = set()
        keywords: list[str] = []
        for w in words:
            if (w and len(w) > 2 and w not in self.STOP_WORDS and w not in seen):
                seen.add(w)
                keywords.append(w)
                if len(keywords) >= max_keywords:
                    break

        return " ".join(keywords)

    def _extract_relevant_sections(
        self, content: str, query: str, max_chars: int = 4000, context_lines: int = 10
    ) -> str:
        """Extract sections of content that contain query keywords.

        Instead of taking the first N chars, finds sections around keyword matches.
        """
        # Extract keywords using shared method
        keywords = self._extract_search_keywords(query).split()

        if not keywords:
            return content[:max_chars]

        lines = content.split('\n')

        # Find lines containing keywords
        matched_indices: set[int] = set()
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                matched_indices.add(i)

        if not matched_indices:
            return content[:max_chars]

        # Expand to include context around matches
        expanded_indices: set[int] = set()
        for idx in matched_indices:
            for offset in range(-context_lines, context_lines + 1):
                new_idx = idx + offset
                if 0 <= new_idx < len(lines):
                    expanded_indices.add(new_idx)

        # Build sections from contiguous line groups
        sorted_indices = sorted(expanded_indices)
        sections: list[str] = []
        current_section: list[str] = []
        last_idx = -999

        for idx in sorted_indices:
            if idx > last_idx + 1 and current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(lines[idx])
            last_idx = idx

        if current_section:
            sections.append('\n'.join(current_section))

        # Join sections with separator, respecting max_chars
        result_parts: list[str] = []
        total_chars = 0
        for section in sections:
            if total_chars + len(section) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    result_parts.append(section[:remaining] + "...")
                break
            result_parts.append(section)
            total_chars += len(section) + 5  # +5 for separator

        return "\n\n[...]\n\n".join(result_parts) if result_parts else content[:max_chars]

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

    def generate_manifest(
        self,
        run: Run,
        results: dict[str, str],
    ) -> FileManifest:
        """Generate a file manifest from run results.

        Per SPEC.md Section 2.7:
        - Code generation produces file manifests (not raw text)
        - Manifests contain paths, content, language, hash, source_nodes

        Args:
            run: The completed run
            results: Map of node_id -> result content

        Returns:
            FileManifest with generated files
        """
        manifest = FileManifest(run_id=run.run_id)

        for node_id, result in results.items():
            node = run.get_node(node_id)
            if not node:
                continue

            # Extract file entries from result
            # Look for code blocks with file paths
            files = self._extract_files_from_result(result, node_id)
            for entry in files:
                try:
                    manifest.add_file(entry)
                except ValueError:
                    # File already exists, update it
                    manifest.update_file(entry)

        return manifest

    def _extract_files_from_result(
        self,
        result: str,
        source_node: str,
    ) -> list[FileEntry]:
        """Extract file entries from a result string.

        Looks for code blocks with file paths in comments or headers.
        """
        import re

        files: list[FileEntry] = []

        # Pattern: ```language\n// filepath: path\n...code...```
        pattern = r'```(\w+)\s*\n(?:\/\/\s*filepath:\s*|#\s*filepath:\s*)?([^\n]*)\n(.*?)```'

        for match in re.finditer(pattern, result, re.DOTALL):
            language = match.group(1)
            path_hint = match.group(2).strip()
            content = match.group(3)

            # Validate path_hint looks like a file path (has extension, no code syntax)
            def is_valid_path(p: str) -> bool:
                if not p:
                    return False
                # Must have a file extension
                if not re.search(r'\.\w+$', p):
                    return False
                # Should not contain code syntax indicators
                if any(c in p for c in ['{', '}', '(', ')', '=', ';', ',']):
                    return False
                # Should not start with common code keywords
                code_keywords = ['export', 'import', 'const', 'let', 'var', 'function',
                                 'class', 'interface', 'type', 'def', 'async', 'return']
                first_word = p.split()[0].lower() if p.split() else ''
                if first_word in code_keywords:
                    return False
                return True

            # Try to extract path from first comment if path_hint is invalid
            if not is_valid_path(path_hint):
                path_hint = ''
                # Look for path in content
                path_match = re.search(r'(?:\/\/|#)\s*(?:file|path)?:?\s*(\S+\.\w+)', content)
                if path_match:
                    path_hint = path_match.group(1)

            if path_hint and is_valid_path(path_hint):
                files.append(FileEntry(
                    path=path_hint,
                    content=content,
                    language=language,
                    source_nodes=[source_node],
                ))

        return files

    async def verify_manifest(
        self,
        manifest: FileManifest,
        level: VerificationLevel = VerificationLevel.BASIC,
    ) -> VerificationResult:
        """Verify a file manifest.

        Per SPEC.md Section 2.6:
        - Run verification checks on manifest
        - Return structured result with pass/fail status

        Args:
            manifest: The file manifest to verify
            level: Verification level (off, basic, build, strict)

        Returns:
            VerificationResult with check results
        """
        config = VerificationConfig(level=level)
        return await self.verification_layer.verify(manifest, config)

    async def _attempt_repair(
        self,
        run: Run,
        manifest: FileManifest,
        failures: list,  # List of CheckResult
        verify_level: VerificationLevel,
        max_local_retries: int = 2,
        max_escalations: int = 10,
    ) -> FileManifest | None:
        """Attempt to repair verification failures.

        Per SPEC.md Section 2.6.3:
        - Syntax/lint: Local repair only
        - Type errors: Local repair with sibling context
        - Integration: Escalate to parent coordinator
        - Contract mismatch: Parent coordination + contract update

        Returns repaired manifest if successful, None otherwise.
        """
        from shad.verification.layer import ErrorClassification, RepairAction

        # Track repair attempts
        local_retries: dict[str, int] = {}  # file_path -> retry count
        escalation_count = 0

        for failure in failures:
            for error in failure.errors:
                # Classify error
                classification = ErrorClassification.classify(failure.check_name, error)
                repair_action = RepairAction.for_classification(classification)

                logger.info(f"[REPAIR] Error classified as {classification.value}, "
                           f"action: {repair_action.scope}")

                if repair_action.scope == "escalate":
                    escalation_count += 1
                    if escalation_count > max_escalations:
                        logger.warning(f"[REPAIR] Max escalations ({max_escalations}) reached")
                        return None
                    # For escalations, we'd need to re-run the parent node with error context
                    # For now, log and continue
                    logger.warning(f"[REPAIR] Escalation needed for: {error}")
                    continue

                # Local repair - extract file path from error
                file_path = self._extract_file_from_error(error)
                if not file_path:
                    continue

                # Check retry count
                if file_path not in local_retries:
                    local_retries[file_path] = 0
                if local_retries[file_path] >= max_local_retries:
                    logger.warning(f"[REPAIR] Max local retries for {file_path}")
                    continue

                local_retries[file_path] += 1

                # Get the file entry
                file_entry = manifest.get_file(file_path)
                if not file_entry:
                    continue

                # Build repair context
                repair_context = f"Error in {file_path}:\n{error}\n\n"
                if repair_action.needs_sibling_context:
                    # Include contracts/types context
                    sibling_context = self._get_sibling_context(manifest, file_path)
                    repair_context += f"Related files context:\n{sibling_context}\n\n"

                repair_context += f"Original code:\n{file_entry.content}\n\n"
                repair_context += "Please fix the error and provide the corrected code."

                # Call LLM for repair
                try:
                    repaired_content, tokens = await self.llm.answer_task(
                        task=f"Fix the following code error:\n{repair_context}",
                        context="",
                    )
                    run.total_tokens += tokens

                    # Extract code from response
                    repaired_code = self._extract_code_from_response(repaired_content, file_entry.language)
                    if repaired_code:
                        # Update manifest with repaired code
                        new_entry = FileEntry(
                            path=file_entry.path,
                            content=repaired_code,
                            language=file_entry.language,
                            source_nodes=file_entry.source_nodes + ["repair"],
                        )
                        manifest.update_file(new_entry)
                        logger.info(f"[REPAIR] Updated {file_path} with repaired code")

                except Exception as e:
                    logger.error(f"[REPAIR] Failed to repair {file_path}: {e}")

        return manifest

    def _extract_file_from_error(self, error: str) -> str | None:
        """Extract file path from error message."""
        import re
        # Match patterns like "path/to/file.py:10:" or "path/to/file.ts: error"
        match = re.search(r'([^\s:]+\.\w+)(?::\d+)?:', error)
        if match:
            return match.group(1)
        return None

    def _get_sibling_context(self, manifest: FileManifest, file_path: str) -> str:
        """Get context from sibling files (contracts/types)."""
        context_parts = []
        for entry in manifest.files:
            # Include type/contract files
            if entry.path != file_path:
                if any(kw in entry.path.lower() for kw in ["types", "contract", "interface", "schema"]):
                    context_parts.append(f"// {entry.path}\n{entry.content[:2000]}")
        return "\n\n".join(context_parts[:3])  # Limit to 3 files

    def _extract_code_from_response(self, response: str, language: str) -> str | None:
        """Extract code block from LLM response."""
        import re
        # Match code block with specified language
        pattern = rf'```(?:{language}|{language[:2]})?\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try generic code block
        match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    async def _get_current_vault_hashes(self, run: Run) -> dict[str, str]:
        """Get current hashes for vault files used in the run.

        Used for delta verification during resume.
        """
        hashes: dict[str, str] = {}

        if not self.retriever.available:
            return hashes

        # Get all used notes from tracked nodes
        for node_id in run.nodes:
            info = self.delta_verifier.get_node_info(node_id)
            if info:
                for path in info.used_notes:
                    if path not in hashes:
                        # Get content and hash it
                        content = await self.retriever.get(path)
                        if content:
                            file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                            hashes[path] = file_hash

        return hashes

    def _track_node_context(
        self,
        node_id: str,
        context: str,
        vault_path: str | None = None,
    ) -> None:
        """Track context used by a node for delta verification.

        Per SPEC.md Section 2.8.2: Store per completed node:
        - used_notes[]
        - used_note_hashes{}
        - subset_fingerprint
        """
        import re

        # Extract note paths from context (wikilink format)
        used_notes: list[str] = []
        note_hashes: dict[str, str] = {}

        # Find wikilinks: [[path/to/note]]
        wikilinks = re.findall(r'\[\[([^\]]+)\]\]', context)
        for link in wikilinks:
            path = link if link.endswith(".md") else f"{link}.md"
            used_notes.append(path)
            # Generate simple hash from context portion
            note_hashes[path] = hashlib.sha256(path.encode()).hexdigest()[:16]

        if used_notes:
            self.delta_verifier.track_node(
                node_id=node_id,
                used_notes=used_notes,
                note_hashes=note_hashes,
            )
