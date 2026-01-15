"""History Manager - Structured run artifacts storage.

The "black box flight recorder" for debugging, replay, and learning.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shad.models.run import DAGNode, NodeStatus, Run, RunStatus
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Manages structured, append-only run artifacts.

    Directory structure per run:
        History/Runs/<run_id>/
        ├── run.manifest.json      # Inputs, versions, config hashes
        ├── events.jsonl           # Node lifecycle events
        ├── dag.json               # DAG structure with statuses
        ├── decisions/
        │   ├── routing.json       # Skill routing decision
        │   └── decomposition/     # Per-node decomposition decisions
        ├── metrics/
        │   ├── nodes.jsonl        # Per-node metrics
        │   └── summary.json       # Rollup metrics
        ├── errors/                # Error records with context
        ├── artifacts/             # Large payloads (referenced by hash)
        ├── replay/
        │   └── manifest.json      # Deterministic replay bundle
        ├── final.report.md        # Human-readable output
        └── final.summary.json     # Machine-readable output
    """

    def __init__(self, base_path: Path | None = None):
        settings = get_settings()
        self.base_path = base_path or settings.history_path
        self.runs_path = self.base_path / "Runs"
        self.runs_path.mkdir(parents=True, exist_ok=True)

    def get_run_path(self, run_id: str) -> Path:
        """Get the path for a run's artifacts."""
        return self.runs_path / run_id

    def save_run(self, run: Run) -> Path:
        """
        Save a complete run with all artifacts.

        Returns the path to the run directory.
        """
        run_path = self.get_run_path(run.run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_path / "decisions").mkdir(exist_ok=True)
        (run_path / "decisions" / "decomposition").mkdir(exist_ok=True)
        (run_path / "metrics").mkdir(exist_ok=True)
        (run_path / "errors").mkdir(exist_ok=True)
        (run_path / "artifacts").mkdir(exist_ok=True)
        (run_path / "replay").mkdir(exist_ok=True)

        # Save manifest
        self._save_manifest(run, run_path)

        # Save DAG
        self._save_dag(run, run_path)

        # Save metrics
        self._save_metrics(run, run_path)

        # Save final outputs
        self._save_final_report(run, run_path)
        self._save_final_summary(run, run_path)

        # Save replay manifest
        self._save_replay_manifest(run, run_path)

        logger.info(f"Saved run {run.run_id} to {run_path}")
        return run_path

    def load_run(self, run_id: str) -> Run:
        """
        Load a run from history.

        Raises FileNotFoundError if run doesn't exist.
        """
        run_path = self.get_run_path(run_id)

        if not run_path.exists():
            raise FileNotFoundError(f"Run {run_id} not found")

        # Load manifest
        manifest_path = run_path / "run.manifest.json"
        with manifest_path.open() as f:
            manifest = json.load(f)

        # Load DAG
        dag_path = run_path / "dag.json"
        with dag_path.open() as f:
            dag_data = json.load(f)

        # Reconstruct Run object
        from shad.models import Budget, RunConfig

        config = RunConfig(
            goal=manifest["config"]["goal"],
            vault_path=manifest["config"].get("vault_path"),
            budget=Budget(**manifest["config"].get("budget", {})),
            voice=manifest["config"].get("voice"),
        )

        run = Run(
            run_id=run_id,
            config=config,
            status=RunStatus(manifest["status"]),
            root_node_id=dag_data.get("root_node_id"),
            total_tokens=manifest.get("total_tokens", 0),
        )

        # Parse timestamps
        if manifest.get("created_at"):
            run.created_at = datetime.fromisoformat(manifest["created_at"])
        if manifest.get("started_at"):
            run.started_at = datetime.fromisoformat(manifest["started_at"])
        if manifest.get("completed_at"):
            run.completed_at = datetime.fromisoformat(manifest["completed_at"])

        run.stop_reason = manifest.get("stop_reason")
        run.error = manifest.get("error")
        run.final_result = manifest.get("final_result")

        # Load nodes
        for node_data in dag_data.get("nodes", []):
            node = DAGNode(
                node_id=node_data["node_id"],
                parent_id=node_data.get("parent_id"),
                depth=node_data.get("depth", 0),
                task=node_data["task"],
                status=NodeStatus(node_data["status"]),
                result=node_data.get("result"),
                children=node_data.get("children", []),
                cache_key=node_data.get("cache_key"),
                cache_hit=node_data.get("cache_hit", False),
                tokens_used=node_data.get("tokens_used", 0),
                error=node_data.get("error"),
            )

            if node_data.get("start_time"):
                node.start_time = datetime.fromisoformat(node_data["start_time"])
            if node_data.get("end_time"):
                node.end_time = datetime.fromisoformat(node_data["end_time"])

            run.add_node(node)

        return run

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent runs with basic metadata."""
        runs = []

        for run_dir in sorted(self.runs_path.iterdir(), reverse=True)[:limit]:
            if not run_dir.is_dir():
                continue

            manifest_path = run_dir / "run.manifest.json"
            if not manifest_path.exists():
                continue

            try:
                with manifest_path.open() as f:
                    manifest = json.load(f)
                runs.append({
                    "run_id": run_dir.name,
                    "goal": manifest.get("config", {}).get("goal", "")[:80],
                    "status": manifest.get("status"),
                    "created_at": manifest.get("created_at"),
                })
            except Exception as e:
                logger.warning(f"Failed to load manifest for {run_dir.name}: {e}")

        return runs

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Append an event to the run's event log."""
        run_path = self.get_run_path(run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        events_path = run_path / "events.jsonl"
        event["timestamp"] = datetime.now(UTC).isoformat()

        with events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def _save_manifest(self, run: Run, run_path: Path) -> None:
        """Save run manifest."""
        manifest = {
            "run_id": run.run_id,
            "version": "1.0",
            "config": {
                "goal": run.config.goal,
                "vault_path": run.config.vault_path,
                "budget": run.config.budget.model_dump(),
                "voice": run.config.voice,
            },
            "status": run.status.value,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "total_tokens": run.total_tokens,
            "stop_reason": run.stop_reason.value if run.stop_reason else None,
            "error": run.error,
            "final_result": run.final_result,
        }

        manifest_path = run_path / "run.manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

    def _save_dag(self, run: Run, run_path: Path) -> None:
        """Save DAG structure."""
        nodes_data = []

        for node in run.nodes.values():
            node_data = {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "task": node.task,
                "status": node.status.value,
                "result": node.result,
                "children": node.children,
                "cache_key": node.cache_key,
                "cache_hit": node.cache_hit,
                "tokens_used": node.tokens_used,
                "start_time": node.start_time.isoformat() if node.start_time else None,
                "end_time": node.end_time.isoformat() if node.end_time else None,
                "error": node.error,
                "stop_reason": node.stop_reason.value if node.stop_reason else None,
            }
            nodes_data.append(node_data)

        dag = {
            "root_node_id": run.root_node_id,
            "nodes": nodes_data,
        }

        dag_path = run_path / "dag.json"
        with dag_path.open("w") as f:
            json.dump(dag, f, indent=2)

    def _save_metrics(self, run: Run, run_path: Path) -> None:
        """Save run metrics."""
        metrics_path = run_path / "metrics"

        # Save per-node metrics
        nodes_metrics_path = metrics_path / "nodes.jsonl"
        with nodes_metrics_path.open("w") as f:
            for node in run.nodes.values():
                metric = {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "status": node.status.value,
                    "tokens_used": node.tokens_used,
                    "cache_hit": node.cache_hit,
                    "duration_ms": node.duration_ms(),
                }
                f.write(json.dumps(metric) + "\n")

        # Save summary metrics
        summary = {
            "total_nodes": len(run.nodes),
            "completed_nodes": len(run.completed_nodes()),
            "failed_nodes": len(run.failed_nodes()),
            "max_depth_reached": max((n.depth for n in run.nodes.values()), default=0),
            "total_tokens": run.total_tokens,
            "cache_hits": sum(1 for n in run.nodes.values() if n.cache_hit),
            "total_duration_ms": None,
        }

        if run.started_at and run.completed_at:
            summary["total_duration_ms"] = int(
                (run.completed_at - run.started_at).total_seconds() * 1000
            )

        summary_path = metrics_path / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

    def _save_final_report(self, run: Run, run_path: Path) -> None:
        """Save human-readable final report."""
        report_lines = [
            f"# Run Report: {run.run_id}",
            "",
            f"**Goal:** {run.config.goal}",
            "",
            f"**Status:** {run.status.value}",
            "",
        ]

        if run.stop_reason:
            report_lines.append(f"**Stop Reason:** {run.stop_reason.value}")
            report_lines.append("")

        if run.error:
            report_lines.append(f"**Error:** {run.error}")
            report_lines.append("")

        # Metrics
        report_lines.extend([
            "## Metrics",
            "",
            f"- Total Nodes: {len(run.nodes)}",
            f"- Completed: {len(run.completed_nodes())}",
            f"- Failed: {len(run.failed_nodes())}",
            f"- Total Tokens: {run.total_tokens}",
            "",
        ])

        # Result
        if run.final_result:
            report_lines.extend([
                "## Result",
                "",
                run.final_result,
                "",
            ])

        # Resume command if partial
        if run.status.value == "partial":
            report_lines.extend([
                "## Resume",
                "",
                "```bash",
                f"shad resume {run.run_id}",
                "```",
                "",
            ])

        report_path = run_path / "final.report.md"
        with report_path.open("w") as f:
            f.write("\n".join(report_lines))

    def _save_final_summary(self, run: Run, run_path: Path) -> None:
        """Save machine-readable final summary."""
        summary: dict[str, Any] = {
            "run_id": run.run_id,
            "status": run.status.value,
            "goal": run.config.goal,
            "result": run.final_result,
            "citations": run.citations,
            "stop_reason": run.stop_reason.value if run.stop_reason else None,
            "error": run.error,
            "metrics": {
                "total_nodes": len(run.nodes),
                "completed_nodes": len(run.completed_nodes()),
                "failed_nodes": len(run.failed_nodes()),
                "total_tokens": run.total_tokens,
            },
            "suggested_next_actions": []
        }

        # Add suggested actions based on status
        actions: list[dict[str, str]] = []
        if run.status.value == "partial":
            actions.append({
                "action": "resume",
                "command": f"shad resume {run.run_id}",
                "reason": run.stop_reason.value if run.stop_reason else "partial completion",
            })
        elif run.status.value == "failed":
            actions.append({
                "action": "retry",
                "command": f"shad run \"{run.config.goal}\" --max-depth {run.config.budget.max_depth + 1}",
                "reason": "previous run failed",
            })
        summary["suggested_next_actions"] = actions

        summary_path = run_path / "final.summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

    def _save_replay_manifest(self, run: Run, run_path: Path) -> None:
        """Save replay manifest for deterministic replay."""
        replay = {
            "run_id": run.run_id,
            "config": run.config.model_dump(),
            "node_count": len(run.nodes),
            "cache_keys": [n.cache_key for n in run.nodes.values() if n.cache_key],
        }

        replay_path = run_path / "replay" / "manifest.json"
        with replay_path.open("w") as f:
            json.dump(replay, f, indent=2)
