"""Shad CLI - Command-line interface for Shannon's Daemon."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from shad import __version__
from shad.engine import LLMProvider, RLMEngine
from shad.history import HistoryManager
from shad.models import Budget, RunConfig
from shad.models.run import NodeStatus, Run, RunStatus

console = Console()


def run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.version_option(version=__version__, prog_name="shad")
def cli() -> None:
    """Shad - Shannon's Daemon: Personal AI Infrastructure for long-context reasoning."""
    pass


@cli.command()
@click.argument("goal")
@click.option("--notebook", "-n", help="Notebook ID to use for context")
@click.option("--max-depth", "-d", default=3, help="Maximum recursion depth")
@click.option("--max-nodes", default=50, help="Maximum DAG nodes")
@click.option("--max-time", "-t", default=300, help="Maximum wall time in seconds")
@click.option("--max-tokens", default=100000, help="Maximum tokens")
@click.option("--voice", "-v", help="Voice for output rendering")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
def run(
    goal: str,
    notebook: str | None,
    max_depth: int,
    max_nodes: int,
    max_time: int,
    max_tokens: int,
    voice: str | None,
    output: str | None,
) -> None:
    """Execute a reasoning task.

    Example:
        shad run "What are the key themes in the research paper?" --notebook abc123
    """
    config = RunConfig(
        goal=goal,
        notebook_id=notebook,
        budget=Budget(
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_wall_time=max_time,
            max_tokens=max_tokens,
        ),
        voice=voice,
    )

    console.print(Panel(f"[bold]Goal:[/bold] {goal}", title="Shad Run", border_style="blue"))

    engine = RLMEngine(llm_provider=LLMProvider())
    history = HistoryManager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing run...", total=None)

        result = run_async(engine.execute(config))
        progress.update(task, description="Complete")

    # Save history
    history.save_run(result)

    # Display results
    _display_run_result(result)

    # Write output if requested
    if output:
        output_path = Path(output)
        with output_path.open("w") as f:
            f.write(result.final_result or "No result")
        console.print(f"\n[dim]Output written to {output_path}[/dim]")

    # Exit with appropriate code
    if result.status == RunStatus.FAILED:
        sys.exit(1)
    elif result.status == RunStatus.PARTIAL:
        sys.exit(2)


@cli.command()
@click.argument("run_id")
def status(run_id: str) -> None:
    """Check the status of a run.

    Example:
        shad status abc12345
    """
    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    _display_run_status(run_data)


@cli.group()
def trace() -> None:
    """Inspect run traces."""
    pass


@trace.command("tree")
@click.argument("run_id")
def trace_tree(run_id: str) -> None:
    """Display the DAG tree for a run.

    Example:
        shad trace tree abc12345
    """
    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    _display_dag_tree(run_data)


@trace.command("node")
@click.argument("run_id")
@click.argument("node_id")
def trace_node(run_id: str, node_id: str) -> None:
    """Inspect a specific node in a run.

    Example:
        shad trace node abc12345 node1234
    """
    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    node = run_data.get_node(node_id)
    if not node:
        console.print(f"[red]Node {node_id} not found in run {run_id}[/red]")
        sys.exit(1)

    _display_node_detail(node)


@cli.command()
@click.argument("run_id")
@click.option("--max-depth", "-d", type=int, help="Override max depth")
@click.option("--max-nodes", type=int, help="Override max nodes")
@click.option("--max-time", "-t", type=int, help="Override max time")
def resume(
    run_id: str,
    max_depth: int | None,
    max_nodes: int | None,
    max_time: int | None,
) -> None:
    """Resume a partial or failed run.

    Example:
        shad resume abc12345 --max-depth 4
    """
    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    if run_data.status not in (RunStatus.PARTIAL, RunStatus.FAILED):
        console.print(f"[yellow]Run {run_id} is {run_data.status.value}, cannot resume[/yellow]")
        sys.exit(1)

    # Override budgets if specified
    if max_depth:
        run_data.config.budget.max_depth = max_depth
    if max_nodes:
        run_data.config.budget.max_nodes = max_nodes
    if max_time:
        run_data.config.budget.max_wall_time = max_time

    console.print(Panel(f"Resuming run {run_id}", title="Shad Resume", border_style="yellow"))

    engine = RLMEngine(llm_provider=LLMProvider())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Resuming run...", total=None)

        result = run_async(engine.resume(run_data))
        progress.update(task, description="Complete")

    # Save updated history
    history.save_run(result)

    _display_run_result(result)


@cli.command()
@click.argument("run_id")
def debug(run_id: str) -> None:
    """Enter debug mode for a run.

    Example:
        shad debug abc12345
    """
    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    console.print(Panel(f"Debug mode for run {run_id}", title="Shad Debug", border_style="magenta"))

    # Display comprehensive debug info
    _display_run_status(run_data)
    console.print()
    _display_dag_tree(run_data)
    console.print()

    # Show metrics
    table = Table(title="Run Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Nodes", str(len(run_data.nodes)))
    table.add_row("Completed Nodes", str(len(run_data.completed_nodes())))
    table.add_row("Failed Nodes", str(len(run_data.failed_nodes())))
    table.add_row("Total Tokens", str(run_data.total_tokens))

    if run_data.started_at and run_data.completed_at:
        duration = (run_data.completed_at - run_data.started_at).total_seconds()
        table.add_row("Duration", f"{duration:.2f}s")

    console.print(table)


def _display_run_result(run: Run) -> None:
    """Display run result with formatting."""
    status_colors = {
        RunStatus.COMPLETE: "green",
        RunStatus.PARTIAL: "yellow",
        RunStatus.FAILED: "red",
        RunStatus.ABORTED: "red",
        RunStatus.RUNNING: "blue",
        RunStatus.PENDING: "dim",
    }

    color = status_colors.get(run.status, "white")
    console.print(f"\n[bold {color}]Status: {run.status.value}[/bold {color}]")

    if run.stop_reason:
        console.print(f"[dim]Stop reason: {run.stop_reason.value}[/dim]")

    if run.error:
        console.print(f"[red]Error: {run.error}[/red]")

    if run.final_result:
        console.print(Panel(run.final_result, title="Result", border_style="green"))

    # Show resume command if partial
    if run.status == RunStatus.PARTIAL:
        console.print(f"\n[yellow]To resume: shad resume {run.run_id}[/yellow]")

    console.print(f"\n[dim]Run ID: {run.run_id}[/dim]")


def _display_run_status(run: Run) -> None:
    """Display run status summary."""
    table = Table(title=f"Run {run.run_id[:8]}...")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Goal", run.config.goal[:80] + "..." if len(run.config.goal) > 80 else run.config.goal)
    table.add_row("Status", f"[bold]{run.status.value}[/bold]")
    table.add_row("Nodes", f"{len(run.completed_nodes())}/{len(run.nodes)}")
    table.add_row("Tokens", str(run.total_tokens))

    if run.created_at:
        table.add_row("Created", run.created_at.isoformat())
    if run.stop_reason:
        table.add_row("Stop Reason", run.stop_reason.value)

    console.print(table)


def _display_dag_tree(run: Run) -> None:
    """Display the DAG as a tree."""
    if not run.root_node_id:
        console.print("[dim]No DAG available[/dim]")
        return

    root = run.get_node(run.root_node_id)
    if not root:
        console.print("[dim]Root node not found[/dim]")
        return

    tree = Tree(f"[bold]{_node_label(root)}[/bold]")
    _build_tree(run, root, tree)
    console.print(tree)


def _build_tree(run: Run, node: Any, tree: Tree) -> None:
    """Recursively build tree display."""
    for child_id in node.children:
        child = run.get_node(child_id)
        if child:
            branch = tree.add(_node_label(child))
            _build_tree(run, child, branch)


def _node_label(node: Any) -> str:
    """Generate a label for a node in the tree."""
    status_icons = {
        NodeStatus.SUCCEEDED: "[green]✓[/green]",
        NodeStatus.FAILED: "[red]✗[/red]",
        NodeStatus.CACHE_HIT: "[cyan]⚡[/cyan]",
        NodeStatus.STARTED: "[yellow]⟳[/yellow]",
        NodeStatus.PRUNED: "[dim]✂[/dim]",
    }

    icon = status_icons.get(node.status, "[dim]○[/dim]")
    task_preview = node.task[:50] + "..." if len(node.task) > 50 else node.task
    return f"{icon} [{node.node_id}] {task_preview}"


def _display_node_detail(node: Any) -> None:
    """Display detailed information about a node."""
    console.print(Panel(f"Node {node.node_id}", title="Node Detail", border_style="cyan"))

    table = Table()
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Task", node.task)
    table.add_row("Status", node.status.value)
    table.add_row("Depth", str(node.depth))
    table.add_row("Parent", node.parent_id or "None")
    table.add_row("Children", ", ".join(node.children) or "None")
    table.add_row("Cache Hit", str(node.cache_hit))
    table.add_row("Tokens", str(node.tokens_used))

    if node.duration_ms():
        table.add_row("Duration", f"{node.duration_ms()}ms")
    if node.error:
        table.add_row("Error", f"[red]{node.error}[/red]")

    console.print(table)

    if node.result:
        console.print(Panel(node.result, title="Result", border_style="green"))


if __name__ == "__main__":
    cli()
