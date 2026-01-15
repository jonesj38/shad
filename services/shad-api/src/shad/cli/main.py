"""Shad CLI - Command-line interface for Shannon's Daemon."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from shad import __version__
from shad.engine import LLMProvider, RLMEngine
from shad.history import HistoryManager
from shad.mcp import ObsidianMCPClient
from shad.models import Budget, RunConfig
from shad.models.run import NodeStatus, Run, RunStatus
from shad.utils.config import get_settings

console = Console()


def run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def get_api_url() -> str:
    """Get the Shad API URL."""
    settings = get_settings()
    return f"http://{settings.api_host}:{settings.api_port}"


@click.group()
@click.version_option(version=__version__, prog_name="shad")
def cli() -> None:
    """Shad - Shannon's Daemon: Personal AI Infrastructure.

    \b
    Usage:
        shad run "your goal or question"              # Run a reasoning task
        shad run "question" --vault /path/to/vault    # Run with Obsidian vault context
        shad status <run_id>                          # Check run status
        shad trace tree <run_id>                      # View execution DAG

    \b
    Examples:
        shad run "Summarize the key points"
        shad run "What are the main themes?" --vault ~/Documents/MyVault
        shad status abc12345
    """
    pass


@cli.command("run")
@click.argument("goal")
@click.option("--vault", "-v", help="Obsidian vault path for context")
@click.option("--max-depth", "-d", default=3, help="Maximum recursion depth")
@click.option("--max-nodes", default=50, help="Maximum DAG nodes")
@click.option("--max-time", "-t", default=300, help="Maximum wall time in seconds")
@click.option("--max-tokens", default=100000, help="Maximum tokens")
@click.option("--voice", help="Voice for output rendering")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--api", default=None, help="Shad API URL (uses local engine if not specified)")
@click.option("--no-code-mode", is_flag=True, help="Disable Code Mode (LLM-generated retrieval scripts)")
@click.option("--strategy", "-s", type=click.Choice(["software", "research", "analysis", "planning"]),
              help="Override automatic strategy selection")
@click.option("--verify", type=click.Choice(["off", "basic", "build", "strict"]), default="basic",
              help="Verification level (default: basic)")
@click.option("--write-files", is_flag=True, help="Write output files to disk (for software strategy)")
@click.option("--output-dir", type=click.Path(), help="Output directory for files (requires --write-files)")
def run(
    goal: str,
    vault: str | None,
    max_depth: int,
    max_nodes: int,
    max_time: int,
    max_tokens: int,
    voice: str | None,
    output: str | None,
    api: str | None,
    no_code_mode: bool,
    strategy: str | None,
    verify: str,
    write_files: bool,
    output_dir: str | None,
) -> None:
    """Execute a reasoning task.

    \b
    Examples:
        shad run "Explain quantum computing"
        shad run "Summarize research" --vault ~/Notes
        shad run "Build REST API" --strategy software --verify strict --write-files
    """
    # If API specified, use remote execution
    if api:
        _run_via_api(goal, vault, max_depth, api)
        return

    config = RunConfig(
        goal=goal,
        vault_path=vault,
        budget=Budget(
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_wall_time=max_time,
            max_tokens=max_tokens,
        ),
        voice=voice,
        strategy_override=strategy,
        verify_level=verify,
        write_files=write_files,
        output_path=output_dir,
    )

    console.print(Panel(f"[bold]Goal:[/bold] {goal}", title="Shad Run", border_style="blue"))

    # Initialize MCP client if vault specified
    mcp_client: ObsidianMCPClient | None = None
    vault_path: Path | None = None
    if vault:
        # Resolve vault path (support relative paths)
        vault_path = Path(vault).expanduser()
        if not vault_path.is_absolute():
            vault_path = Path.cwd() / vault_path
        console.print(f"[dim][CONTEXT] Using vault: {vault_path}[/dim]")
        mcp_client = ObsidianMCPClient(vault_path=vault_path)

        # Show Code Mode status
        use_code_mode = not no_code_mode
        if use_code_mode:
            console.print("[dim][CODE_MODE] Enabled - LLM will generate custom retrieval scripts[/dim]")
        else:
            console.print("[dim][CODE_MODE] Disabled - using direct search[/dim]")
    else:
        console.print("[dim][CONTEXT] No vault specified[/dim]")
        use_code_mode = False

    # Display strategy and verification options
    if strategy:
        console.print(f"[dim][STRATEGY] Override: {strategy}[/dim]")
    console.print(f"[dim][VERIFY] Level: {verify}[/dim]")
    if write_files:
        console.print(f"[dim][OUTPUT] Write files enabled{f' → {output_dir}' if output_dir else ''}[/dim]")

    async def _execute_run() -> Run:
        """Execute run."""
        engine = RLMEngine(
            llm_provider=LLMProvider(),
            mcp_client=mcp_client,
            vault_path=vault_path,
            use_code_mode=use_code_mode,
        )
        return await engine.execute(config)

    history = HistoryManager()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing run...", total=None)

        result = run_async(_execute_run())
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

    # Write manifest files if requested
    if write_files and result.metadata.get("manifest"):
        from shad.output.manifest import FileManifest, ManifestWriter

        manifest_data = result.metadata["manifest"]
        manifest = FileManifest.from_dict(manifest_data)
        output_root = Path(output_dir) if output_dir else Path.cwd() / "output"
        output_root.mkdir(parents=True, exist_ok=True)

        writer = ManifestWriter(output_root=output_root)
        report = writer.write(manifest)

        if report.success:
            console.print(f"\n[green]✓ Wrote {len(report.written)} files to {output_root}[/green]")
            for path in report.written[:5]:
                console.print(f"  [dim]→ {path}[/dim]")
            if len(report.written) > 5:
                console.print(f"  [dim]... and {len(report.written) - 5} more[/dim]")
        else:
            console.print(f"\n[yellow]⚠ Wrote {len(report.written)} files, skipped {len(report.skipped)}[/yellow]")
            for error in report.errors[:3]:
                console.print(f"  [red]{error}[/red]")

    # Exit with appropriate code
    if result.status == RunStatus.FAILED:
        sys.exit(1)
    elif result.status == RunStatus.PARTIAL:
        sys.exit(2)


def _run_via_api(goal: str, vault: str | None, max_depth: int, api: str) -> None:
    """Run a goal via the API."""
    try:
        with httpx.Client(timeout=120.0) as client:
            payload: dict[str, Any] = {"goal": goal, "budget": {"max_depth": max_depth}}
            if vault:
                payload["vault_path"] = vault

            response = client.post(f"{api}/v1/run", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        console.print("[red]Could not connect to Shad API. Is it running?[/red]")
        console.print(f"[dim]Tried: {api}[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    status = data.get("status", "unknown")
    result = data.get("result")
    error = data.get("error")

    if status == "complete" and result:
        console.print(Panel(result, title="Result", border_style="green"))
    elif error:
        console.print(f"[red]Error: {error}[/red]")
    else:
        console.print(f"[yellow]Status: {status}[/yellow]")
        if result:
            console.print(Panel(result, title="Partial Result", border_style="yellow"))

    console.print(f"\n[dim]Run ID: {data.get('run_id', 'unknown')}[/dim]")


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
@click.option("--replay", type=str, help="Replay mode: 'stale', node_id, or 'subtree:node_id'")
def resume(
    run_id: str,
    max_depth: int | None,
    max_nodes: int | None,
    max_time: int | None,
    replay: str | None,
) -> None:
    """Resume a partial or failed run with delta verification.

    \b
    Examples:
        shad resume abc12345 --max-depth 4
        shad resume abc12345 --replay stale     # Re-run stale nodes only
        shad resume abc12345 --replay node123   # Re-run specific node
        shad resume abc12345 --replay subtree:node123  # Re-run subtree
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
    if replay:
        console.print(f"[dim][REPLAY] Mode: {replay}[/dim]")

    async def _resume_run() -> Run:
        """Resume run."""
        engine = RLMEngine(llm_provider=LLMProvider())
        return await engine.resume(run_data, replay_mode=replay)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Resuming run...", total=None)

        result = run_async(_resume_run())
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


# =============================================================================
# Vault Commands
# =============================================================================

@cli.command("vault")
@click.option("--api", default="http://localhost:8000", help="Shad API URL")
def vault_status(api: str) -> None:
    """Check Obsidian vault connection status.

    \b
    Example:
        shad vault
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{api}/v1/vault/status")
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        console.print("[red]Could not connect to Shad API. Is it running?[/red]")
        console.print(f"[dim]Tried: {api}[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if data.get("connected"):
        console.print("[green]✓ Obsidian vault connected[/green]")
        if data.get("vault_path"):
            console.print(f"[dim]Path: {data['vault_path']}[/dim]")
    else:
        console.print("[yellow]⚠ Obsidian vault not connected[/yellow]")
        console.print("[dim]Configure OBSIDIAN_VAULT_PATH in your environment[/dim]")


@cli.command("search")
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--api", default="http://localhost:8000", help="Shad API URL")
def search(query: str, limit: int, api: str) -> None:
    """Search the Obsidian vault.

    \b
    Example:
        shad search "machine learning"
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{api}/v1/vault/search",
                params={"query": query, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        console.print("[red]Could not connect to Shad API. Is it running?[/red]")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            console.print("[yellow]Obsidian vault not connected[/yellow]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    results = data.get("results", [])
    if not results:
        console.print(f"[dim]No results found for '{query}'[/dim]")
        return

    console.print(Panel(f"Search: {query}", title="Vault Search", border_style="blue"))

    for i, result in enumerate(results, 1):
        path = result.get("path", "Unknown")
        content = result.get("content", "")[:200]
        score = result.get("score", 0)

        console.print(f"\n[bold cyan]{i}. {path}[/bold cyan] [dim](score: {score:.2f})[/dim]")
        console.print(f"   {content}...")


# =============================================================================
# Export Command
# =============================================================================

@cli.command()
@click.argument("run_id")
@click.option("--output", "-o", type=click.Path(), default="./output", help="Output directory")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def export(run_id: str, output: str, overwrite: bool) -> None:
    """Export files from a completed run.

    \b
    Examples:
        shad export abc12345
        shad export abc12345 --output ./my-project
        shad export abc12345 --overwrite
    """
    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    manifest_data = run_data.metadata.get("manifest")
    if not manifest_data:
        console.print(f"[yellow]Run {run_id} has no file manifest to export[/yellow]")
        sys.exit(1)

    from shad.output.manifest import FileManifest, ManifestWriter

    manifest = FileManifest.from_dict(manifest_data)
    output_root = Path(output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    writer = ManifestWriter(output_root=output_root, overwrite=overwrite)
    report = writer.write(manifest)

    if report.success:
        console.print(f"[green]✓ Exported {len(report.written)} files to {output_root}[/green]")
        for path in report.written[:10]:
            console.print(f"  [dim]→ {path}[/dim]")
        if len(report.written) > 10:
            console.print(f"  [dim]... and {len(report.written) - 10} more[/dim]")
    else:
        console.print(f"[yellow]⚠ Exported {len(report.written)} files, skipped {len(report.skipped)}[/yellow]")
        for error in report.errors[:5]:
            console.print(f"  [red]{error}[/red]")


# =============================================================================
# Vault Ingestion Command
# =============================================================================

@cli.group("ingest")
def ingest() -> None:
    """Ingest sources into vault."""
    pass


@ingest.command("github")
@click.argument("url")
@click.option("--vault", "-v", required=True, type=click.Path(), help="Target vault path")
@click.option("--preset", type=click.Choice(["mirror", "docs", "deep"]), default="docs",
              help="Ingestion preset (default: docs)")
def ingest_github(url: str, vault: str, preset: str) -> None:
    """Ingest a GitHub repository into vault.

    \b
    Examples:
        shad ingest github https://github.com/user/repo --vault ~/MyVault
        shad ingest github https://github.com/user/repo --vault ~/MyVault --preset deep
    """
    from shad.vault.ingestion import IngestPreset, VaultIngester

    vault_path = Path(vault).expanduser()
    if not vault_path.exists():
        console.print(f"[red]Vault path does not exist: {vault_path}[/red]")
        sys.exit(1)

    console.print(Panel(f"Ingesting {url}", title="Vault Ingestion", border_style="blue"))
    console.print(f"[dim]Preset: {preset}[/dim]")
    console.print(f"[dim]Target: {vault_path}[/dim]")

    async def _ingest() -> Any:
        ingester = VaultIngester(vault_path=vault_path)
        return await ingester.ingest_github(url, preset=IngestPreset(preset))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=None)

        try:
            result = run_async(_ingest())
            progress.update(task, description="Complete")
        except Exception as e:
            progress.update(task, description="Failed")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    console.print(f"[green]✓ Ingested: {result.snapshot_id}[/green]")
    console.print(f"[dim]Files: {len(result.files_processed)}[/dim]")


# =============================================================================
# Server Management
# =============================================================================


@cli.group()
def server() -> None:
    """Manage Shad server (Redis + API).

    \b
    Commands:
        shad server start   - Start Redis and API server
        shad server stop    - Stop all services
        shad server status  - Check service status
    """
    pass


@server.command("start")
@click.option("--foreground", "-f", is_flag=True, help="Run API in foreground (don't daemonize)")
def server_start(foreground: bool) -> None:
    """Start the Shad server (Redis + API)."""
    import os
    import subprocess
    import time

    shad_home = os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad"))
    repo_dir = os.path.join(shad_home, "repo")
    pid_file = os.path.join(shad_home, "shad-api.pid")

    # Check if already running
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)  # Check if process exists
            console.print(f"[yellow]Shad API already running (PID {pid})[/yellow]")
            console.print("[dim]Use 'shad server stop' to stop it first[/dim]")
            return
        except OSError:
            os.remove(pid_file)  # Stale PID file

    # Start Redis via Docker Compose
    console.print("[blue]Starting Redis...[/blue]")
    compose_file = os.path.join(repo_dir, "docker-compose.yml")

    if os.path.exists(compose_file):
        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d", "redis"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Failed to start Redis: {result.stderr}[/red]")
            sys.exit(1)
        console.print("[green]✓ Redis started[/green]")
    else:
        console.print("[yellow]docker-compose.yml not found, assuming Redis is already running[/yellow]")

    # Wait for Redis to be ready
    time.sleep(1)

    # Start API server
    console.print("[blue]Starting Shad API...[/blue]")

    api_dir = os.path.join(repo_dir, "services", "shad-api")
    venv_python = os.path.join(shad_home, "venv", "bin", "python")

    if not os.path.exists(venv_python):
        # Fallback to system python if venv doesn't exist
        venv_python = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(api_dir, "src")

    if foreground:
        # Run in foreground
        console.print("[dim]Running in foreground. Press Ctrl+C to stop.[/dim]")
        try:
            subprocess.run(
                [venv_python, "-m", "uvicorn", "shad.api.main:app",
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=api_dir,
                env=env,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
    else:
        # Run in background
        log_file = os.path.join(shad_home, "shad-api.log")

        with open(log_file, "a") as log:
            proc = subprocess.Popen(
                [venv_python, "-m", "uvicorn", "shad.api.main:app",
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=api_dir,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        # Save PID
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))

        console.print(f"[green]✓ Shad API started (PID {proc.pid})[/green]")
        console.print(f"[dim]Log file: {log_file}[/dim]")
        console.print("[dim]API URL: http://localhost:8000[/dim]")

    console.print("\n[green]Shad is ready![/green]")
    console.print("[dim]Try: shad run \"Hello, Shad!\"[/dim]")


@server.command("stop")
def server_stop() -> None:
    """Stop the Shad server (Redis + API)."""
    import os
    import signal
    import subprocess

    shad_home = os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad"))
    repo_dir = os.path.join(shad_home, "repo")
    pid_file = os.path.join(shad_home, "shad-api.pid")

    stopped_api = False
    stopped_redis = False

    # Stop API server
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]✓ Stopped Shad API (PID {pid})[/green]")
            stopped_api = True
        except OSError:
            console.print("[dim]API was not running[/dim]")
        os.remove(pid_file)
    else:
        console.print("[dim]No API PID file found[/dim]")

    # Stop Redis via Docker Compose
    compose_file = os.path.join(repo_dir, "docker-compose.yml")
    if os.path.exists(compose_file):
        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "stop", "redis"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]✓ Stopped Redis[/green]")
            stopped_redis = True
        else:
            console.print(f"[yellow]Could not stop Redis: {result.stderr}[/yellow]")

    if stopped_api or stopped_redis:
        console.print("\n[green]Shad server stopped[/green]")
    else:
        console.print("\n[dim]Nothing to stop[/dim]")


@server.command("status")
def server_status() -> None:
    """Check Shad server status."""
    import os
    import subprocess

    shad_home = os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad"))
    pid_file = os.path.join(shad_home, "shad-api.pid")

    table = Table(title="Shad Server Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    # Check API
    api_running = False
    api_details = ""
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            api_running = True
            api_details = f"PID {pid}, http://localhost:8000"
        except OSError:
            api_details = "Stale PID file"
    else:
        api_details = "Not started"

    table.add_row(
        "Shad API",
        "[green]Running[/green]" if api_running else "[red]Stopped[/red]",
        api_details,
    )

    # Check Redis
    redis_running = False
    redis_details = ""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=shad-redis", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            redis_running = True
            redis_details = result.stdout.strip()
        else:
            redis_details = "Container not running"
    except Exception as e:
        redis_details = f"Docker error: {e}"

    table.add_row(
        "Redis",
        "[green]Running[/green]" if redis_running else "[red]Stopped[/red]",
        redis_details,
    )

    # Check Claude CLI
    claude_available = False
    claude_details = ""
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            claude_available = True
            claude_details = result.stdout.strip().split("\n")[0]
        else:
            claude_details = "Not configured"
    except FileNotFoundError:
        claude_details = "Not installed"

    table.add_row(
        "Claude CLI",
        "[green]Available[/green]" if claude_available else "[yellow]Missing[/yellow]",
        claude_details,
    )

    console.print(table)

    # Overall status
    if api_running and redis_running:
        console.print("\n[green]All services running. Shad is ready![/green]")
    elif api_running or redis_running:
        console.print("\n[yellow]Some services are not running.[/yellow]")
        console.print("[dim]Run 'shad server start' to start all services.[/dim]")
    else:
        console.print("\n[red]Shad is not running.[/red]")
        console.print("[dim]Run 'shad server start' to start.[/dim]")


@server.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
def server_logs(follow: bool, lines: int) -> None:
    """View Shad API logs."""
    import os
    import subprocess

    shad_home = os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad"))
    log_file = os.path.join(shad_home, "shad-api.log")

    if not os.path.exists(log_file):
        console.print("[dim]No log file found. Start the server first.[/dim]")
        return

    if follow:
        subprocess.run(["tail", "-f", log_file])
    else:
        subprocess.run(["tail", "-n", str(lines), log_file])


# =============================================================================
# Sources Management
# =============================================================================


@cli.group()
def sources() -> None:
    """Manage content sources for automated ingestion.

    \b
    Commands:
        shad sources add       - Add a new source
        shad sources list      - List all sources
        shad sources remove    - Remove a source
        shad sources sync      - Sync sources now
        shad sources status    - Show scheduler status
    """
    pass


@sources.command("add")
@click.argument("source_type", type=click.Choice(["github", "url", "feed", "folder"]))
@click.argument("location")
@click.option("--vault", "-v", required=True, help="Target vault path")
@click.option("--schedule", "-s", type=click.Choice(["manual", "hourly", "daily", "weekly", "monthly"]),
              default="daily", help="Sync schedule")
@click.option("--preset", "-p", default="docs", help="Ingestion preset (for github)")
def sources_add(
    source_type: str,
    location: str,
    vault: str,
    schedule: str,
    preset: str,
) -> None:
    """Add a new content source.

    \b
    Examples:
        shad sources add github https://github.com/org/repo --vault ~/MyVault
        shad sources add url https://docs.example.com --vault ~/MyVault --schedule weekly
        shad sources add feed https://blog.example.com/rss --vault ~/MyVault --schedule hourly
        shad sources add folder ~/Projects/docs --vault ~/MyVault
    """
    from shad.sources import SourceManager, SourceSchedule, SourceType

    manager = SourceManager()

    type_map = {
        "github": SourceType.GITHUB,
        "url": SourceType.URL,
        "feed": SourceType.FEED,
        "folder": SourceType.FOLDER,
    }
    schedule_map = {
        "manual": SourceSchedule.MANUAL,
        "hourly": SourceSchedule.HOURLY,
        "daily": SourceSchedule.DAILY,
        "weekly": SourceSchedule.WEEKLY,
        "monthly": SourceSchedule.MONTHLY,
    }

    is_folder = source_type == "folder"
    source = manager.add_source(
        source_type=type_map[source_type],
        url=None if is_folder else location,
        path=location if is_folder else None,
        vault_path=vault,
        schedule=schedule_map[schedule],
        preset=preset,
    )

    console.print(f"[green]✓ Added source: {source.id}[/green]")
    console.print(f"[dim]Type: {source_type}, Schedule: {schedule}[/dim]")


@sources.command("list")
def sources_list() -> None:
    """List all configured sources."""
    from shad.sources import SourceManager

    manager = SourceManager()
    source_list = manager.list_sources()

    if not source_list:
        console.print("[dim]No sources configured.[/dim]")
        console.print("[dim]Add one with: shad sources add github <url> --vault <path>[/dim]")
        return

    table = Table(title="Configured Sources")
    table.add_column("ID", style="cyan")
    table.add_column("Type")
    table.add_column("Location")
    table.add_column("Schedule")
    table.add_column("Last Sync")
    table.add_column("Status")

    for s in source_list:
        location = s.url or s.path or ""
        if len(location) > 40:
            location = "..." + location[-37:]

        last_sync = s.last_sync.strftime("%Y-%m-%d %H:%M") if s.last_sync else "Never"

        if not s.enabled:
            status = "[dim]Disabled[/dim]"
        elif s.last_error:
            status = "[red]Error[/red]"
        elif s.is_due():
            status = "[yellow]Due[/yellow]"
        else:
            status = "[green]OK[/green]"

        table.add_row(
            s.id[:8],
            s.type.value,
            location,
            s.schedule.value,
            last_sync,
            status,
        )

    console.print(table)


@sources.command("remove")
@click.argument("source_id")
def sources_remove(source_id: str) -> None:
    """Remove a source by ID."""
    from shad.sources import SourceManager

    manager = SourceManager()

    # Allow partial ID matching
    for s in manager.list_sources():
        if s.id.startswith(source_id):
            source_id = s.id
            break

    if manager.remove_source(source_id):
        console.print(f"[green]✓ Removed source: {source_id}[/green]")
    else:
        console.print(f"[red]Source not found: {source_id}[/red]")


@sources.command("sync")
@click.option("--all", "-a", "sync_all", is_flag=True, help="Sync all sources, not just due ones")
@click.option("--daemon", "-d", is_flag=True, help="Run as daemon (continuous sync)")
@click.option("--interval", "-i", default=60, help="Check interval in seconds (for daemon mode)")
def sources_sync(sync_all: bool, daemon: bool, interval: int) -> None:
    """Sync sources now.

    \b
    Examples:
        shad sources sync           # Sync due sources
        shad sources sync --all     # Sync all sources
        shad sources sync --daemon  # Run continuously
    """
    from shad.sources import SourceManager, SourceScheduler

    if daemon:
        console.print(f"[blue]Starting source scheduler (interval: {interval}s)[/blue]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")

        def on_sync(result: any) -> None:
            console.print(
                f"[dim]Sync: {result.successful} ok, {result.failed} failed, "
                f"{result.skipped} skipped[/dim]"
            )

        scheduler = SourceScheduler(check_interval=interval, on_sync=on_sync)

        try:
            run_async(scheduler.start())
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped[/yellow]")
    else:
        manager = SourceManager()
        sources_to_sync = manager.list_sources() if sync_all else manager.get_due_sources()

        if not sources_to_sync:
            console.print("[dim]No sources to sync[/dim]")
            return

        console.print(f"[blue]Syncing {len(sources_to_sync)} sources...[/blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for source in sources_to_sync:
                task = progress.add_task(f"Syncing {source.type.value}: {source.url or source.path}...", total=None)
                result = run_async(manager.sync_source(source))
                if result.success:
                    progress.update(task, description=f"[green]✓[/green] {source.url or source.path}")
                else:
                    progress.update(task, description=f"[red]✗[/red] {source.url or source.path}")
                progress.remove_task(task)

        console.print("[green]Sync complete[/green]")


@sources.command("status")
def sources_status() -> None:
    """Show scheduler and sources status."""
    from shad.sources import SourceScheduler

    scheduler = SourceScheduler()
    status = scheduler.get_status()

    console.print("\n[bold]Sources Status[/bold]")
    console.print(f"Total sources: {status['total_sources']}")
    console.print(f"Due for sync: {status['due_sources']}")

    if status["sources"]:
        console.print()
        table = Table(title="Source Details")
        table.add_column("ID", style="cyan")
        table.add_column("Type")
        table.add_column("URL/Path", max_width=50)
        table.add_column("Schedule")
        table.add_column("Last Sync")
        table.add_column("Next Sync")
        table.add_column("Due")

        for s in status["sources"]:
            # Truncate long URLs
            url = s["url"] or ""
            if len(url) > 50:
                url = url[:47] + "..."

            table.add_row(
                s["id"][:8],
                s["type"],
                url,
                s["schedule"],
                s["last_sync"][:16] if s["last_sync"] else "Never",
                s["next_sync"][:16] if s["next_sync"] else "Manual",
                "[yellow]Yes[/yellow]" if s["is_due"] else "[dim]No[/dim]",
            )

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
    status_str = run.status.value if hasattr(run.status, 'value') else run.status
    table.add_row("Status", f"[bold]{status_str}[/bold]")
    table.add_row("Nodes", f"{len(run.completed_nodes())}/{len(run.nodes)}")
    table.add_row("Tokens", str(run.total_tokens))

    if run.created_at:
        table.add_row("Created", run.created_at.isoformat())
    if run.stop_reason:
        stop_reason_str = run.stop_reason.value if hasattr(run.stop_reason, 'value') else run.stop_reason
        table.add_row("Stop Reason", stop_reason_str)

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
