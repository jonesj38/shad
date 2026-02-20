"""Shad CLI - Command-line interface for Shannon's Daemon."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
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
from shad.engine.llm import ModelTier
from shad.history import HistoryManager
from shad.models import Budget, ModelConfig, RunConfig
from shad.models.run import NodeStatus, Run, RunStatus, StopReason
from shad.retrieval import FilesystemRetriever, QmdRetriever, get_retriever
from shad.utils.config import get_settings

console = Console()


def _get_system_specs() -> tuple[int, float]:
    """Return (cpu_count, memory_gb) best-effort."""
    cpu_count = os.cpu_count() or 4
    mem_gb = 8.0
    try:
        system = platform.system().lower()
        if system == "darwin":
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, timeout=2)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip() or 0)
                mem_gb = max(mem_gb, mem_bytes / (1024 ** 3))
        elif system == "linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        parts = line.split()
                        mem_kb = int(parts[1])
                        mem_gb = max(mem_gb, mem_kb / (1024 ** 2))
                        break
    except Exception:
        pass
    return cpu_count, mem_gb


def _suggest_profile(cpu_count: int, mem_gb: float) -> str:
    if cpu_count >= 12 and mem_gb >= 32:
        return "deep"
    if cpu_count <= 4 or mem_gb <= 8:
        return "fast"
    return "balanced"


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
    Core Commands:
        run <goal>           Execute a reasoning task
        status <run_id>      Check the status of a run
        resume <run_id>      Resume a partial/failed run
        export <run_id>      Export files from a completed run
        debug <run_id>       Enter debug mode for a run

    \b
    Project Setup:
        init [path]              Initialize project permissions for Claude Code
        check-permissions [path] Verify project permissions are configured

    \b
    Vault Commands:
        vault                Check Obsidian vault connection
        search <query>       Search the Obsidian vault
        ingest ...           Ingest sources into vault (see: shad ingest --help)

    \b
    Server Management:
        server start|stop|status|logs    Manage Shad server

    \b
    Source Management:
        sources add|list|remove|sync|status    Manage content sources

    \b
    Trace & Debug:
        trace tree <run_id>              View DAG tree
        trace node <run_id> <node_id>    Inspect specific node

    \b
    Quick Start:
        shad init                                    # Set up project permissions
        shad run "Build a REST API" --vault ~/Notes  # Run with vault context
        shad status <run_id>                         # Check progress
    """
    pass


@cli.command("run")
@click.argument("goal")
@click.option("--vault", "-v", multiple=True, help="Vault path(s) for context (can specify multiple)")
@click.option("--retriever", "-r", type=click.Choice(["auto", "qmd", "filesystem"]), default="auto",
              help="Retrieval backend (auto detects qmd)")
@click.option("--profile", type=click.Choice(["fast", "balanced", "deep"], case_sensitive=False),
              help="Preset budget profile: fast, balanced, or deep")
@click.option("--auto-profile", is_flag=True, help="Auto-select profile based on machine specs")
@click.option("--dry-run", is_flag=True, help="Show budgets/models and exit")
@click.option("--max-depth", "-d", default=3, help="Maximum recursion depth")
@click.option("--max-nodes", default=50, help="Maximum DAG nodes")
@click.option("--max-time", "-t", default=1200, help="Maximum wall time in seconds")
@click.option("--max-tokens", default=100000, help="Maximum tokens")
@click.option("--voice", help="Voice for output rendering")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--api", default=None, help="Shad API URL (uses local engine if not specified)")
@click.option("--no-code-mode", is_flag=True, help="Disable Code Mode (LLM-generated retrieval scripts)")
@click.option("--qmd-hybrid/--no-qmd-hybrid", default=True, help="Use qmd hybrid search with reranking (BM25 + vector + RRF + reranker, default: on)")
@click.option("--strategy", "-s", type=click.Choice(["software", "research", "analysis", "planning"]),
              help="Override automatic strategy selection")
@click.option("--verify", type=click.Choice(["off", "basic", "build", "strict"]), default="basic",
              help="Verification level (default: basic)")
@click.option("--write-files", is_flag=True, help="Write output files to disk (for software strategy)")
@click.option("--output-dir", type=click.Path(), help="Output directory for files (requires --write-files)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress verbose output")
@click.option("--orchestrator-model", "-O", help="Model for planning/synthesis (e.g., opus, sonnet, haiku)")
@click.option("--worker-model", "-W", help="Model for mid-depth execution (e.g., opus, sonnet, haiku)")
@click.option("--leaf-model", "-L", help="Model for fast parallel execution (e.g., opus, sonnet, haiku)")
def run(
    goal: str,
    vault: tuple[str, ...],
    retriever: str,
    profile: str | None,
    auto_profile: bool,
    dry_run: bool,
    max_depth: int,
    max_nodes: int,
    max_time: int,
    max_tokens: int,
    voice: str | None,
    output: str | None,
    api: str | None,
    no_code_mode: bool,
    qmd_hybrid: bool,
    strategy: str | None,
    verify: str,
    write_files: bool,
    output_dir: str | None,
    quiet: bool,
    orchestrator_model: str | None,
    worker_model: str | None,
    leaf_model: str | None,
) -> None:
    """Execute a reasoning task.

    \b
    Examples:
        shad run "Explain quantum computing"
        shad run "Summarize research" --vault ~/Notes
        shad run "Build REST API" --vault ~/Project --vault ~/Patterns
        shad run "Complex task" -O opus -W sonnet -L haiku
        shad run "Fast summary" --profile fast
        shad run "Auto profile" --auto-profile
    """
    # Configure logging (verbose by default, --quiet to suppress)
    if not quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # Set shad loggers to INFO
        logging.getLogger("shad").setLevel(logging.INFO)

    # Apply budget profile if provided (can be overridden by explicit flags)
    if profile:
        profile_key = profile.lower()
        presets = {
            "fast": {"max_depth": 2, "max_nodes": 25, "max_time": 600, "max_tokens": 800000},
            "balanced": {"max_depth": 3, "max_nodes": 50, "max_time": 1200, "max_tokens": 2000000},
            "deep": {"max_depth": 4, "max_nodes": 80, "max_time": 1800, "max_tokens": 3000000},
        }
        if profile_key in presets:
            preset = presets[profile_key]
            if max_depth == 3:
                max_depth = preset["max_depth"]
            if max_nodes == 50:
                max_nodes = preset["max_nodes"]
            if max_time == 1200:
                max_time = preset["max_time"]
            if max_tokens == 100000:
                max_tokens = preset["max_tokens"]
            console.print(
                f"[dim][PROFILE] {profile_key} preset applied (depth={max_depth}, nodes={max_nodes}, time={max_time}s)[/dim]"
            )
    elif auto_profile:
        cpu_count, mem_gb = _get_system_specs()
        profile_key = _suggest_profile(cpu_count, mem_gb)
        presets = {
            "fast": {"max_depth": 2, "max_nodes": 25, "max_time": 600, "max_tokens": 800000},
            "balanced": {"max_depth": 3, "max_nodes": 50, "max_time": 1200, "max_tokens": 2000000},
            "deep": {"max_depth": 4, "max_nodes": 80, "max_time": 1800, "max_tokens": 3000000},
        }
        preset = presets[profile_key]
        if max_depth == 3:
            max_depth = preset["max_depth"]
        if max_nodes == 50:
            max_nodes = preset["max_nodes"]
        if max_time == 1200:
            max_time = preset["max_time"]
        if max_tokens == 100000:
            max_tokens = preset["max_tokens"]
        console.print(
            f"[dim][PROFILE] auto-selected {profile_key} (depth={max_depth}, nodes={max_nodes}, time={max_time}s)[/dim]"
        )
    else:
        # Suggest a profile based on machine specs when defaults are used
        if max_depth == 3 and max_nodes == 50 and max_time == 1200 and max_tokens == 100000:
            cpu_count, mem_gb = _get_system_specs()
            suggestion = _suggest_profile(cpu_count, mem_gb)
            console.print(
                f"[dim][HINT] For this machine ({cpu_count} CPU / {mem_gb:.1f} GB), try --profile {suggestion}[/dim]"
            )

    if dry_run:
        console.print(
            f"[dim][DRY RUN] depth={max_depth}, nodes={max_nodes}, time={max_time}s, tokens={max_tokens}[/dim]"
        )
        return

    # If API specified, use remote execution
    if api:
        _run_via_api(goal, vault[0] if vault else None, max_depth, api)
        return

    # Build vault paths from CLI args or env fallback
    vault_paths: list[Path] = []
    collection_names: dict[str, Path] = {}

    if vault:
        for v in vault:
            vp = Path(v).expanduser()
            if not vp.is_absolute():
                vp = Path.cwd() / vp
            vault_paths.append(vp)
            collection_names[vp.name] = vp
    elif get_settings().obsidian_vault_path:
        # Fall back to env var
        vp = Path(get_settings().obsidian_vault_path).expanduser()
        vault_paths.append(vp)
        collection_names[vp.name] = vp

    # Primary vault path (first one) for backward compatibility
    primary_vault = vault_paths[0] if vault_paths else None

    config = RunConfig(
        goal=goal,
        vault_path=str(primary_vault) if primary_vault else None,
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

    # Initialize retriever
    retriever_instance = None
    collections: list[str] = []

    if vault_paths:
        source = "CLI" if vault else "env"
        if len(vault_paths) == 1:
            console.print(f"[dim][CONTEXT] Using vault ({source}): {vault_paths[0]}[/dim]")
        else:
            console.print(f"[dim][CONTEXT] Using {len(vault_paths)} vaults ({source}):[/dim]")
            for vp in vault_paths:
                console.print(f"[dim]  - {vp.name}: {vp}[/dim]")
        collections = list(collection_names.keys())

        # Get retriever based on preference
        retriever_instance = get_retriever(
            paths=vault_paths,
            collection_names=collection_names,
            prefer=retriever,
        )
        console.print(f"[dim][RETRIEVER] Using {type(retriever_instance).__name__}[/dim]")

        # Auto-provision qmd collections for vault paths
        if isinstance(retriever_instance, QmdRetriever) and collection_names:
            console.print("[dim][QMD] Ensuring collections are provisioned and indexed...[/dim]")
            provision_results = run_async(retriever_instance.ensure_collections(collection_names))
            for cname, success in provision_results.items():
                if success:
                    console.print(f"[dim][QMD] ✓ {cname}[/dim]")
                else:
                    console.print(f"[yellow][QMD] ✗ {cname} (failed to provision)[/yellow]")

        # Show Code Mode status
        use_code_mode = not no_code_mode
        if qmd_hybrid:
            console.print("[dim][QMD_HYBRID] Enabled - BM25 + vector + RRF fusion + reranking[/dim]")
        elif use_code_mode:
            console.print("[dim][CODE_MODE] Enabled - LLM will generate custom retrieval scripts[/dim]")
        else:
            console.print("[dim][CODE_MODE] Disabled - using direct search[/dim]")
    else:
        console.print("[dim][CONTEXT] No vault specified (use --vault or set OBSIDIAN_VAULT_PATH)[/dim]")
        use_code_mode = False

    # Display strategy and verification options
    if strategy:
        console.print(f"[dim][STRATEGY] Override: {strategy}[/dim]")
    console.print(f"[dim][VERIFY] Level: {verify}[/dim]")
    if write_files:
        console.print(f"[dim][OUTPUT] Write files enabled{f' → {output_dir}' if output_dir else ''}[/dim]")

    # Create model config if any model overrides specified
    model_config: ModelConfig | None = None
    if orchestrator_model or worker_model or leaf_model:
        model_config = ModelConfig(
            orchestrator_model=orchestrator_model,
            worker_model=worker_model,
            leaf_model=leaf_model,
        )
        console.print("[dim][MODELS] Custom model selection:[/dim]")
        if orchestrator_model:
            console.print(f"[dim]  Orchestrator: {orchestrator_model}[/dim]")
        if worker_model:
            console.print(f"[dim]  Worker: {worker_model}[/dim]")
        if leaf_model:
            console.print(f"[dim]  Leaf: {leaf_model}[/dim]")

    async def _execute_run() -> Run:
        """Execute run."""
        engine = RLMEngine(
            llm_provider=LLMProvider(model_config=model_config),
            retriever=retriever_instance,
            vault_path=primary_vault,
            collections=collections,
            use_code_mode=use_code_mode,
            use_qmd_hybrid=qmd_hybrid,
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
    """Inspect run traces and DAG execution.

    \b
    Commands:
        tree <run_id>              Show DAG tree structure
        node <run_id> <node_id>    Inspect a specific node

    \b
    Examples:
        shad trace tree abc12345
        shad trace node abc12345 node1234
    """
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
@click.option("--profile", type=click.Choice(["fast", "balanced", "deep"], case_sensitive=False),
              help="Preset budget profile: fast, balanced, or deep")
@click.option("--auto-profile", is_flag=True, help="Auto-select profile based on machine specs")
@click.option("--max-depth", "-d", type=int, help="Override max depth")
@click.option("--max-nodes", type=int, help="Override max nodes")
@click.option("--max-time", "-t", type=int, help="Override max time")
@click.option("--max-tokens", type=int, help="Override max tokens")
@click.option("--replay", type=str, help="Replay mode: 'stale', node_id, or 'subtree:node_id'")
@click.option("--orchestrator-model", "-O", help="Override orchestrator model (e.g., opus, sonnet)")
@click.option("--worker-model", "-W", help="Override worker model")
@click.option("--leaf-model", "-L", help="Override leaf model")
def resume(
    run_id: str,
    profile: str | None,
    auto_profile: bool,
    max_depth: int | None,
    max_nodes: int | None,
    max_time: int | None,
    max_tokens: int | None,
    replay: str | None,
    orchestrator_model: str | None,
    worker_model: str | None,
    leaf_model: str | None,
) -> None:
    """Resume a partial or failed run with delta verification.

    \b
    Examples:
        shad resume abc12345 --max-depth 4
        shad resume abc12345 --replay stale     # Re-run stale nodes only
        shad resume abc12345 --replay node123   # Re-run specific node
        shad resume abc12345 -O opus -W sonnet  # Override models
        shad resume abc12345 --profile deep
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("shad").setLevel(logging.INFO)

    history = HistoryManager()

    try:
        run_data = history.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run {run_id} not found[/red]")
        sys.exit(1)

    if run_data.status not in (RunStatus.PARTIAL, RunStatus.FAILED):
        console.print(f"[yellow]Run {run_id} is {run_data.status.value}, cannot resume[/yellow]")
        sys.exit(1)

    # Apply profile preset if provided
    if profile:
        profile_key = profile.lower()
        presets = {
            "fast": {"max_depth": 2, "max_nodes": 25, "max_time": 600, "max_tokens": 800000},
            "balanced": {"max_depth": 3, "max_nodes": 50, "max_time": 1200, "max_tokens": 2000000},
            "deep": {"max_depth": 4, "max_nodes": 80, "max_time": 1800, "max_tokens": 3000000},
        }
        if profile_key in presets:
            preset = presets[profile_key]
            run_data.config.budget.max_depth = preset["max_depth"]
            run_data.config.budget.max_nodes = preset["max_nodes"]
            run_data.config.budget.max_wall_time = preset["max_time"]
            run_data.config.budget.max_tokens = preset["max_tokens"]
            console.print(f"[dim][PROFILE] {profile_key} preset applied[/dim]")
    elif auto_profile:
        cpu_count, mem_gb = _get_system_specs()
        profile_key = _suggest_profile(cpu_count, mem_gb)
        presets = {
            "fast": {"max_depth": 2, "max_nodes": 25, "max_time": 600, "max_tokens": 800000},
            "balanced": {"max_depth": 3, "max_nodes": 50, "max_time": 1200, "max_tokens": 2000000},
            "deep": {"max_depth": 4, "max_nodes": 80, "max_time": 1800, "max_tokens": 3000000},
        }
        preset = presets[profile_key]
        run_data.config.budget.max_depth = preset["max_depth"]
        run_data.config.budget.max_nodes = preset["max_nodes"]
        run_data.config.budget.max_wall_time = preset["max_time"]
        run_data.config.budget.max_tokens = preset["max_tokens"]
        console.print(f"[dim][PROFILE] auto-selected {profile_key}[/dim]")

    # Override budgets if specified
    if max_depth:
        run_data.config.budget.max_depth = max_depth
    if max_nodes:
        run_data.config.budget.max_nodes = max_nodes
    if max_time:
        run_data.config.budget.max_wall_time = max_time
    if max_tokens:
        run_data.config.budget.max_tokens = max_tokens

    console.print(Panel(f"Resuming run {run_id}", title="Shad Resume", border_style="yellow"))
    if replay:
        console.print(f"[dim][REPLAY] Mode: {replay}[/dim]")

    async def _resume_run() -> Run:
        """Resume run using original or overridden model config."""
        # Start with original model config
        model_config = run_data.config.model_config_override

        # Apply any overrides
        if orchestrator_model or worker_model or leaf_model:
            if model_config:
                # Update existing config with overrides
                model_config = ModelConfig(
                    orchestrator_model=orchestrator_model or model_config.orchestrator_model,
                    worker_model=worker_model or model_config.worker_model,
                    leaf_model=leaf_model or model_config.leaf_model,
                )
            else:
                # Create new config from overrides
                model_config = ModelConfig(
                    orchestrator_model=orchestrator_model,
                    worker_model=worker_model,
                    leaf_model=leaf_model,
                )
            console.print(f"[dim]Using models: O={model_config.orchestrator_model}, W={model_config.worker_model}, L={model_config.leaf_model}[/dim]")
        elif model_config:
            console.print(f"[dim]Using original models: O={model_config.orchestrator_model}, W={model_config.worker_model}, L={model_config.leaf_model}[/dim]")

        engine = RLMEngine(llm_provider=LLMProvider(model_config=model_config))
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
# Models Command
# =============================================================================


@cli.command("models")
@click.option("--refresh", is_flag=True, help="Force refresh from Anthropic API")
@click.option("--ollama", is_flag=True, help="Also list locally installed Ollama models")
def list_models(refresh: bool, ollama: bool) -> None:
    """List available models (Claude and optionally Ollama).

    Shows available Claude models from the Anthropic API (cached for 24 hours).
    Use --refresh to force a fresh fetch from the API.
    Use --ollama to also list locally installed Ollama models.

    \b
    Examples:
        shad models              # List Claude models
        shad models --refresh    # Force refresh from API
        shad models --ollama     # Include Ollama models
    """
    from shad.utils.models import (
        get_available_models,
        get_default_models,
        is_ollama_available,
        list_ollama_models,
    )

    try:
        models = get_available_models(force_refresh=refresh)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if not models:
        console.print("[yellow]No Claude models available[/yellow]")
        console.print("[dim]Set ANTHROPIC_API_KEY to fetch models from API[/dim]")

    # Build Claude models table
    if models:
        table = Table(title="Claude Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Shorthand", style="green")
        table.add_column("Display Name")

        for model in models:
            shorthand = model.shorthand or "-"
            display_name = model.display_name or model.id
            table.add_row(model.id, shorthand, display_name)

        console.print(table)

    # List Ollama models if requested
    if ollama:
        if is_ollama_available():
            ollama_models = list_ollama_models()
            if ollama_models:
                console.print()
                ollama_table = Table(title="Ollama Models (Local)")
                ollama_table.add_column("Model ID", style="magenta")
                ollama_table.add_column("Display Name")

                for model in ollama_models:
                    display_name = model.display_name or model.id
                    ollama_table.add_row(model.id, display_name)

                console.print(ollama_table)
            else:
                console.print("\n[yellow]No Ollama models installed[/yellow]")
                console.print("[dim]Run 'ollama pull <model>' to install models[/dim]")
        else:
            console.print("\n[yellow]Ollama not available[/yellow]")
            console.print("[dim]Install Ollama from https://ollama.com[/dim]")

    # Show defaults
    defaults = get_default_models()
    console.print("\n[bold]Current Defaults:[/bold]")
    console.print(f"  Orchestrator: [cyan]{defaults['orchestrator']}[/cyan]")
    console.print(f"  Worker:       [cyan]{defaults['worker']}[/cyan]")
    console.print(f"  Leaf:         [cyan]{defaults['leaf']}[/cyan]")

    console.print("\n[dim]Use: shad run \"task\" -O <model> -W <model> -L <model>[/dim]")
    if not ollama:
        console.print("[dim]Use --ollama to see locally installed Ollama models[/dim]")


# =============================================================================
# Vault Commands
# =============================================================================

@cli.command("vault")
def vault_status() -> None:
    """Check vault and retriever status.

    \b
    Example:
        shad vault
    """
    # Check retriever availability
    qmd_retriever = QmdRetriever()
    if qmd_retriever.available:
        console.print("[green]✓ qmd retriever available[/green]")

        # Get qmd status
        async def get_status() -> dict:
            return await qmd_retriever.status()

        status = run_async(get_status())
        if status.get("collections"):
            console.print("[dim]Collections:[/dim]")
            for coll in status.get("collections", []):
                name = coll.get("name", "unknown")
                path = coll.get("path", "")
                console.print(f"  - {name}: {path}")
    else:
        console.print("[yellow]⚠ qmd not installed (falling back to filesystem search)[/yellow]")
        console.print("[dim]Install with: bun install -g https://github.com/tobi/qmd[/dim]")

    # Show default vault from env
    default_vault = get_settings().obsidian_vault_path
    if default_vault:
        console.print(f"\n[dim]Default vault (env): {default_vault}[/dim]")
    else:
        console.print("\n[dim]No default vault configured[/dim]")
        console.print("[dim]Set OBSIDIAN_VAULT_PATH or use --vault with commands[/dim]")


@cli.command("search")
@click.argument("query")
@click.option("--vault", "-v", multiple=True, help="Vault path(s) to search")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "bm25", "vector"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output results as JSON")
def search(query: str, vault: tuple[str, ...], limit: int, mode: str, as_json: bool) -> None:
    """Search vault(s) for matching documents.

    \b
    Examples:
        shad search "machine learning"
        shad search "authentication" --vault ~/Notes
        shad search "API design" --mode bm25 --limit 20
    """
    # Build vault paths
    vault_paths: list[Path] = []
    collection_names: dict[str, Path] = {}

    if vault:
        for v in vault:
            vp = Path(v).expanduser()
            if not vp.is_absolute():
                vp = Path.cwd() / vp
            vault_paths.append(vp)
            collection_names[vp.name] = vp
    elif get_settings().obsidian_vault_path:
        vp = Path(get_settings().obsidian_vault_path).expanduser()
        vault_paths.append(vp)
        collection_names[vp.name] = vp
    else:
        console.print("[yellow]No vault specified. Use --vault or set OBSIDIAN_VAULT_PATH[/yellow]")
        sys.exit(1)

    # Get retriever
    retriever = get_retriever(paths=vault_paths, collection_names=collection_names)

    async def do_search() -> list:
        return await retriever.search(
            query,
            mode=mode,
            collections=list(collection_names.keys()),
            limit=limit,
        )

    results = run_async(do_search())

    if as_json:
        import json as json_mod

        output = [result.to_dict() for result in results]
        click.echo(json_mod.dumps(output, ensure_ascii=False))
        return

    if not results:
        console.print(f"[dim]No results found for '{query}'[/dim]")
        return

    console.print(Panel(f"Search: {query}", title="Vault Search", border_style="blue"))
    console.print(f"[dim]Mode: {mode} | Backend: {type(retriever).__name__}[/dim]\n")

    for i, result in enumerate(results, 1):
        path = result.path
        collection = result.collection
        content = result.content[:200] if result.content else ""
        score = result.score

        # Show collection if multiple vaults
        if len(vault_paths) > 1 and collection:
            console.print(f"[bold cyan]{i}. [{collection}] {path}[/bold cyan] [dim](score: {score:.2f})[/dim]")
        else:
            console.print(f"[bold cyan]{i}. {path}[/bold cyan] [dim](score: {score:.2f})[/dim]")

        if result.snippet:
            console.print(f"   {result.snippet[:200]}...")
        elif content:
            console.print(f"   {content}...")


# =============================================================================
# Context Command
# =============================================================================


CONTEXT_SYNTHESIS_PROMPT = """You are a context synthesizer. Given the following documents retrieved from a knowledge vault, \
produce a concise, information-dense brief relevant to the query. Include key facts, decisions, \
and context. Do NOT include filler or meta-commentary. Target {max_chars} characters.

Query: {query}

Retrieved documents:
{documents}"""


@cli.command("context")
@click.argument("query")
@click.option("--vault", "-v", multiple=True, help="Vault path(s) to search")
@click.option("--limit", "-l", default=10, help="Max retrieval results")
@click.option("--max-chars", default=4000, help="Max output chars for brief")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "bm25", "vector"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, default=False, help="JSON output")
@click.option("--leaf-model", "-L", default=None, help="Model for synthesis (default: leaf tier)")
def context(
    query: str,
    vault: tuple[str, ...],
    limit: int,
    max_chars: int,
    mode: str,
    as_json: bool,
    leaf_model: str | None,
) -> None:
    """Retrieve and synthesize vault context into a compact brief.

    Faster than `shad run` (no DAG/decomposition), richer than `shad search`
    (includes LLM synthesis). Returns a concise context brief from vault documents.

    \b
    Examples:
        shad context "BSV authentication decisions" -v ~/Notes
        shad context "API design patterns" -v ~/vault --max-chars 2000
        shad context "project architecture" -v ~/vault --json
    """
    import json as json_mod

    # Build vault paths
    vault_paths: list[Path] = []
    collection_names: dict[str, Path] = {}

    if vault:
        for v in vault:
            vp = Path(v).expanduser()
            if not vp.is_absolute():
                vp = Path.cwd() / vp
            vault_paths.append(vp)
            collection_names[vp.name] = vp
    elif get_settings().obsidian_vault_path:
        vp = Path(get_settings().obsidian_vault_path).expanduser()
        vault_paths.append(vp)
        collection_names[vp.name] = vp
    else:
        console.print("[yellow]No vault specified. Use --vault or set OBSIDIAN_VAULT_PATH[/yellow]")
        sys.exit(1)

    # Retrieve
    retriever = get_retriever(paths=vault_paths, collection_names=collection_names)

    async def do_search() -> list:
        return await retriever.search(
            query,
            mode=mode,
            collections=list(collection_names.keys()),
            limit=limit,
        )

    results = run_async(do_search())

    if not results:
        output = {
            "brief": "",
            "sources": [],
            "query": query,
            "chars": 0,
            "retrieval_count": 0,
            "synthesis_model": None,
        }
        if as_json:
            click.echo(json_mod.dumps(output, ensure_ascii=False))
        else:
            console.print(f"[dim]No results found for '{query}'[/dim]")
        return

    # Read full content of top results for synthesis
    documents_text = ""
    sources = []
    for result in results:
        doc_content = result.content or result.snippet or ""
        if doc_content:
            documents_text += f"\n---\nFile: {result.path}\n{doc_content}\n"
        sources.append({"path": result.path, "score": round(result.score, 4)})

    # Synthesize via LLM
    model_config = ModelConfig(leaf_model=leaf_model) if leaf_model else None
    llm = LLMProvider(model_config=model_config)
    synthesis_model = llm.get_model_for_tier(ModelTier.LEAF)

    synthesis_prompt = CONTEXT_SYNTHESIS_PROMPT.format(
        max_chars=max_chars,
        query=query,
        documents=documents_text,
    )

    async def do_synthesize() -> tuple[str, int]:
        return await llm.complete(
            prompt=synthesis_prompt,
            tier=ModelTier.LEAF,
            max_tokens=max_chars // 2,  # rough char-to-token ratio
            temperature=0.3,
        )

    brief, _tokens = run_async(do_synthesize())

    # Truncate to max_chars
    if len(brief) > max_chars:
        brief = brief[:max_chars]

    output = {
        "brief": brief,
        "sources": sources,
        "query": query,
        "chars": len(brief),
        "retrieval_count": len(results),
        "synthesis_model": synthesis_model,
    }

    if as_json:
        click.echo(json_mod.dumps(output, ensure_ascii=False))
    else:
        console.print(Panel(brief, title=f"Context: {query}", border_style="green"))
        console.print(f"\n[dim]Sources ({len(sources)}):[/dim]")
        for s in sources:
            console.print(f"  [cyan]{s['path']}[/cyan] [dim](score: {s['score']:.2f})[/dim]")
        console.print(f"\n[dim]{len(brief)} chars | {len(results)} docs | model: {synthesis_model}[/dim]")


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
    """Ingest external sources into your Obsidian vault.

    \b
    Commands:
        github <url>    Ingest a GitHub repository

    \b
    Arguments:
        url             GitHub repository URL (e.g., https://github.com/user/repo)

    \b
    Options:
        -v, --vault     Target vault path (required)
        --preset        Ingestion preset: mirror, docs, deep (default: docs)

    \b
    Examples:
        shad ingest github https://github.com/user/repo --vault ~/MyVault
        shad ingest github https://github.com/user/repo -v ~/MyVault --preset deep
    """
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
        start     Start Redis and API server
        stop      Stop all services
        status    Check service status
        logs      View API server logs

    \b
    Start Options:
        -f, --foreground    Run in foreground (don't daemonize)

    \b
    Logs Options:
        -f, --follow        Follow log output (like tail -f)
        -n, --lines N       Number of lines to show (default: 50)

    \b
    Examples:
        shad server start
        shad server start --foreground
        shad server status
        shad server logs -f
        shad server stop
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
        add <type> <location>    Add a new source
        list                     List all configured sources
        remove <source_id>       Remove a source by ID
        sync                     Sync sources now
        status                   Show scheduler and sources status

    \b
    Source Types (for 'add'):
        github    GitHub repository URL
        url       Web page URL
        feed      RSS/Atom feed URL
        folder    Local folder path

    \b
    Add Options:
        -v, --vault PATH         Target vault path (required)
        -s, --schedule SCHED     Sync schedule: manual, hourly, daily, weekly, monthly
        -p, --preset PRESET      Ingestion preset (for github): mirror, docs, deep

    \b
    Sync Options:
        -a, --all               Sync all sources, not just due ones
        -d, --daemon            Run as daemon (continuous sync)
        -i, --interval SECS     Check interval in seconds (for daemon mode)

    \b
    Examples:
        shad sources add github https://github.com/org/repo --vault ~/MyVault
        shad sources add url https://docs.example.com -v ~/MyVault -s weekly
        shad sources add feed https://blog.example.com/rss -v ~/MyVault -s hourly
        shad sources list
        shad sources sync --all
        shad sources sync --daemon --interval 300
        shad sources remove abc123
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
        if run.stop_reason == StopReason.BUDGET_TIME:
            console.print(
                "[yellow]Hit max wall time. Consider re-running with --max-time (e.g., 1800) or set DEFAULT_MAX_WALL_TIME in ~/.shad/.env[/yellow]"
            )

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


# =============================================================================
# Project Initialization
# =============================================================================


SHAD_PERMISSIONS = [
    # File operations - shad needs to read/write generated code
    "Read",
    "Edit",
    "Write",
    # Bash - for running builds, tests, linters
    "Bash(npm:*)",
    "Bash(npx:*)",
    "Bash(node:*)",
    "Bash(python:*)",
    "Bash(pip:*)",
    "Bash(pytest:*)",
    "Bash(ruff:*)",
    "Bash(mypy:*)",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(mkdir:*)",
    "Bash(cp:*)",
    "Bash(mv:*)",
]


@cli.command("init")
@click.argument("path", default=".", type=click.Path())
@click.option("--force", "-f", is_flag=True, help="Overwrite existing .claude settings")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def init(path: str, force: bool, yes: bool) -> None:
    """Initialize a project for use with shad.

    Creates .claude/settings.json with permissions that allow shad to:
    - Read, write, and edit files
    - Run builds, tests, and linters
    - Execute common development commands

    \b
    Examples:
        shad init                    # Initialize current directory
        shad init ~/projects/myapp   # Initialize specific project
        shad init --force            # Overwrite existing settings
    """
    import json

    project_path = Path(path).expanduser().resolve()
    claude_dir = project_path / ".claude"
    settings_file = claude_dir / "settings.json"

    if not project_path.exists():
        console.print(f"[red]Path does not exist: {project_path}[/red]")
        sys.exit(1)

    # Check for existing settings
    if settings_file.exists() and not force:
        console.print(f"[yellow].claude/settings.json already exists at {project_path}[/yellow]")
        console.print("[dim]Use --force to overwrite[/dim]")
        sys.exit(1)

    # Show what permissions will be granted
    console.print(Panel(
        f"[bold]Project:[/bold] {project_path}",
        title="Shad Project Initialization",
        border_style="blue"
    ))

    console.print("\n[bold]The following permissions will be granted to Claude Code:[/bold]\n")

    table = Table(show_header=False, box=None)
    table.add_column("Permission", style="cyan")
    table.add_column("Description")

    table.add_row("Read", "Read any file in the project")
    table.add_row("Edit", "Edit any file in the project")
    table.add_row("Write", "Create new files in the project")
    table.add_row("Bash(npm/npx/node:*)", "Run Node.js commands")
    table.add_row("Bash(python/pip:*)", "Run Python commands")
    table.add_row("Bash(pytest/ruff/mypy:*)", "Run test and lint tools")
    table.add_row("Bash(git status/diff/log:*)", "Read-only git commands")
    table.add_row("Bash(ls/cat/mkdir/cp/mv:*)", "Basic file operations")

    console.print(table)
    console.print()

    # Confirm unless --yes
    if not yes:
        if not click.confirm("Create .claude/settings.json with these permissions?"):
            console.print("[dim]Aborted[/dim]")
            sys.exit(0)

    # Create settings
    settings = {
        "_comment": "Shad project permissions - allows automated code generation",
        "permissions": {
            "allow": SHAD_PERMISSIONS,
        },
    }

    # Write settings
    claude_dir.mkdir(parents=True, exist_ok=True)
    with settings_file.open("w") as f:
        json.dump(settings, f, indent=2)

    console.print(f"\n[green]Created {settings_file}[/green]")
    console.print("\n[dim]You can now run shad commands in this project without permission prompts.[/dim]")
    console.print("[dim]To revoke permissions, delete .claude/settings.json[/dim]")


@cli.command("doctor")
@click.option("--fix", is_flag=True, help="Attempt to fix common issues (qmd install)")
def doctor(fix: bool) -> None:
    """Run environment checks for common issues.

    
    Examples:
        shad doctor
    """
    import shutil
    import subprocess

    console.print(Panel("Shad Doctor", border_style="blue"))

    # Check shad binary (self)
    console.print(f"[green]✔[/green] Shad CLI available")

    # Check qmd
    qmd_path = shutil.which("qmd")
    if qmd_path:
        console.print(f"[green]✔[/green] qmd detected: {qmd_path}")
        try:
            result = subprocess.run(["qmd", "status", "--json"], capture_output=True, timeout=5)
            if result.returncode == 0:
                console.print("[green]✔[/green] qmd status OK")
            else:
                console.print("[yellow]⚠[/yellow] qmd status returned non-zero")
        except Exception:
            console.print("[yellow]⚠[/yellow] qmd status check failed")
    else:
        console.print("[yellow]⚠[/yellow] qmd not found (filesystem search only)")
        if fix:
            qmd_repo = os.environ.get("QMD_REPO", "https://github.com/jonesj38/qmd#feat/openai-embeddings")
            bun = shutil.which("bun")
            npm = shutil.which("npm")
            if bun or npm:
                if click.confirm(f"Install qmd from {qmd_repo} now?"):
                    try:
                        if bun:
                            subprocess.run(["bun", "install", "-g", qmd_repo], check=False)
                        else:
                            subprocess.run(["npm", "install", "-g", qmd_repo], check=False)
                        console.print("[green]✔[/green] qmd install attempted")
                    except Exception:
                        console.print("[red]✖[/red] qmd install failed")
            else:
                console.print("[yellow]⚠[/yellow] bun/npm not found; cannot install qmd automatically")

    # Check Redis
    redis_url = get_settings().redis_url
    try:
        hostport = redis_url.split("//", 1)[-1]
        host, port = hostport.split(":", 1)
        port = int(port.split("/", 1)[0])
        import socket
        with socket.create_connection((host, port), timeout=2):
            console.print(f"[green]✔[/green] Redis reachable at {host}:{port}")
    except Exception:
        console.print(f"[yellow]⚠[/yellow] Redis not reachable at {redis_url}")

    # Check vault env
    vault_path = get_settings().obsidian_vault_path
    if vault_path:
        console.print(f"[green]✔[/green] OBSIDIAN_VAULT_PATH set: {vault_path}")
    else:
        console.print("[yellow]⚠[/yellow] OBSIDIAN_VAULT_PATH not set (pass --vault)")

    # Offer to register vault + embed if fixing and vault set
    if fix and vault_path and qmd_path:
        if click.confirm(f"Register vault {vault_path} with qmd and embed now?"):
            try:
                subprocess.run(["qmd", "collection", "add", vault_path, "--name", Path(vault_path).name], check=False)
                subprocess.run(["qmd", "embed"], check=False, env={**os.environ, "QMD_OPENAI": "1"})
                console.print("[green]✔[/green] qmd collection add + embed attempted")
            except Exception:
                console.print("[red]✖[/red] qmd embed failed")

    # Suggest profile based on machine specs
    cpu_count, mem_gb = _get_system_specs()
    suggestion = _suggest_profile(cpu_count, mem_gb)
    console.print(
        f"[dim][HINT] Suggested profile for this machine: {suggestion} ({cpu_count} CPU / {mem_gb:.1f} GB)[/dim]"
    )


@cli.command("check-permissions")
@click.argument("path", default=".", type=click.Path())
def check_permissions(path: str) -> None:
    """Check if a project has shad permissions configured.

    \b
    Examples:
        shad check-permissions
        shad check-permissions ~/projects/myapp
    """
    import json

    project_path = Path(path).expanduser().resolve()
    settings_file = project_path / ".claude" / "settings.json"

    if not settings_file.exists():
        console.print(f"[yellow]No .claude/settings.json found at {project_path}[/yellow]")
        console.print("[dim]Run 'shad init' to set up permissions[/dim]")
        sys.exit(1)

    try:
        with settings_file.open() as f:
            settings = json.load(f)
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON in {settings_file}[/red]")
        sys.exit(1)

    permissions = settings.get("permissions", {}).get("allow", [])

    if not permissions:
        console.print(f"[yellow]No permissions configured in {settings_file}[/yellow]")
        console.print("[dim]Run 'shad init --force' to set up permissions[/dim]")
        sys.exit(1)

    console.print(f"[green]Permissions configured at {project_path}[/green]\n")

    # Check which shad permissions are present
    missing = []
    for perm in SHAD_PERMISSIONS:
        if perm in permissions:
            console.print(f"  [green]✓[/green] {perm}")
        else:
            console.print(f"  [yellow]○[/yellow] {perm} [dim](not configured)[/dim]")
            missing.append(perm)

    if missing:
        console.print(f"\n[yellow]Missing {len(missing)} shad permissions[/yellow]")
        console.print("[dim]Run 'shad init --force' to update[/dim]")
    else:
        console.print("\n[green]All shad permissions are configured[/green]")


if __name__ == "__main__":
    cli()
