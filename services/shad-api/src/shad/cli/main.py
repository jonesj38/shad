"""Shad CLI - Command-line interface for Shannon's Daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shlex
import socket
import sys
import time
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
from shad.engine.strategies import StrategySelector, StrategyType
from shad.history import HistoryManager
from shad.models import Budget, ModelConfig, RunConfig
from shad.models.run import NodeStatus, Run, RunStatus, StopReason
from shad.retrieval import QmdRetriever, get_retriever
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
            with open("/proc/meminfo", encoding="utf-8") as f:
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


def _get_local_api_base_url() -> str:
    """Return a loopback URL for local health checks."""
    settings = get_settings()
    return f"http://127.0.0.1:{settings.api_port}"


def _api_health_url() -> str:
    """Return the local API health-check URL."""
    return f"{_get_local_api_base_url()}/v1/health"


def _api_listen_port() -> int:
    """Return the configured API listen port."""
    return int(get_settings().api_port)


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Best-effort local port occupancy check."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _find_next_available_port(start_port: int, host: str = "127.0.0.1", max_tries: int = 20) -> int | None:
    """Find the next available local port after start_port."""
    for candidate in range(start_port + 1, start_port + max_tries + 1):
        if not _port_in_use(candidate, host=host):
            return candidate
    return None


def _pid_is_running(pid: int) -> bool:
    """Return whether a PID is currently alive."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid_file(pid_file: Path) -> int | None:
    """Read a PID file safely."""
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _api_is_healthy(base_url: str | None = None, timeout: float = 1.5) -> bool:
    """Return whether the local API health endpoint is responding."""
    url = f"{base_url or _get_local_api_base_url()}/v1/health"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            return response.status_code == 200
    except Exception:
        return False


def _api_supports_async_runs(base_url: str | None = None, timeout: float = 1.5) -> bool:
    """Return whether the API exposes the async run surface."""
    url = f"{base_url or _get_local_api_base_url()}/openapi.json"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
        payload = response.json()
    except Exception:
        return False

    paths = payload.get("paths", {})
    runs_entry = paths.get("/v1/runs", {})
    return "post" in runs_entry and "get" in runs_entry


def _wait_for_api_health(
    timeout_s: float = 20.0,
    require_async_runs: bool = False,
    base_url: str | None = None,
) -> bool:
    """Wait until the API becomes healthy, optionally requiring async routes."""
    deadline = time.time() + timeout_s
    resolved_base_url = base_url or _get_local_api_base_url()

    while time.time() < deadline:
        if _api_is_healthy(resolved_base_url):
            if not require_async_runs or _api_supports_async_runs(resolved_base_url):
                return True
        time.sleep(0.5)
    return False


def _find_repo_dir(shad_home: Path) -> Path:
    """Resolve the Shad repo root for docker-compose and dev installs."""
    env_repo = os.environ.get("SHAD_REPO_PATH")
    candidates: list[Path] = []

    if env_repo:
        candidates.append(Path(env_repo).expanduser())
    candidates.append(shad_home / "repo")
    candidates.append(Path.cwd())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "docker-compose.yml").exists() and (candidate / "services" / "shad-api").exists():
            return candidate

    return (shad_home / "repo").resolve()


def _find_api_dir(repo_dir: Path) -> Path:
    """Resolve the API service directory for uvicorn startup."""
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "src" / "shad" / "api" / "main.py").exists():
            return candidate

    return repo_dir / "services" / "shad-api"


def _server_meta_path(shad_home: Path) -> Path:
    """Path to managed server metadata."""
    return shad_home / "shad-api.json"


def _read_server_meta(shad_home: Path) -> dict[str, Any]:
    """Load managed server metadata, tolerating legacy pid-only installs."""
    meta_path = _server_meta_path(shad_home)
    pid_file = shad_home / "shad-api.pid"

    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass

    legacy_pid = _read_pid_file(pid_file)
    if legacy_pid is None:
        return {}

    port = _api_listen_port()
    return {
        "pid": legacy_pid,
        "port": port,
        "api_url": f"http://127.0.0.1:{port}",
    }


def _write_server_meta(shad_home: Path, *, pid: int, port: int) -> None:
    """Persist managed server metadata."""
    payload = {
        "pid": pid,
        "port": port,
        "api_url": f"http://127.0.0.1:{port}",
        "updated_at": time.time(),
    }
    _server_meta_path(shad_home).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (shad_home / "shad-api.pid").write_text(str(pid), encoding="utf-8")


def _clear_server_meta(shad_home: Path) -> None:
    """Remove managed server metadata files."""
    _server_meta_path(shad_home).unlink(missing_ok=True)
    (shad_home / "shad-api.pid").unlink(missing_ok=True)


def _resolve_collection_inputs(
    collection: tuple[str, ...],
    legacy_collection: tuple[str, ...],
) -> tuple[list[Path], dict[str, Path], str]:
    """Resolve collection paths from CLI options or environment."""
    _all_paths = collection or legacy_collection
    collection_paths: list[Path] = []
    collection_names: dict[str, Path] = {}

    if _all_paths:
        source = "CLI"
        for value in _all_paths:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            collection_paths.append(path)
            collection_names[path.name] = path
    elif get_settings().default_collection_path:
        source = "env"
        path = Path(get_settings().default_collection_path).expanduser()
        collection_paths.append(path)
        collection_names[path.name] = path
    else:
        source = "none"

    return collection_paths, collection_names, source


def _recommended_profile(explicit: str | None, auto_profile: bool) -> tuple[str | None, str | None]:
    """Determine the profile choice and how it was selected."""
    if explicit:
        return explicit.lower(), "explicit"
    if auto_profile:
        cpu_count, mem_gb = _get_system_specs()
        return _suggest_profile(cpu_count, mem_gb), "auto"
    cpu_count, mem_gb = _get_system_specs()
    return _suggest_profile(cpu_count, mem_gb), "suggested"


def _slugify_goal(goal: str) -> str:
    """Create a directory-friendly slug from a goal."""
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    return slug[:48] or "shad-output"


def _build_run_command_preview(
    goal: str,
    collection_paths: list[Path],
    strategy: str,
    profile: str | None,
    verify: str,
    write_files: bool,
    output_dir: str | None,
) -> str:
    """Build a recommended shad run command."""
    parts = ["shad", "run", shlex.quote(goal)]
    for path in collection_paths:
        parts.extend(["--collection", shlex.quote(str(path))])
    parts.extend(["--strategy", strategy])
    if profile:
        parts.extend(["--profile", profile])
    parts.extend(["--verify", verify])
    if write_files:
        parts.append("--write-files")
    if output_dir:
        parts.extend(["--output-dir", shlex.quote(output_dir)])
    return " ".join(parts)


def _render_runs_table(runs: list[dict[str, Any]], title: str = "Shad Runs") -> None:
    """Render a run listing table."""
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title=title)
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Goal")

    for run in runs:
        run_id = str(run.get("run_id", ""))[:8]
        status_value = str(run.get("status", "unknown"))
        created = str(run.get("created_at", "") or "unknown")
        goal = str(run.get("goal", "") or "")

        if created and created != "unknown":
            created = created.replace("T", " ")[:19]
        if len(goal) > 70:
            goal = goal[:67] + "..."

        table.add_row(run_id, status_value, created, goal)

    console.print(table)


@click.group()
@click.version_option(version=__version__, prog_name="shad")
def cli() -> None:
    """Shad - Shannon's Daemon: Personal AI Infrastructure.

    \b
    Core Commands:
        run <goal>           Execute a reasoning task
        runs                 List recent runs
        plan <goal>          Preflight a task and recommend a run command
        status <run_id>      Check the status of a run
        cancel <run_id>      Cancel a remote async run
        resume <run_id>      Resume a partial/failed run
        export <run_id>      Export files from a completed run
        debug <run_id>       Enter debug mode for a run

    \b
    Project Setup:
        init [path]              Initialize project permissions for Claude Code
        check-permissions [path] Verify project permissions are configured

    \b
    Collection Commands:
        search <query>       Search collection(s)
        ingest ...           Ingest sources into a collection (see: shad ingest --help)

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
        shad plan "Build a REST API" --collection ~/Notes
        shad run "Build a REST API" --collection ~/Notes  # Run with collection context
        shad status <run_id>                         # Check progress
    """
    pass


@cli.command("run")
@click.argument("goal")
@click.option("--collection", "-c", multiple=True, help="Collection path(s) for context (can specify multiple)")
@click.option("--vault", "legacy_collection", "-v", multiple=True, hidden=True, help="[deprecated: use --collection] Collection path(s)")
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
@click.option("--decay-halflife", default=30.0, type=float, metavar="DAYS",
              help="Half-life for temporal score decay in days (0 = disable decay, default: 30)")
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
@click.option("--gemini", is_flag=True, help="Use Gemini CLI instead of Claude Code")
def run(
    goal: str,
    collection: tuple[str, ...],
    legacy_collection: tuple[str, ...],
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
    decay_halflife: float,
    strategy: str | None,
    verify: str,
    write_files: bool,
    output_dir: str | None,
    quiet: bool,
    orchestrator_model: str | None,
    worker_model: str | None,
    leaf_model: str | None,
    gemini: bool,
) -> None:
    """Execute a reasoning task.

    \b
    Examples:
        shad run "Explain quantum computing"
        shad run "Summarize research" --collection ~/Notes
        shad run "Build REST API" --collection ~/Project --collection ~/Patterns
        shad run "Complex task" -O opus -W sonnet -L haiku
        shad run "Fast summary" --profile fast
        shad run "Auto profile" --auto-profile
        shad run "Using Gemini" --gemini -O gemini-3-pro-preview
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

    if write_files and output and not output_dir:
        output_dir = output
        output = None
        console.print("[yellow][HINT] Interpreting --output as --output-dir because --write-files is enabled[/yellow]")

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

    # If API specified, use remote execution
    if api:
        _run_via_api(
            api=api,
            payload={
                "goal": goal,
                "collection_path": (collection or legacy_collection)[0] if (collection or legacy_collection) else None,
                "collection_paths": list(collection or legacy_collection) if (collection or legacy_collection) else None,
                "budget": {
                    "max_depth": max_depth,
                    "max_nodes": max_nodes,
                    "max_wall_time": max_time,
                    "max_tokens": max_tokens,
                },
                "voice": voice,
                "strategy": strategy,
                "verify": verify,
                "write_files": write_files,
                "output_path": output_dir,
                "use_gemini_cli": gemini,
                "orchestrator_model": orchestrator_model,
                "worker_model": worker_model,
                "leaf_model": leaf_model,
            },
        )
        return

    # Merge --collection and deprecated --vault
    collection_paths, collection_names, source = _resolve_collection_inputs(collection, legacy_collection)

    # Primary collection path (first one) for backward compatibility
    primary_collection = collection_paths[0] if collection_paths else None

    config = RunConfig(
        goal=goal,
        collection_path=str(primary_collection) if primary_collection else None,
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

    if collection_paths:
        if len(collection_paths) == 1:
            console.print(f"[dim][CONTEXT] Using collection ({source}): {collection_paths[0]}[/dim]")
        else:
            console.print(f"[dim][CONTEXT] Using {len(collection_paths)} collections ({source}):[/dim]")
            for cp in collection_paths:
                console.print(f"[dim]  - {cp.name}: {cp}[/dim]")
        collections = list(collection_names.keys())

        # Get retriever based on preference
        retriever_instance = get_retriever(
            paths=collection_paths,
            collection_names=collection_names,
            prefer=retriever,
        )
        console.print(f"[dim][RETRIEVER] Using {type(retriever_instance).__name__}[/dim]")

        # Auto-provision qmd collections for collection paths
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
        console.print("[dim][CONTEXT] No collection specified (use --collection or set SHAD_COLLECTION_PATH)[/dim]")
        use_code_mode = False

    # Display strategy and verification options
    if strategy:
        console.print(f"[dim][STRATEGY] Override: {strategy}[/dim]")
    console.print(f"[dim][VERIFY] Level: {verify}[/dim]")
    if write_files:
        console.print(f"[dim][OUTPUT] Write files enabled{f' → {output_dir}' if output_dir else ''}[/dim]")

    # Provider info
    if gemini:
        console.print("[dim][PROVIDER] Using Gemini CLI[/dim]")
    else:
        console.print("[dim][PROVIDER] Using Claude Code CLI[/dim]")

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

    if dry_run:
        llm = LLMProvider(model_config=model_config)
        orch = llm.get_model_for_tier(ModelTier.ORCHESTRATOR)
        work = llm.get_model_for_tier(ModelTier.WORKER)
        leaf = llm.get_model_for_tier(ModelTier.LEAF)
        retriever_name = type(retriever_instance).__name__ if retriever_instance else "None"

        table = Table(title="Shad Dry Run", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        table.add_row("Budget", f"depth={max_depth}, nodes={max_nodes}, time={max_time}s, tokens={max_tokens}")
        table.add_row("Retriever", retriever_name)
        table.add_row("Collections", str(len(collection_paths)) if collection_paths else "none")
        table.add_row("Models", f"O={orch}, W={work}, L={leaf}")
        table.add_row("Mode", "qmd-hybrid" if qmd_hybrid else ("code-mode" if use_code_mode else "direct"))
        console.print(table)
        return

    history = HistoryManager()

    async def _execute_run() -> Run:
        """Execute run."""
        engine = RLMEngine(
            llm_provider=LLMProvider(
                model_config=model_config,
                use_gemini_cli=gemini,
                use_claude_code=not gemini,
            ),
            retriever=retriever_instance,
            collection_path=primary_collection,
            collections=collections,
            use_code_mode=use_code_mode,
            use_qmd_hybrid=qmd_hybrid,
            decay_halflife_days=decay_halflife,
            history=history,
        )
        return await engine.execute(config)

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


@cli.command("plan")
@click.argument("goal")
@click.option("--collection", "-c", multiple=True, help="Collection path(s) for context (can specify multiple)")
@click.option("--vault", "legacy_collection", "-v", multiple=True, hidden=True, help="[deprecated: use --collection] Collection path(s)")
@click.option("--retriever", "-r", type=click.Choice(["auto", "qmd", "filesystem"]), default="auto",
              help="Retrieval backend (auto detects qmd)")
@click.option("--profile", type=click.Choice(["fast", "balanced", "deep"], case_sensitive=False),
              help="Preset budget profile override")
@click.option("--auto-profile", is_flag=True, help="Auto-select profile based on machine specs")
@click.option("--strategy", "-s", type=click.Choice(["software", "research", "analysis", "planning"]),
              help="Override automatic strategy selection")
@click.option("--verify", type=click.Choice(["off", "basic", "build", "strict"]),
              help="Verification level override")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output plan as JSON")
def plan(
    goal: str,
    collection: tuple[str, ...],
    legacy_collection: tuple[str, ...],
    retriever: str,
    profile: str | None,
    auto_profile: bool,
    strategy: str | None,
    verify: str | None,
    as_json: bool,
) -> None:
    """Preflight a task and recommend a run command.

    Examples:
        shad plan "Build a task app" --collection ~/TeamDocs
        shad plan "Analyze this architecture" --collection ~/Docs --json
    """
    import json as json_mod

    collection_paths, collection_names, source = _resolve_collection_inputs(collection, legacy_collection)
    if not collection_paths:
        console.print("[yellow]No collection specified. Use --collection or set SHAD_COLLECTION_PATH[/yellow]")
        sys.exit(1)

    profile_choice, profile_source = _recommended_profile(profile, auto_profile)
    selector = StrategySelector()
    override = StrategyType(strategy) if strategy else None
    strategy_result = selector.select(goal, override=override)
    recommended_strategy = strategy_result.strategy_type.value
    recommended_verify = verify or ("strict" if recommended_strategy == "software" else "basic")
    should_write_files = recommended_strategy == "software"
    recommended_output_dir = f"./{_slugify_goal(goal)}" if should_write_files else None

    retriever_instance = get_retriever(
        paths=collection_paths,
        collection_names=collection_names,
        prefer=retriever,
    )

    async def do_search() -> list:
        return await retriever_instance.search(
            goal,
            mode="hybrid",
            collections=list(collection_names.keys()),
            limit=5,
        )

    results = run_async(do_search())
    command_preview = _build_run_command_preview(
        goal=goal,
        collection_paths=collection_paths,
        strategy=recommended_strategy,
        profile=profile_choice,
        verify=recommended_verify,
        write_files=should_write_files,
        output_dir=recommended_output_dir,
    )

    plan_data = {
        "goal": goal,
        "collections": [str(path) for path in collection_paths],
        "collection_source": source,
        "retriever": type(retriever_instance).__name__,
        "strategy": recommended_strategy,
        "strategy_confidence": round(strategy_result.confidence, 2),
        "matched_keywords": strategy_result.matched_keywords,
        "profile": profile_choice,
        "profile_source": profile_source,
        "verify": recommended_verify,
        "write_files": should_write_files,
        "output_dir": recommended_output_dir,
        "retrieval_hits": len(results),
        "top_hits": [
            {
                "path": result.path,
                "collection": result.collection,
                "score": round(result.score, 4),
            }
            for result in results[:3]
        ],
        "recommended_command": command_preview,
    }

    if as_json:
        click.echo(json_mod.dumps(plan_data, ensure_ascii=False, indent=2))
        return

    console.print(Panel(f"[bold]Goal:[/bold] {goal}", title="Shad Plan", border_style="cyan"))
    console.print(f"[dim][CONTEXT] Using {len(collection_paths)} collection(s) from {source}[/dim]")
    console.print(f"[dim][RETRIEVER] {type(retriever_instance).__name__}[/dim]")
    console.print(f"[dim][STRATEGY] {recommended_strategy} (confidence {strategy_result.confidence:.2f})[/dim]")
    console.print(f"[dim][PROFILE] {profile_choice} ({profile_source})[/dim]")
    console.print(f"[dim][VERIFY] {recommended_verify}[/dim]")
    if should_write_files and recommended_output_dir:
        console.print(f"[dim][OUTPUT] write files -> {recommended_output_dir}[/dim]")

    if results:
        console.print("\n[bold]Retrieval Check[/bold]")
        for idx, result in enumerate(results[:3], 1):
            prefix = f"[{result.collection}] " if result.collection else ""
            console.print(f"{idx}. [cyan]{prefix}{result.path}[/cyan] [dim](score: {result.score:.2f})[/dim]")
    else:
        console.print("\n[yellow]No retrieval hits for this goal. Validate the collection with `shad search` or qmd indexing before the full run.[/yellow]")

    console.print("\n[bold]Recommended Command[/bold]")
    console.print(f"[green]{command_preview}[/green]")


@cli.command("runs")
@click.option("--api", default=None, help="Shad API URL (defaults to local history if omitted)")
@click.option("--limit", default=20, help="Maximum runs to show")
@click.option("--active", is_flag=True, help="Show only pending/running runs")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output runs as JSON")
def list_runs_command(api: str | None, limit: int, active: bool, as_json: bool) -> None:
    """List recent runs from local history or the API.

    Examples:
        shad runs
        shad runs --active
        shad runs --api http://localhost:8000
    """
    import json as json_mod

    runs: list[dict[str, Any]]
    title: str

    if api:
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{api}/v1/runs", params={"limit": limit})
                response.raise_for_status()
                payload = response.json()
        except httpx.ConnectError:
            console.print("[red]Could not connect to Shad API. Is it running?[/red]")
            console.print(f"[dim]Tried: {api}[/dim]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

        runs = payload.get("runs", [])
        title = "Remote Runs"
    else:
        history = HistoryManager()
        runs = history.list_runs(limit=limit)
        title = "Local Runs"

    if active:
        runs = [run for run in runs if run.get("status") in {"pending", "running"}]

    if as_json:
        click.echo(json_mod.dumps(runs, ensure_ascii=False, indent=2))
        return

    _render_runs_table(runs, title=title)


@cli.group("collection", invoke_without_command=True)
@click.option("--collection", "-c", multiple=True, help="Collection path(s) to inspect")
@click.option("--vault", "legacy_collection", "-v", multiple=True, hidden=True, help="[deprecated: use --collection]")
@click.pass_context
def collection_group(ctx: click.Context, collection: tuple[str, ...], legacy_collection: tuple[str, ...]) -> None:
    """Manage collections and retriever status.

    \b
    Subcommands:
        list           List all indexed collections
        add <path>     Add a directory as a collection
        remove <name>  Remove a collection
        embed          Generate vector embeddings

    \b
    Without a subcommand, shows collection and retriever status.

    \b
    Examples:
        shad collection                              # Show status
        shad collection list                         # List all collections
        shad collection add ~/Notes --name my-notes  # Add collection
        shad collection remove my-notes              # Remove collection
        shad collection embed                        # Generate embeddings
    """
    if ctx.invoked_subcommand is not None:
        return

    import shutil

    collection_paths, collection_names, source = _resolve_collection_inputs(collection, legacy_collection)
    retriever = get_retriever(
        paths=collection_paths or None,
        collection_names=collection_names or None,
    ) if collection_paths else QmdRetriever()

    console.print(Panel("Collection Status", border_style="blue"))
    console.print(f"[dim][RETRIEVER] {type(retriever).__name__}[/dim]")
    console.print(f"[dim][QMD] {'available' if shutil.which('qmd') else 'not installed'}[/dim]")

    if collection_paths:
        console.print(f"[dim][SOURCE] {source}[/dim]")
        for path in collection_paths:
            exists = "exists" if path.exists() else "missing"
            console.print(f"[dim]  - {path} ({exists})[/dim]")
    else:
        console.print("[yellow]No collection specified. Use --collection or set SHAD_COLLECTION_PATH[/yellow]")

    if isinstance(retriever, QmdRetriever) and retriever.available:
        async def get_status() -> dict[str, Any]:
            return await retriever.status()

        status_info = run_async(get_status())
        collections_list = status_info.get("collections", [])
        if collections_list:
            console.print("\n[bold]Indexed Collections[/bold]")
            for coll in collections_list:
                console.print(f"[dim]- {coll.get('name', 'unknown')}: {coll.get('path', '')}[/dim]")


@collection_group.command("list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
def collection_list(as_json: bool) -> None:
    """List all indexed collections."""
    retriever = QmdRetriever()
    if not retriever.available:
        console.print("[red]qmd is not installed. Install it for collection management.[/red]")
        sys.exit(1)

    collections_data = run_async(retriever.list_collections())

    if as_json:
        click.echo(json.dumps(collections_data, ensure_ascii=False, indent=2))
        return

    if not collections_data:
        console.print("[yellow]No collections found. Use 'shad collection add <path>' to create one.[/yellow]")
        return

    table = Table(title="Collections", border_style="blue")
    table.add_column("Name", style="cyan bold")
    table.add_column("Path")
    table.add_column("Pattern", style="dim")
    table.add_column("Files", justify="right")

    for coll in collections_data:
        table.add_row(
            coll.get("name", ""),
            coll.get("path", ""),
            coll.get("pattern", "**/*.md"),
            str(coll.get("files", "")),
        )

    console.print(table)


@collection_group.command("add")
@click.argument("path")
@click.option("--name", "-n", default=None, help="Collection name (defaults to directory name)")
@click.option("--mask", "-m", default="**/*.md", help="File glob pattern (default: **/*.md)")
def collection_add(path: str, name: str | None, mask: str) -> None:
    """Add a directory as a searchable collection.

    \b
    Examples:
        shad collection add ~/Notes
        shad collection add ~/Code --name my-code --mask "**/*.py"
        shad collection add ./docs --mask "**/*.md,**/*.txt"
    """
    retriever = QmdRetriever()
    if not retriever.available:
        console.print("[red]qmd is not installed. Install it for collection management.[/red]")
        sys.exit(1)

    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        console.print(f"[red]Path does not exist: {target_path}[/red]")
        sys.exit(1)

    coll_name = name or target_path.name

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Adding collection '{coll_name}'...", total=None)
        success = run_async(retriever.add_collection(str(target_path), name=coll_name, mask=mask))

    if success:
        console.print(f"[green]✓[/green] Collection '{coll_name}' added: {target_path}")
        console.print(f"[dim]  Pattern: {mask}[/dim]")
        console.print("\n[dim]Run 'shad collection embed' to generate vector embeddings.[/dim]")
    else:
        console.print("[red]Failed to add collection. It may already exist.[/red]")
        console.print("[dim]Use 'shad collection list' to check existing collections.[/dim]")
        sys.exit(1)


@collection_group.command("remove")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def collection_remove(name: str, yes: bool) -> None:
    """Remove a collection from the index.

    This only removes the index entry — it does NOT delete any files.

    \b
    Examples:
        shad collection remove my-notes
        shad collection remove old-docs -y
    """
    retriever = QmdRetriever()
    if not retriever.available:
        console.print("[red]qmd is not installed.[/red]")
        sys.exit(1)

    if not yes:
        click.confirm(f"Remove collection '{name}' from the index?", abort=True)

    async def do_remove() -> bool:
        try:
            process = await asyncio.create_subprocess_exec(
                "qmd", "collection", "remove", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                console.print(f"[red]{stderr.decode().strip()}[/red]")
                return False
            return True
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False

    if run_async(do_remove()):
        console.print(f"[green]✓[/green] Collection '{name}' removed from index")
    else:
        sys.exit(1)


@collection_group.command("embed")
@click.option("--force", "-f", is_flag=True, help="Force re-embedding of all chunks")
def collection_embed(force: bool) -> None:
    """Generate vector embeddings for all collections.

    Processes any chunks that don't yet have vector embeddings.
    Uses the configured embedding provider (local GPU by default).

    \b
    Examples:
        shad collection embed         # Embed new chunks only
        shad collection embed --force  # Re-embed everything
    """
    import shutil
    import subprocess

    if not shutil.which("qmd"):
        console.print("[red]qmd is not installed. Install it for embedding support.[/red]")
        sys.exit(1)

    # Run qmd embed with live output (streaming to terminal)
    args = ["qmd", "embed"]
    if force:
        args.append("-f")

    console.print(Panel("Generating Embeddings", border_style="blue"))
    console.print(f"[dim]Running: {' '.join(args)}[/dim]\n")

    result = subprocess.run(args, env={**os.environ, "QMD_OPENAI": "0"})
    if result.returncode == 0:
        console.print("\n[green]✓[/green] Embeddings generated successfully")
    else:
        console.print("\n[red]Embedding failed[/red]")
        sys.exit(result.returncode)


def _run_via_api(api: str, payload: dict[str, Any]) -> None:
    """Run a goal via the API."""
    import json as json_mod
    import time

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{api}/v1/runs", json=payload)
            response.raise_for_status()
            data = response.json()

            run_id = data["run_id"]
            event_offset = 0

            def _poll_until_complete() -> dict[str, Any]:
                nonlocal event_offset

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task_id = progress.add_task("Queued...", total=None)

                    while True:
                        events_response = client.get(
                            f"{api}/v1/runs/{run_id}/events",
                            params={"since": event_offset, "limit": 100},
                        )
                        events_response.raise_for_status()
                        events_data = events_response.json()
                        event_offset = events_data["next_offset"]

                        for event in events_data["events"]:
                            event_type = event.get("type", "event")
                            if event_type == "run_started":
                                progress.update(task_id, description="Running...")
                            elif event_type == "run_completed":
                                progress.update(task_id, description="Finalizing...")
                            elif event_type == "run_failed":
                                progress.update(task_id, description="Failed")
                            console.print(f"[dim][API] {event_type}[/dim]")

                        status_response = client.get(f"{api}/v1/runs/{run_id}")
                        status_response.raise_for_status()
                        run_data: dict[str, Any] = status_response.json()
                        status_value = run_data.get("status", "unknown")

                        if status_value == "running":
                            progress.update(task_id, description="Running...")
                        elif status_value == "pending":
                            progress.update(task_id, description="Queued...")

                        if status_value in {"complete", "partial", "failed", "aborted"}:
                            progress.update(task_id, description="Complete")
                            return run_data

                        time.sleep(2)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Queued...", total=None)
                progress.stop()

            try:
                with client.stream("GET", f"{api}/v1/runs/{run_id}/events/stream") as stream:
                    if stream.status_code != 200:
                        data = _poll_until_complete()
                    else:
                        terminal = False
                        for line in stream.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            event = json_mod.loads(line[6:])
                            event_type = event.get("type", "event")
                            console.print(f"[dim][API] {event_type}[/dim]")
                            if event_type in {"run_completed", "run_failed"}:
                                terminal = True
                                break

                        if not terminal:
                            data = _poll_until_complete()
                        else:
                            status_response = client.get(f"{api}/v1/runs/{run_id}")
                            status_response.raise_for_status()
                            data = status_response.json()
            except Exception:
                data = _poll_until_complete()
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


@cli.command("cancel")
@click.argument("run_id")
@click.option("--api", default=None, help="Shad API URL (defaults to local server)")
def cancel_run(run_id: str, api: str | None) -> None:
    """Cancel a remote async run.

    Example:
        shad cancel abc12345
        shad cancel abc12345 --api http://localhost:8000
    """
    api_url = api or get_api_url()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{api_url}/v1/runs/{run_id}/cancel")
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        console.print("[red]Could not connect to Shad API. Is it running?[/red]")
        console.print(f"[dim]Tried: {api_url}[/dim]")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            detail = e.response.text
        console.print(f"[red]Cancel failed: {detail or e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Cancel request processed for run {run_id}[/green]")
    console.print(f"[dim]Status: {data.get('status', 'unknown')}[/dim]")
    if data.get("error"):
        console.print(f"[dim]Message: {data['error']}[/dim]")


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

        engine = RLMEngine(llm_provider=LLMProvider(model_config=model_config), history=history)
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
# Collection Commands
# =============================================================================

@cli.command("search")
@click.argument("query")
@click.option("--collection", "-c", multiple=True, help="Collection path(s) to search")
@click.option("--vault", "legacy_collection", "-v", multiple=True, hidden=True, help="[deprecated: use --collection]")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "bm25", "vector"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output results as JSON")
def search(
    query: str,
    collection: tuple[str, ...],
    legacy_collection: tuple[str, ...],
    limit: int,
    mode: str,
    as_json: bool,
) -> None:
    """Search collection(s) for matching documents.

    \b
    Examples:
        shad search "machine learning"
        shad search "authentication" --collection ~/Notes
        shad search "API design" --mode bm25 --limit 20
    """
    # Build collection paths (merge --collection and deprecated --vault)
    _all_paths = collection or legacy_collection
    collection_paths: list[Path] = []
    collection_names: dict[str, Path] = {}

    if _all_paths:
        for v in _all_paths:
            vp = Path(v).expanduser()
            if not vp.is_absolute():
                vp = Path.cwd() / vp
            collection_paths.append(vp)
            collection_names[vp.name] = vp
    elif get_settings().default_collection_path:
        vp = Path(get_settings().default_collection_path).expanduser()
        collection_paths.append(vp)
        collection_names[vp.name] = vp
    else:
        console.print("[yellow]No collection specified. Use --collection or set SHAD_COLLECTION_PATH[/yellow]")
        sys.exit(1)

    # Get retriever
    retriever = get_retriever(paths=collection_paths, collection_names=collection_names)

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

    console.print(Panel(f"Search: {query}", title="Collection Search", border_style="blue"))
    console.print(f"[dim]Mode: {mode} | Backend: {type(retriever).__name__}[/dim]\n")

    for i, result in enumerate(results, 1):
        path = result.path
        collection = result.collection
        content = result.content[:200] if result.content else ""
        score = result.score

        # Show collection if multiple collections
        if len(collection_paths) > 1 and result.collection:
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


CONTEXT_SYNTHESIS_PROMPT = """You are a context synthesizer. Given the following documents retrieved from a knowledge collection, \
produce a concise, information-dense brief relevant to the query. Include key facts, decisions, \
and context. Do NOT include filler or meta-commentary. Target {max_chars} characters.

Query: {query}

Retrieved documents:
{documents}"""


@cli.command("context")
@click.argument("query")
@click.option("--collection", "-c", multiple=True, help="Collection path(s) to search")
@click.option("--vault", "legacy_collection", "-v", multiple=True, hidden=True, help="[deprecated: use --collection]")
@click.option("--limit", "-l", default=10, help="Max retrieval results")
@click.option("--max-chars", default=4000, help="Max output chars for brief")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "bm25", "vector"]), default="hybrid",
              help="Search mode (default: hybrid)")
@click.option("--json", "as_json", is_flag=True, default=False, help="JSON output")
@click.option("--leaf-model", "-L", default=None, help="Model for synthesis (default: leaf tier)")
def context(
    query: str,
    collection: tuple[str, ...],
    legacy_collection: tuple[str, ...],
    limit: int,
    max_chars: int,
    mode: str,
    as_json: bool,
    leaf_model: str | None,
) -> None:
    """Retrieve and synthesize collection context into a compact brief.

    Faster than `shad run` (no DAG/decomposition), richer than `shad search`
    (includes LLM synthesis). Returns a concise context brief from collection documents.

    \b
    Examples:
        shad context "BSV authentication decisions" -c ~/Notes
        shad context "API design patterns" --collection ~/docs --max-chars 2000
        shad context "project architecture" --collection ~/docs --json
    """
    import json as json_mod

    # Build collection paths (merge --collection and deprecated --vault)
    _all_paths = collection or legacy_collection
    collection_paths: list[Path] = []
    collection_names: dict[str, Path] = {}

    if _all_paths:
        for v in _all_paths:
            vp = Path(v).expanduser()
            if not vp.is_absolute():
                vp = Path.cwd() / vp
            collection_paths.append(vp)
            collection_names[vp.name] = vp
    elif get_settings().default_collection_path:
        vp = Path(get_settings().default_collection_path).expanduser()
        collection_paths.append(vp)
        collection_names[vp.name] = vp
    else:
        console.print("[yellow]No collection specified. Use --collection or set SHAD_COLLECTION_PATH[/yellow]")
        sys.exit(1)

    # Retrieve
    retriever = get_retriever(paths=collection_paths, collection_names=collection_names)

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
# Collection Ingestion Command
# =============================================================================

@cli.group("ingest")
def ingest() -> None:
    """Ingest external sources into a collection.

    \b
    Commands:
        github <url>    Ingest a GitHub repository

    \b
    Arguments:
        url             GitHub repository URL (e.g., https://github.com/user/repo)

    \b
    Options:
        -c, --collection     Target collection path (required)
        --preset             Ingestion preset: mirror, docs, deep (default: docs)

    \b
    Examples:
        shad ingest github https://github.com/user/repo --collection ~/MyCollection
        shad ingest github https://github.com/user/repo -c ~/MyCollection --preset deep
    """
    pass


@ingest.command("github")
@click.argument("url")
@click.option("--collection", "-c", required=True, type=click.Path(), help="Target collection path")
@click.option("--vault", "legacy_collection", "-v", type=click.Path(), hidden=True, help="[deprecated: use --collection]")
@click.option("--preset", type=click.Choice(["mirror", "docs", "deep"]), default="docs",
              help="Ingestion preset (default: docs)")
def ingest_github(
    url: str,
    collection: str | None,
    legacy_collection: str | None,
    preset: str,
) -> None:
    """Ingest a GitHub repository into a collection.

    \b
    Examples:
        shad ingest github https://github.com/user/repo --collection ~/MyCollection
        shad ingest github https://github.com/user/repo -c ~/MyCollection --preset deep
    """
    from shad.vault.ingestion import IngestPreset, VaultIngester

    target = collection or legacy_collection
    if not target:
        console.print("[red]Target collection path required (use --collection)[/red]")
        sys.exit(1)
    target_path = Path(target).expanduser()
    if not target_path.exists():
        console.print(f"[red]Collection path does not exist: {target_path}[/red]")
        sys.exit(1)

    console.print(Panel(f"Ingesting {url}", title="Collection Ingestion", border_style="blue"))
    console.print(f"[dim]Preset: {preset}[/dim]")
    console.print(f"[dim]Target: {target_path}[/dim]")

    async def _ingest() -> Any:
        ingester = VaultIngester(collection_path=target_path)
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
@click.option("--port", type=int, default=None, help="API port override (default: configured SHAD_API_PORT or 8000)")
@click.option("--auto-port/--no-auto-port", default=True, help="Auto-fallback to the next free port if the requested port is occupied")
def server_start(foreground: bool, port: int | None, auto_port: bool) -> None:
    """Start the Shad server (Redis + API)."""
    import subprocess

    shad_home = Path(os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad")))
    repo_dir = _find_repo_dir(shad_home)
    api_dir = _find_api_dir(repo_dir)
    log_file = shad_home / "shad-api.log"
    api_port = int(port or _api_listen_port())
    meta = _read_server_meta(shad_home)
    managed_pid = int(meta.get("pid", 0)) if meta.get("pid") else None
    managed_port = int(meta.get("port", api_port)) if meta.get("port") else api_port
    api_base_url = f"http://127.0.0.1:{api_port}"
    api_health_url = f"{api_base_url}/v1/health"

    # Check if already running
    if managed_pid and _pid_is_running(managed_pid) and _api_is_healthy(meta.get("api_url", api_base_url)):
        console.print(f"[yellow]Shad API already running (PID {managed_pid})[/yellow]")
        console.print(f"[dim]API URL: {meta.get('api_url', api_base_url)}[/dim]")
        if managed_port != api_port:
            console.print(f"[dim]Use --port {managed_port} to target the managed instance.[/dim]")
        return
    if meta:
        _clear_server_meta(shad_home)

    if _port_in_use(api_port):
        if _api_is_healthy(api_base_url):
            if _api_supports_async_runs(api_base_url):
                console.print(f"[yellow]A healthy Shad API is already listening on port {api_port}.[/yellow]")
                console.print(f"[dim]API URL: {api_base_url}[/dim]")
                console.print("[dim]This listener is not managed by the current server metadata.[/dim]")
                return

            if not auto_port:
                console.print(f"[red]Port {api_port} is already serving an older or incompatible Shad API surface.[/red]")
                console.print("[dim]Stop the existing process, choose another port with --port, or set SHAD_API_PORT before starting a new server.[/dim]")
                return

            fallback_port = _find_next_available_port(api_port)
            if fallback_port is None:
                console.print(f"[red]Port {api_port} is occupied by an incompatible Shad API and no nearby free port was found.[/red]")
                sys.exit(1)

            console.print(f"[yellow]Port {api_port} is occupied by an incompatible Shad API. Falling back to port {fallback_port}.[/yellow]")
            api_port = fallback_port
            api_base_url = f"http://127.0.0.1:{api_port}"
            api_health_url = f"{api_base_url}/v1/health"
        else:
            if not auto_port:
                console.print(f"[red]Port {api_port} is already in use by another process.[/red]")
                console.print("[dim]Stop the existing listener, choose another port with --port, or set SHAD_API_PORT before starting Shad.[/dim]")
                sys.exit(1)

            fallback_port = _find_next_available_port(api_port)
            if fallback_port is None:
                console.print(f"[red]Port {api_port} is occupied and no nearby free port was found.[/red]")
                sys.exit(1)

            console.print(f"[yellow]Port {api_port} is in use. Falling back to port {fallback_port}.[/yellow]")
            api_port = fallback_port
            api_base_url = f"http://127.0.0.1:{api_port}"
            api_health_url = f"{api_base_url}/v1/health"

    if managed_pid and _pid_is_running(managed_pid):
        console.print(f"[yellow]Shad API already running (PID {managed_pid})[/yellow]")
        console.print(f"[dim]API URL: {api_base_url}[/dim]")
        return

    # Start Redis via Docker Compose
    console.print("[blue]Starting Redis...[/blue]")
    compose_file = repo_dir / "docker-compose.yml"

    if compose_file.exists():
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d", "redis"],
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

    venv_python = shad_home / "venv" / "bin" / "python"

    if not venv_python.exists():
        venv_python = Path(sys.executable)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(api_dir / "src")

    if foreground:
        # Run in foreground
        console.print("[dim]Running in foreground. Press Ctrl+C to stop.[/dim]")
        console.print(f"[dim]Health check: {api_health_url}[/dim]")
        try:
            subprocess.run(
                [str(venv_python), "-m", "uvicorn", "shad.api.main:app",
                 "--host", "0.0.0.0", "--port", str(api_port)],
                cwd=str(api_dir),
                env=env,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
    else:
        # Run in background
        with log_file.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [str(venv_python), "-m", "uvicorn", "shad.api.main:app",
                 "--host", "0.0.0.0", "--port", str(api_port)],
                cwd=str(api_dir),
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        console.print(f"[dim]Waiting for API health check: {api_health_url}[/dim]")
        if _wait_for_api_health(require_async_runs=True, base_url=api_base_url):
            _write_server_meta(shad_home, pid=proc.pid, port=api_port)
            console.print(f"[green]✓ Shad API started (PID {proc.pid})[/green]")
            console.print(f"[dim]Log file: {log_file}[/dim]")
            console.print(f"[dim]API URL: {api_base_url}[/dim]")
        else:
            try:
                proc.terminate()
            except Exception:
                pass
            _clear_server_meta(shad_home)
            console.print(f"[red]Shad API did not become healthy within timeout (PID {proc.pid})[/red]")
            console.print(f"[dim]Check logs: {log_file}[/dim]")
            sys.exit(1)

    console.print("\n[green]Shad is ready![/green]")
    console.print("[dim]Try: shad run \"Hello, Shad!\"[/dim]")


@server.command("stop")
def server_stop() -> None:
    """Stop the Shad server (Redis + API)."""
    import signal
    import subprocess

    shad_home = Path(os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad")))
    repo_dir = _find_repo_dir(shad_home)
    meta = _read_server_meta(shad_home)

    stopped_api = False
    stopped_redis = False

    # Stop API server
    pid = int(meta.get("pid", 0)) if meta.get("pid") else None
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]✓ Stopped Shad API (PID {pid})[/green]")
            stopped_api = True
        except OSError:
            console.print("[dim]API was not running[/dim]")
        _clear_server_meta(shad_home)
    else:
        console.print("[dim]No managed API metadata found; not stopping externally managed listeners.[/dim]")

    # Stop Redis via Docker Compose
    compose_file = repo_dir / "docker-compose.yml"
    if compose_file.exists():
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "stop", "redis"],
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
@click.option("--port", type=int, default=None, help="Inspect a specific API port (default: managed port or configured SHAD_API_PORT)")
def server_status(port: int | None) -> None:
    """Check Shad server status."""
    import subprocess

    shad_home = Path(os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad")))
    meta = _read_server_meta(shad_home)
    managed_port = int(meta.get("port", _api_listen_port())) if meta else _api_listen_port()
    api_port = int(port or managed_port)
    api_base_url = f"http://127.0.0.1:{api_port}"

    table = Table(title="Shad Server Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    # Check API
    api_running = _api_is_healthy(api_base_url)
    pid = int(meta.get("pid", 0)) if meta.get("pid") else None
    pid_running = pid is not None and _pid_is_running(pid)
    api_details = "Not started"

    if api_running:
        route_details = "async runs ready" if _api_supports_async_runs(api_base_url) else "legacy API surface"
        if pid_running and api_port == managed_port:
            api_details = f"PID {pid}, {api_base_url} ({route_details})"
        elif _port_in_use(api_port):
            api_details = f"{api_base_url} ({route_details}, managed externally)"
        else:
            api_details = f"{api_base_url} ({route_details})"
    elif meta:
        api_details = f"Stale PID file ({pid})" if pid is not None else "Stale PID file"
    elif _port_in_use(api_port):
        api_details = f"Port {api_port} in use but health check failed"

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
    claude_installed = False
    claude_details = ""
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            claude_installed = True
            claude_details = f"{result.stdout.strip().splitlines()[0]} (auth not checked)"
        else:
            claude_details = "Not configured"
    except FileNotFoundError:
        claude_details = "Not installed"

    table.add_row(
        "Claude CLI",
        "[green]Installed[/green]" if claude_installed else "[yellow]Missing[/yellow]",
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
    import subprocess

    shad_home = Path(os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad")))
    log_file = shad_home / "shad-api.log"

    if not log_file.exists():
        console.print("[dim]No log file found. Start the server first.[/dim]")
        return

    if follow:
        subprocess.run(["tail", "-f", str(log_file)])
    else:
        subprocess.run(["tail", "-n", str(lines), str(log_file)])


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
        -c, --collection PATH    Target collection path (required)
        -s, --schedule SCHED     Sync schedule: manual, hourly, daily, weekly, monthly
        -p, --preset PRESET      Ingestion preset (for github): mirror, docs, deep

    \b
    Sync Options:
        -a, --all               Sync all sources, not just due ones
        -d, --daemon            Run as daemon (continuous sync)
        -i, --interval SECS     Check interval in seconds (for daemon mode)

    \b
    Examples:
        shad sources add github https://github.com/org/repo --collection ~/MyCollection
        shad sources add url https://docs.example.com -c ~/MyCollection -s weekly
        shad sources add feed https://blog.example.com/rss -c ~/MyCollection -s hourly
        shad sources list
        shad sources sync --all
        shad sources sync --daemon --interval 300
        shad sources remove abc123
    """
    pass


@sources.command("add")
@click.argument("source_type", type=click.Choice(["github", "url", "feed", "folder"]))
@click.argument("location")
@click.option("--collection", "-c", required=True, help="Target collection path")
@click.option("--vault", "legacy_collection", hidden=True, help="[deprecated: use --collection]")
@click.option("--schedule", "-s", type=click.Choice(["manual", "hourly", "daily", "weekly", "monthly"]),
              default="daily", help="Sync schedule")
@click.option("--preset", "-p", default="docs", help="Ingestion preset (for github)")
def sources_add(
    source_type: str,
    location: str,
    collection: str | None,
    legacy_collection: str | None,
    schedule: str,
    preset: str,
) -> None:
    """Add a new content source.

    \b
    Examples:
        shad sources add github https://github.com/org/repo --collection ~/MyCollection
        shad sources add url https://docs.example.com --collection ~/MyCollection --schedule weekly
        shad sources add feed https://blog.example.com/rss --collection ~/MyCollection --schedule hourly
        shad sources add folder ~/Projects/docs --collection ~/MyCollection
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

    target = collection or legacy_collection
    if not target:
        console.print("[red]Target collection path required (use --collection)[/red]")
        sys.exit(1)

    is_folder = source_type == "folder"
    source = manager.add_source(
        source_type=type_map[source_type],
        url=None if is_folder else location,
        path=location if is_folder else None,
        collection_path=target,
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
        console.print("[dim]Add one with: shad sources add github <url> --collection <path>[/dim]")
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

        def on_sync(result: Any) -> None:
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
    console.print("[green]✔[/green] Shad CLI available")

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
        host, port_str = hostport.split(":", 1)
        port_int = int(port_str.split("/", 1)[0])
        import socket
        with socket.create_connection((host, port_int), timeout=2):
            console.print(f"[green]✔[/green] Redis reachable at {host}:{port_int}")
    except Exception:
        console.print(f"[yellow]⚠[/yellow] Redis not reachable at {redis_url}")

    # Check collection env
    collection_path = get_settings().default_collection_path
    if collection_path:
        console.print(f"[green]✔[/green] SHAD_COLLECTION_PATH set: {collection_path}")
    else:
        console.print("[yellow]⚠[/yellow] SHAD_COLLECTION_PATH not set (pass --collection)")

    # Offer to register collection + embed if fixing and collection set
    if fix and collection_path and qmd_path:
        if click.confirm(f"Register collection {collection_path} with qmd and embed now?"):
            try:
                subprocess.run(["qmd", "collection", "add", collection_path, "--name", Path(collection_path).name], check=False)
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


# Top-level embed alias for convenience
@cli.command("embed")
@click.option("--force", "-f", is_flag=True, help="Force re-embedding of all chunks")
@click.pass_context
def embed_alias(ctx: click.Context, force: bool) -> None:
    """Generate vector embeddings (alias for 'shad collection embed').

    \b
    Examples:
        shad embed          # Embed new chunks only
        shad embed --force  # Re-embed everything
    """
    ctx.invoke(collection_embed, force=force)


if __name__ == "__main__":
    cli()
