"""FastAPI application for Shad API."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from shad import __version__
from shad.cache import RedisCache
from shad.engine import LLMProvider, RLMEngine
from shad.history import HistoryManager
from shad.integrations import N8NClient
from shad.learnings import LearningsStore
from shad.models import Budget, RunConfig
from shad.models.run import Run, RunStatus
from shad.retrieval import RetrievalLayer, get_retriever
from shad.skills import SkillRouter
from shad.utils.config import get_settings
from shad.verification import HITLQueue
from shad.voice import VoiceRenderer

logger = logging.getLogger(__name__)

# Global instances
engine: RLMEngine | None = None
history: HistoryManager | None = None
cache: RedisCache | None = None
skill_router: SkillRouter | None = None
voice_renderer: VoiceRenderer | None = None
n8n_client: N8NClient | None = None
hitl_queue: HITLQueue | None = None
learnings_store: LearningsStore | None = None
retriever: RetrievalLayer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global engine, history, cache, skill_router, voice_renderer
    global n8n_client, hitl_queue, learnings_store, retriever

    settings = get_settings()

    # Initialize components
    llm_provider = LLMProvider()

    # Initialize cache
    cache = RedisCache()
    await cache.connect()

    # Initialize retriever
    from pathlib import Path
    vault_path = Path(settings.obsidian_vault_path) if settings.obsidian_vault_path else None
    retriever = get_retriever(
        paths=[vault_path] if vault_path else None,
        collection_names={vault_path.name: vault_path} if vault_path else None,
    )
    logger.info(f"Initialized retriever: {type(retriever).__name__}")

    # Initialize engine with cache and retriever
    collections = [vault_path.name] if vault_path else None
    engine = RLMEngine(
        llm_provider=llm_provider,
        cache=cache,
        retriever=retriever,
        vault_path=vault_path,
        collections=collections,
    )
    history = HistoryManager()

    # Initialize skill router
    skill_router = SkillRouter()
    skill_router.load_skills()

    # Initialize voice renderer
    voice_renderer = VoiceRenderer(llm_provider=llm_provider)

    # Initialize n8n client
    n8n_client = N8NClient()
    await n8n_client.connect()

    # Initialize HITL queue
    hitl_queue = HITLQueue()

    # Initialize learnings store
    learnings_store = LearningsStore()

    logger.info("Shad API initialized")

    yield

    # Cleanup
    if cache:
        await cache.disconnect()
    if n8n_client:
        await n8n_client.disconnect()

    logger.info("Shad API shutting down")


# Routers - defined before create_app
health_router = APIRouter(tags=["Health"])
run_router = APIRouter(prefix="/v1", tags=["Runs"])


# Request/Response models
class RunRequest(BaseModel):
    """Request model for creating a run."""

    goal: str = Field(..., description="The goal/task to accomplish")
    vault_path: str | None = Field(default=None, description="Obsidian vault path for context")
    budget: dict[str, int] | None = Field(default=None, description="Budget overrides")
    voice: str | None = Field(default=None, description="Voice for output rendering")
    strategy: str | None = Field(default=None, description="Strategy override: software, research, analysis, planning")
    verify: str | None = Field(default="basic", description="Verification level: off, basic, build, strict")
    write_files: bool = Field(default=False, description="Write output files to disk")
    output_path: str | None = Field(default=None, description="Output directory for files")


class RunResponse(BaseModel):
    """Response model for run operations."""

    run_id: str
    status: str
    goal: str
    result: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    total_tokens: int = 0
    nodes_count: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0

    @classmethod
    def from_run(cls, run: Run) -> RunResponse:
        """Create response from Run model."""
        return cls(
            run_id=run.run_id,
            status=run.status.value,
            goal=run.config.goal,
            result=run.final_result,
            stop_reason=run.stop_reason.value if run.stop_reason else None,
            error=run.error,
            total_tokens=run.total_tokens,
            nodes_count=len(run.nodes),
            completed_nodes=len(run.completed_nodes()),
            failed_nodes=len(run.failed_nodes()),
        )


class ResumeRequest(BaseModel):
    """Request model for resuming a run."""

    budget_overrides: dict[str, int] | None = Field(
        default=None, description="Budget overrides for resumed run"
    )
    replay: str | None = Field(
        default=None, description="Replay mode: 'stale', node_id, or 'subtree:node_id'"
    )


@health_router.get("/v1/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "service": "shad-api",
    }


@run_router.post("/run", response_model=RunResponse)
async def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    """
    Execute a reasoning task.

    This creates a new run and executes it synchronously.
    For long-running tasks, consider using the async endpoint.
    """
    if engine is None or history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Build budget from defaults and overrides
    settings = get_settings()
    budget_dict = {
        "max_depth": settings.default_max_depth,
        "max_nodes": settings.default_max_nodes,
        "max_wall_time": settings.default_max_wall_time,
        "max_tokens": settings.default_max_tokens,
    }

    if request.budget:
        budget_dict.update(request.budget)

    config = RunConfig(
        goal=request.goal,
        vault_path=request.vault_path,
        budget=Budget(**budget_dict),
        voice=request.voice,
        strategy_override=request.strategy,
        verify_level=request.verify,
        write_files=request.write_files,
        output_path=request.output_path,
    )

    # Execute run
    try:
        run = await engine.execute(config)
    except Exception as e:
        logger.exception(f"Run execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Save to history in background
    background_tasks.add_task(history.save_run, run)

    return RunResponse.from_run(run)


@run_router.get("/run/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Get run status and results."""
    if history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        run = history.load_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from e

    return RunResponse.from_run(run)


@run_router.post("/run/{run_id}/resume", response_model=RunResponse)
async def resume_run(
    run_id: str, request: ResumeRequest, background_tasks: BackgroundTasks
) -> RunResponse:
    """Resume a partial or failed run."""
    if engine is None or history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        run = history.load_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from e

    if run.status not in (RunStatus.PARTIAL, RunStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} has status {run.status.value}, cannot resume",
        )

    # Apply budget overrides
    if request.budget_overrides:
        for key, value in request.budget_overrides.items():
            if hasattr(run.config.budget, key):
                setattr(run.config.budget, key, value)

    # Resume execution
    try:
        run = await engine.resume(run, replay_mode=request.replay)
    except Exception as e:
        logger.exception(f"Run resume failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Save updated history
    background_tasks.add_task(history.save_run, run)

    return RunResponse.from_run(run)


@run_router.get("/runs")
async def list_runs(limit: int = 50) -> dict[str, Any]:
    """List recent runs."""
    if history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    runs = history.list_runs(limit=limit)
    return {"runs": runs, "count": len(runs)}


# Additional routers
skills_router = APIRouter(prefix="/v1/skills", tags=["Skills"])
admin_router = APIRouter(prefix="/v1/admin", tags=["Admin"])
vault_router = APIRouter(prefix="/v1/vault", tags=["Vault"])


@skills_router.get("")
async def list_skills() -> dict[str, Any]:
    """List all available skills."""
    if skill_router is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {"skills": skill_router.list_skills()}


@skills_router.post("/route")
async def route_goal(goal: str) -> dict[str, Any]:
    """Route a goal to appropriate skill(s)."""
    if skill_router is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    decision = skill_router.route(goal)
    return decision.to_dict()


@admin_router.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    if cache is None:
        return {"connected": False}

    return await cache.get_stats()


@admin_router.get("/hitl/queue")
async def get_hitl_queue(limit: int = 50) -> dict[str, Any]:
    """Get pending HITL review items."""
    if hitl_queue is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    items = hitl_queue.list_pending(limit=limit)
    return {
        "items": [item.to_dict() for item in items],
        "stats": hitl_queue.get_stats(),
    }


@admin_router.post("/hitl/{item_id}/approve")
async def approve_hitl(item_id: str, reviewer: str, notes: str = "") -> dict[str, bool]:
    """Approve a HITL review item."""
    if hitl_queue is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    success = hitl_queue.approve(item_id, reviewer, notes)
    if not success:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return {"approved": True}


@admin_router.post("/hitl/{item_id}/reject")
async def reject_hitl(item_id: str, reviewer: str, notes: str = "") -> dict[str, bool]:
    """Reject a HITL review item."""
    if hitl_queue is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    success = hitl_queue.reject(item_id, reviewer, notes)
    if not success:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return {"rejected": True}


@admin_router.get("/learnings/stats")
async def learnings_stats() -> dict[str, Any]:
    """Get learnings store statistics."""
    if learnings_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return learnings_store.get_stats()


@admin_router.get("/voices")
async def list_voices() -> dict[str, Any]:
    """List available voices."""
    if voice_renderer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {"voices": voice_renderer.list_voices()}


# Vault router - vault operations
@vault_router.get("/status")
async def vault_status() -> dict[str, Any]:
    """Get retriever status."""
    if retriever is None:
        return {"available": False}

    status = await retriever.status()
    return status


@vault_router.get("/search")
async def search_vault(
    query: str,
    limit: int = 10,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Search the vault."""
    if retriever is None or not retriever.available:
        raise HTTPException(status_code=503, detail="Retriever not available")

    results = await retriever.search(query, mode=mode, limit=limit)
    return {
        "results": [r.to_dict() for r in results],
        "query": query,
        "count": len(results),
    }


@vault_router.get("/note/{path:path}")
async def read_note(path: str, collection: str | None = None) -> dict[str, Any]:
    """Read a note from the vault."""
    if retriever is None or not retriever.available:
        raise HTTPException(status_code=503, detail="Retriever not available")

    content = await retriever.get(path, collection=collection)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Note {path} not found")

    return {
        "path": path,
        "content": content,
        "collection": collection,
    }


@vault_router.get("/notes")
async def list_vault_notes(directory: str = "", recursive: bool = False) -> dict[str, Any]:
    """List notes in the vault.

    Note: This endpoint has limited functionality with the new retriever architecture.
    Use search for finding notes instead.
    """
    if retriever is None or not retriever.available:
        raise HTTPException(status_code=503, detail="Retriever not available")

    # For now, return a basic status since the retriever doesn't have list_notes
    status = await retriever.status()
    return {
        "message": "Use search to find notes",
        "retriever": type(retriever).__name__,
        "status": status,
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Shad API",
        description="Shannon's Daemon - Personal AI Infrastructure for long-context reasoning with Obsidian",
        version=__version__,
        lifespan=lifespan,
    )

    # Include routes
    application.include_router(health_router)
    application.include_router(run_router)
    application.include_router(skills_router)
    application.include_router(admin_router)
    application.include_router(vault_router)

    return application


# Create default app instance
app = create_app()
