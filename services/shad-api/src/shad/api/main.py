"""FastAPI application for Shad API."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shad import __version__
from shad.cache import RedisCache
from shad.engine import LLMProvider, RLMEngine
from shad.history import HistoryManager
from shad.integrations import N8NClient
from shad.learnings import LearningsStore
from shad.models import Budget, RunConfig
from shad.models.run import Run, RunStatus, StopReason
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
run_tasks: dict[str, asyncio.Task[None]] = {}


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
    collection_path = Path(settings.default_collection_path) if settings.default_collection_path else None
    retriever = get_retriever(
        paths=[collection_path] if collection_path else None,
        collection_names={collection_path.name: collection_path} if collection_path else None,
    )
    logger.info(f"Initialized retriever: {type(retriever).__name__}")

    # Initialize engine with cache and retriever
    collections = [collection_path.name] if collection_path else None
    engine = RLMEngine(
        llm_provider=llm_provider,
        cache=cache,
        retriever=retriever,
        collection_path=collection_path,
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
    collection_path: str | None = Field(default=None, description="Collection path for context")
    collection_paths: list[str] | None = Field(default=None, description="Collection paths for multi-collection context")
    budget: dict[str, int] | None = Field(default=None, description="Budget overrides")
    voice: str | None = Field(default=None, description="Voice for output rendering")
    strategy: str | None = Field(default=None, description="Strategy override: software, research, analysis, planning")
    verify: str | None = Field(default="basic", description="Verification level: off, basic, build, strict")
    write_files: bool = Field(default=False, description="Write output files to disk")
    output_path: str | None = Field(default=None, description="Output directory for files")
    use_gemini_cli: bool = Field(default=False, description="Use Gemini CLI instead of Claude Code")
    orchestrator_model: str | None = Field(default=None, description="Override orchestrator model")
    worker_model: str | None = Field(default=None, description="Override worker model")
    leaf_model: str | None = Field(default=None, description="Override leaf model")


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


class RunAcceptedResponse(BaseModel):
    """Response model for accepted async runs."""

    run_id: str
    status: str
    goal: str


class RunEventsResponse(BaseModel):
    """Response model for run events."""

    run_id: str
    events: list[dict[str, Any]]
    total: int
    next_offset: int


def _build_run_config(request: RunRequest) -> RunConfig:
    """Build run configuration from request."""
    settings = get_settings()
    budget_dict = {
        "max_depth": settings.default_max_depth,
        "max_nodes": settings.default_max_nodes,
        "max_wall_time": settings.default_max_wall_time,
        "max_tokens": settings.default_max_tokens,
    }

    if request.budget:
        budget_dict.update(request.budget)

    model_config = None
    if request.orchestrator_model or request.worker_model or request.leaf_model:
        from shad.models import ModelConfig

        model_config = ModelConfig(
            orchestrator_model=request.orchestrator_model,
            worker_model=request.worker_model,
            leaf_model=request.leaf_model,
        )

    return RunConfig(
        goal=request.goal,
        collection_path=(request.collection_paths[0] if request.collection_paths else request.collection_path),
        budget=Budget(**budget_dict),
        voice=request.voice,
        strategy_override=request.strategy,
        verify_level=request.verify,
        write_files=request.write_files,
        output_path=request.output_path,
        model_config_override=model_config,
    )


def _get_collection_paths(
    config: RunConfig,
    collection_paths: list[str] | None = None,
) -> list[Path]:
    """Resolve collection paths from request or run state."""
    raw_paths = collection_paths or []
    if not raw_paths and config.collection_path:
        raw_paths = [config.collection_path]
    return [Path(path).expanduser() for path in raw_paths]


def _build_engine_for_config(
    config: RunConfig,
    use_gemini_cli: bool = False,
    collection_paths: list[str] | None = None,
) -> RLMEngine:
    """Build a fresh engine instance for one run."""
    resolved_paths = _get_collection_paths(config, collection_paths)
    collection_path = resolved_paths[0] if resolved_paths else None
    collection_names = {path.name: path for path in resolved_paths}
    local_retriever = get_retriever(
        paths=resolved_paths or None,
        collection_names=collection_names or None,
    )
    collections = list(collection_names.keys()) if collection_names else None

    return RLMEngine(
        llm_provider=LLMProvider(
            model_config=config.model_config_override,
            use_gemini_cli=use_gemini_cli,
            use_claude_code=not use_gemini_cli,
        ),
        cache=cache,
        retriever=local_retriever,
        collection_path=collection_path,
        collections=collections,
    )


def _append_run_event(run_id: str, event_type: str, **payload: Any) -> None:
    """Append a structured run event if history is available."""
    if history is None:
        return
    history.append_event(run_id, {"type": event_type, **payload})


def _is_terminal_status(status_value: str) -> bool:
    """Return whether a run status is terminal."""
    return status_value in {"complete", "partial", "failed", "aborted"}


async def _execute_run_in_background(run: Run, use_gemini_cli: bool = False) -> None:
    """Execute an async run and persist status transitions."""
    if history is None:
        return

    try:
        run.status = RunStatus.RUNNING
        history.save_run(run)
        _append_run_event(run.run_id, "run_started", status=run.status.value)

        run_engine = _build_engine_for_config(
            run.config,
            use_gemini_cli=use_gemini_cli,
            collection_paths=run.metadata.get("collection_paths"),
        )
        completed_run = await run_engine.execute(run.config)
        completed_run.run_id = run.run_id
        completed_run.created_at = run.created_at
        history.save_run(completed_run)
        _append_run_event(
            completed_run.run_id,
            "run_completed",
            status=completed_run.status.value,
            stop_reason=completed_run.stop_reason.value if completed_run.stop_reason else None,
            total_tokens=completed_run.total_tokens,
        )
    except asyncio.CancelledError:
        logger.info("Async run cancelled: %s", run.run_id)
        run.status = RunStatus.ABORTED
        run.stop_reason = StopReason.ABORTED
        run.error = "Run cancelled by user"
        history.save_run(run)
        _append_run_event(run.run_id, "run_cancelled", status=run.status.value)
        raise
    except Exception as e:
        logger.exception("Async run execution failed: %s", e)
        run.status = RunStatus.FAILED
        run.error = str(e)
        history.save_run(run)
        _append_run_event(run.run_id, "run_failed", status=run.status.value, error=str(e))
    finally:
        run_tasks.pop(run.run_id, None)


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

    config = _build_run_config(request)

    # Execute run
    try:
        run = await _build_engine_for_config(
            config,
            use_gemini_cli=request.use_gemini_cli,
            collection_paths=request.collection_paths,
        ).execute(config)
    except Exception as e:
        logger.exception(f"Run execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Save to history in background
    background_tasks.add_task(history.save_run, run)

    return RunResponse.from_run(run)


@run_router.post("/runs", response_model=RunAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_async_run(request: RunRequest) -> RunAcceptedResponse:
    """Create a run and execute it asynchronously."""
    if history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    config = _build_run_config(request)
    run = Run(config=config, status=RunStatus.PENDING)
    if request.collection_paths:
        run.metadata["collection_paths"] = request.collection_paths
    history.save_run(run)
    _append_run_event(run.run_id, "run_queued", status=run.status.value)

    task = asyncio.create_task(_execute_run_in_background(run, use_gemini_cli=request.use_gemini_cli))
    run_tasks[run.run_id] = task

    return RunAcceptedResponse(run_id=run.run_id, status=run.status.value, goal=run.config.goal)


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


@run_router.get("/runs/{run_id}", response_model=RunResponse)
async def get_async_run(run_id: str) -> RunResponse:
    """Get async run status and results."""
    return await get_run(run_id)


@run_router.get("/runs/{run_id}/events", response_model=RunEventsResponse)
async def get_run_events(
    run_id: str,
    since: int = Query(default=0, ge=0, description="Zero-based event offset"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum events to return"),
) -> RunEventsResponse:
    """Get pollable events for a run."""
    if history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        events, total = history.load_events(run_id, since=since, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from e

    return RunEventsResponse(
        run_id=run_id,
        events=events,
        total=total,
        next_offset=min(total, since + len(events)),
    )


@run_router.get("/runs/{run_id}/events/stream")
async def stream_run_events(
    run_id: str,
    since: int = Query(default=0, ge=0, description="Zero-based event offset"),
    poll_interval: float = Query(default=1.0, ge=0.2, le=10.0, description="Seconds between polls"),
) -> StreamingResponse:
    """Stream run events as server-sent events."""
    if history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        history.load_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from e

    async def event_stream() -> AsyncGenerator[str, None]:
        offset = since

        while True:
            try:
                events, total = history.load_events(run_id, since=offset, limit=100)
                run = history.load_run(run_id)
            except FileNotFoundError:
                yield f"data: {json.dumps({'type': 'run_missing', 'run_id': run_id})}\n\n"
                break

            for event in events:
                yield f"data: {json.dumps(event)}\n\n"

            offset = total

            if _is_terminal_status(run.status.value):
                break

            yield ": keep-alive\n\n"
            await asyncio.sleep(poll_interval)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@run_router.post("/runs/{run_id}/cancel", response_model=RunResponse)
async def cancel_async_run(run_id: str) -> RunResponse:
    """Cancel a pending or running async run."""
    if history is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        run = history.load_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from e

    if run.status in (RunStatus.COMPLETE, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.ABORTED):
        return RunResponse.from_run(run)

    task = run_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    else:
        run.status = RunStatus.ABORTED
        run.stop_reason = StopReason.ABORTED
        run.error = "Run cancelled by user"
        history.save_run(run)
        _append_run_event(run.run_id, "run_cancelled", status=run.status.value)

    updated_run = history.load_run(run_id)
    return RunResponse.from_run(updated_run)


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
collection_router = APIRouter(prefix="/v1/collection", tags=["Collection"])


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


# Collection router - collection operations
@collection_router.get("/status")
async def collection_status() -> dict[str, Any]:
    """Get retriever status."""
    if retriever is None:
        return {"available": False}

    status = await retriever.status()
    return status


@collection_router.get("/search")
async def search_collection(
    query: str,
    limit: int = 10,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Search the collection."""
    if retriever is None or not retriever.available:
        raise HTTPException(status_code=503, detail="Retriever not available")

    results = await retriever.search(query, mode=mode, limit=limit)
    return {
        "results": [r.to_dict() for r in results],
        "query": query,
        "count": len(results),
    }


@collection_router.get("/note/{path:path}")
async def read_note(path: str, collection: str | None = None) -> dict[str, Any]:
    """Read a note from the collection."""
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


@collection_router.get("/notes")
async def list_collection_notes(directory: str = "", recursive: bool = False) -> dict[str, Any]:
    """List notes in the collection.

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
        description="Shannon's Daemon - Personal AI Infrastructure for long-context reasoning with Collection",
        version=__version__,
        lifespan=lifespan,
    )

    # Include routes
    application.include_router(health_router)
    application.include_router(run_router)
    application.include_router(skills_router)
    application.include_router(admin_router)
    application.include_router(collection_router)

    return application


# Create default app instance
app = create_app()
