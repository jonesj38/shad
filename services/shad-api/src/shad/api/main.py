"""FastAPI application for Shad API."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from shad import __version__
from shad.engine import LLMProvider, RLMEngine
from shad.history import HistoryManager
from shad.models import Budget, RunConfig
from shad.models.run import Run, RunStatus
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)

# Global instances
engine: RLMEngine | None = None
history: HistoryManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global engine, history

    # Initialize components
    llm_provider = LLMProvider()
    engine = RLMEngine(llm_provider=llm_provider)
    history = HistoryManager()

    logger.info("Shad API initialized")

    yield

    # Cleanup
    logger.info("Shad API shutting down")


# Routers - defined before create_app
health_router = APIRouter(tags=["Health"])
run_router = APIRouter(prefix="/v1", tags=["Runs"])


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Shad API",
        description="Shannon's Daemon - Personal AI Infrastructure for long-context reasoning",
        version=__version__,
        lifespan=lifespan,
    )

    # Include routes
    app.include_router(health_router)
    app.include_router(run_router)

    return app


# Request/Response models
class RunRequest(BaseModel):
    """Request model for creating a run."""

    goal: str = Field(..., description="The goal/task to accomplish")
    notebook_id: str | None = Field(default=None, description="Notebook ID for context")
    budgets: dict[str, int] | None = Field(default=None, description="Budget overrides")
    voice: str | None = Field(default=None, description="Voice for output rendering")


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

    if request.budgets:
        budget_dict.update(request.budgets)

    config = RunConfig(
        goal=request.goal,
        notebook_id=request.notebook_id,
        budget=Budget(**budget_dict),
        voice=request.voice,
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
        run = await engine.resume(run)
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


# Create default app instance
app = create_app()
