from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shad.history.manager import HistoryManager
from shad.models.run import Run, RunConfig, RunStatus, StopReason


def test_load_run_reconciles_stale_running_run_past_wall_time_budget(tmp_path):
    history = HistoryManager(base_path=tmp_path)
    run = Run(
        config=RunConfig(goal="stale run", budget={"max_wall_time": 1}),
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC) - timedelta(seconds=10),
    )

    history.save_run(run)

    loaded = history.load_run(run.run_id)

    assert loaded.status == RunStatus.PARTIAL
    assert loaded.stop_reason == StopReason.BUDGET_TIME
    assert loaded.completed_at is not None
    assert loaded.error == "Run exceeded max wall time before finalizing state"

    reloaded = history.load_run(run.run_id)
    assert reloaded.status == RunStatus.PARTIAL
    assert reloaded.stop_reason == StopReason.BUDGET_TIME


def test_list_runs_reconciles_stale_running_run_before_returning_summary(tmp_path):
    history = HistoryManager(base_path=tmp_path)
    run = Run(
        config=RunConfig(goal="summary stale run", budget={"max_wall_time": 1}),
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC) - timedelta(seconds=10),
    )

    history.save_run(run)

    listed = history.list_runs(limit=10)

    assert listed[0]["run_id"] == run.run_id
    assert listed[0]["status"] == RunStatus.PARTIAL.value
