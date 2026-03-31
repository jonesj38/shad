"""Tests for StrEnum serialization in shad.models.run and shad.models.goal.

Follows the patterns in test_taxonomy.py.

Covers:
  - RunStatus: all variants, str values, StrEnum identity, invalid construction
  - NodeStatus: all variants, str values, StrEnum identity, invalid construction
  - StopReason: all variants, str values, StrEnum identity, invalid construction
  - RiskLevel: all variants, str values, StrEnum identity, invalid construction
  - Pydantic JSON round-trips: all enum fields serialize to/from their string values
"""

from __future__ import annotations

import json

import pytest

from shad.models.goal import GoalSpec, RiskLevel
from shad.models.run import DAGNode, NodeStatus, Run, RunConfig, RunStatus, StopReason


# ---------------------------------------------------------------------------
# RunStatus
# ---------------------------------------------------------------------------


class TestRunStatusEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in RunStatus}
        assert names == {"PENDING", "RUNNING", "COMPLETE", "PARTIAL", "FAILED", "ABORTED"}

    def test_values_are_lowercase_strings(self) -> None:
        for member in RunStatus:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    @pytest.mark.parametrize("member,expected_value", [
        (RunStatus.PENDING,  "pending"),
        (RunStatus.RUNNING,  "running"),
        (RunStatus.COMPLETE, "complete"),
        (RunStatus.PARTIAL,  "partial"),
        (RunStatus.FAILED,   "failed"),
        (RunStatus.ABORTED,  "aborted"),
    ])
    def test_str_returns_value(self, member: RunStatus, expected_value: str) -> None:
        assert str(member) == expected_value

    @pytest.mark.parametrize("value,expected", [
        ("pending",  RunStatus.PENDING),
        ("running",  RunStatus.RUNNING),
        ("complete", RunStatus.COMPLETE),
        ("partial",  RunStatus.PARTIAL),
        ("failed",   RunStatus.FAILED),
        ("aborted",  RunStatus.ABORTED),
    ])
    def test_construction_from_value_roundtrip(self, value: str, expected: RunStatus) -> None:
        assert RunStatus(value) is expected

    @pytest.mark.parametrize("bad", [
        "PENDING", "RUNNING", "invalid", "", "   ", "pending_status", "done",
    ])
    def test_invalid_value_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError):
            RunStatus(bad)

    def test_raises_for_none(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            RunStatus(None)  # type: ignore[arg-type]

    def test_raises_for_integer(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            RunStatus(42)  # type: ignore[arg-type]

    def test_valid_values_cover_all_members(self) -> None:
        for member in RunStatus:
            assert RunStatus(member.value) is member


# ---------------------------------------------------------------------------
# NodeStatus
# ---------------------------------------------------------------------------


class TestNodeStatusEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in NodeStatus}
        assert names == {
            "CREATED", "READY", "STARTED", "SUCCEEDED",
            "FAILED", "PRUNED", "CACHE_HIT",
        }

    def test_values_are_lowercase_strings(self) -> None:
        for member in NodeStatus:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    @pytest.mark.parametrize("member,expected_value", [
        (NodeStatus.CREATED,   "created"),
        (NodeStatus.READY,     "ready"),
        (NodeStatus.STARTED,   "started"),
        (NodeStatus.SUCCEEDED, "succeeded"),
        (NodeStatus.FAILED,    "failed"),
        (NodeStatus.PRUNED,    "pruned"),
        (NodeStatus.CACHE_HIT, "cache_hit"),
    ])
    def test_str_returns_value(self, member: NodeStatus, expected_value: str) -> None:
        assert str(member) == expected_value

    @pytest.mark.parametrize("value,expected", [
        ("created",   NodeStatus.CREATED),
        ("ready",     NodeStatus.READY),
        ("started",   NodeStatus.STARTED),
        ("succeeded", NodeStatus.SUCCEEDED),
        ("failed",    NodeStatus.FAILED),
        ("pruned",    NodeStatus.PRUNED),
        ("cache_hit", NodeStatus.CACHE_HIT),
    ])
    def test_construction_from_value_roundtrip(self, value: str, expected: NodeStatus) -> None:
        assert NodeStatus(value) is expected

    @pytest.mark.parametrize("bad", [
        "CREATED", "SUCCEEDED", "invalid", "", "cache-hit", "done",
    ])
    def test_invalid_value_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError):
            NodeStatus(bad)

    def test_dag_node_defaults_to_created(self) -> None:
        node = DAGNode(task="do something")
        assert node.status is NodeStatus.CREATED

    def test_valid_values_cover_all_members(self) -> None:
        for member in NodeStatus:
            assert NodeStatus(member.value) is member


# ---------------------------------------------------------------------------
# StopReason
# ---------------------------------------------------------------------------


class TestStopReasonEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in StopReason}
        assert names == {
            "COMPLETE", "BUDGET_DEPTH", "BUDGET_NODES", "BUDGET_TIME",
            "BUDGET_TOKENS", "NOVELTY_PRUNED", "ERROR", "ABORTED",
        }

    def test_values_are_lowercase_strings(self) -> None:
        for member in StopReason:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    @pytest.mark.parametrize("member,expected_value", [
        (StopReason.COMPLETE,       "complete"),
        (StopReason.BUDGET_DEPTH,   "budget_depth"),
        (StopReason.BUDGET_NODES,   "budget_nodes"),
        (StopReason.BUDGET_TIME,    "budget_time"),
        (StopReason.BUDGET_TOKENS,  "budget_tokens"),
        (StopReason.NOVELTY_PRUNED, "novelty_pruned"),
        (StopReason.ERROR,          "error"),
        (StopReason.ABORTED,        "aborted"),
    ])
    def test_str_returns_value(self, member: StopReason, expected_value: str) -> None:
        assert str(member) == expected_value

    @pytest.mark.parametrize("value,expected", [
        ("complete",       StopReason.COMPLETE),
        ("budget_depth",   StopReason.BUDGET_DEPTH),
        ("budget_nodes",   StopReason.BUDGET_NODES),
        ("budget_time",    StopReason.BUDGET_TIME),
        ("budget_tokens",  StopReason.BUDGET_TOKENS),
        ("novelty_pruned", StopReason.NOVELTY_PRUNED),
        ("error",          StopReason.ERROR),
        ("aborted",        StopReason.ABORTED),
    ])
    def test_construction_from_value_roundtrip(self, value: str, expected: StopReason) -> None:
        assert StopReason(value) is expected

    @pytest.mark.parametrize("bad", [
        "COMPLETE", "BUDGET", "budget-depth", "invalid", "", "done",
    ])
    def test_invalid_value_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError):
            StopReason(bad)

    def test_valid_values_cover_all_members(self) -> None:
        for member in StopReason:
            assert StopReason(member.value) is member


# ---------------------------------------------------------------------------
# RiskLevel
# ---------------------------------------------------------------------------


class TestRiskLevelEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in RiskLevel}
        assert names == {"LOW", "MEDIUM", "HIGH"}

    def test_values_are_lowercase_strings(self) -> None:
        for member in RiskLevel:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()

    @pytest.mark.parametrize("member,expected_value", [
        (RiskLevel.LOW,    "low"),
        (RiskLevel.MEDIUM, "medium"),
        (RiskLevel.HIGH,   "high"),
    ])
    def test_str_returns_value(self, member: RiskLevel, expected_value: str) -> None:
        assert str(member) == expected_value

    @pytest.mark.parametrize("value,expected", [
        ("low",    RiskLevel.LOW),
        ("medium", RiskLevel.MEDIUM),
        ("high",   RiskLevel.HIGH),
    ])
    def test_construction_from_value_roundtrip(self, value: str, expected: RiskLevel) -> None:
        assert RiskLevel(value) is expected

    @pytest.mark.parametrize("bad", [
        "LOW", "HIGH", "MEDIUM", "critical", "none", "", "   ",
    ])
    def test_invalid_value_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError):
            RiskLevel(bad)

    def test_goal_spec_defaults_to_low_risk(self) -> None:
        spec = GoalSpec.from_goal("explain this concept")
        assert spec.risk_level is RiskLevel.LOW

    def test_goal_spec_accepts_explicit_risk_level(self) -> None:
        spec = GoalSpec(
            raw_goal="drop production database",
            normalized_goal="drop production database",
            risk_level=RiskLevel.HIGH,
        )
        assert spec.risk_level is RiskLevel.HIGH

    def test_valid_values_cover_all_members(self) -> None:
        for member in RiskLevel:
            assert RiskLevel(member.value) is member


# ---------------------------------------------------------------------------
# Pydantic JSON round-trips
# ---------------------------------------------------------------------------


class TestPydanticJsonRoundTrips:
    """All enum fields serialize to their string value in JSON and restore correctly."""

    def test_run_status_serializes_to_string(self) -> None:
        run = Run(config=RunConfig(goal="test"), status=RunStatus.RUNNING)
        data = json.loads(run.model_dump_json())
        assert data["status"] == "running"

    def test_run_status_round_trip(self) -> None:
        run = Run(config=RunConfig(goal="test"), status=RunStatus.PARTIAL)
        restored = Run.model_validate_json(run.model_dump_json())
        assert restored.status is RunStatus.PARTIAL

    def test_run_default_status_is_pending(self) -> None:
        run = Run(config=RunConfig(goal="test"))
        data = json.loads(run.model_dump_json())
        assert data["status"] == "pending"

    def test_node_status_serializes_to_string(self) -> None:
        node = DAGNode(task="subtask")
        data = json.loads(node.model_dump_json())
        assert data["status"] == "created"

    def test_node_status_round_trip(self) -> None:
        node = DAGNode(task="subtask", status=NodeStatus.SUCCEEDED)
        restored = DAGNode.model_validate_json(node.model_dump_json())
        assert restored.status is NodeStatus.SUCCEEDED

    @pytest.mark.parametrize("status", list(NodeStatus))
    def test_all_node_statuses_round_trip(self, status: NodeStatus) -> None:
        node = DAGNode(task="subtask", status=status)
        restored = DAGNode.model_validate_json(node.model_dump_json())
        assert restored.status is status

    def test_stop_reason_serializes_to_string(self) -> None:
        node = DAGNode(task="subtask", stop_reason=StopReason.BUDGET_TOKENS)
        data = json.loads(node.model_dump_json())
        assert data["stop_reason"] == "budget_tokens"

    def test_stop_reason_round_trip(self) -> None:
        node = DAGNode(task="subtask", stop_reason=StopReason.ERROR)
        restored = DAGNode.model_validate_json(node.model_dump_json())
        assert restored.stop_reason is StopReason.ERROR

    def test_stop_reason_none_preserved(self) -> None:
        node = DAGNode(task="subtask")
        assert node.stop_reason is None
        restored = DAGNode.model_validate_json(node.model_dump_json())
        assert restored.stop_reason is None

    def test_run_stop_reason_round_trip(self) -> None:
        run = Run(config=RunConfig(goal="test"), stop_reason=StopReason.BUDGET_DEPTH)
        restored = Run.model_validate_json(run.model_dump_json())
        assert restored.stop_reason is StopReason.BUDGET_DEPTH

    def test_risk_level_round_trip(self) -> None:
        spec = GoalSpec(
            raw_goal="analyze security",
            normalized_goal="analyze security",
            risk_level=RiskLevel.MEDIUM,
        )
        restored = GoalSpec.model_validate_json(spec.model_dump_json())
        assert restored.risk_level is RiskLevel.MEDIUM

    def test_pydantic_rejects_invalid_run_status_in_json(self) -> None:
        run = Run(config=RunConfig(goal="test"))
        bad_json = run.model_dump_json().replace('"pending"', '"not_a_status"')
        with pytest.raises(Exception):
            Run.model_validate_json(bad_json)

    def test_pydantic_rejects_invalid_node_status_in_json(self) -> None:
        node = DAGNode(task="subtask")
        bad_json = node.model_dump_json().replace('"created"', '"garbage"')
        with pytest.raises(Exception):
            DAGNode.model_validate_json(bad_json)
