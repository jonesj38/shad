"""Integration test: memory_type filter through the full search pipeline.

What this covers
----------------
The full pipeline path in ``QmdRetriever.search()``:

    query → subprocess (stubbed) → _parse_results → rerank_with_metadata
          → memory_types client-side filter → ranked results returned

Contracts verified
------------------
1. **Filter + decay ordering**: After decay reranking, the memory_types filter
   must remove non-matching results without disrupting the ranked order of the
   surviving results.
2. **Filter applied after decay**: Decay scores must be applied to all results
   first; only then is the type filter applied — surviving results carry
   decay-adjusted scores, not raw qmd scores.
3. **Server-side + client-side combined**: Passing both ``memory_type``
   (server-side ``--filter`` flag) and ``memory_types`` (client-side list)
   simultaneously must honour both constraints.
4. **Score preservation**: Every result that survives the filter must expose
   its ``memory_type`` in ``metadata`` and carry the correct score
   (decay-adjusted when a decay config is present, raw otherwise).
5. **Empty pipeline**: When the type filter removes all results the pipeline
   must return ``[]`` without error.
6. **No filter regression**: With ``memory_types=None`` every result passes
   through regardless of type, matching the no-filter baseline.

Architecture reference
----------------------
``services/shad-api/src/shad/retrieval/qmd.py`` lines 382–394 — the decay
branch followed by the client-side type filter.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from shad.retrieval.qmd import QmdRetriever
from shad.vault.decay import DecayConfig, DecayCurve
from shad.vault.shadow_index import MemoryType

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_HALFLIFE_DAYS = 90.0
_HALFLIFE_SECONDS = _HALFLIFE_DAYS * 86_400.0
_EXP_CONFIG = DecayConfig(halflife_seconds=_HALFLIFE_SECONDS, curve=DecayCurve.EXPONENTIAL)

# Fixed reference "now" so expected scores can be derived by formula.
_NOW = datetime(2026, 3, 31, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(
    path: str,
    score: float,
    memory_type: str | None = None,
    ingested_at: str | None = None,
) -> dict:
    """Build a minimal qmd JSON result item."""
    d: dict = {
        "file": path,
        "score": score,
        "content": f"Content of {path}",
        "title": path,
        "snippet": f"Snippet from {path}",
    }
    if memory_type is not None:
        d["memory_type"] = memory_type
    if ingested_at is not None:
        d["ingested_at"] = ingested_at
    return d


def _fake_proc(items: list[dict]) -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = json.dumps(items).encode()
    proc.stderr = b""
    return proc


async def _search(
    items: list[dict],
    memory_types: list[str] | None = None,
    memory_type: str | None = None,
    decay_config: DecayConfig | None = None,
) -> list:
    retriever = QmdRetriever(decay_config=decay_config)
    with (
        patch("shad.retrieval.qmd.shutil.which", return_value="/usr/bin/qmd"),
        patch("shad.retrieval.qmd.subprocess.run", return_value=_fake_proc(items)),
    ):
        return await retriever.search(
            "integration test query",
            mode="bm25",
            memory_type=memory_type,
            memory_types=memory_types,
        )


# ---------------------------------------------------------------------------
# 1. Filter applied after decay — ranked order among survivors is correct
# ---------------------------------------------------------------------------

class TestFilterAppliedAfterDecay:
    """memory_types filter must operate on decay-adjusted scores, not raw scores.

    If decay is applied first and filter second, surviving results must be
    ordered by *decayed* score, not the original qmd score.
    """

    async def test_survivors_sorted_by_decay_adjusted_score(self) -> None:
        """After filtering, remaining results must be in decay-adjusted score order.

        Setup (halflife = 90 d, exponential, reference = _NOW):
          ep-fresh.md  EPISODIC  score=0.60  age=1d   → decayed ≈ 0.60 × 0.992 ≈ 0.595
          ep-stale.md  EPISODIC  score=0.90  age=180d → decayed ≈ 0.90 × 0.25   = 0.225
          sm.md        SEMANTIC  score=0.80  age=3d   → (filtered out)

        Raw qmd order:    ep-stale > sm > ep-fresh
        After decay:      ep-fresh > sm > ep-stale
        After type filter (EPISODIC only): [ep-fresh, ep-stale]
        """
        items = [
            _item("ep-stale.md", 0.90, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=180)).isoformat()),
            _item("sm.md",       0.80, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=3)).isoformat()),
            _item("ep-fresh.md", 0.60, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=1)).isoformat()),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert len(results) == 2
        paths = [r.path for r in results]
        assert "sm.md" not in paths, "Semantic result must be excluded"
        assert paths.index("ep-fresh.md") < paths.index("ep-stale.md"), (
            f"ep-fresh must rank above ep-stale after decay; got {paths}"
        )

    async def test_decay_scores_are_applied_to_survivors(self) -> None:
        """Surviving results must carry decay-adjusted scores, not raw qmd scores."""
        items = [
            _item("ep-stale.md", 0.90, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=180)).isoformat()),
            _item("sm.md",       0.80, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=3)).isoformat()),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert len(results) == 1
        survivor = results[0]
        assert survivor.path == "ep-stale.md"
        # Decay factor at 180 d = 0.5^(180/90) = 0.25
        expected_decayed = 0.90 * math.pow(0.5, 180.0 / _HALFLIFE_DAYS)
        assert survivor.score == pytest.approx(expected_decayed, abs=1e-3)
        assert survivor.score < 0.90, "Score must be lower than the raw qmd score"

    async def test_original_score_in_metadata_after_filter(self) -> None:
        """Survivors must still expose original_score in metadata (set by reranker)."""
        items = [
            _item("ep.md", 0.90, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=90)).isoformat()),
            _item("sm.md", 0.80, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=90)).isoformat()),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert len(results) == 1
        assert results[0].metadata.get("original_score") == pytest.approx(0.90)

    async def test_multiple_type_survivors_sorted_by_decayed_score(self) -> None:
        """OR filter with two types must keep survivors in decay-score order."""
        items = [
            _item("sm-stale.md", 0.95, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=360)).isoformat()),
            _item("ep-fresh.md", 0.50, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=2)).isoformat()),
            _item("wk.md",       0.70, MemoryType.WORKING.value,
                  (_NOW - timedelta(days=5)).isoformat()),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.SEMANTIC.value, MemoryType.EPISODIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert len(results) == 2
        paths = [r.path for r in results]
        assert "wk.md" not in paths

        # sm-stale at 360d → 0.95 × 0.5^4 ≈ 0.0594
        # ep-fresh at 2d   → 0.50 × 0.5^(2/90) ≈ 0.492
        assert paths.index("ep-fresh.md") < paths.index("sm-stale.md"), (
            f"ep-fresh (less stale) must rank above sm-stale; got {paths}"
        )


# ---------------------------------------------------------------------------
# 2. Server-side + client-side filter combined
# ---------------------------------------------------------------------------

class TestServerSideAndClientSideCombined:
    """Passing both memory_type (--filter) and memory_types (client list) must
    honour both constraints independently.

    memory_type adds a ``--filter`` flag to the qmd command (server-side).
    memory_types post-filters the parsed results (client-side).
    """

    async def test_server_filter_flag_present_and_client_filter_applied(self) -> None:
        """Both --filter in qmd args and client-side filtering must be active."""
        items = [
            _item("ep.md", 0.90, MemoryType.EPISODIC.value),
            _item("sm.md", 0.80, MemoryType.SEMANTIC.value),
            _item("wk.md", 0.70, MemoryType.WORKING.value),
        ]

        retriever = QmdRetriever()
        mock_run = MagicMock(return_value=_fake_proc(items))

        with (
            patch("shad.retrieval.qmd.shutil.which", return_value="/usr/bin/qmd"),
            patch("shad.retrieval.qmd.subprocess.run", mock_run),
        ):
            results = await retriever.search(
                "combined filter query",
                mode="bm25",
                memory_type=MemoryType.EPISODIC.value,     # server-side
                memory_types=[MemoryType.EPISODIC.value],  # client-side
            )

        captured_args: list[str] = mock_run.call_args.args[0]
        # Server-side: --filter present
        assert "--filter" in captured_args
        fi = captured_args.index("--filter")
        assert captured_args[fi + 1] == f"memory_type={MemoryType.EPISODIC.value}"

        # Client-side: only episodic survives (qmd stub returns all three types)
        assert len(results) == 1
        assert results[0].path == "ep.md"

    async def test_client_filter_can_narrow_server_filtered_results(self) -> None:
        """Client-side filter can further restrict what server-side filter returns."""
        # Simulate qmd honouring server-side filter: only episodic + semantic returned
        items = [
            _item("ep.md", 0.90, MemoryType.EPISODIC.value),
            _item("sm.md", 0.80, MemoryType.SEMANTIC.value),
        ]

        results = await _search(
            items,
            memory_type=None,                              # no server-side filter
            memory_types=[MemoryType.EPISODIC.value],      # client-side narrows to episodic
        )

        assert len(results) == 1
        assert results[0].path == "ep.md"
        assert results[0].metadata.get("memory_type") == MemoryType.EPISODIC.value


# ---------------------------------------------------------------------------
# 3. Score and metadata preservation through the full pipeline
# ---------------------------------------------------------------------------

class TestScoreAndMetadataPreservation:
    """Every surviving result must carry the correct score and memory_type metadata."""

    async def test_memory_type_in_metadata_after_decay_and_filter(self) -> None:
        """memory_type must remain accessible in metadata after full pipeline."""
        items = [
            _item("ep.md", 0.85, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=30)).isoformat()),
            _item("sm.md", 0.75, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=30)).isoformat()),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert len(results) == 2
        by_path = {r.path: r for r in results}
        assert by_path["ep.md"].metadata.get("memory_type") == MemoryType.EPISODIC.value
        assert by_path["sm.md"].metadata.get("memory_type") == MemoryType.SEMANTIC.value

    async def test_raw_scores_preserved_without_decay(self) -> None:
        """Without decay_config, surviving results must carry the unmodified qmd score."""
        items = [
            _item("ep.md", 0.91, MemoryType.EPISODIC.value),
            _item("sm.md", 0.77, MemoryType.SEMANTIC.value),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=None,
        )

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.91)

    async def test_all_four_memory_types_survive_full_filter(self) -> None:
        """Requesting all four types must return all typed results."""
        all_types = [m.value for m in MemoryType]
        items = [_item(f"{t}.md", round(0.9 - i * 0.1, 1), t) for i, t in enumerate(all_types)]

        results = await _search(items, memory_types=all_types)

        assert len(results) == len(all_types)
        result_types = {r.metadata.get("memory_type") for r in results}
        assert result_types == set(all_types)

    async def test_ranked_order_without_decay_matches_qmd_emission_order(self) -> None:
        """With no decay and a passing type filter the qmd emission order is preserved."""
        items = [
            _item("ep1.md", 0.95, MemoryType.EPISODIC.value),
            _item("ep2.md", 0.80, MemoryType.EPISODIC.value),
            _item("ep3.md", 0.60, MemoryType.EPISODIC.value),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=None,
        )

        assert [r.path for r in results] == ["ep1.md", "ep2.md", "ep3.md"]


# ---------------------------------------------------------------------------
# 4. Empty-result edge cases
# ---------------------------------------------------------------------------

class TestEmptyResultEdgeCases:
    """The pipeline must handle empty outputs at every stage without error."""

    async def test_filter_removes_all_results_returns_empty_list(self) -> None:
        """When every result is excluded by the type filter the pipeline returns []."""
        items = [
            _item("sm.md", 0.90, MemoryType.SEMANTIC.value),
            _item("pr.md", 0.80, MemoryType.PROCEDURAL.value),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert results == []

    async def test_empty_qmd_response_with_filter_returns_empty_list(self) -> None:
        results = await _search(
            [],
            memory_types=[MemoryType.SEMANTIC.value],
            decay_config=_EXP_CONFIG,
        )
        assert results == []

    async def test_all_untyped_with_active_filter_returns_empty_list(self) -> None:
        items = [
            _item("a.md", 0.90, memory_type=None),
            _item("b.md", 0.80, memory_type=None),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.WORKING.value],
            decay_config=None,
        )

        assert results == []

    async def test_filter_removes_all_after_decay_no_exception(self) -> None:
        """Even if decay reorders results to all low scores, type filter must work."""
        items = [
            _item("sm.md", 0.50, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=360)).isoformat()),
        ]

        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value],
            decay_config=_EXP_CONFIG,
        )

        assert results == []


# ---------------------------------------------------------------------------
# 5. No-filter regression — memory_types=None must pass everything through
# ---------------------------------------------------------------------------

class TestNoFilterRegression:
    """memory_types=None must return all results unchanged (no filter active)."""

    async def test_all_types_pass_through_with_none_filter(self) -> None:
        items = [
            _item("ep.md", 0.90, MemoryType.EPISODIC.value),
            _item("sm.md", 0.80, MemoryType.SEMANTIC.value),
            _item("wk.md", 0.70, MemoryType.WORKING.value),
            _item("pr.md", 0.60, MemoryType.PROCEDURAL.value),
            _item("no-type.md", 0.50, memory_type=None),
        ]

        results = await _search(items, memory_types=None, decay_config=None)

        assert len(results) == 5
        paths = {r.path for r in results}
        assert paths == {"ep.md", "sm.md", "wk.md", "pr.md", "no-type.md"}

    async def test_none_filter_with_decay_all_results_present(self) -> None:
        """decay_config + memory_types=None must rerank but not drop any result."""
        items = [
            _item("ep.md", 0.90, MemoryType.EPISODIC.value,
                  (_NOW - timedelta(days=180)).isoformat()),
            _item("sm.md", 0.50, MemoryType.SEMANTIC.value,
                  (_NOW - timedelta(days=1)).isoformat()),
            _item("no-type.md", 0.40, memory_type=None),
        ]

        results = await _search(items, memory_types=None, decay_config=_EXP_CONFIG)

        assert len(results) == 3
        paths = {r.path for r in results}
        assert paths == {"ep.md", "sm.md", "no-type.md"}

    async def test_empty_memory_types_list_equivalent_to_none(self) -> None:
        """memory_types=[] must behave identically to memory_types=None."""
        items = [
            _item("ep.md", 0.90, MemoryType.EPISODIC.value),
            _item("sm.md", 0.80, MemoryType.SEMANTIC.value),
        ]

        results_none  = await _search(items, memory_types=None,  decay_config=None)
        results_empty = await _search(items, memory_types=[],    decay_config=None)

        assert len(results_none) == len(results_empty) == 2
        assert {r.path for r in results_none} == {r.path for r in results_empty}
