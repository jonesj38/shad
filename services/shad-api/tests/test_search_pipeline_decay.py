"""Integration test: full search pipeline calls temporal decay reranking.

What this covers
----------------
``QmdRetriever.search()`` with a ``decay_config`` must:

1. Pass parsed results through ``rerank_with_metadata`` — verified via spy.
2. Return results sorted by *decay-adjusted* score, not the raw qmd score.
3. Set ``metadata["original_score"]`` on every result to the raw qmd score.
4. Leave results with no ``ingested_at`` timestamp undecayed (score preserved).

The test stubs out the ``subprocess.run`` call so that no real ``qmd`` binary
is needed; the fake JSON payload mimics what qmd would return in ``--json``
mode.  Timestamps are set relative to ``datetime.now(UTC)`` so the test
remains valid regardless of when it runs.

Architecture reference: ``services/shad-api/src/shad/retrieval/qmd.py``
lines 370-372 — the decay branch that calls ``rerank_with_metadata``.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pytest

from shad.retrieval.qmd import QmdRetriever
from shad.vault.decay import DecayConfig, DecayCurve
from shad.vault.reranker import rerank_with_metadata as _real_rerank_with_metadata

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_HALFLIFE_DAYS = 90.0
_HALFLIFE_SECONDS = _HALFLIFE_DAYS * 86_400.0
_EXP_CONFIG = DecayConfig(halflife_seconds=_HALFLIFE_SECONDS, curve=DecayCurve.EXPONENTIAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qmd_json(
    path: str,
    score: float,
    ingested_at: str | None = None,
) -> dict:
    """Build a single qmd JSON result item."""
    item: dict = {
        "file": path,
        "score": score,
        "content": f"Content of {path}",
        "title": path,
        "snippet": f"Snippet from {path}",
    }
    if ingested_at is not None:
        item["ingested_at"] = ingested_at
    return item


def _fake_subprocess_result(items: list[dict]) -> MagicMock:
    """Return a mock subprocess.CompletedProcess carrying *items* as JSON."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = json.dumps(items).encode()
    proc.stderr = b""
    return proc


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestSearchPipelineCallsDecayReranking:
    """Verify the full path: QmdRetriever.search → _parse_results → rerank_with_metadata."""

    async def _run_search(
        self,
        items: list[dict],
        decay_config: DecayConfig | None,
    ):
        """Execute QmdRetriever.search with stubbed subprocess output."""
        retriever = QmdRetriever(decay_config=decay_config)
        fake_proc = _fake_subprocess_result(items)

        with (
            patch("shad.retrieval.qmd.shutil.which", return_value="/usr/bin/qmd"),
            patch("shad.retrieval.qmd.subprocess.run", return_value=fake_proc),
        ):
            return await retriever.search("integration test query", mode="bm25")

    async def test_rerank_with_metadata_is_called_when_decay_config_set(self) -> None:
        """Spy confirms rerank_with_metadata is invoked exactly once."""
        now = datetime.now(UTC)
        items = [
            _make_qmd_json("a.md", score=0.9, ingested_at=(now - timedelta(days=180)).isoformat()),
            _make_qmd_json("b.md", score=0.5, ingested_at=(now - timedelta(days=1)).isoformat()),
        ]

        with patch(
            "shad.vault.reranker.rerank_with_metadata",
            wraps=_real_rerank_with_metadata,
        ) as spy:
            await self._run_search(items, decay_config=_EXP_CONFIG)

        assert spy.call_count == 1
        _results_arg, _config_arg = spy.call_args.args
        assert _config_arg is _EXP_CONFIG

    async def test_rerank_not_called_when_decay_config_is_none(self) -> None:
        """Without a decay_config the reranker is never invoked."""
        now = datetime.now(UTC)
        items = [
            _make_qmd_json("a.md", score=0.9, ingested_at=(now - timedelta(days=180)).isoformat()),
        ]

        with patch(
            "shad.vault.reranker.rerank_with_metadata",
            wraps=_real_rerank_with_metadata,
        ) as spy:
            await self._run_search(items, decay_config=None)

        assert spy.call_count == 0

    async def test_results_reordered_by_decay_adjusted_score(self) -> None:
        """High raw-score but very stale doc must rank below a fresh lower-score doc.

        Setup (halflife = 90 days, exponential):
          old-doc.md   score=0.90  age≈180d  → decayed ≈ 0.90 × 0.25  = 0.225
          fresh-doc.md score=0.50  age≈1d    → decayed ≈ 0.50 × 0.992 ≈ 0.496
          no-ts-doc.md score=0.30  no ts     → no decay              = 0.300

        Raw qmd order:   old-doc (0.9) > fresh-doc (0.5) > no-ts-doc (0.3)
        After reranking: fresh-doc     > no-ts-doc       > old-doc
        """
        now = datetime.now(UTC)
        items = [
            # qmd returns them in raw-score order (highest first)
            _make_qmd_json(
                "old-doc.md",
                score=0.90,
                ingested_at=(now - timedelta(days=180)).isoformat(),
            ),
            _make_qmd_json(
                "fresh-doc.md",
                score=0.50,
                ingested_at=(now - timedelta(days=1)).isoformat(),
            ),
            _make_qmd_json(
                "no-ts-doc.md",
                score=0.30,
                ingested_at=None,
            ),
        ]

        results = await self._run_search(items, decay_config=_EXP_CONFIG)

        assert len(results) == 3
        ranked_paths = [r.path for r in results]

        # fresh-doc must overtake old-doc after decay
        assert ranked_paths.index("fresh-doc.md") < ranked_paths.index("old-doc.md"), (
            f"Expected fresh-doc to rank above old-doc, got order: {ranked_paths}"
        )
        # no-ts-doc (0.30, no decay) must rank above heavily-decayed old-doc (~0.225)
        assert ranked_paths.index("no-ts-doc.md") < ranked_paths.index("old-doc.md"), (
            f"Expected no-ts-doc to rank above old-doc, got order: {ranked_paths}"
        )

    async def test_scores_are_decay_adjusted_not_raw(self) -> None:
        """The live score on each result must be less than the raw qmd score when decayed."""
        now = datetime.now(UTC)
        items = [
            _make_qmd_json(
                "old-doc.md",
                score=0.90,
                ingested_at=(now - timedelta(days=180)).isoformat(),
            ),
            _make_qmd_json(
                "fresh-doc.md",
                score=0.50,
                ingested_at=(now - timedelta(days=1)).isoformat(),
            ),
        ]

        results = await self._run_search(items, decay_config=_EXP_CONFIG)
        by_path = {r.path: r for r in results}

        # Both docs have timestamps so both must be penalised
        assert by_path["old-doc.md"].score < 0.90
        assert by_path["fresh-doc.md"].score < 0.50

        # Approximately: old-doc at 180d ≈ 2 halflives → factor 0.25
        expected_old = 0.90 * math.pow(0.5, 180.0 / _HALFLIFE_DAYS)
        assert by_path["old-doc.md"].score == pytest.approx(expected_old, abs=1e-3)

    async def test_original_score_stored_in_metadata(self) -> None:
        """Every result must carry its raw qmd score in metadata['original_score']."""
        now = datetime.now(UTC)
        items = [
            _make_qmd_json(
                "old-doc.md",
                score=0.90,
                ingested_at=(now - timedelta(days=180)).isoformat(),
            ),
            _make_qmd_json(
                "fresh-doc.md",
                score=0.50,
                ingested_at=(now - timedelta(days=1)).isoformat(),
            ),
            _make_qmd_json("no-ts-doc.md", score=0.30, ingested_at=None),
        ]

        results = await self._run_search(items, decay_config=_EXP_CONFIG)
        by_path = {r.path: r for r in results}

        assert by_path["old-doc.md"].metadata["original_score"] == pytest.approx(0.90)
        assert by_path["fresh-doc.md"].metadata["original_score"] == pytest.approx(0.50)
        assert by_path["no-ts-doc.md"].metadata["original_score"] == pytest.approx(0.30)

    async def test_no_timestamp_result_score_is_preserved(self) -> None:
        """A result without ingested_at must come through with its raw score unchanged."""
        items = [_make_qmd_json("no-ts-doc.md", score=0.65, ingested_at=None)]

        results = await self._run_search(items, decay_config=_EXP_CONFIG)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.65)
        assert results[0].metadata["original_score"] == pytest.approx(0.65)

    async def test_output_is_sorted_highest_decay_score_first(self) -> None:
        """Results list must always be in descending decay-adjusted score order."""
        now = datetime.now(UTC)
        # Provide items in worst possible order (ascending raw score)
        items = [
            _make_qmd_json(
                "c.md",
                score=0.30,
                ingested_at=(now - timedelta(days=2)).isoformat(),
            ),
            _make_qmd_json(
                "b.md",
                score=0.60,
                ingested_at=(now - timedelta(days=2)).isoformat(),
            ),
            _make_qmd_json(
                "a.md",
                score=0.90,
                ingested_at=(now - timedelta(days=2)).isoformat(),
            ),
        ]

        results = await self._run_search(items, decay_config=_EXP_CONFIG)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by decay-adjusted score: {scores}"
        )

    async def test_without_decay_config_raw_order_is_preserved(self) -> None:
        """With decay_config=None the pipeline must return results in qmd's original order."""
        now = datetime.now(UTC)
        items = [
            # Deliberately: old doc first (highest raw score) — should stay first
            _make_qmd_json(
                "old-doc.md",
                score=0.90,
                ingested_at=(now - timedelta(days=365)).isoformat(),
            ),
            _make_qmd_json(
                "fresh-doc.md",
                score=0.50,
                ingested_at=(now - timedelta(days=1)).isoformat(),
            ),
        ]

        results = await self._run_search(items, decay_config=None)

        assert len(results) == 2
        assert results[0].path == "old-doc.md"
        assert results[0].score == pytest.approx(0.90)
        assert results[1].path == "fresh-doc.md"

    async def test_empty_qmd_response_returns_empty_list(self) -> None:
        """An empty result set from qmd must propagate cleanly — no reranker call."""
        with patch(
            "shad.vault.reranker.rerank_with_metadata",
            wraps=_real_rerank_with_metadata,
        ) as spy:
            results = await self._run_search([], decay_config=_EXP_CONFIG)

        assert results == []
        # rerank_with_metadata is guarded by ``if results:`` in qmd.py
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# memory_type filter parameter
# ---------------------------------------------------------------------------

class TestMemoryTypeFilter:
    """Verify that the memory_type param passes --filter memory_type=<value> to qmd.

    Architecture reference: ``services/shad-api/src/shad/retrieval/qmd.py``
    lines 326-327 — the ``--filter`` branch gated on ``memory_type``.
    """

    async def _run_with_captured_args(
        self,
        items: list[dict],
        memory_type: str | None = None,
        decay_config: DecayConfig | None = None,
    ) -> tuple[list, list[str]]:
        """Run search with stubbed subprocess and return (results, qmd_cmd_args)."""
        retriever = QmdRetriever(decay_config=decay_config)
        fake_proc = _fake_subprocess_result(items)
        mock_run = MagicMock(return_value=fake_proc)

        with (
            patch("shad.retrieval.qmd.shutil.which", return_value="/usr/bin/qmd"),
            patch("shad.retrieval.qmd.subprocess.run", mock_run),
        ):
            results = await retriever.search(
                "test query", mode="bm25", memory_type=memory_type
            )

        # subprocess.run(args, ...) — args is the first positional argument
        captured_args: list[str] = mock_run.call_args.args[0]
        return results, captured_args

    async def test_memory_type_appends_filter_flag(self) -> None:
        """memory_type='episodic_memory' must add --filter memory_type=episodic_memory."""
        _, args = await self._run_with_captured_args([], memory_type="episodic_memory")
        assert "--filter" in args
        filter_idx = args.index("--filter")
        assert args[filter_idx + 1] == "memory_type=episodic_memory"

    async def test_no_memory_type_omits_filter_flag(self) -> None:
        """memory_type=None (default) must not add --filter to the qmd command."""
        _, args = await self._run_with_captured_args([], memory_type=None)
        assert "--filter" not in args

    @pytest.mark.parametrize(
        "memory_type_val",
        [
            "semantic_memory",
            "episodic_memory",
            "working_memory",
            "procedural_memory",
        ],
    )
    async def test_each_memory_type_value_produces_correct_filter(
        self, memory_type_val: str
    ) -> None:
        """Every MemoryType value must produce the correct --filter argument."""
        _, args = await self._run_with_captured_args([], memory_type=memory_type_val)
        assert "--filter" in args
        filter_idx = args.index("--filter")
        assert args[filter_idx + 1] == f"memory_type={memory_type_val}"

    async def test_filter_flag_appears_after_core_command_args(self) -> None:
        """--filter must follow qmd <subcommand> <query> in the args list."""
        _, args = await self._run_with_captured_args([], memory_type="semantic_memory")
        # Minimum: ["qmd", <subcommand>, <query>, ..., "--filter", ...]
        assert args[0] == "qmd"
        assert "--filter" in args
        filter_idx = args.index("--filter")
        assert filter_idx >= 3

    async def test_memory_type_filter_combined_with_decay_config(self) -> None:
        """memory_type filter and decay_config must both be active simultaneously.

        The --filter flag must appear in qmd args AND results must be
        reranked by decay-adjusted score (fresh doc b.md above stale a.md).
        """
        now = datetime.now(UTC)
        items = [
            _make_qmd_json(
                "a.md",
                score=0.90,
                ingested_at=(now - timedelta(days=180)).isoformat(),
            ),
            _make_qmd_json(
                "b.md",
                score=0.50,
                ingested_at=(now - timedelta(days=1)).isoformat(),
            ),
        ]

        results, args = await self._run_with_captured_args(
            items,
            memory_type="episodic_memory",
            decay_config=_EXP_CONFIG,
        )

        # Filter present in command
        assert "--filter" in args
        filter_idx = args.index("--filter")
        assert args[filter_idx + 1] == "memory_type=episodic_memory"

        # Decay applied — fresh doc overtakes stale doc
        assert len(results) == 2
        paths = [r.path for r in results]
        assert paths.index("b.md") < paths.index("a.md"), (
            f"Expected b.md (fresh) to rank above a.md (stale); got {paths}"
        )

    async def test_memory_type_filter_without_decay_preserves_raw_order(self) -> None:
        """With decay_config=None the filter is still sent but order stays as qmd returns it."""
        now = datetime.now(UTC)
        items = [
            _make_qmd_json(
                "high.md",
                score=0.90,
                ingested_at=(now - timedelta(days=365)).isoformat(),
            ),
            _make_qmd_json(
                "low.md",
                score=0.40,
                ingested_at=(now - timedelta(days=1)).isoformat(),
            ),
        ]

        results, args = await self._run_with_captured_args(
            items,
            memory_type="semantic_memory",
            decay_config=None,
        )

        # Filter present
        assert "memory_type=semantic_memory" in args[args.index("--filter") + 1]

        # No reranking — qmd order is preserved
        assert results[0].path == "high.md"
        assert results[1].path == "low.md"

    async def test_empty_results_with_memory_type_filter_returns_empty(self) -> None:
        """An empty qmd response with memory_type set must return an empty list."""
        results, args = await self._run_with_captured_args(
            [], memory_type="working_memory", decay_config=_EXP_CONFIG
        )
        assert results == []
        assert "memory_type=working_memory" in args[args.index("--filter") + 1]


# ---------------------------------------------------------------------------
# Baseline: no memory_type filter returns ALL result types unfiltered
# ---------------------------------------------------------------------------

# One representative item per MemoryType, plus one with no classification.
_ALL_MEMORY_TYPES_PAYLOAD: list[dict] = [
    _make_qmd_json("episodic/feed-item.md", score=0.90),
    _make_qmd_json("semantic/github-repo.md", score=0.80),
    _make_qmd_json("working/scratch-note.md", score=0.70),
    _make_qmd_json("procedural/how-to.md", score=0.60),
    _make_qmd_json("unclassified/no-type.md", score=0.50),
]

_ALL_EXPECTED_PATHS = {item["file"] for item in _ALL_MEMORY_TYPES_PAYLOAD}


class TestNoMemoryTypeFilterBaseline:
    """Verify that search() with memory_type=None (the default) returns every
    document qmd emits, regardless of its memory classification.

    Contract:
    - No ``--filter`` flag is added to the qmd command.
    - Every item in the qmd response appears in the result list.
    - No result is silently dropped by client-side type filtering.
    - Result scores and paths are unaltered.

    Architecture reference: ``services/shad-api/src/shad/retrieval/qmd.py``
    line 326 — ``if memory_type:`` guard that skips ``--filter`` when falsy.
    """

    async def _run_unfiltered(
        self,
        items: list[dict],
        decay_config: "DecayConfig | None" = None,
    ) -> tuple[list, list[str]]:
        """Run search with memory_type=None and return (results, cmd_args)."""
        retriever = QmdRetriever(decay_config=decay_config)
        fake_proc = _fake_subprocess_result(items)
        mock_run = MagicMock(return_value=fake_proc)

        with (
            patch("shad.retrieval.qmd.shutil.which", return_value="/usr/bin/qmd"),
            patch("shad.retrieval.qmd.subprocess.run", mock_run),
        ):
            results = await retriever.search("baseline query", mode="bm25", memory_type=None)

        captured_args: list[str] = mock_run.call_args.args[0]
        return results, captured_args

    # ------------------------------------------------------------------
    # Command-argument contract
    # ------------------------------------------------------------------

    async def test_no_filter_flag_in_cmd_args(self) -> None:
        """memory_type=None must not add --filter to the qmd command."""
        _, args = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        assert "--filter" not in args

    async def test_no_filter_flag_with_empty_payload(self) -> None:
        """Even with an empty result set, --filter must be absent."""
        _, args = await self._run_unfiltered([])
        assert "--filter" not in args

    # ------------------------------------------------------------------
    # Result completeness
    # ------------------------------------------------------------------

    async def test_all_items_returned(self) -> None:
        """Every item in the qmd payload must appear in the result list."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        assert len(results) == len(_ALL_MEMORY_TYPES_PAYLOAD)

    async def test_all_paths_present_in_results(self) -> None:
        """Result paths must exactly match the set of paths qmd returned."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        returned_paths = {r.path for r in results}
        assert returned_paths == _ALL_EXPECTED_PATHS

    async def test_episodic_item_passes_through(self) -> None:
        """The episodic-classified item must appear in unfiltered results."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        paths = [r.path for r in results]
        assert "episodic/feed-item.md" in paths

    async def test_semantic_item_passes_through(self) -> None:
        """The semantic-classified item must appear in unfiltered results."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        paths = [r.path for r in results]
        assert "semantic/github-repo.md" in paths

    async def test_working_memory_item_passes_through(self) -> None:
        """The working-memory item must appear in unfiltered results."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        paths = [r.path for r in results]
        assert "working/scratch-note.md" in paths

    async def test_procedural_item_passes_through(self) -> None:
        """The procedural item must appear in unfiltered results."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        paths = [r.path for r in results]
        assert "procedural/how-to.md" in paths

    async def test_unclassified_item_passes_through(self) -> None:
        """A result with no memory_type annotation must also pass through."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD)
        paths = [r.path for r in results]
        assert "unclassified/no-type.md" in paths

    # ------------------------------------------------------------------
    # Score integrity
    # ------------------------------------------------------------------

    async def test_raw_scores_are_unmodified_without_decay(self) -> None:
        """With decay_config=None the score on each result equals the raw qmd score."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD, decay_config=None)
        by_path = {r.path: r for r in results}

        assert by_path["episodic/feed-item.md"].score == pytest.approx(0.90)
        assert by_path["semantic/github-repo.md"].score == pytest.approx(0.80)
        assert by_path["working/scratch-note.md"].score == pytest.approx(0.70)
        assert by_path["procedural/how-to.md"].score == pytest.approx(0.60)
        assert by_path["unclassified/no-type.md"].score == pytest.approx(0.50)

    async def test_raw_order_preserved_without_decay(self) -> None:
        """With no filter and no decay the results must arrive in qmd's emission order."""
        results, _ = await self._run_unfiltered(_ALL_MEMORY_TYPES_PAYLOAD, decay_config=None)
        returned_paths = [r.path for r in results]
        expected_paths = [item["file"] for item in _ALL_MEMORY_TYPES_PAYLOAD]
        assert returned_paths == expected_paths

    # ------------------------------------------------------------------
    # Larger mixed payload
    # ------------------------------------------------------------------

    async def test_large_mixed_payload_count_unchanged(self) -> None:
        """50 items covering all four types must all survive the unfiltered pipeline."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        types_cycle = [
            "episodic_memory",
            "semantic_memory",
            "working_memory",
            "procedural_memory",
        ]
        items = [
            _make_qmd_json(
                f"{types_cycle[i % 4]}/doc-{i:03d}.md",
                score=round(1.0 - i * 0.01, 2),
                ingested_at=(now - timedelta(days=i)).isoformat(),
            )
            for i in range(50)
        ]

        results, _ = await self._run_unfiltered(items)
        assert len(results) == 50

    async def test_large_mixed_payload_all_paths_present(self) -> None:
        """Every path in a 50-item mixed payload must appear in unfiltered results."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        types_cycle = [
            "episodic_memory",
            "semantic_memory",
            "working_memory",
            "procedural_memory",
        ]
        items = [
            _make_qmd_json(
                f"{types_cycle[i % 4]}/doc-{i:03d}.md",
                score=round(1.0 - i * 0.01, 2),
                ingested_at=(now - timedelta(days=i)).isoformat(),
            )
            for i in range(50)
        ]

        results, _ = await self._run_unfiltered(items)
        expected = {item["file"] for item in items}
        returned = {r.path for r in results}
        assert returned == expected
