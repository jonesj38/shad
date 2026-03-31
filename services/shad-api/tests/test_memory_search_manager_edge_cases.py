"""Edge-case tests for MemorySearchManager.search.

Three specific edge cases under test:

1. **Empty results when no result matches the requested type** — the manager
   returns an empty list rather than raising or leaking unmatched items.

2. **Null / missing memory_type on result entries** — results whose metadata
   carries ``memory_type=None`` *or* no ``memory_type`` key at all are excluded
   when a filter is active.

3. **Invalid / unrecognised memory_type values in metadata** — results whose
   metadata ``memory_type`` is an arbitrary string not in the allowed set are
   excluded, not surfaced as a different type or cause an exception.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from shad.retrieval.layer import RetrievalResult, SearchOpts
from shad.vault.search import MemorySearchManager
from shad.vault.shadow_index import MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(path: str, score: float, memory_type: object = "MISSING") -> RetrievalResult:
    """Build a RetrievalResult.

    Pass ``memory_type=None`` to put an explicit ``None`` in metadata.
    Omit the argument (sentinel ``"MISSING"``) to leave the key absent entirely.
    Pass any string to set that value.
    """
    meta: dict = {}
    if memory_type != "MISSING":
        meta["memory_type"] = memory_type
    return RetrievalResult(path=path, content=f"content of {path}", score=score, metadata=meta)


def _manager(results: list[RetrievalResult]) -> MemorySearchManager:
    """Return a MemorySearchManager backed by a stub retriever."""
    retriever = AsyncMock()
    retriever.search = AsyncMock(return_value=results)
    return MemorySearchManager(retriever)


# ---------------------------------------------------------------------------
# 1. Empty results when no result matches the requested type
# ---------------------------------------------------------------------------


class TestEmptyResultsWhenNoTypeMatch:
    """search() must return [] when a filter is active but no result qualifies."""

    async def test_returns_empty_list_when_all_results_are_wrong_type(self) -> None:
        results = [
            _result("sm.md", 0.9, memory_type=MemoryType.SEMANTIC.value),
            _result("pr.md", 0.8, memory_type=MemoryType.PROCEDURAL.value),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.EPISODIC))
        assert out == []

    async def test_returns_empty_list_when_retriever_returns_nothing(self) -> None:
        manager = _manager([])
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.WORKING))
        assert out == []

    async def test_returns_empty_list_for_each_type_when_no_results(self) -> None:
        for mt in MemoryType:
            manager = _manager([])
            out = await manager.search(SearchOpts(query="q", memory_type=mt))
            assert out == [], f"Expected [] for {mt}"

    async def test_returns_empty_list_with_multi_type_filter_and_no_matching_results(self) -> None:
        results = [
            _result("wk.md", 0.9, memory_type=MemoryType.WORKING.value),
        ]
        manager = _manager(results)
        out = await manager.search(
            SearchOpts(
                query="q",
                memory_type=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            )
        )
        assert out == []

    async def test_empty_result_count_not_limit(self) -> None:
        """Empty output must be an empty list, not a list of Nones or errors."""
        manager = _manager([])
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.SEMANTIC, limit=5))
        assert out == []
        assert len(out) == 0


# ---------------------------------------------------------------------------
# 2. Null / missing memory_type on result entries
# ---------------------------------------------------------------------------


class TestNullOrMissingMemoryTypeOnResults:
    """Results with None or absent memory_type are excluded when a filter is active."""

    async def test_result_with_none_memory_type_excluded_by_single_filter(self) -> None:
        results = [
            _result("typed.md",   0.9, memory_type=MemoryType.EPISODIC.value),
            _result("null-mt.md", 0.8, memory_type=None),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.EPISODIC))
        paths = [r.path for r in out]
        assert "typed.md" in paths
        assert "null-mt.md" not in paths

    async def test_result_without_memory_type_key_excluded_by_single_filter(self) -> None:
        results = [
            _result("typed.md",   0.9, memory_type=MemoryType.SEMANTIC.value),
            _result("no-key.md",  0.8),  # memory_type key absent
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.SEMANTIC))
        paths = [r.path for r in out]
        assert "typed.md" in paths
        assert "no-key.md" not in paths

    async def test_all_untyped_results_filtered_out_leaving_empty_list(self) -> None:
        results = [
            _result("a.md", 0.9, memory_type=None),
            _result("b.md", 0.8),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.EPISODIC))
        assert out == []

    async def test_none_memory_type_excluded_by_multi_type_filter(self) -> None:
        results = [
            _result("ep.md",      0.9, memory_type=MemoryType.EPISODIC.value),
            _result("null-mt.md", 0.8, memory_type=None),
            _result("sm.md",      0.7, memory_type=MemoryType.SEMANTIC.value),
        ]
        manager = _manager(results)
        out = await manager.search(
            SearchOpts(
                query="q",
                memory_type=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            )
        )
        paths = [r.path for r in out]
        assert "null-mt.md" not in paths
        assert len(out) == 2

    async def test_missing_key_and_none_both_excluded_together(self) -> None:
        """Mixing absent-key and None-value results — both must be excluded."""
        results = [
            _result("typed.md",   0.9, memory_type=MemoryType.WORKING.value),
            _result("null-mt.md", 0.8, memory_type=None),
            _result("no-key.md",  0.7),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.WORKING))
        assert len(out) == 1
        assert out[0].path == "typed.md"

    async def test_untyped_results_preserved_when_no_filter(self) -> None:
        """With no filter, results lacking memory_type must still be returned."""
        results = [
            _result("typed.md",   0.9, memory_type=MemoryType.SEMANTIC.value),
            _result("null-mt.md", 0.8, memory_type=None),
            _result("no-key.md",  0.7),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=None))
        assert len(out) == 3


# ---------------------------------------------------------------------------
# 3. Invalid / unrecognised memory_type values in metadata
# ---------------------------------------------------------------------------


class TestInvalidMemoryTypeValuesOnResults:
    """Results with unknown memory_type strings in metadata are excluded."""

    async def test_unknown_string_excluded_by_single_type_filter(self) -> None:
        results = [
            _result("valid.md",   0.9, memory_type=MemoryType.EPISODIC.value),
            _result("garbage.md", 0.8, memory_type="not_a_real_type"),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.EPISODIC))
        paths = [r.path for r in out]
        assert "valid.md" in paths
        assert "garbage.md" not in paths

    async def test_empty_string_excluded_by_filter(self) -> None:
        results = [
            _result("valid.md",  0.9, memory_type=MemoryType.SEMANTIC.value),
            _result("empty.md",  0.8, memory_type=""),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.SEMANTIC))
        paths = [r.path for r in out]
        assert "valid.md" in paths
        assert "empty.md" not in paths

    async def test_wrong_numeric_type_in_metadata_excluded(self) -> None:
        results = [
            _result("valid.md",   0.9, memory_type=MemoryType.WORKING.value),
            _result("numeric.md", 0.8, memory_type=42),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.WORKING))
        paths = [r.path for r in out]
        assert "valid.md" in paths
        assert "numeric.md" not in paths

    async def test_all_invalid_types_leave_empty_result(self) -> None:
        results = [
            _result("a.md", 0.9, memory_type="bad_value_1"),
            _result("b.md", 0.8, memory_type="bad_value_2"),
            _result("c.md", 0.7, memory_type=""),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=MemoryType.EPISODIC))
        assert out == []

    async def test_invalid_type_excluded_by_multi_type_filter(self) -> None:
        results = [
            _result("ep.md",      0.9, memory_type=MemoryType.EPISODIC.value),
            _result("sm.md",      0.8, memory_type=MemoryType.SEMANTIC.value),
            _result("garbage.md", 0.7, memory_type="totally_unknown"),
        ]
        manager = _manager(results)
        out = await manager.search(
            SearchOpts(
                query="q",
                memory_type=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            )
        )
        paths = [r.path for r in out]
        assert "ep.md" in paths
        assert "sm.md" in paths
        assert "garbage.md" not in paths
        assert len(out) == 2

    async def test_invalid_type_preserved_when_no_filter(self) -> None:
        """Without a filter, even results with invalid type strings pass through."""
        results = [
            _result("valid.md",   0.9, memory_type=MemoryType.SEMANTIC.value),
            _result("garbage.md", 0.8, memory_type="not_a_real_type"),
        ]
        manager = _manager(results)
        out = await manager.search(SearchOpts(query="q", memory_type=None))
        assert len(out) == 2
