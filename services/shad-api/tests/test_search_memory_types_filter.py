"""Tests for QmdRetriever.search with a list of memory_types.

Contract under test
-------------------
``QmdRetriever.search(query, memory_types=[...])`` must:

1. **Single type** — return only results whose ``metadata["memory_type"]``
   matches the single requested type; exclude all others.
2. **Multiple types (OR logic)** — return results matching *any* of the listed
   types; exclude results whose type is not in the list.
3. **Empty list** — no filter applied; all results are returned.
4. **None (default)** — no filter applied; all results are returned.
5. **No matching results** — return an empty list when no result's type is
   in the specified list.
6. **Untyped results** — results whose qmd JSON carries no ``memory_type``
   field are excluded when a filter list is active, preserved when there is
   none.
7. **memory_type value preserved** — every included result must expose its
   type via ``metadata["memory_type"]`` so callers can inspect it downstream.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from shad.retrieval.qmd import QmdRetriever
from shad.vault.shadow_index import MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    path: str,
    score: float,
    memory_type: str | None = None,
) -> dict:
    """Build a minimal qmd JSON result item.

    When *memory_type* is given it is included in the item dict, simulating
    the frontmatter field that qmd surfaces on each result.
    """
    item: dict = {
        "file": path,
        "score": score,
        "content": f"Content of {path}",
        "title": path,
        "snippet": f"Snippet from {path}",
    }
    if memory_type is not None:
        item["memory_type"] = memory_type
    return item


def _fake_proc(items: list[dict]) -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = json.dumps(items).encode()
    proc.stderr = b""
    return proc


async def _search(
    items: list[dict],
    memory_types: list[str] | None = None,
) -> list:
    """Run QmdRetriever.search with stubbed subprocess output."""
    retriever = QmdRetriever()
    with (
        patch("shad.retrieval.qmd.shutil.which", return_value="/usr/bin/qmd"),
        patch("shad.retrieval.qmd.subprocess.run", return_value=_fake_proc(items)),
    ):
        return await retriever.search("test query", mode="bm25", memory_types=memory_types)


# ---------------------------------------------------------------------------
# Single-type filter
# ---------------------------------------------------------------------------

class TestSingleMemoryTypeFilter:
    """search(memory_types=[one_type]) must return only results of that type."""

    async def test_returns_only_matching_type(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("wk.md", 0.7, memory_type=MemoryType.WORKING.value),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        assert len(results) == 1
        assert results[0].path == "ep.md"

    async def test_excludes_all_non_matching_types(self) -> None:
        items = [
            _make_item("ep1.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("ep2.md", 0.8, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md",  0.7, memory_type=MemoryType.SEMANTIC.value),
            _make_item("pr.md",  0.6, memory_type=MemoryType.PROCEDURAL.value),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        paths = [r.path for r in results]
        assert "sm.md" not in paths
        assert "pr.md" not in paths

    async def test_all_returned_results_match_requested_type(self) -> None:
        items = [
            _make_item("ep1.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("ep2.md", 0.8, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md",  0.7, memory_type=MemoryType.SEMANTIC.value),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        assert all(
            r.metadata.get("memory_type") == MemoryType.EPISODIC.value
            for r in results
        )

    @pytest.mark.parametrize(
        "mt",
        [
            MemoryType.WORKING.value,
            MemoryType.EPISODIC.value,
            MemoryType.SEMANTIC.value,
            MemoryType.PROCEDURAL.value,
        ],
    )
    async def test_each_type_filters_independently(self, mt: str) -> None:
        """Filter for each variant yields exactly the one matching result."""
        all_types = [m.value for m in MemoryType]
        items = [_make_item(f"{t}.md", 0.5, memory_type=t) for t in all_types]
        results = await _search(items, memory_types=[mt])
        assert len(results) == 1
        assert results[0].metadata.get("memory_type") == mt


# ---------------------------------------------------------------------------
# Multiple types — OR logic
# ---------------------------------------------------------------------------

class TestMultipleMemoryTypesOrLogic:
    """search(memory_types=[a, b]) must return results matching a OR b."""

    async def test_returns_results_matching_any_listed_type(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("wk.md", 0.7, memory_type=MemoryType.WORKING.value),
            _make_item("pr.md", 0.6, memory_type=MemoryType.PROCEDURAL.value),
        ]
        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value],
        )
        paths = {r.path for r in results}
        assert "ep.md" in paths
        assert "sm.md" in paths
        assert len(results) == 2

    async def test_excludes_types_absent_from_list(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("wk.md", 0.7, memory_type=MemoryType.WORKING.value),
            _make_item("pr.md", 0.6, memory_type=MemoryType.PROCEDURAL.value),
        ]
        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value],
        )
        excluded_types = {MemoryType.WORKING.value, MemoryType.PROCEDURAL.value}
        for r in results:
            assert r.metadata.get("memory_type") not in excluded_types

    async def test_all_four_types_returns_everything(self) -> None:
        all_types = [m.value for m in MemoryType]
        items = [_make_item(f"{t}.md", 0.5, memory_type=t) for t in all_types]
        results = await _search(items, memory_types=all_types)
        assert len(results) == len(all_types)

    async def test_three_of_four_types_excludes_fourth(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("wk.md", 0.7, memory_type=MemoryType.WORKING.value),
            _make_item("pr.md", 0.6, memory_type=MemoryType.PROCEDURAL.value),
        ]
        results = await _search(
            items,
            memory_types=[
                MemoryType.EPISODIC.value,
                MemoryType.SEMANTIC.value,
                MemoryType.WORKING.value,
            ],
        )
        assert len(results) == 3
        assert all(
            r.metadata.get("memory_type") != MemoryType.PROCEDURAL.value
            for r in results
        )

    async def test_only_included_types_appear_in_results(self) -> None:
        """Every result's memory_type must be one of the requested types."""
        requested = [MemoryType.SEMANTIC.value, MemoryType.PROCEDURAL.value]
        items = [
            _make_item("sm.md", 0.9, memory_type=MemoryType.SEMANTIC.value),
            _make_item("ep.md", 0.8, memory_type=MemoryType.EPISODIC.value),
            _make_item("pr.md", 0.7, memory_type=MemoryType.PROCEDURAL.value),
            _make_item("wk.md", 0.6, memory_type=MemoryType.WORKING.value),
        ]
        results = await _search(items, memory_types=requested)
        for r in results:
            assert r.metadata.get("memory_type") in requested, (
                f"{r.path} has type {r.metadata.get('memory_type')!r}, not in {requested}"
            )


# ---------------------------------------------------------------------------
# No filter — empty list / None
# ---------------------------------------------------------------------------

class TestNoMemoryTypeFilter:
    """memory_types=None or memory_types=[] must return all results unfiltered."""

    async def test_none_returns_all_results(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("wk.md", 0.7, memory_type=MemoryType.WORKING.value),
        ]
        results = await _search(items, memory_types=None)
        assert len(results) == 3

    async def test_empty_list_returns_all_results(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
        ]
        results = await _search(items, memory_types=[])
        assert len(results) == 2

    async def test_default_param_returns_all_results(self) -> None:
        """Calling search without memory_types must not apply any type filter."""
        items = [
            _make_item("a.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("b.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("c.md", 0.7, memory_type=MemoryType.PROCEDURAL.value),
        ]
        results = await _search(items)  # memory_types omitted
        assert len(results) == 3

    async def test_untyped_results_preserved_when_no_filter(self) -> None:
        items = [
            _make_item("typed.md",   0.9, memory_type=MemoryType.SEMANTIC.value),
            _make_item("untyped.md", 0.7, memory_type=None),
        ]
        results = await _search(items, memory_types=None)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# No matching results
# ---------------------------------------------------------------------------

class TestNoMatchingResults:
    async def test_returns_empty_when_no_type_matches(self) -> None:
        items = [
            _make_item("sm.md", 0.9, memory_type=MemoryType.SEMANTIC.value),
            _make_item("pr.md", 0.8, memory_type=MemoryType.PROCEDURAL.value),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        assert results == []

    async def test_returns_empty_when_all_items_untyped(self) -> None:
        items = [
            _make_item("a.md", 0.9, memory_type=None),
            _make_item("b.md", 0.8, memory_type=None),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        assert results == []

    async def test_returns_empty_list_with_empty_input_and_filter(self) -> None:
        results = await _search([], memory_types=[MemoryType.SEMANTIC.value])
        assert results == []


# ---------------------------------------------------------------------------
# Results without memory_type metadata
# ---------------------------------------------------------------------------

class TestResultsWithoutMemoryType:
    """Untyped results should be excluded when a filter list is active."""

    async def test_untyped_excluded_when_filter_active(self) -> None:
        items = [
            _make_item("typed.md",   0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("untyped.md", 0.8, memory_type=None),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        paths = [r.path for r in results]
        assert "typed.md" in paths
        assert "untyped.md" not in paths

    async def test_untyped_excluded_with_multi_type_filter(self) -> None:
        items = [
            _make_item("ep.md",      0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md",      0.8, memory_type=MemoryType.SEMANTIC.value),
            _make_item("untyped.md", 0.7, memory_type=None),
        ]
        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value],
        )
        paths = [r.path for r in results]
        assert "untyped.md" not in paths
        assert len(results) == 2

    async def test_mixed_typed_and_untyped_only_typed_returned(self) -> None:
        items = [
            _make_item("ep1.md",     0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("no-type-1",  0.85, memory_type=None),
            _make_item("ep2.md",     0.8, memory_type=MemoryType.EPISODIC.value),
            _make_item("no-type-2",  0.75, memory_type=None),
            _make_item("sm.md",      0.7, memory_type=MemoryType.SEMANTIC.value),
        ]
        results = await _search(items, memory_types=[MemoryType.EPISODIC.value])
        assert len(results) == 2
        assert all(
            r.metadata.get("memory_type") == MemoryType.EPISODIC.value
            for r in results
        )


# ---------------------------------------------------------------------------
# memory_type preserved in result metadata
# ---------------------------------------------------------------------------

class TestMemoryTypeInResultMetadata:
    """memory_type must be accessible via result.metadata["memory_type"]."""

    async def test_memory_type_accessible_per_result(self) -> None:
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
        ]
        results = await _search(
            items,
            memory_types=[MemoryType.EPISODIC.value, MemoryType.SEMANTIC.value],
        )
        by_path = {r.path: r for r in results}
        assert by_path["ep.md"].metadata.get("memory_type") == MemoryType.EPISODIC.value
        assert by_path["sm.md"].metadata.get("memory_type") == MemoryType.SEMANTIC.value

    async def test_memory_type_round_trips_all_variants(self) -> None:
        all_types = [m.value for m in MemoryType]
        items = [_make_item(f"{t}.md", 0.5, memory_type=t) for t in all_types]
        results = await _search(items, memory_types=all_types)
        result_types = {r.metadata.get("memory_type") for r in results}
        assert result_types == set(all_types)

    async def test_memory_type_present_without_filter(self) -> None:
        """Even without a filter, memory_type must be in metadata when supplied."""
        items = [
            _make_item("ep.md", 0.9, memory_type=MemoryType.EPISODIC.value),
            _make_item("sm.md", 0.8, memory_type=MemoryType.SEMANTIC.value),
        ]
        results = await _search(items, memory_types=None)
        by_path = {r.path: r for r in results}
        assert by_path["ep.md"].metadata.get("memory_type") == MemoryType.EPISODIC.value
        assert by_path["sm.md"].metadata.get("memory_type") == MemoryType.SEMANTIC.value
