"""Tests for MemorySearchManager.search with a single memory_type filter.

Contract under test
-------------------
``MemorySearchManager.search(SearchOpts(memory_type=MemoryType.X))`` must:

1. Return only results whose ``metadata["memory_type"]`` equals ``MemoryType.X.value``.
2. Exclude results of every other type (WORKING, SEMANTIC, EPISODIC, PROCEDURAL).
3. Exclude results whose ``metadata`` lacks a ``memory_type`` key.
4. Return an empty list when no result matches the requested type.
5. Preserve ``metadata["memory_type"]`` on every returned result.
6. Respect ``SearchOpts.limit`` even after filtering.
7. Work correctly and independently for each of the four MemoryType variants.
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

_ALL_TYPES: list[MemoryType] = list(MemoryType)


def _result(path: str, score: float, memory_type: str | None = None) -> RetrievalResult:
    """Build a minimal RetrievalResult.

    Omit *memory_type* to leave the metadata key absent entirely.
    Pass ``None`` explicitly to store ``{"memory_type": None}`` in metadata.
    """
    meta: dict = {}
    if memory_type is not None:
        meta["memory_type"] = memory_type
    return RetrievalResult(
        path=path,
        content=f"content:{path}",
        score=score,
        metadata=meta,
    )


def _manager_with(results: list[RetrievalResult]) -> MemorySearchManager:
    """Return a MemorySearchManager whose retriever stub always returns *results*."""
    stub = AsyncMock()
    stub.search = AsyncMock(return_value=results)
    return MemorySearchManager(stub)


def _one_of_each(base_score: float = 0.9) -> list[RetrievalResult]:
    """Return four results — one per MemoryType — in descending score order."""
    return [
        _result(f"{mt.value}.md", round(base_score - i * 0.1, 2), mt.value)
        for i, mt in enumerate(_ALL_TYPES)
    ]


# ---------------------------------------------------------------------------
# 1. Each type filters independently
# ---------------------------------------------------------------------------


class TestEachTypeFiltersIndependently:
    """Filtering by a single MemoryType must return exactly the matching result(s)."""

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_single_match_from_one_of_each(self, mt: MemoryType) -> None:
        """When one result per type exists, only the requested type is returned."""
        results = _one_of_each()
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        assert len(out) == 1
        assert out[0].metadata.get("memory_type") == mt.value

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_returns_all_matching_results_not_just_one(self, mt: MemoryType) -> None:
        """When multiple results share the same type, all are returned."""
        results = [
            _result(f"{mt.value}-a.md", 0.9, mt.value),
            _result(f"{mt.value}-b.md", 0.8, mt.value),
            _result(f"{mt.value}-c.md", 0.7, mt.value),
            # noise: one result of a different type
            _result("other.md", 0.6, _other_type(mt).value),
        ]
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        assert len(out) == 3
        assert all(r.metadata.get("memory_type") == mt.value for r in out)

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_all_returned_results_have_correct_type(self, mt: MemoryType) -> None:
        """Every result in the output must carry the requested memory_type."""
        results = _one_of_each()
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        for r in out:
            assert r.metadata.get("memory_type") == mt.value, (
                f"{r.path!r} has memory_type={r.metadata.get('memory_type')!r}; "
                f"expected {mt.value!r}"
            )


# ---------------------------------------------------------------------------
# 2. Non-matching types are excluded
# ---------------------------------------------------------------------------


class TestNonMatchingTypesExcluded:
    """Results of every other type must be absent from the output."""

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_excludes_every_other_type(self, mt: MemoryType) -> None:
        """Results of the three non-matching types must not appear in the output."""
        results = _one_of_each()
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        returned_paths = {r.path for r in out}
        other_types = [t for t in _ALL_TYPES if t != mt]
        for other in other_types:
            assert f"{other.value}.md" not in returned_paths, (
                f"Result for {other.value!r} must not appear when filtering by {mt.value!r}"
            )

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_excludes_results_without_memory_type_key(self, mt: MemoryType) -> None:
        """Results missing the metadata 'memory_type' key are excluded when filtering."""
        results = [
            _result(f"{mt.value}.md", 0.9, mt.value),
            _result("no-key.md", 0.85),  # absent key
        ]
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        paths = [r.path for r in out]
        assert f"{mt.value}.md" in paths
        assert "no-key.md" not in paths


# ---------------------------------------------------------------------------
# 3. Empty results
# ---------------------------------------------------------------------------


class TestEmptyResultsForSingleTypeFilter:
    """The manager must return [] when no result matches the filter."""

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_empty_list_when_no_type_matches(self, mt: MemoryType) -> None:
        """With only results of other types, output is []."""
        other_types = [t for t in _ALL_TYPES if t != mt]
        results = [_result(f"{t.value}.md", 0.8, t.value) for t in other_types]
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        assert out == []

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_empty_list_when_retriever_returns_nothing(self, mt: MemoryType) -> None:
        manager = _manager_with([])
        out = await manager.search(SearchOpts(query="q", memory_type=mt))
        assert out == []

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_empty_list_when_all_results_are_untyped(self, mt: MemoryType) -> None:
        results = [_result("a.md", 0.9), _result("b.md", 0.8)]  # no memory_type key
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))
        assert out == []


# ---------------------------------------------------------------------------
# 4. Metadata preserved on returned results
# ---------------------------------------------------------------------------


class TestMemoryTypePreservedInMetadata:
    """memory_type must be accessible via result.metadata on every returned result."""

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_metadata_memory_type_matches_filter(self, mt: MemoryType) -> None:
        results = [_result(f"{mt.value}.md", 0.88, mt.value)]
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt))

        assert len(out) == 1
        assert out[0].metadata.get("memory_type") == mt.value

    async def test_all_four_types_metadata_correct_when_all_requested(self) -> None:
        """Filtering by each type one at a time yields consistent metadata for all four."""
        results = _one_of_each()
        for mt in _ALL_TYPES:
            manager = _manager_with(results)
            out = await manager.search(SearchOpts(query="q", memory_type=mt))
            assert len(out) == 1
            assert out[0].metadata["memory_type"] == mt.value, (
                f"Metadata mismatch for {mt}"
            )


# ---------------------------------------------------------------------------
# 5. Limit respected after filtering
# ---------------------------------------------------------------------------


class TestLimitRespectedAfterFiltering:
    """The output length must not exceed SearchOpts.limit."""

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_limit_one_returns_at_most_one(self, mt: MemoryType) -> None:
        results = [
            _result(f"{mt.value}-1.md", 0.9, mt.value),
            _result(f"{mt.value}-2.md", 0.8, mt.value),
            _result(f"{mt.value}-3.md", 0.7, mt.value),
        ]
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt, limit=1))

        assert len(out) <= 1

    @pytest.mark.parametrize("mt", _ALL_TYPES)
    async def test_limit_two_returns_at_most_two(self, mt: MemoryType) -> None:
        results = [
            _result(f"{mt.value}-{i}.md", round(0.9 - i * 0.05, 2), mt.value)
            for i in range(5)
        ]
        manager = _manager_with(results)
        out = await manager.search(SearchOpts(query="q", memory_type=mt, limit=2))

        assert len(out) <= 2
        assert all(r.metadata.get("memory_type") == mt.value for r in out)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _other_type(mt: MemoryType) -> MemoryType:
    """Return any MemoryType variant that is not *mt*."""
    for candidate in _ALL_TYPES:
        if candidate != mt:
            return candidate
    raise AssertionError("unreachable")  # pragma: no cover
