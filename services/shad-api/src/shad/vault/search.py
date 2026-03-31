"""Vault-aware search manager.

Wraps a :class:`~shad.retrieval.layer.RetrievalLayer` and adds
post-retrieval filtering by :class:`~shad.vault.shadow_index.MemoryType`.

Usage::

    from shad.vault.search import MemorySearchManager
    from shad.retrieval.layer import SearchOpts
    from shad.vault.shadow_index import MemoryType

    manager = MemorySearchManager(retriever)
    results = await manager.search(
        SearchOpts(query="auth patterns", memory_type=MemoryType.SEMANTIC)
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shad.retrieval.layer import RetrievalLayer, RetrievalResult, SearchOpts
from shad.vault.shadow_index import MemoryType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# How many extra results to fetch when filtering by memory_type so that
# after the filter we still have a chance of returning `limit` items.
_FILTER_OVERSAMPLE = 4


class MemorySearchManager:
    """Vault-aware search: delegates to a RetrievalLayer, then filters by memory_type.

    Args:
        retriever: Any object that satisfies the
            :class:`~shad.retrieval.layer.RetrievalLayer` protocol.
    """

    def __init__(self, retriever: RetrievalLayer) -> None:
        self._retriever = retriever

    async def search(self, opts: SearchOpts) -> list[RetrievalResult]:
        """Run a search and optionally post-filter results by memory_type.

        The underlying retriever is called with ``opts.query``, ``opts.mode``,
        ``opts.limit``, and ``opts.min_score``.  When ``opts.memory_type`` is
        set, the fetch limit is multiplied by :data:`_FILTER_OVERSAMPLE` so
        that enough raw candidates are retrieved before discarding
        non-matching results.

        Post-filtering compares ``result.metadata["memory_type"]`` (the
        string value stored at ingest time, e.g. ``"episodic"``) against
        the allowed set derived from ``opts.memory_type``.  Results whose
        metadata lacks a ``memory_type`` key are excluded when a filter is
        active.

        Args:
            opts: Search parameters including query, mode, limit, min_score,
                and optional memory_type filter.

        Returns:
            Up to ``opts.limit`` :class:`~shad.retrieval.layer.RetrievalResult`
            objects sorted by descending relevance score.
        """
        filtering = opts.memory_type is not None
        fetch_limit = opts.limit * _FILTER_OVERSAMPLE if filtering else opts.limit

        raw: list[RetrievalResult] = await self._retriever.search(
            query=opts.query,
            mode=opts.mode,
            limit=fetch_limit,
            min_score=opts.min_score,
        )

        if not filtering:
            return raw

        allowed: set[str] = _normalise_memory_type_filter(opts.memory_type)

        filtered = [
            r for r in raw if r.metadata.get("memory_type") in allowed
        ]

        if len(filtered) < opts.limit and len(raw) == fetch_limit:
            logger.debug(
                "memory_type filter reduced results from %d to %d "
                "(limit=%d, allowed=%s); consider increasing oversample factor",
                len(raw),
                len(filtered),
                opts.limit,
                allowed,
            )

        return filtered[: opts.limit]


def _normalise_memory_type_filter(
    memory_type: MemoryType | list[MemoryType] | None,
) -> set[str]:
    """Convert a SearchOpts.memory_type value to a set of string enum values.

    The metadata stored on RetrievalResult uses ``MemoryType.value`` strings
    (e.g. ``"episodic"``), so comparison is done against those strings.
    """
    if memory_type is None:
        return set()
    if isinstance(memory_type, list):
        return {mt.value for mt in memory_type}
    return {memory_type.value}
