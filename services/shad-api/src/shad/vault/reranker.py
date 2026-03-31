"""Temporal-decay reranker for retrieval results.

Applies decay to retrieval scores based on document age, preserving
the original score in result metadata under the key ``original_score``.

Typical use-case: re-sort episodic search results so that stale hits
yield to fresher ones even when their raw relevance score was higher.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from shad.retrieval.layer import RetrievalResult
from shad.vault.decay import DecayConfig, DecayCurve, apply_decay

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0


def temporal_decay_score(
    base_score: float,
    age_days: float,
    halflife_days: float = 90.0,
) -> float:
    """Return *base_score* discounted by exponential half-life decay.

    Uses the formula ``base_score * 0.5 ^ (age_days / halflife_days)``, so
    the score is exactly halved at ``age_days == halflife_days``.

    Args:
        base_score:    Raw relevance score in [0, 1].
        age_days:      Age of the document in days.  Negative values are
                       treated as zero (no penalty for future-dated entries).
        halflife_days: Number of days after which the score is halved.
                       Must be > 0.  Defaults to 90 days.

    Returns:
        Decayed score in [0, 1].

    Raises:
        ValueError: If *halflife_days* is not strictly positive.
    """
    config = DecayConfig(
        halflife_seconds=halflife_days * _SECONDS_PER_DAY,
        curve=DecayCurve.EXPONENTIAL,
    )
    return apply_decay(base_score, age_days * _SECONDS_PER_DAY, config)


def rerank(
    results: list[RetrievalResult],
    ages: list[float | None],
    config: DecayConfig,
) -> list[RetrievalResult]:
    """Apply temporal decay to *results* and re-sort by adjusted score.

    Each returned ``RetrievalResult`` has:

    - ``score``: decay-adjusted relevance score (replaces the raw score).
    - ``metadata["original_score"]``: the raw score **before** adjustment.

    The input ``results`` list is **not** mutated.

    Args:
        results:  Retrieval results to rerank.
        ages:     Age in seconds for each result, parallel to *results*.
                  Negative values are treated as zero (no penalty).
                  ``None`` means no timestamp was available; decay is skipped
                  and the original score is preserved unchanged.
        config:   Decay configuration (halflife + curve shape).

    Returns:
        New list of ``RetrievalResult`` sorted by decay-adjusted score,
        highest first.

    Raises:
        ValueError: If ``len(results) != len(ages)``.
    """
    if len(results) != len(ages):
        raise ValueError(
            f"results and ages must have the same length, "
            f"got {len(results)} and {len(ages)}"
        )

    adjusted: list[RetrievalResult] = []
    for result, age in zip(results, ages, strict=False):
        original_score = result.score
        if age is None:
            logger.debug(
                "No timestamp for result %r — skipping decay, preserving score %.4f",
                result.path,
                original_score,
            )
            decayed = original_score
        else:
            decayed = apply_decay(original_score, age, config)
        new_metadata = {**result.metadata, "original_score": original_score}
        adjusted.append(
            RetrievalResult(
                path=result.path,
                content=result.content,
                score=decayed,
                snippet=result.snippet,
                collection=result.collection,
                docid=result.docid,
                matched_line=result.matched_line,
                metadata=new_metadata,
            )
        )

    return sorted(adjusted, key=lambda r: r.score, reverse=True)


def _age_seconds_from_result(
    result: RetrievalResult,
    now: datetime,
) -> float | None:
    """Parse the ``ingested_at`` timestamp from result metadata and return age in seconds.

    Args:
        result: A retrieval result whose ``metadata`` may contain ``"ingested_at"``.
        now:    Reference time for computing elapsed seconds.

    Returns:
        Age in seconds, or ``None`` if no parseable timestamp is present.
    """
    raw = result.metadata.get("ingested_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (now - ts).total_seconds()
    except (ValueError, TypeError):
        logger.debug("Could not parse ingested_at %r for %r", raw, result.path)
        return None


def rerank_with_metadata(
    results: list[RetrievalResult],
    config: DecayConfig,
    now: datetime | None = None,
) -> list[RetrievalResult]:
    """Apply temporal decay to *results* using timestamps from their metadata.

    Extracts ``metadata["ingested_at"]`` from each result to compute age, then
    delegates to :func:`rerank`.  Results without a parseable timestamp are not
    penalised (their score is preserved unchanged).

    Args:
        results: Retrieval results to rerank.
        config:  Decay configuration (halflife + curve shape).
        now:     Reference timestamp.  Defaults to ``datetime.now(UTC)``.

    Returns:
        New list of ``RetrievalResult`` sorted by decay-adjusted score,
        highest first.
    """
    reference = now if now is not None else datetime.now(UTC)
    ages = [_age_seconds_from_result(r, reference) for r in results]
    return rerank(results, ages, config)
