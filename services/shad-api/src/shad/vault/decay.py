"""Temporal decay for episodic memory scoring.

Provides configurable decay so that older query results contribute less
to history-pain calculations in gap detection.  Modelled on the
exponential half-life formula used in network ban-score systems (see
teranode/connmgr/dynamicbanscore-go) and the openclaw temporal-decay
implementation.

Two curve shapes are supported:

- EXPONENTIAL  score * 0.5 ^ (age / halflife)   — asymptotic, never zero
- LINEAR       max(0, score * (1 - age/(2*halflife)))  — reaches zero at 2*halflife

Both curves are normalised so that score is exactly halved at age == halflife.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from shad.vault.gap_detection import QueryHistoryAnalyzer
    from shad.vault.shadow_index import SnapshotEntry


@runtime_checkable
class DecayFunction(Protocol):
    """Callable protocol for temporal decay functions.

    Matches the signature of :func:`apply_decay` so that custom decay
    implementations can be substituted wherever a decay function is expected.
    """

    def __call__(
        self, score: float, age_seconds: float, config: DecayConfig
    ) -> float: ...


class DecayCurve(StrEnum):
    """Shape of the decay curve."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"


_SECONDS_PER_DAY: float = 86_400.0

# Sentinel halflife used for SEMANTIC memory: 100 years — effectively no decay.
_NO_DECAY_HALFLIFE: float = 100 * 365 * _SECONDS_PER_DAY


@dataclass(frozen=True)
class DecayConfig:
    """Parameters that control temporal score decay.

    Attributes:
        halflife_seconds: Time (in seconds) after which a score is halved.
            Must be > 0.
        curve: Mathematical shape of the decay.
    """

    halflife_seconds: float
    curve: DecayCurve = DecayCurve.EXPONENTIAL

    def __post_init__(self) -> None:
        if self.halflife_seconds <= 0:
            raise ValueError(
                f"halflife_seconds must be > 0, got {self.halflife_seconds}"
            )

    @classmethod
    def default_for(cls, memory_type: object) -> "DecayConfig":
        """Return a :class:`DecayConfig` with sensible defaults for *memory_type*.

        Defaults per type:

        - **EPISODIC**   — 90-day exponential halflife.  Time-bound events
          naturally fade; mirrors the openclaw ``halfLifeDays=30–90`` range.
        - **WORKING**    — 1-day linear halflife.  In-progress knowledge is
          extremely short-lived; linear curve reaches zero at 2 days.
        - **PROCEDURAL** — 365-day exponential halflife.  Skills degrade
          slowly over time.
        - **SEMANTIC**   — Effectively no decay (100-year halflife).  Factual
          reference content is timeless; matches ``SnapshotDecayScorer``
          behaviour of passing ``age_seconds=0`` for SEMANTIC entries.

        Args:
            memory_type: A :class:`~shad.vault.shadow_index.MemoryType`
                instance (or any object whose ``value`` attribute is one of
                ``"episodic_memory"``, ``"working_memory"``,
                ``"procedural_memory"``, ``"semantic_memory"``).

        Returns:
            A :class:`DecayConfig` appropriate for the supplied type.
        """
        # Lazy import avoids a circular dependency at module load time.
        from shad.vault.shadow_index import MemoryType

        _DEFAULTS: dict[MemoryType, DecayConfig] = {
            MemoryType.EPISODIC: cls(
                halflife_seconds=90 * _SECONDS_PER_DAY,
                curve=DecayCurve.EXPONENTIAL,
            ),
            MemoryType.WORKING: cls(
                halflife_seconds=1 * _SECONDS_PER_DAY,
                curve=DecayCurve.LINEAR,
            ),
            MemoryType.PROCEDURAL: cls(
                halflife_seconds=365 * _SECONDS_PER_DAY,
                curve=DecayCurve.EXPONENTIAL,
            ),
            MemoryType.SEMANTIC: cls(
                halflife_seconds=_NO_DECAY_HALFLIFE,
                curve=DecayCurve.EXPONENTIAL,
            ),
        }
        return _DEFAULTS.get(memory_type, _DEFAULTS[MemoryType.SEMANTIC])  # type: ignore[arg-type]


def linear_age_factor(age_seconds: float, max_age_seconds: float) -> float:
    """Return a linear freshness factor in [0, 1] based on age and a hard cutoff.

    Edge cases:
    - ``max_age_seconds <= 0``  → 0.0  (invalid / disabled window)
    - ``age_seconds < 0``       → 1.0  (event is in the future; treat as brand-new)
    - ``age_seconds >= max_age_seconds`` → 0.0  (past the window)
    - Otherwise: ``1.0 - age_seconds / max_age_seconds``

    Unlike :func:`apply_decay`, this function uses a hard cutoff at *max_age_seconds*
    rather than a halflife parameterisation, and it returns a raw factor (not a
    score-weighted value).

    Args:
        age_seconds:     Elapsed seconds since the event.  Negative values mean
                         the event is timestamped in the future.
        max_age_seconds: Window beyond which the factor drops to zero.

    Returns:
        Freshness factor in [0, 1].
    """
    if max_age_seconds <= 0:
        return 0.0
    if age_seconds < 0:
        return 1.0
    if age_seconds >= max_age_seconds:
        return 0.0
    return 1.0 - age_seconds / max_age_seconds


def decay_score(
    base_score: float,
    days_since_access: int,
    decay_rate: float = 0.95,
) -> float:
    """Return *base_score* discounted by a per-day decay rate.

    Args:
        base_score:        Raw score before decay.
        days_since_access: Number of whole days since the memory was last accessed.
        decay_rate:        Fraction of score retained each day (default 0.95).

    Returns:
        ``base_score * decay_rate ** days_since_access``
    """
    days = max(0, days_since_access)
    return base_score * decay_rate**days


def apply_decay(score: float, age_seconds: float, config: DecayConfig) -> float:
    """Return *score* adjusted for the age of the underlying observation.

    Args:
        score:        Raw score in [0, 1].
        age_seconds:  Elapsed time in seconds since the observation was
                      recorded.  Negative values are treated as zero (no
                      decay for future-dated entries).
        config:       Decay parameters (halflife + curve shape).

    Returns:
        Decayed score in [0, 1].
    """
    age = max(0.0, age_seconds)

    if config.curve is DecayCurve.EXPONENTIAL:
        # score * 0.5^(age/halflife)  ≡  score * exp(-ln2 * age/halflife)
        factor = math.pow(0.5, age / config.halflife_seconds)
    else:  # LINEAR
        # Halved at t=halflife, zero at t=2*halflife
        factor = max(0.0, 1.0 - age / (2.0 * config.halflife_seconds))

    return max(0.0, min(1.0, score * factor))


@dataclass(frozen=True)
class DecayScore:
    """Temporal decay score for a single :class:`SnapshotEntry`.

    Attributes:
        snapshot_id:      Identifies the scored snapshot.
        age_score:        Freshness component in [0, 1].  Derived from
                          ``ingested_at`` age via :func:`apply_decay`.
                          Always 1.0 for SEMANTIC entries (age-irrelevant).
        frequency_score:  Query-frequency bonus in [0, 1].  Normalised count
                          of history queries whose topic matches at least one
                          of the snapshot's ``entry_paths``.
        combined_score:   Blended score in [0, 1]:
                          ``min(1.0, age_score + frequency_score * frequency_weight)``.
                          Frequently-queried entries can partially recover from
                          age-based decay.
        is_stale:         ``True`` when ``combined_score`` falls below
                          ``staleness_threshold``.
    """

    snapshot_id: str
    age_score: float
    frequency_score: float
    combined_score: float
    is_stale: bool


class SnapshotDecayScorer:
    """Scores :class:`SnapshotEntry` objects using age decay and query frequency.

    Age component
    ~~~~~~~~~~~~~
    For EPISODIC entries the real ``ingested_at`` age is fed to
    :func:`apply_decay`.  For SEMANTIC entries (timeless reference content)
    ``age_seconds=0.0`` is used, so the age component is always 1.0.

    Frequency component
    ~~~~~~~~~~~~~~~~~~~
    The scorer counts how many queries in the supplied
    :class:`~shad.vault.gap_detection.QueryHistoryAnalyzer` have a topic that
    is a substring of (or is contained in) any path in
    ``SnapshotEntry.entry_paths``.  The raw count is normalised to [0, 1] by
    dividing by ``max_query_count`` (capped at 1.0).

    Combined score
    ~~~~~~~~~~~~~~
    ``combined_score = min(1.0, age_score + frequency_score * frequency_weight)``

    The frequency bonus is additive: a hot snapshot can partially rescue itself
    from age-based decay, but the bonus is capped so the combined score never
    exceeds 1.0.

    Staleness
    ~~~~~~~~~
    ``is_stale = combined_score < staleness_threshold``
    """

    #: Default staleness threshold — entries scoring below this are stale.
    DEFAULT_STALENESS_THRESHOLD: float = 0.2
    #: Default maximum query count used to normalise frequency to [0, 1].
    DEFAULT_MAX_QUERY_COUNT: int = 10
    #: Default weight applied to the frequency bonus.
    DEFAULT_FREQUENCY_WEIGHT: float = 0.3

    def __init__(
        self,
        config: DecayConfig | None,
        query_analyzer: QueryHistoryAnalyzer,
        staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD,
        frequency_weight: float = DEFAULT_FREQUENCY_WEIGHT,
        max_query_count: int = DEFAULT_MAX_QUERY_COUNT,
    ) -> None:
        """
        Args:
            config: Decay parameters used for all entries.  Pass ``None`` to
                    derive a per-entry :class:`DecayConfig` from each entry's
                    ``memory_type`` via :meth:`DecayConfig.default_for`.
        """
        if not 0.0 <= staleness_threshold <= 1.0:
            raise ValueError(
                f"staleness_threshold must be in [0, 1], got {staleness_threshold}"
            )
        if not 0.0 <= frequency_weight <= 1.0:
            raise ValueError(
                f"frequency_weight must be in [0, 1], got {frequency_weight}"
            )
        if max_query_count <= 0:
            raise ValueError(
                f"max_query_count must be > 0, got {max_query_count}"
            )
        self.config = config
        self.query_analyzer = query_analyzer
        self.staleness_threshold = staleness_threshold
        self.frequency_weight = frequency_weight
        self.max_query_count = max_query_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        entry: SnapshotEntry,
        now: datetime | None = None,
    ) -> DecayScore:
        """Compute a :class:`DecayScore` for *entry*.

        Args:
            entry: The snapshot to score.
            now:   Reference timestamp for age calculation.  Defaults to
                   ``datetime.now(UTC)`` when ``None``.

        Returns:
            A :class:`DecayScore` with all components populated.
        """
        # Lazy import avoids circular dependency at module load time.
        from shad.vault.shadow_index import MemoryType

        reference = now if now is not None else datetime.now(UTC)

        # Resolve the effective config: per-entry type default when none given.
        effective_config = (
            self.config
            if self.config is not None
            else DecayConfig.default_for(entry.memory_type)
        )

        # Age component — SEMANTIC is timeless, always 1.0.
        if entry.memory_type is MemoryType.SEMANTIC:
            age_score = apply_decay(1.0, age_seconds=0.0, config=effective_config)
        else:
            age_seconds = (reference - entry.ingested_at).total_seconds()
            age_score = apply_decay(1.0, age_seconds=age_seconds, config=effective_config)

        # Frequency component — queries whose topic appears in any entry path.
        raw_count = self._match_query_count(entry.entry_paths)
        frequency_score = min(1.0, raw_count / self.max_query_count)

        combined = min(1.0, age_score + frequency_score * self.frequency_weight)
        return DecayScore(
            snapshot_id=entry.snapshot_id,
            age_score=age_score,
            frequency_score=frequency_score,
            combined_score=combined,
            is_stale=combined < self.staleness_threshold,
        )

    def score_all(
        self,
        entries: list[SnapshotEntry],
        now: datetime | None = None,
    ) -> list[DecayScore]:
        """Score all entries in *entries*, preserving order.

        Args:
            entries: Snapshots to score.
            now:     Reference timestamp forwarded to :meth:`score`.

        Returns:
            Parallel list of :class:`DecayScore` objects.
        """
        reference = now if now is not None else datetime.now(UTC)
        return [self.score(entry, now=reference) for entry in entries]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _match_query_count(self, entry_paths: list[str]) -> int:
        """Return total query hits whose topic overlaps any of *entry_paths*.

        Matching rule (case-insensitive substring):
        - topic is a substring of some path, OR
        - some path segment is a substring of topic.
        """
        freq = self.query_analyzer.get_query_frequency()
        if not freq or not entry_paths:
            return 0

        paths_lower = [p.lower() for p in entry_paths]
        total = 0
        for topic, count in freq.items():
            topic_lower = topic.lower()
            for path in paths_lower:
                if topic_lower in path or path in topic_lower:
                    total += count
                    break  # count each query topic at most once per entry
        return total
