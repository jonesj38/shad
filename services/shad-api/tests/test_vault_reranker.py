"""Tests for the temporal-decay reranker.

Contract under test
-------------------
``rerank(results, ages, config)`` must:

1. **Preserve original scores** — every returned result carries
   ``metadata["original_score"]`` equal to the score of the corresponding
   input result.

2. **Sort by decay-adjusted score** — the returned list is ordered
   highest-adjusted-score first; the raw order of the input is irrelevant.

3. **Not mutate inputs** — the original ``RetrievalResult`` objects and their
   metadata dicts are left unchanged.

4. **Overwrite stale original_score metadata** — if an input result already
   has ``metadata["original_score"]`` from a prior call, the new value
   replaces it so it always reflects the score at the time of *this* call.

Sources / prior art:
  - openclaw/src/memory/temporal-decay.ts (openclaw@2026-02-22) — same
    half-life contract; stores raw score alongside adjusted score.
  - teranode/connmgr/dynamicbanscore-go  — exponential Halflife = 60 s
    reference implementation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from shad.retrieval.layer import RetrievalResult
from shad.vault.decay import DecayConfig, DecayCurve
from shad.vault.reranker import rerank, rerank_with_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(
    score: float,
    path: str = "doc.md",
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        path=path,
        content="",
        score=score,
        metadata=dict(metadata) if metadata else {},
    )


EXP_CFG = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.EXPONENTIAL)
LIN_CFG = DecayConfig(halflife_seconds=3600.0, curve=DecayCurve.LINEAR)


# ---------------------------------------------------------------------------
# Preserving original scores in metadata
# ---------------------------------------------------------------------------

class TestOriginalScorePreservedInMetadata:
    def test_single_result_original_score_stored(self) -> None:
        r = make_result(score=0.9, path="a.md")
        [out] = rerank([r], ages=[0.0], config=EXP_CFG)
        assert out.metadata["original_score"] == pytest.approx(0.9)

    def test_multiple_results_each_preserve_own_score(self) -> None:
        scores = [0.9, 0.6, 0.3]
        results = [make_result(s, path=f"{i}.md") for i, s in enumerate(scores)]
        ages = [0.0, 0.0, 0.0]

        ranked = rerank(results, ages=ages, config=EXP_CFG)

        # Collect by path to match inputs regardless of output order
        by_path = {r.path: r for r in ranked}
        for i, expected in enumerate(scores):
            assert by_path[f"{i}.md"].metadata["original_score"] == pytest.approx(expected)

    def test_original_score_equals_input_score_not_adjusted(self) -> None:
        """The stored original_score must be the raw score, not the decayed one."""
        r = make_result(score=0.8, path="b.md")
        age = 3600.0  # one halflife → score halved
        [out] = rerank([r], ages=[age], config=EXP_CFG)

        assert out.metadata["original_score"] == pytest.approx(0.8)
        assert out.score == pytest.approx(0.4)  # decay applied to live score
        assert out.metadata["original_score"] != pytest.approx(out.score)

    def test_original_score_preserved_across_both_curve_types(self) -> None:
        r = make_result(score=0.7, path="c.md")
        age = 3600.0

        [exp_out] = rerank([r], ages=[age], config=EXP_CFG)
        [lin_out] = rerank([r], ages=[age], config=LIN_CFG)

        assert exp_out.metadata["original_score"] == pytest.approx(0.7)
        assert lin_out.metadata["original_score"] == pytest.approx(0.7)

    def test_stale_original_score_in_metadata_is_overwritten(self) -> None:
        """If a result already has metadata['original_score'], it must be replaced."""
        r = make_result(score=0.5, path="d.md", metadata={"original_score": 0.99})
        [out] = rerank([r], ages=[0.0], config=EXP_CFG)
        assert out.metadata["original_score"] == pytest.approx(0.5)

    def test_other_metadata_keys_are_preserved(self) -> None:
        r = make_result(score=0.4, path="e.md", metadata={"tags": ["foo", "bar"], "lang": "en"})
        [out] = rerank([r], ages=[0.0], config=EXP_CFG)
        assert out.metadata["tags"] == ["foo", "bar"]
        assert out.metadata["lang"] == "en"
        assert "original_score" in out.metadata


# ---------------------------------------------------------------------------
# Sorting by decay-adjusted score
# ---------------------------------------------------------------------------

class TestSortedByDecayAdjustedScore:
    def test_sorted_descending_all_same_age(self) -> None:
        results = [
            make_result(score=0.3, path="low.md"),
            make_result(score=0.9, path="high.md"),
            make_result(score=0.6, path="mid.md"),
        ]
        ranked = rerank(results, ages=[60.0, 60.0, 60.0], config=EXP_CFG)
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_decay_adjusted_not_original(self) -> None:
        """Live scores in output must equal apply_decay(original, age)."""
        import math
        r = make_result(score=1.0, path="x.md")
        age = 7200.0  # two halflives → 0.25
        [out] = rerank([r], ages=[age], config=EXP_CFG)
        expected = math.pow(0.5, age / EXP_CFG.halflife_seconds)
        assert out.score == pytest.approx(expected)

    def test_older_result_ranked_below_fresher_same_original_score(self) -> None:
        fresh = make_result(score=0.8, path="fresh.md")
        stale = make_result(score=0.8, path="stale.md")
        ranked = rerank([stale, fresh], ages=[7200.0, 0.0], config=EXP_CFG)
        assert ranked[0].path == "fresh.md"
        assert ranked[1].path == "stale.md"

    def test_fresher_lower_original_score_can_beat_stale_high_score(self) -> None:
        """A fresh doc with score 0.5 can beat a stale doc with score 0.9."""
        fresh = make_result(score=0.5, path="fresh.md")
        # stale doc at 4 halflives → 0.9 * 0.5^4 = 0.9 * 0.0625 = 0.05625
        stale = make_result(score=0.9, path="stale.md")
        ranked = rerank([stale, fresh], ages=[14400.0, 0.0], config=EXP_CFG)
        assert ranked[0].path == "fresh.md"
        assert ranked[0].metadata["original_score"] == pytest.approx(0.5)

    def test_output_order_independent_of_input_order(self) -> None:
        """Same results in different input order must produce the same ranked list."""
        a = make_result(score=0.9, path="a.md")
        b = make_result(score=0.6, path="b.md")
        c = make_result(score=0.3, path="c.md")
        ages = [0.0, 0.0, 0.0]

        ranked_abc = rerank([a, b, c], ages=ages, config=EXP_CFG)
        ranked_cba = rerank([c, b, a], ages=list(reversed(ages)), config=EXP_CFG)

        assert [r.path for r in ranked_abc] == [r.path for r in ranked_cba]

    def test_linear_curve_results_also_sorted_correctly(self) -> None:
        results = [
            make_result(score=0.5, path="fresh.md"),
            make_result(score=0.9, path="stale.md"),
        ]
        # At t = 2 * halflife the linear score is zero, so fresh wins despite lower score
        ranked = rerank(results, ages=[0.0, 2 * LIN_CFG.halflife_seconds], config=LIN_CFG)
        assert ranked[0].path == "fresh.md"
        assert ranked[1].score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Input not mutated
# ---------------------------------------------------------------------------

class TestInputNotMutated:
    def test_original_result_score_unchanged(self) -> None:
        r = make_result(score=0.75, path="orig.md")
        rerank([r], ages=[3600.0], config=EXP_CFG)
        assert r.score == pytest.approx(0.75)

    def test_original_result_metadata_unchanged(self) -> None:
        r = make_result(score=0.5, path="orig.md", metadata={"key": "val"})
        rerank([r], ages=[3600.0], config=EXP_CFG)
        assert "original_score" not in r.metadata
        assert r.metadata == {"key": "val"}

    def test_input_list_not_mutated(self) -> None:
        results = [make_result(0.5, path="p.md")]
        original_id = id(results[0])
        rerank(results, ages=[0.0], config=EXP_CFG)
        assert id(results[0]) == original_id


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input_returns_empty_list(self) -> None:
        assert rerank([], ages=[], config=EXP_CFG) == []

    def test_mismatched_lengths_raise_value_error(self) -> None:
        results = [make_result(0.5, path="x.md")]
        with pytest.raises(ValueError, match="same length"):
            rerank(results, ages=[], config=EXP_CFG)

    def test_mismatched_lengths_too_many_ages(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            rerank([], ages=[1.0, 2.0], config=EXP_CFG)

    def test_negative_age_treated_as_zero_no_decay(self) -> None:
        r = make_result(score=0.8, path="f.md")
        [out] = rerank([r], ages=[-100.0], config=EXP_CFG)
        assert out.score == pytest.approx(0.8)
        assert out.metadata["original_score"] == pytest.approx(0.8)

    def test_zero_score_stays_zero(self) -> None:
        r = make_result(score=0.0, path="z.md")
        [out] = rerank([r], ages=[3600.0], config=EXP_CFG)
        assert out.score == pytest.approx(0.0)
        assert out.metadata["original_score"] == pytest.approx(0.0)

    def test_age_zero_score_unchanged(self) -> None:
        r = make_result(score=0.65, path="now.md")
        [out] = rerank([r], ages=[0.0], config=EXP_CFG)
        assert out.score == pytest.approx(0.65)
        assert out.metadata["original_score"] == pytest.approx(0.65)

    @pytest.mark.parametrize("n", [1, 2, 5, 20])
    def test_output_length_equals_input_length(self, n: int) -> None:
        results = [make_result(float(i) / n, path=f"{i}.md") for i in range(n)]
        ages = [float(i * 100) for i in range(n)]
        assert len(rerank(results, ages=ages, config=EXP_CFG)) == n

    def test_all_zero_ages_order_by_original_score(self) -> None:
        results = [
            make_result(score=0.2, path="low.md"),
            make_result(score=0.8, path="high.md"),
            make_result(score=0.5, path="mid.md"),
        ]
        ranked = rerank(results, ages=[0.0, 0.0, 0.0], config=EXP_CFG)
        assert [r.path for r in ranked] == ["high.md", "mid.md", "low.md"]


# ---------------------------------------------------------------------------
# Mixed episodic (decayed) vs semantic (undecayed) ranking
#
# Semantic results model queries whose relevance does not decay with time
# (vector similarity is timeless).  Episodic results are retrieval hits from
# time-stamped observations and must be penalised by age.  The caller signals
# this distinction purely through the age value: semantic results receive
# age=0 (no penalty) while episodic results receive their true elapsed age.
#
# Key contract: a semantic entry at 0.8 must outrank an episodic entry whose
# base score is 0.9 once the episodic entry is old enough that decay drives it
# below 0.8.  Reference: openclaw/src/memory/temporal-decay.ts — same half-
# life contract used for mixed-source re-ranking (openclaw@2026-02-22).
# ---------------------------------------------------------------------------

class TestMixedEpisodicSemanticRanking:
    """Verify merged ranking when episodic results are decayed and semantic are not."""

    def test_semantic_0_8_beats_episodic_0_9_after_two_halflives(self) -> None:
        """Core scenario: semantic 0.8 (age=0) must rank above episodic 0.9 (age=2 HL).

        Exponential decay at 2 halflives: 0.9 * 0.5^2 = 0.225 < 0.8.
        """
        episodic = make_result(score=0.9, path="episodic.md", metadata={"source_type": "episodic"})
        semantic = make_result(score=0.8, path="semantic.md", metadata={"source_type": "semantic"})

        two_halflives = 2.0 * EXP_CFG.halflife_seconds  # 7200 s
        ranked = rerank([episodic, semantic], ages=[two_halflives, 0.0], config=EXP_CFG)

        assert ranked[0].path == "semantic.md"
        assert ranked[1].path == "episodic.md"
        assert ranked[0].score == pytest.approx(0.8)
        assert ranked[1].score == pytest.approx(0.9 * math.pow(0.5, 2))

    def test_episodic_original_score_preserved_after_losing_to_semantic(self) -> None:
        """The episodic result's metadata must still carry its original score."""
        episodic = make_result(score=0.9, path="episodic.md")
        semantic = make_result(score=0.8, path="semantic.md")

        ranked = rerank([episodic, semantic], ages=[2.0 * EXP_CFG.halflife_seconds, 0.0], config=EXP_CFG)

        by_path = {r.path: r for r in ranked}
        assert by_path["episodic.md"].metadata["original_score"] == pytest.approx(0.9)
        assert by_path["semantic.md"].metadata["original_score"] == pytest.approx(0.8)

    def test_fresh_episodic_still_beats_lower_semantic(self) -> None:
        """A brand-new episodic result (age≈0) keeps its score advantage over semantic."""
        episodic = make_result(score=0.9, path="episodic.md")
        semantic = make_result(score=0.8, path="semantic.md")

        ranked = rerank([episodic, semantic], ages=[0.0, 0.0], config=EXP_CFG)

        assert ranked[0].path == "episodic.md"
        assert ranked[1].path == "semantic.md"

    def test_crossover_point_episodic_just_below_semantic(self) -> None:
        """At the exact age where decayed episodic < semantic, semantic must lead.

        Crossover: 0.9 * 0.5^(age/HL) == 0.8  →  age = HL * log2(0.9/0.8).
        One second past crossover the episodic entry should fall behind.
        """
        crossover_age = EXP_CFG.halflife_seconds * math.log2(0.9 / 0.8)
        age_just_past = crossover_age + 1.0  # nudge past crossover

        episodic = make_result(score=0.9, path="episodic.md")
        semantic = make_result(score=0.8, path="semantic.md")

        ranked = rerank([episodic, semantic], ages=[age_just_past, 0.0], config=EXP_CFG)

        assert ranked[0].path == "semantic.md"
        assert ranked[0].score > ranked[1].score

    def test_none_age_preserves_original_score(self) -> None:
        """None age must not apply decay — score passes through unchanged."""
        r = make_result(score=0.7, path="no-ts.md")
        [out] = rerank([r], ages=[None], config=EXP_CFG)
        assert out.score == pytest.approx(0.7)
        assert out.metadata["original_score"] == pytest.approx(0.7)

    def test_none_age_emits_debug_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A debug-level warning must be logged when age is None."""
        import logging
        r = make_result(score=0.6, path="missing-ts.md")
        with caplog.at_level(logging.DEBUG, logger="shad.vault.reranker"):
            rerank([r], ages=[None], config=EXP_CFG)
        assert any("missing-ts.md" in m for m in caplog.messages)
        assert any("skipping decay" in m.lower() for m in caplog.messages)

    def test_mixed_none_and_real_ages_ranked_correctly(self) -> None:
        """None-aged result keeps full score; decayed result may rank below it."""
        no_ts = make_result(score=0.6, path="no-ts.md")
        stale = make_result(score=0.9, path="stale.md")
        # stale at 2 halflives → 0.9 * 0.25 = 0.225
        ranked = rerank([stale, no_ts], ages=[2 * EXP_CFG.halflife_seconds, None], config=EXP_CFG)
        assert ranked[0].path == "no-ts.md"
        assert ranked[0].score == pytest.approx(0.6)
        assert ranked[1].score == pytest.approx(0.9 * math.pow(0.5, 2))

    def test_three_mixed_results_ranked_correctly(self) -> None:
        """Blend of two episodic entries at different ages and one semantic entry.

        Expected order (exponential, halflife=3600 s):
          semantic  0.8  age=0     → 0.8000
          episodic1 0.95 age=1 HL  → 0.475
          episodic2 0.9  age=2 HL  → 0.225
        """
        ep1 = make_result(score=0.95, path="ep1.md", metadata={"source_type": "episodic"})
        ep2 = make_result(score=0.90, path="ep2.md", metadata={"source_type": "episodic"})
        sem = make_result(score=0.80, path="sem.md", metadata={"source_type": "semantic"})

        hl = EXP_CFG.halflife_seconds
        ranked = rerank([ep1, ep2, sem], ages=[hl, 2 * hl, 0.0], config=EXP_CFG)

        assert [r.path for r in ranked] == ["sem.md", "ep1.md", "ep2.md"]
        assert ranked[0].score == pytest.approx(0.80)
        assert ranked[1].score == pytest.approx(0.95 * 0.5)
        assert ranked[2].score == pytest.approx(0.90 * 0.25)


# ---------------------------------------------------------------------------
# Newer memories rank higher than older memories with equal relevance scores
#
# When two memories share an identical raw relevance score, temporal decay must
# tip the ranking in favour of the fresher one.  These tests use
# ``rerank_with_metadata`` so the age is derived from the ``ingested_at`` field
# that real memory entries carry — exercising the full path from timestamp
# parsing through decay to sorted output.
# ---------------------------------------------------------------------------

# 90-day halflife — matches the default in temporal_decay_score()
_DAY_CFG = DecayConfig(
    halflife_seconds=90 * 86_400.0,
    curve=DecayCurve.EXPONENTIAL,
)

_NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)


def _memory_result(path: str, score: float, ingested_at: datetime) -> RetrievalResult:
    return RetrievalResult(
        path=path,
        content="",
        score=score,
        metadata={"ingested_at": ingested_at.isoformat()},
    )


class TestNewerMemoryRanksHigherWithEqualRelevance:
    """newer memories beat older ones when raw relevance scores are equal."""

    def test_newer_memory_ranks_first(self) -> None:
        """The newer memory must appear at index 0 after reranking."""
        newer = _memory_result("newer.md", score=0.8, ingested_at=_NOW - timedelta(days=1))
        older = _memory_result("older.md", score=0.8, ingested_at=_NOW - timedelta(days=180))

        ranked = rerank_with_metadata([older, newer], config=_DAY_CFG, now=_NOW)

        assert ranked[0].path == "newer.md"
        assert ranked[1].path == "older.md"

    def test_newer_memory_has_higher_combined_score(self) -> None:
        """The newer memory's decay-adjusted score must be strictly greater."""
        newer = _memory_result("newer.md", score=0.8, ingested_at=_NOW - timedelta(days=1))
        older = _memory_result("older.md", score=0.8, ingested_at=_NOW - timedelta(days=180))

        ranked = rerank_with_metadata([older, newer], config=_DAY_CFG, now=_NOW)

        newer_score = next(r.score for r in ranked if r.path == "newer.md")
        older_score = next(r.score for r in ranked if r.path == "older.md")
        assert newer_score > older_score

    def test_equal_scores_original_score_preserved_for_both(self) -> None:
        """Both results must carry original_score == 0.8 regardless of ranking."""
        newer = _memory_result("newer.md", score=0.8, ingested_at=_NOW - timedelta(days=1))
        older = _memory_result("older.md", score=0.8, ingested_at=_NOW - timedelta(days=180))

        ranked = rerank_with_metadata([older, newer], config=_DAY_CFG, now=_NOW)

        by_path = {r.path: r for r in ranked}
        assert by_path["newer.md"].metadata["original_score"] == pytest.approx(0.8)
        assert by_path["older.md"].metadata["original_score"] == pytest.approx(0.8)

    def test_combined_score_equals_expected_decay(self) -> None:
        """Verify the exact decay formula: score * 0.5^(age_days / halflife_days)."""
        age_days = 45.0  # half a halflife → factor = 0.5^0.5 ≈ 0.7071
        older_age_days = 180.0  # two halflives → factor = 0.25

        newer = _memory_result(
            "newer.md", score=0.8, ingested_at=_NOW - timedelta(days=age_days)
        )
        older = _memory_result(
            "older.md", score=0.8, ingested_at=_NOW - timedelta(days=older_age_days)
        )

        ranked = rerank_with_metadata([older, newer], config=_DAY_CFG, now=_NOW)

        by_path = {r.path: r for r in ranked}
        halflife_days = _DAY_CFG.halflife_seconds / 86_400.0
        assert by_path["newer.md"].score == pytest.approx(
            0.8 * math.pow(0.5, age_days / halflife_days)
        )
        assert by_path["older.md"].score == pytest.approx(
            0.8 * math.pow(0.5, older_age_days / halflife_days)
        )

    def test_three_memories_equal_relevance_ordered_by_recency(self) -> None:
        """Three memories with identical relevance scores must be ordered newest → oldest."""
        m1 = _memory_result("1day.md", score=0.75, ingested_at=_NOW - timedelta(days=1))
        m2 = _memory_result("30day.md", score=0.75, ingested_at=_NOW - timedelta(days=30))
        m3 = _memory_result("270day.md", score=0.75, ingested_at=_NOW - timedelta(days=270))

        ranked = rerank_with_metadata([m3, m1, m2], config=_DAY_CFG, now=_NOW)

        assert [r.path for r in ranked] == ["1day.md", "30day.md", "270day.md"]
        # Scores must be strictly decreasing
        assert ranked[0].score > ranked[1].score > ranked[2].score
