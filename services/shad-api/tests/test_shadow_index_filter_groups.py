"""Unit tests for filter_groups_with_content_diff.

Contract under test
-------------------
``filter_groups_with_content_diff(records)`` must:

1. **Empty input** — return an empty list when given no records.
2. **No conflicts** — return an empty list when every event_time bucket has
   at most one distinct content value (identical duplicates are not conflicts).
3. **Single conflict group** — return exactly one group when records for one
   event_time carry two or more distinct content values.
4. **Multiple groups** — return one entry per conflicting event_time bucket,
   in ascending event_time order.
5. **Tie-breaking on ingested_at** — within a returned conflict group every
   record's ``ingested_at`` is preserved, so callers can resolve the conflict
   by selecting ``max(group, key=lambda r: r.ingested_at)``.

Sources / prior art:
  - teranode/stores/utxo/process-conflicting.go — bi-temporal conflict
    detection; same "group by event slot, pick latest system-time" pattern.
  - openclaw/src/memory/temporal-decay.test.ts (openclaw@2026-02-22) — tests
    for a parallel TypeScript implementation of the same conflict model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.shadow_index import BiTemporalRecord, filter_groups_with_content_diff


# ---------------------------------------------------------------------------
# Shared timestamps
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)


def rec(event_time: datetime, ingested_at: datetime, content: str) -> BiTemporalRecord:
    """Shorthand constructor for BiTemporalRecord."""
    return BiTemporalRecord(event_time=event_time, ingested_at=ingested_at, content=content)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_list_returns_empty_list(self) -> None:
        assert filter_groups_with_content_diff([]) == []

    def test_return_type_is_list(self) -> None:
        result = filter_groups_with_content_diff([])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# No conflicts
# ---------------------------------------------------------------------------

class TestNoConflicts:
    def test_single_record_no_conflict(self) -> None:
        assert filter_groups_with_content_diff([rec(T0, T0, "hello")]) == []

    def test_two_records_same_event_time_identical_content(self) -> None:
        """Duplicate content at the same event_time is NOT a conflict."""
        records = [
            rec(T0, T0, "identical"),
            rec(T0, T0 + timedelta(hours=1), "identical"),
        ]
        assert filter_groups_with_content_diff(records) == []

    def test_exact_duplicate_records_not_flagged_as_conflict(self) -> None:
        """Two records with the same event_time, ingested_at, AND content are NOT a conflict.

        A perfect duplicate (same on all three fields) represents idempotent
        re-ingestion, not a genuine bi-temporal divergence.  The content set
        for the bucket has size 1, so no conflict group is emitted.
        """
        records = [
            rec(T0, T0, "identical"),
            rec(T0, T0, "identical"),
        ]
        assert filter_groups_with_content_diff(records) == []

    def test_records_at_different_event_times_no_shared_bucket(self) -> None:
        """Each event_time has exactly one record — nothing can conflict."""
        records = [rec(T0, T0, "alpha"), rec(T1, T1, "beta"), rec(T2, T2, "gamma")]
        assert filter_groups_with_content_diff(records) == []

    def test_many_duplicates_across_one_event_time_no_conflict(self) -> None:
        records = [rec(T0, T0 + timedelta(hours=i), "same") for i in range(5)]
        assert filter_groups_with_content_diff(records) == []

    def test_mixed_event_times_all_unique_content_per_bucket(self) -> None:
        """T0 has two records with identical content; T1 has one — no group conflicts."""
        records = [
            rec(T0, T0, "x"),
            rec(T0, T0 + timedelta(hours=1), "x"),  # duplicate — not a conflict
            rec(T1, T1, "y"),
        ]
        assert filter_groups_with_content_diff(records) == []

    def test_different_event_times_never_flagged_regardless_of_content_similarity(
        self,
    ) -> None:
        """Records at different event_time values are NEVER a conflict, no matter
        how similar (or identical) their content is.

        Conflict detection is strictly intra-bucket (same event_time).  Two
        records that share the same content but differ in event_time represent
        two *separate observations* — each bucket has at most one distinct
        content value, so no conflict group may be emitted.

        Scenarios covered:
          1. Completely identical content strings across all three buckets.
          2. Near-identical content (one character changed) across two buckets.
          3. Prefix relationship — one record's content is a prefix of another's.
        """
        # 1. Identical content across three distinct event_time buckets.
        identical = [
            rec(T0, T0, "same content"),
            rec(T1, T1, "same content"),
            rec(T2, T2, "same content"),
        ]
        assert filter_groups_with_content_diff(identical) == []

        # 2. Near-identical content (differ only by trailing character).
        near_identical = [
            rec(T0, T0, "content-v1"),
            rec(T1, T1, "content-v2"),
        ]
        assert filter_groups_with_content_diff(near_identical) == []

        # 3. Prefix relationship between contents.
        prefix_relationship = [
            rec(T0, T0, "hello"),
            rec(T1, T1, "hello world"),
        ]
        assert filter_groups_with_content_diff(prefix_relationship) == []


# ---------------------------------------------------------------------------
# Single conflict group
# ---------------------------------------------------------------------------

class TestSingleConflictGroup:
    def test_two_records_same_event_time_different_content(self) -> None:
        records = [
            rec(T0, T0, "version-a"),
            rec(T0, T0 + timedelta(hours=1), "version-b"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 1

    def test_same_event_time_different_content_flagged_as_conflict(self) -> None:
        """Two records sharing event_time but with different content are a conflict.

        Both records must appear together in a single returned group — this is
        the core invariant of bi-temporal conflict detection.
        """
        r1 = rec(T0, T0, "content-alpha")
        r2 = rec(T0, T0 + timedelta(seconds=1), "content-beta")
        groups = filter_groups_with_content_diff([r1, r2])
        assert len(groups) == 1, "exactly one conflict group expected"
        assert r1 in groups[0], "first record must be in the conflict group"
        assert r2 in groups[0], "second record must be in the conflict group"

    def test_conflict_group_contains_all_conflicting_records(self) -> None:
        r1 = rec(T0, T0, "v1")
        r2 = rec(T0, T0 + timedelta(hours=2), "v2")
        [group] = filter_groups_with_content_diff([r1, r2])
        assert r1 in group
        assert r2 in group

    def test_group_size_equals_number_of_input_records_for_that_bucket(self) -> None:
        """All three records (two contents: alpha ×2, beta ×1) appear in the group."""
        records = [
            rec(T0, T0, "alpha"),
            rec(T0, T0 + timedelta(hours=1), "alpha"),  # duplicate content
            rec(T0, T0 + timedelta(hours=2), "beta"),   # genuine divergence
        ]
        [group] = filter_groups_with_content_diff(records)
        assert len(group) == 3

    def test_conflict_identified_only_for_conflicting_event_time(self) -> None:
        """T0 conflicts; T1 does not — only one group is returned."""
        records = [
            rec(T0, T0, "a"),
            rec(T0, T0 + timedelta(hours=1), "b"),
            rec(T1, T1, "c"),  # T1 has a single record — no conflict
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 1
        assert all(r.event_time == T0 for r in groups[0])

    def test_non_conflicting_event_times_excluded_from_output(self) -> None:
        """T2 (no conflict) must not appear in any returned group."""
        records = [
            rec(T0, T0, "x"),
            rec(T0, T0 + timedelta(hours=1), "y"),
            rec(T2, T2, "z"),
        ]
        [group] = filter_groups_with_content_diff(records)
        assert all(r.event_time != T2 for r in group)


# ---------------------------------------------------------------------------
# Multiple conflict groups
# ---------------------------------------------------------------------------

class TestMultipleConflictGroups:
    def test_two_independent_conflict_groups(self) -> None:
        records = [
            rec(T0, T0, "a"),
            rec(T0, T0 + timedelta(hours=1), "b"),
            rec(T1, T1, "x"),
            rec(T1, T1 + timedelta(hours=1), "y"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 2

    def test_groups_returned_in_ascending_event_time_order(self) -> None:
        """Groups must be ordered by event_time ascending regardless of insertion order."""
        records = [
            rec(T2, T2, "late-a"),
            rec(T0, T0, "early-a"),
            rec(T2, T2 + timedelta(hours=1), "late-b"),
            rec(T0, T0 + timedelta(hours=1), "early-b"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 2
        assert all(r.event_time == T0 for r in groups[0])
        assert all(r.event_time == T2 for r in groups[1])

    def test_non_conflicting_bucket_excluded_from_multiple_groups(self) -> None:
        """T1 has identical content — must not appear in output alongside T0 and T2 conflicts."""
        records = [
            rec(T0, T0, "a"),
            rec(T0, T0 + timedelta(hours=1), "b"),
            rec(T1, T1, "same"),
            rec(T1, T1 + timedelta(hours=1), "same"),  # no conflict
            rec(T2, T2, "p"),
            rec(T2, T2 + timedelta(hours=1), "q"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 2
        event_times = {r.event_time for group in groups for r in group}
        assert T1 not in event_times

    def test_three_conflict_groups_ordering(self) -> None:
        records = [
            rec(T2, T2, "c1"), rec(T2, T2 + timedelta(hours=1), "c2"),
            rec(T0, T0, "a1"), rec(T0, T0 + timedelta(hours=1), "a2"),
            rec(T1, T1, "b1"), rec(T1, T1 + timedelta(hours=1), "b2"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 3
        ordered_times = [next(r.event_time for r in g) for g in groups]
        assert ordered_times == sorted(ordered_times)

    def test_each_group_contains_only_records_for_its_event_time(self) -> None:
        records = [
            rec(T0, T0, "a"), rec(T0, T0 + timedelta(hours=1), "b"),
            rec(T1, T1, "x"), rec(T1, T1 + timedelta(hours=1), "y"),
        ]
        groups = filter_groups_with_content_diff(records)
        for group in groups:
            event_times_in_group = {r.event_time for r in group}
            assert len(event_times_in_group) == 1, "each group must hold exactly one event_time"


# ---------------------------------------------------------------------------
# Tie-breaking on ingested_at
# ---------------------------------------------------------------------------

class TestTieBreakingOnIngestedAt:
    """Verify that ingested_at values survive into each conflict group so
    callers can resolve conflicts via max(group, key=lambda r: r.ingested_at).
    """

    def test_ingested_at_values_preserved_in_conflict_group(self) -> None:
        t_early = T0
        t_late = T0 + timedelta(hours=8)
        r1 = rec(T0, t_early, "old-content")
        r2 = rec(T0, t_late, "new-content")
        [group] = filter_groups_with_content_diff([r1, r2])
        ingested_times = {r.ingested_at for r in group}
        assert t_early in ingested_times
        assert t_late in ingested_times

    def test_latest_ingested_at_is_resolvable_tiebreak(self) -> None:
        """Caller picks max(group, key=r.ingested_at) to resolve the conflict."""
        r_stale = rec(T0, T0, "stale")
        r_fresh = rec(T0, T0 + timedelta(days=1), "fresh")
        [group] = filter_groups_with_content_diff([r_stale, r_fresh])
        winner = max(group, key=lambda r: r.ingested_at)
        assert winner.content == "fresh"
        assert winner.ingested_at == T0 + timedelta(days=1)

    def test_three_way_conflict_latest_ingested_wins(self) -> None:
        records = [
            rec(T0, T0, "v1"),
            rec(T0, T0 + timedelta(hours=2), "v2"),
            rec(T0, T0 + timedelta(hours=5), "v3"),
        ]
        [group] = filter_groups_with_content_diff(records)
        winner = max(group, key=lambda r: r.ingested_at)
        assert winner.content == "v3"

    def test_identical_content_different_ingested_at_not_a_conflict(self) -> None:
        """Duplicate content with differing ingested_at must NOT form a conflict group.

        Two observations of the same content at the same event_time represent
        idempotent re-ingestion, not a genuine bi-temporal conflict.
        """
        records = [
            rec(T0, T0, "stable"),
            rec(T0, T0 + timedelta(days=7), "stable"),
        ]
        assert filter_groups_with_content_diff(records) == []

    def test_multiple_groups_each_resolvable_by_latest_ingested_at(self) -> None:
        """Verify independent tie-breaking across two separate conflict groups."""
        records = [
            # T0 conflict
            rec(T0, T0, "t0-old"),
            rec(T0, T0 + timedelta(hours=3), "t0-new"),
            # T1 conflict
            rec(T1, T1, "t1-old"),
            rec(T1, T1 + timedelta(hours=6), "t1-new"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 2

        t0_group = next(g for g in groups if g[0].event_time == T0)
        t1_group = next(g for g in groups if g[0].event_time == T1)

        assert max(t0_group, key=lambda r: r.ingested_at).content == "t0-new"
        assert max(t1_group, key=lambda r: r.ingested_at).content == "t1-new"


# ---------------------------------------------------------------------------
# Mixed content at same event_time (some matching, some differing)
# ---------------------------------------------------------------------------

class TestMixedContentAtSameEventTime:
    """Edge cases where a single event_time bucket contains a mix of records:
    some share identical content and others diverge.

    Contract:
    - A bucket is flagged as a conflict as soon as any two records carry
      different content (regardless of how many duplicates exist).
    - The returned group contains ALL records for that event_time, not just
      the distinct-content representatives.
    """

    def test_many_identical_plus_one_different_is_a_conflict(self) -> None:
        """Five records with content "same" and one outlier must form a group."""
        records = [
            rec(T0, T0 + timedelta(hours=i), "same") for i in range(5)
        ] + [
            rec(T0, T0 + timedelta(hours=5), "different"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 1

    def test_all_records_returned_including_duplicate_content_ones(self) -> None:
        """The group must include the duplicates, not just the unique records."""
        records = [
            rec(T0, T0, "same"),
            rec(T0, T0 + timedelta(hours=1), "same"),  # duplicate of above
            rec(T0, T0 + timedelta(hours=2), "same"),  # triplicate
            rec(T0, T0 + timedelta(hours=3), "other"),
        ]
        [group] = filter_groups_with_content_diff(records)
        assert len(group) == 4, "all 4 records must appear in the group"

    def test_two_content_clusters_of_equal_size(self) -> None:
        """Two records with "a" and two with "b" at the same event_time — one group, four records."""
        records = [
            rec(T0, T0, "a"),
            rec(T0, T0 + timedelta(hours=1), "a"),
            rec(T0, T0 + timedelta(hours=2), "b"),
            rec(T0, T0 + timedelta(hours=3), "b"),
        ]
        [group] = filter_groups_with_content_diff(records)
        assert len(group) == 4
        assert {r.content for r in group} == {"a", "b"}

    def test_mixed_bucket_flagged_while_all_same_bucket_excluded(self) -> None:
        """T0 has matching + differing records; T1 is all identical — only T0 returned."""
        records = [
            # T0: mostly "base", one outlier
            rec(T0, T0, "base"),
            rec(T0, T0 + timedelta(hours=1), "base"),
            rec(T0, T0 + timedelta(hours=2), "outlier"),
            # T1: all identical
            rec(T1, T1, "stable"),
            rec(T1, T1 + timedelta(hours=1), "stable"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 1
        assert all(r.event_time == T0 for r in groups[0])

    def test_mixed_bucket_group_contains_all_t0_records(self) -> None:
        """Every T0 record (duplicates included) must be present in the returned group."""
        r1 = rec(T0, T0, "base")
        r2 = rec(T0, T0 + timedelta(hours=1), "base")
        r3 = rec(T0, T0 + timedelta(hours=2), "outlier")
        records = [r1, r2, r3, rec(T1, T1, "stable"), rec(T1, T1 + timedelta(hours=1), "stable")]
        [group] = filter_groups_with_content_diff(records)
        assert r1 in group
        assert r2 in group
        assert r3 in group

    def test_multiple_mixed_buckets_each_returned(self) -> None:
        """T0 and T2 each have matching+differing content; T1 is uniform — two groups."""
        records = [
            rec(T0, T0, "a"), rec(T0, T0 + timedelta(hours=1), "a"),
            rec(T0, T0 + timedelta(hours=2), "a-prime"),
            rec(T1, T1, "x"), rec(T1, T1 + timedelta(hours=1), "x"),  # no conflict
            rec(T2, T2, "b"), rec(T2, T2 + timedelta(hours=1), "b"),
            rec(T2, T2 + timedelta(hours=2), "b-prime"),
        ]
        groups = filter_groups_with_content_diff(records)
        assert len(groups) == 2
        group_times = {next(iter({r.event_time for r in g})) for g in groups}
        assert T0 in group_times
        assert T2 in group_times
        assert T1 not in group_times

    def test_single_outlier_is_resolvable_as_latest_ingested(self) -> None:
        """The outlier record (latest ingested_at) wins tie-breaking."""
        records = [
            rec(T0, T0, "original"),
            rec(T0, T0 + timedelta(hours=1), "original"),
            rec(T0, T0 + timedelta(hours=10), "corrected"),
        ]
        [group] = filter_groups_with_content_diff(records)
        winner = max(group, key=lambda r: r.ingested_at)
        assert winner.content == "corrected"

    def test_large_duplicate_cluster_plus_one_different_group_size(self) -> None:
        """100 identical records plus 1 different: group size must be 101."""
        n = 100
        records = [rec(T0, T0 + timedelta(minutes=i), "bulk") for i in range(n)]
        records.append(rec(T0, T0 + timedelta(minutes=n), "singleton"))
        [group] = filter_groups_with_content_diff(records)
        assert len(group) == n + 1
