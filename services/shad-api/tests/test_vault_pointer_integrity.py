"""Pointer integrity tests for vault consolidation back-pointers.

Four invariants verified:

1. ``consolidated_into`` never points to a nonexistent entry.
2. No snapshot record points to itself.
3. Re-running synthesis on already-consolidated records is idempotent
   (no double-consolidation — the pointer must not change on a second pass).
4. Orphaned entries (``consolidated_into`` set but the target ID is absent
   from the snapshot table) are detectable by the audit helper.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shad.vault.consolidation import (
    ConsolidationConfig,
    MergeStrategy,
    synthesize_group,
)
from shad.vault.contracts import GroupedNotes
from shad.vault.shadow_index import (
    BiTemporalRecord,
    EpisodicRecord,
    MemoryMetadata,
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
    make_bitemporal_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_SOURCE_ID = "test-source-ptr"
_SOURCE_URL = "https://example.com/ptr-test"


def _btr(content: str, *, age_days: float = 0.0) -> BiTemporalRecord[str]:
    event_time = _BASE - timedelta(days=age_days)
    return make_bitemporal_record(event_time, content, ingestion_time=_BASE)


def _episodic(
    record_id: str,
    content: str = "event",
    *,
    age_days: float = 0.0,
    session_id: str | None = "s1",
) -> EpisodicRecord:
    return EpisodicRecord(
        record_id=record_id,
        record=_btr(content, age_days=age_days),
        metadata=MemoryMetadata(source="test", confidence=0.9),
        session_id=session_id,
    )


def _snapshot(
    snapshot_id: str,
    *,
    age_days: float = 0.0,
) -> SnapshotEntry:
    return SnapshotEntry(
        snapshot_id=snapshot_id,
        source_id=_SOURCE_ID,
        ingested_at=_BASE - timedelta(days=age_days),
        source_revision=f"rev-{snapshot_id[:8]}",
        entry_paths=[f"notes/{snapshot_id}.md"],
        content_hash=snapshot_id,
        memory_type=MemoryType.EPISODIC,
    )


def _make_group(record_ids: list[str], topic: str = "default") -> GroupedNotes:
    """Build a GroupedNotes that triggers consolidation (size == len(record_ids))."""
    now = _BASE
    return GroupedNotes(
        group_id=f"grp-{topic}",
        memory_type=MemoryType.EPISODIC,
        record_ids=tuple(record_ids),
        topic=topic,
        oldest_at=now - timedelta(days=len(record_ids)),
        newest_at=now,
    )


def _fresh_index(tmp_path: Path) -> ShadowIndex:
    """Return a ShadowIndex with a single registered source."""
    index = ShadowIndex(tmp_path / f"idx-{uuid.uuid4().hex[:8]}.sqlite")
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
        )
    )
    return index


def _all_snapshots_with_pointer(index: ShadowIndex) -> list[SnapshotEntry]:
    """Fetch every snapshot that has ``consolidated_into`` set.

    Uses ``get_snapshot`` (which populates the field) rather than
    ``list_snapshots`` (which omits it from the result row mapping).
    """
    conn = index._ensure_conn()  # noqa: SLF001
    rows = conn.execute(
        "SELECT snapshot_id FROM snapshots WHERE consolidated_into IS NOT NULL"
    ).fetchall()
    results = []
    for row in rows:
        snap = index.get_snapshot(row["snapshot_id"])
        if snap is not None:
            results.append(snap)
    return results


def _all_snapshot_ids(index: ShadowIndex) -> set[str]:
    """Return the full set of snapshot_ids present in the DB."""
    conn = index._ensure_conn()  # noqa: SLF001
    rows = conn.execute("SELECT snapshot_id FROM snapshots").fetchall()
    return {row["snapshot_id"] for row in rows}


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


def audit_pointer_integrity(
    index: ShadowIndex,
    known_semantic_ids: set[str] | None = None,
) -> list[str]:
    """Return a list of violation messages for broken ``consolidated_into`` pointers.

    Checks three things for every snapshot whose ``consolidated_into`` is set:

    1. **Self-reference**: ``consolidated_into == snapshot_id``.
    2. **Dangling pointer**: ``consolidated_into`` is not in the snapshot table
       AND not in *known_semantic_ids* (the set of newly-created semantic
       record IDs that live outside the snapshot table).
    3. **Orphan detection**: a snapshot with ``consolidated_into`` set where
       the target does not resolve to any known entity (equivalent to dangling
       pointer when *known_semantic_ids* is empty / not supplied).

    Returns an empty list when the store is consistent.
    """
    all_ids = _all_snapshot_ids(index)
    known = (known_semantic_ids or set()) | all_ids
    violations: list[str] = []

    for snap in _all_snapshots_with_pointer(index):
        target = snap.consolidated_into
        assert target is not None  # filtered above

        if target == snap.snapshot_id:
            violations.append(
                f"SELF_REF: snapshot {snap.snapshot_id!r} points to itself"
            )
        elif target not in known:
            violations.append(
                f"DANGLING: snapshot {snap.snapshot_id!r} → {target!r} (not found)"
            )

    return violations


# ---------------------------------------------------------------------------
# Invariant 1 – consolidated_into never points to a nonexistent entry
# ---------------------------------------------------------------------------


class TestNoPointersToNonexistentEntries:
    """consolidated_into values must resolve to a known entry."""

    def test_valid_pointer_has_no_violations(self, tmp_path: Path) -> None:
        """Marking A → B (where B exists) produces no violations."""
        index = _fresh_index(tmp_path)
        snap_a = _snapshot("snap-a")
        snap_b = _snapshot("snap-b")
        index.add_snapshot(snap_a)
        index.add_snapshot(snap_b)

        index.mark_snapshot_consolidated("snap-a", "snap-b")

        violations = audit_pointer_integrity(index)
        assert violations == []

    def test_pointer_to_nonexistent_snapshot_is_detected(self, tmp_path: Path) -> None:
        """Marking A → 'ghost-id' (not in DB) is a dangling pointer."""
        index = _fresh_index(tmp_path)
        snap_a = _snapshot("snap-a")
        index.add_snapshot(snap_a)

        index.mark_snapshot_consolidated("snap-a", "ghost-id")

        violations = audit_pointer_integrity(index, known_semantic_ids=None)
        assert len(violations) == 1
        assert "DANGLING" in violations[0]
        assert "ghost-id" in violations[0]

    def test_known_semantic_ids_satisfy_dangling_check(self, tmp_path: Path) -> None:
        """If the canonical ID is a known semantic record (not a snapshot),
        supplying it in *known_semantic_ids* clears the dangling violation."""
        index = _fresh_index(tmp_path)
        snap_a = _snapshot("snap-a")
        index.add_snapshot(snap_a)

        semantic_id = "sem-grp-default-abc123"
        index.mark_snapshot_consolidated("snap-a", semantic_id)

        # Without semantic_ids → dangling
        assert len(audit_pointer_integrity(index)) == 1

        # With semantic_ids → clean
        assert audit_pointer_integrity(index, known_semantic_ids={semantic_id}) == []

    def test_synthesize_group_pointer_resolves_with_returned_id(
        self, tmp_path: Path
    ) -> None:
        """synthesize_group writes consolidated_into = merged_record_id.
        Passing the returned merged_record_id as a known_semantic_id clears violations."""
        index = _fresh_index(tmp_path)
        record_ids = ["ep-1", "ep-2", "ep-3"]
        for rid in record_ids:
            index.add_snapshot(_snapshot(rid))

        records = [_episodic(rid, age_days=i) for i, rid in enumerate(record_ids)]
        group = _make_group(record_ids)

        result = synthesize_group(
            group,
            records,
            config=ConsolidationConfig(threshold=3, merge_strategy=MergeStrategy.LATEST_WINS),
            shadow_index=index,
            now=_BASE,
        )

        assert result.success
        assert result.groups_merged == 1
        merged_id = result.metrics.merge_results[0].merged_record_id

        # Raw audit (merged_id is not in snapshot table) → each source is "dangling"
        raw_violations = audit_pointer_integrity(index, known_semantic_ids=None)
        assert len(raw_violations) == len(record_ids)

        # Supplying the merged_id resolves all violations
        clean = audit_pointer_integrity(index, known_semantic_ids={merged_id})
        assert clean == []


# ---------------------------------------------------------------------------
# Invariant 2 – no record points to itself
# ---------------------------------------------------------------------------


class TestNoSelfReferences:
    """A snapshot must never have consolidated_into == its own snapshot_id."""

    def test_self_reference_is_detected(self, tmp_path: Path) -> None:
        """Explicitly writing a self-pointer is flagged."""
        index = _fresh_index(tmp_path)
        snap = _snapshot("snap-self")
        index.add_snapshot(snap)

        index.mark_snapshot_consolidated("snap-self", "snap-self")

        violations = audit_pointer_integrity(index, known_semantic_ids={"snap-self"})
        assert len(violations) == 1
        assert "SELF_REF" in violations[0]
        assert "snap-self" in violations[0]

    def test_normal_consolidation_produces_no_self_references(
        self, tmp_path: Path
    ) -> None:
        """synthesize_group must not set any record's consolidated_into to its own id."""
        index = _fresh_index(tmp_path)
        record_ids = ["ep-a", "ep-b", "ep-c"]
        for rid in record_ids:
            index.add_snapshot(_snapshot(rid))

        records = [_episodic(rid, age_days=i) for i, rid in enumerate(record_ids)]
        group = _make_group(record_ids)

        synthesize_group(
            group,
            records,
            config=ConsolidationConfig(threshold=3),
            shadow_index=index,
            now=_BASE,
        )

        for rid in record_ids:
            snap = index.get_snapshot(rid)
            assert snap is not None
            assert snap.consolidated_into != rid, (
                f"Self-reference detected on {rid!r}: consolidated_into == snapshot_id"
            )

    def test_mark_snapshot_consolidated_allows_self_reference_without_guard(
        self, tmp_path: Path
    ) -> None:
        """mark_snapshot_consolidated has no built-in guard against self-references;
        the audit layer is responsible for detection."""
        index = _fresh_index(tmp_path)
        snap = _snapshot("snap-x")
        index.add_snapshot(snap)

        # The DB write succeeds — no guard in mark_snapshot_consolidated itself.
        result = index.mark_snapshot_consolidated("snap-x", "snap-x")
        assert result is True

        # The audit helper catches it.
        violations = audit_pointer_integrity(index, known_semantic_ids={"snap-x"})
        assert any("SELF_REF" in v for v in violations)


# ---------------------------------------------------------------------------
# Invariant 3 – idempotency: re-running synthesis does not double-consolidate
# ---------------------------------------------------------------------------


class TestSynthesisIdempotency:
    """A second synthesis pass over already-consolidated records must not
    overwrite the existing consolidated_into pointer with a fresh ID."""

    def test_second_pass_does_not_change_pointer(self, tmp_path: Path) -> None:
        """Running synthesize_group twice on the same records must leave
        each snapshot's consolidated_into unchanged after the first pass."""
        index = _fresh_index(tmp_path)
        record_ids = ["ep-1", "ep-2", "ep-3"]
        for rid in record_ids:
            index.add_snapshot(_snapshot(rid))

        records = [_episodic(rid, age_days=i) for i, rid in enumerate(record_ids)]
        group = _make_group(record_ids)
        cfg = ConsolidationConfig(threshold=3, merge_strategy=MergeStrategy.LATEST_WINS)

        # First pass — establishes the pointers.
        result1 = synthesize_group(group, records, config=cfg, shadow_index=index, now=_BASE)
        assert result1.groups_merged == 1
        first_merged_id = result1.metrics.merge_results[0].merged_record_id

        # Capture state after the first pass.
        pointers_after_first = {
            rid: index.get_snapshot(rid).consolidated_into  # type: ignore[union-attr]
            for rid in record_ids
        }

        # Second pass — should be a no-op on the pointers.
        result2 = synthesize_group(group, records, config=cfg, shadow_index=index, now=_BASE)
        second_merged_id = result2.metrics.merge_results[0].merged_record_id

        for rid in record_ids:
            snap = index.get_snapshot(rid)
            assert snap is not None
            assert snap.consolidated_into == pointers_after_first[rid], (
                f"Idempotency violation on {rid!r}: pointer changed from "
                f"{pointers_after_first[rid]!r} to {snap.consolidated_into!r} "
                f"(first_merged={first_merged_id!r}, second_merged={second_merged_id!r})"
            )

    def test_already_consolidated_group_skipped_at_threshold(
        self, tmp_path: Path
    ) -> None:
        """A group that falls below threshold after its records are filtered out
        (e.g., all already-consolidated) must produce groups_merged == 0 on the
        second pass."""
        index = _fresh_index(tmp_path)
        record_ids = ["ep-x", "ep-y", "ep-z"]
        for rid in record_ids:
            index.add_snapshot(_snapshot(rid))

        records = [_episodic(rid, age_days=i) for i, rid in enumerate(record_ids)]
        group = _make_group(record_ids)
        cfg = ConsolidationConfig(threshold=3)

        # First pass.
        result1 = synthesize_group(group, records, config=cfg, shadow_index=index, now=_BASE)
        assert result1.groups_merged == 1
        first_merged_id = result1.metrics.merge_results[0].merged_record_id

        # Build a new group that excludes already-consolidated records — simulating
        # what a well-behaved caller should do before a second pass.
        unconsolidated = [
            rid for rid in record_ids
            if index.get_snapshot(rid) is not None
            and index.get_snapshot(rid).consolidated_into is None  # type: ignore[union-attr]
        ]
        if unconsolidated:
            second_group = _make_group(unconsolidated)
            second_records = [r for r in records if r.record_id in set(unconsolidated)]
            result2 = synthesize_group(
                second_group, second_records, config=cfg, shadow_index=index, now=_BASE
            )
            # Fewer than threshold records → no merge should fire.
            assert result2.groups_merged == 0, (
                "Second pass should not merge a group that is below threshold "
                "after filtering out already-consolidated records"
            )
        else:
            # All records were consolidated on the first pass — no second group to form.
            assert len(unconsolidated) == 0

        # First-pass pointer must be unchanged.
        for rid in record_ids:
            snap = index.get_snapshot(rid)
            assert snap is not None
            assert snap.consolidated_into == first_merged_id


# ---------------------------------------------------------------------------
# Invariant 4 – orphaned entries without valid back-pointers are detected
# ---------------------------------------------------------------------------


class TestOrphanedEntryDetection:
    """The audit helper must surface every snapshot whose consolidated_into
    resolves to nothing (neither a snapshot nor a supplied semantic ID)."""

    def test_no_consolidated_snapshots_means_no_orphans(self, tmp_path: Path) -> None:
        """A freshly populated index with no consolidations has no orphans."""
        index = _fresh_index(tmp_path)
        for i in range(5):
            index.add_snapshot(_snapshot(f"snap-{i}"))

        violations = audit_pointer_integrity(index)
        assert violations == []

    def test_multiple_dangling_pointers_all_reported(self, tmp_path: Path) -> None:
        """Every snapshot with an unresolvable consolidated_into is reported."""
        index = _fresh_index(tmp_path)
        snap_ids = [f"snap-{i}" for i in range(4)]
        for sid in snap_ids:
            index.add_snapshot(_snapshot(sid))

        # Manually write dangling pointers for all four snapshots.
        for sid in snap_ids:
            index.mark_snapshot_consolidated(sid, f"gone-{sid}")

        violations = audit_pointer_integrity(index, known_semantic_ids=None)
        assert len(violations) == len(snap_ids)
        assert all("DANGLING" in v for v in violations)

    def test_mixed_valid_and_orphaned_pointers(self, tmp_path: Path) -> None:
        """Only orphaned entries appear in violations; valid pointers are silent."""
        index = _fresh_index(tmp_path)
        snap_a = _snapshot("snap-a")
        snap_b = _snapshot("snap-b")  # valid canonical target
        snap_c = _snapshot("snap-c")  # will be orphaned
        for s in (snap_a, snap_b, snap_c):
            index.add_snapshot(s)

        index.mark_snapshot_consolidated("snap-a", "snap-b")   # valid
        index.mark_snapshot_consolidated("snap-c", "snap-gone")  # orphan

        violations = audit_pointer_integrity(index)
        assert len(violations) == 1
        assert "snap-c" in violations[0]
        assert "snap-gone" in violations[0]

    def test_synthesize_group_orphans_without_known_semantic_ids(
        self, tmp_path: Path
    ) -> None:
        """After synthesize_group, source snapshots point to a merged_record_id
        that is not in the snapshot table.  The audit detects these as orphans
        unless the caller provides the merged IDs as known_semantic_ids."""
        index = _fresh_index(tmp_path)
        record_ids = ["ep-p", "ep-q", "ep-r"]
        for rid in record_ids:
            index.add_snapshot(_snapshot(rid))

        records = [_episodic(rid, age_days=i) for i, rid in enumerate(record_ids)]
        group = _make_group(record_ids)

        result = synthesize_group(
            group,
            records,
            config=ConsolidationConfig(threshold=3),
            shadow_index=index,
            now=_BASE,
        )
        assert result.success

        merged_id = result.metrics.merge_results[0].merged_record_id

        # Without providing merged_id → all source snapshots appear orphaned.
        orphaned = audit_pointer_integrity(index, known_semantic_ids=None)
        assert len(orphaned) == len(record_ids)
        assert all("DANGLING" in v for v in orphaned)
        assert all(merged_id in v for v in orphaned)

        # Registering merged_id resolves them all.
        clean = audit_pointer_integrity(index, known_semantic_ids={merged_id})
        assert clean == []

    def test_mark_redundant_snapshots_only_marks_eligible_candidates(
        self, tmp_path: Path
    ) -> None:
        """mark_redundant_snapshots skips candidates whose eligible_at is in the
        future, so no orphaned pointer is written prematurely."""
        from shad.vault.contracts import PruneCandidate, PruneReason

        index = _fresh_index(tmp_path)
        snap = _snapshot("snap-future")
        index.add_snapshot(snap)

        future = _BASE + timedelta(days=30)
        candidate = PruneCandidate(
            record_id="snap-future",
            snapshot_id="snap-future",
            reason=PruneReason.SUPERSEDED,
            decay_score=0.1,
            flagged_at=_BASE,
            eligible_at=future,
            superseded_by="sem-xyz",
        )

        marked = index.mark_redundant_snapshots([candidate], now=_BASE)
        assert marked == 0

        # The snapshot must still have no consolidated_into.
        snap_after = index.get_snapshot("snap-future")
        assert snap_after is not None
        assert snap_after.consolidated_into is None

        # No violations.
        violations = audit_pointer_integrity(index)
        assert violations == []
