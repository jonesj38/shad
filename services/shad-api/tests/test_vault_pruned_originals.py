"""Tests verifying that pruned originals are preserved intact with correct consolidated_into references.

Covers:
- mark_snapshot_consolidated preserves the row and sets consolidated_into
- mark_redundant_snapshots: originals survive, consolidated_into is set correctly
- Pinned snapshots are skipped by mark_redundant_snapshots
- Not-yet-eligible candidates are skipped
- Candidates without superseded_by are skipped
- Contrast: prune_snapshots actually deletes rows (proving soft-prune is the safe path)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shad.vault.contracts import PruneCandidate, PruneReason
from shad.vault.shadow_index import (
    MemoryType,
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_ID = "src-prune-test"
_SOURCE_URL = "https://example.com/source"

_NOW = datetime(2026, 3, 31, 12, 0, 0, tzinfo=UTC)
_PAST = _NOW - timedelta(hours=1)  # already eligible
_FUTURE = _NOW + timedelta(hours=1)  # not yet eligible


def _make_index(tmp_path: Path) -> ShadowIndex:
    index = ShadowIndex(tmp_path / "index.sqlite")
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="github",
            update_policy=UpdatePolicy.AUTO,
        )
    )
    return index


def _add_snapshot(
    index: ShadowIndex,
    *,
    snapshot_id: str | None = None,
    content_hash: str | None = None,
    memory_type: MemoryType = MemoryType.EPISODIC,
    ingested_at: datetime | None = None,
) -> SnapshotEntry:
    snap = SnapshotEntry(
        snapshot_id=snapshot_id or str(uuid.uuid4()),
        source_id=_SOURCE_ID,
        ingested_at=ingested_at or _NOW,
        source_revision="rev-" + str(uuid.uuid4())[:8],
        entry_paths=["notes/item.md"],
        content_hash=content_hash or str(uuid.uuid4()),
        memory_type=memory_type,
    )
    index.add_snapshot(snap)
    return snap


def _candidate(
    snapshot_id: str,
    superseded_by: str | None,
    *,
    eligible_at: datetime = _PAST,
) -> PruneCandidate:
    return PruneCandidate(
        record_id=snapshot_id,
        snapshot_id=snapshot_id,
        reason=PruneReason.SUPERSEDED,
        decay_score=0.5,
        flagged_at=_NOW,
        eligible_at=eligible_at,
        superseded_by=superseded_by,
    )


# ---------------------------------------------------------------------------
# mark_snapshot_consolidated — atomic unit tests
# ---------------------------------------------------------------------------


class TestMarkSnapshotConsolidated:
    """ShadowIndex.mark_snapshot_consolidated must update the row without deleting it."""

    def test_row_still_exists_after_marking(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-orig")
        canonical_id = "snap-canonical"

        result = index.mark_snapshot_consolidated(snap.snapshot_id, canonical_id)

        assert result is True
        assert index.get_snapshot(snap.snapshot_id) is not None

    def test_consolidated_into_set_to_canonical_id(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-src")
        canonical_id = "snap-new"

        index.mark_snapshot_consolidated(snap.snapshot_id, canonical_id)
        updated = index.get_snapshot(snap.snapshot_id)

        assert updated is not None
        assert updated.consolidated_into == canonical_id

    def test_returns_false_for_nonexistent_snapshot(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        result = index.mark_snapshot_consolidated("ghost-snap", "canon")
        assert result is False

    def test_other_fields_unchanged_after_marking(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        content_hash = "abc123"
        snap = _add_snapshot(index, snapshot_id="snap-fields", content_hash=content_hash)

        index.mark_snapshot_consolidated(snap.snapshot_id, "canon-id")
        updated = index.get_snapshot(snap.snapshot_id)

        assert updated is not None
        assert updated.content_hash == content_hash
        assert updated.source_id == _SOURCE_ID
        assert updated.memory_type == MemoryType.EPISODIC

    def test_can_overwrite_consolidated_into(self, tmp_path: Path) -> None:
        """Calling mark_snapshot_consolidated twice updates to the latest canonical."""
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-repoint")

        index.mark_snapshot_consolidated(snap.snapshot_id, "first-canon")
        index.mark_snapshot_consolidated(snap.snapshot_id, "second-canon")
        updated = index.get_snapshot(snap.snapshot_id)

        assert updated is not None
        assert updated.consolidated_into == "second-canon"


# ---------------------------------------------------------------------------
# mark_redundant_snapshots — originals preserved, references correct
# ---------------------------------------------------------------------------


class TestMarkRedundantSnapshotsPreservesOriginals:
    """mark_redundant_snapshots must never delete rows; it only sets consolidated_into."""

    def test_original_survives_after_mark_redundant(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        orig = _add_snapshot(index, snapshot_id="orig-snap")
        canonical_id = "canon-snap"
        _add_snapshot(index, snapshot_id=canonical_id)

        candidates = [_candidate(orig.snapshot_id, canonical_id)]
        count = index.mark_redundant_snapshots(candidates, now=_NOW)

        assert count == 1
        assert index.get_snapshot(orig.snapshot_id) is not None, (
            "Original must not be deleted"
        )

    def test_consolidated_into_points_to_correct_canonical(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        orig = _add_snapshot(index, snapshot_id="snap-a")
        canonical_id = "snap-b"
        _add_snapshot(index, snapshot_id=canonical_id)

        index.mark_redundant_snapshots([_candidate(orig.snapshot_id, canonical_id)], now=_NOW)
        updated = index.get_snapshot(orig.snapshot_id)

        assert updated is not None
        assert updated.consolidated_into == canonical_id

    def test_multiple_originals_all_preserved(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        canonical_id = "canon"
        _add_snapshot(index, snapshot_id=canonical_id)

        orig_ids = [f"orig-{i}" for i in range(3)]
        for oid in orig_ids:
            _add_snapshot(index, snapshot_id=oid)

        candidates = [_candidate(oid, canonical_id) for oid in orig_ids]
        count = index.mark_redundant_snapshots(candidates, now=_NOW)

        assert count == 3
        for oid in orig_ids:
            snap = index.get_snapshot(oid)
            assert snap is not None, f"{oid} must not be deleted"
            assert snap.consolidated_into == canonical_id

    def test_each_original_references_its_own_canonical(self, tmp_path: Path) -> None:
        """Two originals superseded by different canonicals must each point to theirs."""
        index = _make_index(tmp_path)
        orig_a = _add_snapshot(index, snapshot_id="orig-a")
        orig_b = _add_snapshot(index, snapshot_id="orig-b")
        _add_snapshot(index, snapshot_id="canon-a")
        _add_snapshot(index, snapshot_id="canon-b")

        candidates = [
            _candidate("orig-a", "canon-a"),
            _candidate("orig-b", "canon-b"),
        ]
        index.mark_redundant_snapshots(candidates, now=_NOW)

        assert index.get_snapshot("orig-a").consolidated_into == "canon-a"
        assert index.get_snapshot("orig-b").consolidated_into == "canon-b"

    def test_canonical_itself_is_never_touched(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        orig = _add_snapshot(index, snapshot_id="orig")
        canonical = _add_snapshot(index, snapshot_id="canonical")

        index.mark_redundant_snapshots([_candidate("orig", "canonical")], now=_NOW)
        canon_after = index.get_snapshot("canonical")

        assert canon_after is not None
        assert canon_after.consolidated_into is None, (
            "Canonical snapshot must not be marked as consolidated"
        )


# ---------------------------------------------------------------------------
# Eligibility and pin guards
# ---------------------------------------------------------------------------


class TestMarkRedundantEligibilityGuards:
    """Candidates must only be marked when eligible and unpinned."""

    def test_future_eligible_at_is_skipped(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-future")
        _add_snapshot(index, snapshot_id="canon")

        candidates = [_candidate(snap.snapshot_id, "canon", eligible_at=_FUTURE)]
        count = index.mark_redundant_snapshots(candidates, now=_NOW)

        assert count == 0
        updated = index.get_snapshot(snap.snapshot_id)
        assert updated is not None
        assert updated.consolidated_into is None, "Not-yet-eligible candidate must not be marked"

    def test_pinned_snapshot_is_skipped(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-pinned")
        _add_snapshot(index, snapshot_id="canon")

        # Pin the snapshot so the guard fires.
        index.pin_snapshot(_SOURCE_URL, snap.snapshot_id)

        candidates = [_candidate(snap.snapshot_id, "canon")]
        count = index.mark_redundant_snapshots(candidates, now=_NOW)

        assert count == 0
        updated = index.get_snapshot(snap.snapshot_id)
        assert updated is not None
        assert updated.consolidated_into is None, "Pinned snapshot must not be marked"

    def test_candidate_without_superseded_by_is_skipped(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-no-canon")

        candidate = PruneCandidate(
            record_id=snap.snapshot_id,
            snapshot_id=snap.snapshot_id,
            reason=PruneReason.STALE,
            decay_score=0.1,
            flagged_at=_NOW,
            eligible_at=_PAST,
            superseded_by=None,  # no canonical
        )
        count = index.mark_redundant_snapshots([candidate], now=_NOW)

        assert count == 0
        updated = index.get_snapshot(snap.snapshot_id)
        assert updated is not None
        assert updated.consolidated_into is None

    def test_empty_candidate_list_returns_zero(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        assert index.mark_redundant_snapshots([], now=_NOW) == 0

    def test_mix_eligible_and_not_eligible(self, tmp_path: Path) -> None:
        """Only the past-eligible candidate is marked; the future one is skipped."""
        index = _make_index(tmp_path)
        _add_snapshot(index, snapshot_id="canon")
        snap_now = _add_snapshot(index, snapshot_id="snap-eligible")
        snap_later = _add_snapshot(index, snapshot_id="snap-not-eligible")

        candidates = [
            _candidate(snap_now.snapshot_id, "canon", eligible_at=_PAST),
            _candidate(snap_later.snapshot_id, "canon", eligible_at=_FUTURE),
        ]
        count = index.mark_redundant_snapshots(candidates, now=_NOW)

        assert count == 1
        assert index.get_snapshot(snap_now.snapshot_id).consolidated_into == "canon"
        assert index.get_snapshot(snap_later.snapshot_id).consolidated_into is None


# ---------------------------------------------------------------------------
# Contrast: prune_snapshots actually deletes rows
# ---------------------------------------------------------------------------


class TestPruneSnapshotsDeletesRows:
    """Contrast suite: prune_snapshots (hard-prune) removes rows entirely.

    These tests prove that mark_redundant_snapshots is the correct non-destructive
    path, while prune_snapshots performs a permanent deletion.
    """

    def test_prune_snapshots_deletes_eligible_row(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-deletable")

        candidates = [_candidate(snap.snapshot_id, None, eligible_at=_PAST)]
        count = index.prune_snapshots(candidates, now=_NOW)

        assert count == 1
        assert index.get_snapshot(snap.snapshot_id) is None, (
            "prune_snapshots must delete the row"
        )

    def test_prune_snapshots_skips_pinned(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-pinned-hard")

        index.pin_snapshot(_SOURCE_URL, snap.snapshot_id)

        candidates = [_candidate(snap.snapshot_id, None, eligible_at=_PAST)]
        count = index.prune_snapshots(candidates, now=_NOW)

        assert count == 0
        assert index.get_snapshot(snap.snapshot_id) is not None

    def test_prune_snapshots_skips_future_eligible(self, tmp_path: Path) -> None:
        index = _make_index(tmp_path)
        snap = _add_snapshot(index, snapshot_id="snap-future-hard")

        candidates = [_candidate(snap.snapshot_id, None, eligible_at=_FUTURE)]
        count = index.prune_snapshots(candidates, now=_NOW)

        assert count == 0
        assert index.get_snapshot(snap.snapshot_id) is not None
