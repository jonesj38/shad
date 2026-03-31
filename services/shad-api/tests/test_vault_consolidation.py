"""Tests for ConsolidationConfig boundary conditions and post-sync callback integration."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shad.sources.config import Source, SourceType
from shad.sources.manager import SourceManager
from shad.vault.consolidation import ConsolidationConfig
from shad.vault.shadow_index import MemoryType, ShadowIndex, SnapshotEntry, SourceEntry, UpdatePolicy


class TestConsolidationConfigPreconditions:
    """ConsolidationConfig must reject invalid construction arguments."""

    @pytest.mark.parametrize("threshold", [0, -1, -100])
    def test_zero_or_negative_threshold_raises_value_error(self, threshold: int) -> None:
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            ConsolidationConfig(threshold=threshold)

    def test_threshold_of_one_is_minimum_valid(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.threshold == 1

    def test_error_message_includes_bad_value(self) -> None:
        with pytest.raises(ValueError, match="0"):
            ConsolidationConfig(threshold=0)

    def test_disabled_with_invalid_threshold_still_raises(self) -> None:
        """The threshold invariant is checked regardless of the enabled flag."""
        with pytest.raises(ValueError):
            ConsolidationConfig(threshold=0, enabled=False)

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        cfg = ConsolidationConfig(threshold=5)
        with pytest.raises((AttributeError, TypeError)):
            cfg.threshold = 10  # type: ignore[misc]


class TestConsolidationThresholdBoundary:
    """Verify the exact-threshold / threshold-minus-one boundary."""

    @pytest.mark.parametrize("threshold", [1, 5, 10, 100])
    def test_fires_at_exactly_threshold(self, threshold: int) -> None:
        cfg = ConsolidationConfig(threshold=threshold)
        assert cfg.should_consolidate(threshold) is True

    @pytest.mark.parametrize("threshold", [2, 5, 10, 100])
    def test_does_not_fire_at_threshold_minus_one(self, threshold: int) -> None:
        cfg = ConsolidationConfig(threshold=threshold)
        assert cfg.should_consolidate(threshold - 1) is False

    def test_minimum_threshold_of_one_fires_at_one(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(1) is True

    def test_minimum_threshold_of_one_does_not_fire_at_zero(self) -> None:
        cfg = ConsolidationConfig(threshold=1)
        assert cfg.should_consolidate(0) is False

    def test_disabled_does_not_fire_at_threshold(self) -> None:
        cfg = ConsolidationConfig(threshold=5, enabled=False)
        assert cfg.should_consolidate(5) is False

    def test_disabled_does_not_fire_above_threshold(self) -> None:
        cfg = ConsolidationConfig(threshold=5, enabled=False)
        assert cfg.should_consolidate(999) is False

    def test_fires_above_threshold(self) -> None:
        cfg = ConsolidationConfig(threshold=10)
        assert cfg.should_consolidate(11) is True


# ---------------------------------------------------------------------------
# Integration: callback fires based on episodic snapshot count after sync
# ---------------------------------------------------------------------------

_THRESHOLD = 3
_SOURCE_ID = "feed-src-001"
_SOURCE_URL = "https://feeds.example.com/tech"


@pytest.fixture
def shadow_index(tmp_path: Path) -> ShadowIndex:
    """ShadowIndex backed by a temp SQLite file with one episodic feed source."""
    index = ShadowIndex(tmp_path / "test_index.sqlite")
    index.add_source(
        SourceEntry(
            source_url=_SOURCE_URL,
            source_id=_SOURCE_ID,
            source_type="feed",
            update_policy=UpdatePolicy.AUTO,
        )
    )
    return index


def _add_episodic_snapshot(
    index: ShadowIndex,
    ingested_at: datetime,
) -> None:
    """Insert one EPISODIC snapshot for the shared test source."""
    index.add_snapshot(
        SnapshotEntry(
            snapshot_id=str(uuid.uuid4()),
            source_id=_SOURCE_ID,
            ingested_at=ingested_at,
            source_revision="rev-" + str(uuid.uuid4())[:8],
            entry_paths=["feeds/item.md"],
            content_hash=str(uuid.uuid4()),
            memory_type=MemoryType.EPISODIC,
        )
    )


class TestConsolidationCallbackAfterSync:
    """Verify that ConsolidationConfig.should_consolidate fires (or not) based on
    ShadowIndex.episodic_count — the number of episodic snapshots added since the
    last consolidation timestamp.
    """

    def test_callback_does_not_fire_below_threshold(
        self, shadow_index: ShadowIndex
    ) -> None:
        """threshold-1 episodic snapshots ingested after baseline → no trigger."""
        cfg = ConsolidationConfig(threshold=_THRESHOLD)
        last_consolidated_at = datetime(2026, 3, 1, tzinfo=UTC)

        for i in range(_THRESHOLD - 1):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=last_consolidated_at + timedelta(hours=i + 1),
            )

        count = shadow_index.episodic_count(_SOURCE_ID, since=last_consolidated_at)
        assert count == _THRESHOLD - 1
        assert cfg.should_consolidate(count) is False

    def test_callback_fires_at_threshold(
        self, shadow_index: ShadowIndex
    ) -> None:
        """Exactly threshold episodic snapshots after sync → trigger fires."""
        cfg = ConsolidationConfig(threshold=_THRESHOLD)
        last_consolidated_at = datetime(2026, 3, 1, tzinfo=UTC)

        for i in range(_THRESHOLD):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=last_consolidated_at + timedelta(hours=i + 1),
            )

        count = shadow_index.episodic_count(_SOURCE_ID, since=last_consolidated_at)
        assert count == _THRESHOLD
        assert cfg.should_consolidate(count) is True

    def test_callback_fires_above_threshold(
        self, shadow_index: ShadowIndex
    ) -> None:
        """threshold+2 episodic snapshots after sync → trigger still fires."""
        cfg = ConsolidationConfig(threshold=_THRESHOLD)
        last_consolidated_at = datetime(2026, 3, 1, tzinfo=UTC)

        for i in range(_THRESHOLD + 2):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=last_consolidated_at + timedelta(hours=i + 1),
            )

        count = shadow_index.episodic_count(_SOURCE_ID, since=last_consolidated_at)
        assert count > _THRESHOLD
        assert cfg.should_consolidate(count) is True

    def test_snapshots_before_baseline_excluded_from_count(
        self, shadow_index: ShadowIndex
    ) -> None:
        """Pre-baseline snapshots do not contribute to the post-sync count.

        This mirrors the teranode consolidator pattern where only incremental
        records since the last consolidation run are considered.
        """
        cfg = ConsolidationConfig(threshold=_THRESHOLD)
        last_consolidated_at = datetime(2026, 3, 10, tzinfo=UTC)

        # Add threshold snapshots *before* the baseline — must be excluded.
        for i in range(_THRESHOLD):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=last_consolidated_at - timedelta(hours=i + 1),
            )

        # Add only threshold-1 snapshots *after* the baseline.
        for i in range(_THRESHOLD - 1):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=last_consolidated_at + timedelta(hours=i + 1),
            )

        count = shadow_index.episodic_count(_SOURCE_ID, since=last_consolidated_at)
        assert count == _THRESHOLD - 1
        assert cfg.should_consolidate(count) is False

    def test_callback_fires_after_advancing_baseline_and_new_sync(
        self, shadow_index: ShadowIndex
    ) -> None:
        """After a consolidation completes, advancing the baseline resets the
        effective count — a subsequent sync must accumulate threshold new
        entries before firing again.
        """
        cfg = ConsolidationConfig(threshold=_THRESHOLD)
        first_baseline = datetime(2026, 3, 1, tzinfo=UTC)
        second_baseline = datetime(2026, 3, 15, tzinfo=UTC)

        # First batch: threshold entries — triggers, then consolidation advances baseline.
        for i in range(_THRESHOLD):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=first_baseline + timedelta(hours=i + 1),
            )
        first_count = shadow_index.episodic_count(_SOURCE_ID, since=first_baseline)
        assert cfg.should_consolidate(first_count) is True

        # Second batch: only threshold-1 entries after the new baseline — no trigger yet.
        for i in range(_THRESHOLD - 1):
            _add_episodic_snapshot(
                shadow_index,
                ingested_at=second_baseline + timedelta(hours=i + 1),
            )
        second_count = shadow_index.episodic_count(_SOURCE_ID, since=second_baseline)
        assert second_count == _THRESHOLD - 1
        assert cfg.should_consolidate(second_count) is False

        # One more snapshot pushes it over — trigger fires again.
        _add_episodic_snapshot(
            shadow_index,
            ingested_at=second_baseline + timedelta(hours=_THRESHOLD + 1),
        )
        final_count = shadow_index.episodic_count(_SOURCE_ID, since=second_baseline)
        assert final_count == _THRESHOLD
        assert cfg.should_consolidate(final_count) is True


# ---------------------------------------------------------------------------
# End-to-end: on_consolidation callback via SourceManager._check_consolidation
# ---------------------------------------------------------------------------


class TestSourceManagerConsolidationCallback:
    """Verify that the on_consolidation callback is NOT invoked when the episodic
    snapshot count after sync is below the configured threshold.

    Uses SourceManager._check_consolidation directly so the test does not
    need to stub the network/ingest layers, while still exercising the real
    ShadowIndex query and ConsolidationConfig.should_consolidate path.
    """

    def _make_feed_source(self) -> Source:
        return Source(
            type=SourceType.FEED,
            url="https://feeds.example.com/shad-test",
            collection_path="/tmp/shad-test-collection",
        )

    def test_callback_not_fired_when_count_below_threshold(
        self, tmp_path: Path
    ) -> None:
        """threshold=5, only 4 episodic snapshots after epoch → callback silent."""
        threshold = 5
        index = ShadowIndex(tmp_path / "index.sqlite")
        source = self._make_feed_source()
        index.add_source(
            SourceEntry(
                source_url=source.url,
                source_id=source.id,
                source_type="feed",
                update_policy=UpdatePolicy.AUTO,
            )
        )

        # Insert threshold - 1 episodic snapshots (below the trigger boundary).
        base = datetime(2026, 3, 1, tzinfo=UTC)
        for i in range(threshold - 1):
            index.add_snapshot(
                SnapshotEntry(
                    snapshot_id=str(uuid.uuid4()),
                    source_id=source.id,
                    ingested_at=base + timedelta(hours=i + 1),
                    source_revision=f"rev-{i}",
                    entry_paths=[f"feeds/item-{i}.md"],
                    content_hash=str(uuid.uuid4()),
                    memory_type=MemoryType.EPISODIC,
                )
            )

        callback = MagicMock()
        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=callback,
        )

        manager._check_consolidation(source)

        callback.assert_not_called()
        index.close()

    def test_callback_not_fired_when_count_is_zero(
        self, tmp_path: Path
    ) -> None:
        """No snapshots at all → callback never fires regardless of threshold."""
        index = ShadowIndex(tmp_path / "index.sqlite")
        source = self._make_feed_source()
        index.add_source(
            SourceEntry(
                source_url=source.url,
                source_id=source.id,
                source_type="feed",
                update_policy=UpdatePolicy.AUTO,
            )
        )

        callback = MagicMock()
        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=1),
            shadow_index=index,
            on_consolidation=callback,
        )

        manager._check_consolidation(source)

        callback.assert_not_called()
        index.close()

    def test_callback_fires_exactly_at_threshold_for_contrast(
        self, tmp_path: Path
    ) -> None:
        """Contrast case: exactly threshold snapshots present → callback IS called."""
        threshold = 3
        index = ShadowIndex(tmp_path / "index.sqlite")
        source = self._make_feed_source()
        index.add_source(
            SourceEntry(
                source_url=source.url,
                source_id=source.id,
                source_type="feed",
                update_policy=UpdatePolicy.AUTO,
            )
        )

        base = datetime(2026, 3, 1, tzinfo=UTC)
        for i in range(threshold):
            index.add_snapshot(
                SnapshotEntry(
                    snapshot_id=str(uuid.uuid4()),
                    source_id=source.id,
                    ingested_at=base + timedelta(hours=i + 1),
                    source_revision=f"rev-{i}",
                    entry_paths=[f"feeds/item-{i}.md"],
                    content_hash=str(uuid.uuid4()),
                    memory_type=MemoryType.EPISODIC,
                )
            )

        callback = MagicMock()
        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=callback,
        )

        manager._check_consolidation(source)

        callback.assert_called_once_with(source, threshold)
        index.close()


# ---------------------------------------------------------------------------
# Custom threshold values from config
# ---------------------------------------------------------------------------


class TestCustomThresholdConfig:
    """ConsolidationConfig with non-default thresholds must gate SourceManager
    consolidation accurately — firing at exactly the configured value and never
    below it, regardless of which custom threshold is supplied.
    """

    def _make_index_and_source(self, tmp_path: Path) -> tuple[ShadowIndex, Source]:
        index = ShadowIndex(tmp_path / "index.sqlite")
        source = Source(
            type=SourceType.FEED,
            url="https://feeds.example.com/custom-threshold",
            collection_path="/tmp/shad-custom",
        )
        index.add_source(
            SourceEntry(
                source_url=source.url,
                source_id=source.id,
                source_type="feed",
                update_policy=UpdatePolicy.AUTO,
            )
        )
        return index, source

    def _insert_snapshots(
        self,
        index: ShadowIndex,
        source_id: str,
        count: int,
        base: datetime,
    ) -> None:
        for i in range(count):
            index.add_snapshot(
                SnapshotEntry(
                    snapshot_id=str(uuid.uuid4()),
                    source_id=source_id,
                    ingested_at=base + timedelta(hours=i + 1),
                    source_revision=f"rev-{i}",
                    entry_paths=[f"feeds/item-{i}.md"],
                    content_hash=str(uuid.uuid4()),
                    memory_type=MemoryType.EPISODIC,
                )
            )

    @pytest.mark.parametrize("threshold", [2, 7, 15, 50])
    def test_fires_at_custom_threshold(self, threshold: int, tmp_path: Path) -> None:
        """Custom threshold N: exactly N episodic entries → callback fires."""
        index, source = self._make_index_and_source(tmp_path)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self._insert_snapshots(index, source.id, threshold, base)

        callback = MagicMock()
        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=callback,
        )
        manager._check_consolidation(source)

        callback.assert_called_once_with(source, threshold)
        index.close()

    @pytest.mark.parametrize("threshold", [2, 7, 15, 50])
    def test_does_not_fire_below_custom_threshold(
        self, threshold: int, tmp_path: Path
    ) -> None:
        """Custom threshold N: N-1 episodic entries → callback stays silent."""
        index, source = self._make_index_and_source(tmp_path)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self._insert_snapshots(index, source.id, threshold - 1, base)

        callback = MagicMock()
        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=callback,
        )
        manager._check_consolidation(source)

        callback.assert_not_called()
        index.close()

    def test_default_threshold_is_ten(self, tmp_path: Path) -> None:
        """The default ConsolidationConfig threshold is 10."""
        cfg = ConsolidationConfig()
        assert cfg.threshold == 10
        assert cfg.should_consolidate(9) is False
        assert cfg.should_consolidate(10) is True

    def test_callback_receives_correct_count_argument(
        self, tmp_path: Path
    ) -> None:
        """on_consolidation is called with the actual episodic count, not the threshold."""
        threshold = 3
        extra = 2  # insert threshold + extra entries
        index, source = self._make_index_and_source(tmp_path)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self._insert_snapshots(index, source.id, threshold + extra, base)

        callback = MagicMock()
        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=callback,
        )
        manager._check_consolidation(source)

        callback.assert_called_once_with(source, threshold + extra)
        index.close()

    def test_last_consolidation_metadata_stamped_after_firing(
        self, tmp_path: Path
    ) -> None:
        """source.metadata['last_consolidation'] is set to an ISO timestamp after firing."""
        threshold = 2
        index, source = self._make_index_and_source(tmp_path)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self._insert_snapshots(index, source.id, threshold, base)

        assert "last_consolidation" not in source.metadata

        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=MagicMock(),
        )
        manager._check_consolidation(source)

        assert "last_consolidation" in source.metadata
        # Value must be a valid ISO-format datetime string.
        stamped = datetime.fromisoformat(source.metadata["last_consolidation"])
        assert stamped.tzinfo is not None
        index.close()

    def test_last_consolidation_metadata_not_set_when_below_threshold(
        self, tmp_path: Path
    ) -> None:
        """Metadata is NOT updated when consolidation does not fire."""
        threshold = 5
        index, source = self._make_index_and_source(tmp_path)
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self._insert_snapshots(index, source.id, threshold - 1, base)

        manager = SourceManager(
            consolidation_config=ConsolidationConfig(threshold=threshold),
            shadow_index=index,
            on_consolidation=MagicMock(),
        )
        manager._check_consolidation(source)

        assert "last_consolidation" not in source.metadata
        index.close()
