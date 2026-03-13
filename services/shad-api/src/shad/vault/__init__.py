"""Collection curation tools for ingestion, indexing, and gap detection."""

from shad.collection.gap_detection import (
    GapDetector,
    GapReport,
    GapScore,
    QueryHistoryAnalyzer,
)
from shad.collection.ingestion import (
    IngestPreset,
    IngestResult,
    SnapshotMetadata,
    VaultIngester,
)
from shad.collection.shadow_index import (
    ShadowIndex,
    SnapshotEntry,
    SourceEntry,
    UpdatePolicy,
)

__all__ = [
    "IngestPreset",
    "IngestResult",
    "SnapshotMetadata",
    "VaultIngester",
    "ShadowIndex",
    "SourceEntry",
    "SnapshotEntry",
    "UpdatePolicy",
    "GapDetector",
    "GapReport",
    "GapScore",
    "QueryHistoryAnalyzer",
]
