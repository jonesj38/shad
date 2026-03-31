"""Feature-area audit extractor.

Reads each target file listed in FeatureAudit, detects which expected
artifacts are present via pattern matching, and returns GapRecord instances
for every artifact that is absent.

Usage::

    from pathlib import Path
    from shad.vault.audit import run_feature_audit

    gaps = run_feature_audit()
    for g in gaps:
        print(f"[{g.feature_area}] {g.file}: missing '{g.what_must_be_added}'")
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from shad.vault.contracts import FeatureAudit, GapRecord

# ---------------------------------------------------------------------------
# Per-artifact detection patterns
# ---------------------------------------------------------------------------
# Keys are the normalised artifact description (lowercase, alphanumeric + space).
# Values are lists of regex patterns; presence is True when ANY pattern matches
# the file content (IGNORECASE | DOTALL).

_ARTIFACT_PATTERNS: dict[str, list[str]] = {
    # retrieval/layer.py
    "retrievallayer protocol": [r"class\s+RetrievalLayer", r"RetrievalLayer\s*\(Protocol"],
    "retrievalresult dataclass": [r"class\s+RetrievalResult"],
    "hybrid search mode": [r'"hybrid"', r"mode.*hybrid"],
    "bm25 search mode": [r'"bm25"', r"mode.*bm25"],
    "vector search mode": [r'"vector"', r"mode.*vector"],
    # retrieval/qmd.py
    "qmdretriever class": [r"class\s+QmdRetriever"],
    "qmd cli detection": [r"shutil\.which\(", r"_qmd_path.*None"],
    "bm25  vector hybrid search": [r"MODE_COMMANDS", r'"bm25".*"vector".*"hybrid"'],
    "score normalisation": [r"normalize_score", r"normalise_score", r"_normalize_scores",
                             r"score\s*/\s*max", r"min\s*\(\s*1\.0\s*,\s*score",
                             r"clamp\s*\("],
    # retrieval/filesystem.py
    "filesystemretriever class": [r"class\s+FilesystemRetriever"],
    "keyword fallback search": [r"_extract_terms", r"keyword", r"fallback.*search",
                                  r"search.*keyword"],
    # vault/gap_detection.py
    "gapscore dataclass": [r"class\s+GapScore"],
    "gapreport dataclass": [r"class\s+GapReport"],
    "gapdetectorcalculate gap score": [r"def\s+calculate_gap_score"],
    "queryhistoryanalyzer": [r"class\s+QueryHistoryAnalyzer"],
    "gapreportto markdown": [r"def\s+to_markdown"],
    # sources/manager.py
    "sourcemanager class": [r"class\s+SourceManager"],
    "add source": [r"def\s+add_source"],
    "remove source": [r"def\s+remove_source"],
    "list sources": [r"def\s+list_sources"],
    "sync sources": [r"def\s+sync_all", r"def\s+sync_sources"],
    # vault/shadow_index.py
    "shadowindex class": [r"class\s+ShadowIndex"],
    "sourceentry dataclass": [r"class\s+SourceEntry"],
    "snapshotentry dataclass": [r"class\s+SnapshotEntry"],
    "updatepolicy enum": [r"class\s+UpdatePolicy"],
    # cli/main.py
    "shad run command": [r'command\("run"\)'],
    "shad status command": [r'command\("status"\)', r'command\(\)\s*\ndef status'],
    "shad search command": [r'command\("search"\)'],
    "shad ingest command": [r'group\("ingest"\)', r'command\("ingest"\)'],
    "shad sources subgroup": [r'@sources\.command', r'cli\.group.*sources'],
    "shad server subgroup": [r'@server\.command', r'cli\.group.*server'],
    "shad models command": [r'command\("models"\)'],
    # sandbox/executor.py
    "codeexecutor class": [r"class\s+CodeExecutor"],
    "sandboxed exec with restricted builtins": [r"_get_safe_builtins", r"restricted.*builtins"],
    "result extraction": [r"__result__"],
    # engine/rlm.py
    "rlm engine": [r"class\s+RLMEngine"],
    "dag execution": [r"DAGNode", r"dag.*execut", r"execute_single_node"],
    "code mode orchestration": [r"_execute_code_mode", r"retrieve_context_code_mode"],
    "budget system": [r"BudgetExhausted", r"_check_budgets"],
    "strategy selection": [r"StrategySelector", r"strategy_selector\.select"],
    # verification/layer.py
    "verificationlayer": [r"class\s+VerificationLayer"],
    "syntax check": [r"class\s+SyntaxCheck"],
    "type check": [r"class\s+TypeCheck", r"\bmypy\b", r"\bpyright\b",
                   r"subprocess.*mypy", r"subprocess.*pyright"],
    "import resolution": [r"class\s+ImportResolutionCheck", r"import.*resolut"],
    "strict  build  basic  off levels": [r"class\s+VerificationLevel",
                                          r'STRICT\s*=', r'BASIC\s*='],
}


def _normalise_key(artifact: str) -> str:
    """Normalise an artifact description to a dict lookup key.

    Strips non-alphanumeric characters (except spaces) and lowercases.
    """
    return re.sub(r"[^a-z0-9 ]", "", artifact.lower()).strip()


def _artifact_present(artifact: str, content: str) -> bool:
    """Return True if *artifact* is detectable in *content*.

    First consults the explicit ``_ARTIFACT_PATTERNS`` table.  If the
    normalised key is absent, falls back to searching for any word from
    the artifact description that is longer than three characters.
    """
    key = _normalise_key(artifact)
    patterns = _ARTIFACT_PATTERNS.get(key)
    if patterns is None:
        # Generic fallback: any substantive word from the description
        words = [w for w in re.split(r"\W+", artifact) if len(w) > 3]
        return any(re.search(re.escape(w), content, re.IGNORECASE) for w in words)
    return any(re.search(p, content, re.IGNORECASE | re.DOTALL) for p in patterns)


def _existing_summary(content: str) -> str:
    """Return a short summary of what is implemented in *content*.

    Collects class and top-level function names (up to four each) so that
    GapRecord.what_exists is informative without being verbose.
    """
    classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
    funcs = re.findall(r"^(?:async\s+)?def\s+(\w+)", content, re.MULTILINE)
    parts: list[str] = []
    if classes:
        parts.append("classes: " + ", ".join(classes[:4]))
    if funcs:
        parts.append("defs: " + ", ".join(funcs[:4]))
    return "; ".join(parts) if parts else "file exists"


def run_feature_audit(
    base_dir: Path | None = None,
    *,
    feature_areas: Sequence[str] | None = None,
) -> list[GapRecord]:
    """Run the feature audit and return GapRecord instances for each gap.

    For every entry in :data:`~shad.vault.contracts.FeatureAudit` (or the
    subset named in *feature_areas*):

    1. Reads the ``target_file`` relative to *base_dir*.
    2. Detects which ``expected_artifacts`` are present using
       :func:`_artifact_present`.
    3. Emits a :class:`~shad.vault.contracts.GapRecord` for every expected
       artifact that is absent.

    Args:
        base_dir: Root directory against which ``target_file`` paths are
            resolved.  Defaults to the ``services/shad-api/`` directory
            (four ``parents`` above this module).
        feature_areas: Optional subset of feature-area names to audit.
            When *None*, all areas in ``FeatureAudit`` are audited.

    Returns:
        List of :class:`GapRecord` objects, one per missing artifact.
        An empty list means every expected artifact was found.
    """
    if base_dir is None:
        # This file lives at services/shad-api/src/shad/vault/audit.py
        # → parents[3] == services/shad-api/
        base_dir = Path(__file__).resolve().parents[3]

    areas: Sequence[str] = (
        feature_areas if feature_areas is not None else list(FeatureAudit.keys())
    )

    gaps: list[GapRecord] = []

    for area in areas:
        entry = FeatureAudit.get(area)
        if entry is None:
            continue

        target = base_dir / entry.target_file

        if not target.exists():
            for artifact in entry.expected_artifacts:
                gaps.append(GapRecord(
                    feature_area=area,
                    file=entry.target_file,
                    what_exists="file not found",
                    what_must_be_added=artifact,
                ))
            continue

        content = target.read_text(encoding="utf-8")
        summary = _existing_summary(content)

        for artifact in entry.expected_artifacts:
            if not _artifact_present(artifact, content):
                gaps.append(GapRecord(
                    feature_area=area,
                    file=entry.target_file,
                    what_exists=summary,
                    what_must_be_added=artifact,
                ))

    return gaps
