"""Tests verifying that invalid memory types are rejected with clear error messages.

Two rejection surfaces:

  1. ``MemoryType.from_string(value)`` — must raise ``ValueError`` with a message
     that names the bad value and enumerates every valid alternative.

  2. ``source_to_memory_type(source_type)`` — must silently fall back to
     ``MemoryType.SEMANTIC`` for any unrecognised source type, never raising.

These tests are deliberately distinct from ``test_memory_type_serialization.py``
(which checks round-trips and the *presence* of ``ValueError``) and from
``test_source_to_memory_type.py`` (which focuses on the mapper's known-type table).
"""

from __future__ import annotations

import pytest

from shad.vault.shadow_index import MemoryType, source_to_memory_type


# ---------------------------------------------------------------------------
# MemoryType.from_string — error message quality
# ---------------------------------------------------------------------------


class TestFromStringErrorMessage:
    """The ValueError from from_string must carry a message useful for diagnosis.

    Implementation: ``f"Invalid MemoryType {value!r}. Valid values: {valid}"``
    where ``valid`` lists every ``member.value`` string.
    """

    @pytest.mark.parametrize("bad_value", [
        "invalid",
        "INVALID",
        # Note: from_string normalises to lowercase and accepts both member .value
        # strings AND member .name strings (case-insensitively).  The test cases
        # here must not match any .value or .name under that normalisation.
        "unknown_memory",
        "semantic_mem",    # truncated value
        "working memory",  # space instead of underscore
        "episodic_mem",    # truncated
        "semantic_mem_x",  # near-miss: extra suffix
        "",
        "   ",
    ])
    def test_raises_value_error(self, bad_value: str) -> None:
        """Invalid strings must always produce ValueError, never return a variant."""
        with pytest.raises(ValueError):
            MemoryType.from_string(bad_value)

    @pytest.mark.parametrize("bad_value", [
        "invalid",
        "not_a_type",
        "garbage",
    ])
    def test_error_message_contains_bad_value(self, bad_value: str) -> None:
        """The error message must include a repr of the rejected input so the caller
        can see exactly what string failed, not just that *something* was wrong."""
        with pytest.raises(ValueError) as exc_info:
            MemoryType.from_string(bad_value)

        message = str(exc_info.value)
        assert repr(bad_value) in message, (
            f"Expected {repr(bad_value)!r} in error message, got: {message!r}"
        )

    def test_error_message_starts_with_invalid_memory_type(self) -> None:
        """Error message must begin with 'Invalid MemoryType' so callers can pattern-match."""
        with pytest.raises(ValueError, match="Invalid MemoryType"):
            MemoryType.from_string("invalid")

    def test_error_message_lists_all_valid_values(self) -> None:
        """Every valid MemoryType value string must appear in the error message so
        the caller knows what to use instead."""
        with pytest.raises(ValueError) as exc_info:
            MemoryType.from_string("invalid")

        message = str(exc_info.value)
        for member in MemoryType:
            assert member.value in message, (
                f'Valid value "{member.value}" missing from error message: {message!r}'
            )

    def test_error_message_contains_valid_values_label(self) -> None:
        """The phrase 'Valid values' must appear so the listing is self-explaining."""
        with pytest.raises(ValueError) as exc_info:
            MemoryType.from_string("unknown_memory")

        assert "Valid values" in str(exc_info.value)

    @pytest.mark.parametrize("bad_value", ["", "   "])
    def test_empty_and_whitespace_error_lists_valid_values(self, bad_value: str) -> None:
        """Empty / whitespace strings produce the same informative message."""
        with pytest.raises(ValueError) as exc_info:
            MemoryType.from_string(bad_value)

        message = str(exc_info.value)
        for member in MemoryType:
            assert member.value in message


# ---------------------------------------------------------------------------
# source_to_memory_type — SEMANTIC fallback for unknown source types
# ---------------------------------------------------------------------------


class TestSourceToMemoryTypeFallback:
    """source_to_memory_type must return MemoryType.SEMANTIC (never raise) for
    any source_type string that is not in the known-type table."""

    @pytest.mark.parametrize("unknown_source_type", [
        "rss",
        "twitter",
        "slack",
        "custom",
        "rss_widget",
        "notion",
        "confluence",
        "s3",
        "database",
        "api",
        "webhook",
        "import",
    ])
    def test_unknown_type_returns_semantic(self, unknown_source_type: str) -> None:
        """Unrecognised source types should silently default to SEMANTIC."""
        result = source_to_memory_type(unknown_source_type)
        assert result is MemoryType.SEMANTIC, (
            f"Expected SEMANTIC for unknown source type {unknown_source_type!r}, got {result!r}"
        )

    def test_fallback_is_specifically_semantic_not_any_type(self) -> None:
        """The fallback must be SEMANTIC, not EPISODIC, WORKING, or PROCEDURAL."""
        result = source_to_memory_type("completely_unknown_type")
        assert result is MemoryType.SEMANTIC
        assert result is not MemoryType.EPISODIC
        assert result is not MemoryType.WORKING
        assert result is not MemoryType.PROCEDURAL

    def test_fallback_returns_memory_type_instance(self) -> None:
        """Return value must be a MemoryType enum instance, not a bare string."""
        result = source_to_memory_type("unknown")
        assert isinstance(result, MemoryType)

    def test_fallback_does_not_raise(self) -> None:
        """source_to_memory_type must never raise for any string input."""
        for bad_type in ["invalid", "INVALID", "123", "feed_v2", "github_enterprise"]:
            # Any of these raising would be a regression
            result = source_to_memory_type(bad_type)
            assert result is MemoryType.SEMANTIC

    def test_near_miss_names_fall_back(self) -> None:
        """Strings that look like known types but differ (typos, variants) fall back to SEMANTIC."""
        near_misses = [
            "github_enterprise",  # superset of "github"
            "github_",            # suffix
            "feeds",              # plural of "feed"
            "session",            # singular of "sessions"
            "folder_",            # trailing char
            "url_source",         # compound
        ]
        for near_miss in near_misses:
            result = source_to_memory_type(near_miss)
            assert result is MemoryType.SEMANTIC, (
                f"Near-miss {near_miss!r} should fall back to SEMANTIC, got {result!r}"
            )
