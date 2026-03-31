"""Tests for the make_bitemporal_record factory.

Verifies three behavioural contracts:
  1. ingested_at is set close to datetime.now(UTC) — it is stamped at call time,
     never back-dated or carried over from the caller.
  2. event_time is preserved verbatim from the argument.
  3. content is preserved verbatim from the argument.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shad.vault.shadow_index import BiTemporalRecord, make_bitemporal_record

# Maximum acceptable gap between the factory call and ingested_at.
# 2 s is generous enough to survive even a heavily loaded CI runner.
_INGESTION_TOLERANCE = timedelta(seconds=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_event_time() -> datetime:
    return datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# ingested_at reflects the current time
# ---------------------------------------------------------------------------

class TestIngestedAtTimestamp:
    """ingested_at must be set by the factory to approximately now."""

    def test_ingested_at_is_close_to_now(self) -> None:
        before = datetime.now(UTC)
        record = make_bitemporal_record(_fixed_event_time(), "hello")
        after = datetime.now(UTC)

        assert before <= record.ingested_at <= after + timedelta(milliseconds=1), (
            f"ingested_at={record.ingested_at} falls outside "
            f"[{before}, {after}]"
        )

    def test_ingested_at_within_tolerance(self) -> None:
        record = make_bitemporal_record(_fixed_event_time(), "payload")
        delta = abs(datetime.now(UTC) - record.ingested_at)
        assert delta <= _INGESTION_TOLERANCE, (
            f"ingested_at drifted {delta.total_seconds():.3f}s from now"
        )

    def test_ingested_at_is_utc(self) -> None:
        record = make_bitemporal_record(_fixed_event_time(), "payload")
        assert record.ingested_at.tzinfo is not None
        assert record.ingested_at.utcoffset() == timedelta(0)

    def test_ingested_at_not_before_call(self) -> None:
        before = datetime.now(UTC)
        record = make_bitemporal_record(_fixed_event_time(), "payload")
        assert record.ingested_at >= before

    def test_ingested_at_not_after_return(self) -> None:
        record = make_bitemporal_record(_fixed_event_time(), "payload")
        after = datetime.now(UTC)
        assert record.ingested_at <= after

    def test_ingested_at_differs_from_event_time(self) -> None:
        """Factory must not accidentally copy event_time into ingested_at."""
        past_event = datetime(2020, 6, 1, tzinfo=UTC)
        record = make_bitemporal_record(past_event, "payload")
        assert record.ingested_at != past_event

    def test_two_consecutive_calls_ingested_at_nondecreasing(self) -> None:
        """Each call stamps a fresh now; the second must be >= the first."""
        r1 = make_bitemporal_record(_fixed_event_time(), "first")
        r2 = make_bitemporal_record(_fixed_event_time(), "second")
        assert r2.ingested_at >= r1.ingested_at

    @pytest.mark.parametrize("delta_days", [-365, -30, -1, 0, 1, 30, 365])
    def test_ingested_at_independent_of_event_time_offset(
        self, delta_days: int
    ) -> None:
        """ingested_at must always be near now, regardless of event_time."""
        event = datetime.now(UTC) + timedelta(days=delta_days)
        record = make_bitemporal_record(event, "payload")
        delta = abs(datetime.now(UTC) - record.ingested_at)
        assert delta <= _INGESTION_TOLERANCE


# ---------------------------------------------------------------------------
# event_time is preserved
# ---------------------------------------------------------------------------

class TestEventTimePreserved:
    """event_time must be stored exactly as passed — no mutation."""

    def test_event_time_preserved_past(self) -> None:
        t = datetime(2019, 3, 15, 12, 34, 56, tzinfo=UTC)
        record = make_bitemporal_record(t, "old event")
        assert record.event_time == t

    def test_event_time_preserved_future(self) -> None:
        t = datetime(2030, 12, 31, 23, 59, 59, tzinfo=UTC)
        record = make_bitemporal_record(t, "future event")
        assert record.event_time == t

    def test_event_time_preserved_now(self) -> None:
        t = datetime.now(UTC)
        record = make_bitemporal_record(t, "now event")
        assert record.event_time == t

    def test_event_time_preserves_microseconds(self) -> None:
        t = datetime(2025, 7, 4, 8, 0, 0, 123456, tzinfo=UTC)
        record = make_bitemporal_record(t, "precise")
        assert record.event_time == t

    @pytest.mark.parametrize("ts", [
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2026, 3, 31, 0, 0, 0, tzinfo=UTC),
        datetime(2099, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    ])
    def test_event_time_preserved_parametrized(self, ts: datetime) -> None:
        record = make_bitemporal_record(ts, "payload")
        assert record.event_time == ts

    def test_event_time_is_not_ingested_at(self) -> None:
        """The two time axes must remain independent."""
        t = datetime(2024, 1, 1, tzinfo=UTC)
        record = make_bitemporal_record(t, "payload")
        assert record.event_time is not record.ingested_at


# ---------------------------------------------------------------------------
# content is preserved
# ---------------------------------------------------------------------------

class TestContentPreserved:
    """content must be stored verbatim."""

    def test_simple_string(self) -> None:
        record = make_bitemporal_record(_fixed_event_time(), "hello world")
        assert record.content == "hello world"

    def test_empty_string(self) -> None:
        record = make_bitemporal_record(_fixed_event_time(), "")
        assert record.content == ""

    def test_multiline_string(self) -> None:
        payload = "line one\nline two\nline three"
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload

    def test_unicode_content(self) -> None:
        payload = "こんにちは 🌍 Ünïcödé"
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload

    def test_json_serialised_content(self) -> None:
        payload = '{"key": "value", "num": 42}'
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload

    def test_large_content_preserved(self) -> None:
        payload = "x" * 100_000
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload
        assert len(record.content) == 100_000

    @pytest.mark.parametrize("payload", [
        "short",
        "with\nnewlines\n",
        "  leading and trailing spaces  ",
        "\t\ttabs\t\t",
        "null-bytes-adjacent \x00 char",
    ])
    def test_content_preserved_parametrized(self, payload: str) -> None:
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload


# ---------------------------------------------------------------------------
# Generic content round-trip
# ---------------------------------------------------------------------------

class TestContentRoundTrip:
    """Factory must preserve any content payload unchanged — no mutation,
    no truncation, no encoding side-effects.

    Each test follows the same round-trip pattern:
        payload → make_bitemporal_record(...) → record.content == payload
    """

    @pytest.mark.parametrize("payload", [
        # Plain ASCII
        "hello world",
        # Empty
        "",
        # Whitespace variants
        "   ",
        "\t\n\r",
        # Newlines embedded
        "first\nsecond\nthird",
        # Unicode: accents, CJK, emoji
        "café résumé",
        "日本語テスト",
        "🚀🌍🎯",
        # Null bytes and non-printable ASCII
        "\x00\x01\x1f",
        "before\x00after",
        # JSON-like strings
        '{"key": "value", "nums": [1, 2, 3]}',
        # Repeated characters (large-ish payload)
        "a" * 10_000,
        # Mixed special characters
        "line1\nline2\ttabbed\r\nwindows",
        # Escaped backslashes
        "path\\to\\file",
    ])
    def test_round_trip_preserves_payload(self, payload: str) -> None:
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload, (
            f"Content mutated: expected {payload!r}, got {record.content!r}"
        )

    def test_round_trip_does_not_share_identity_with_string_interning(self) -> None:
        """Even if Python interns short strings, the *value* must be equal."""
        payload = "interned"
        record = make_bitemporal_record(_fixed_event_time(), payload)
        # Value equality is the contract; identity is irrelevant but value must hold.
        assert record.content == payload

    def test_round_trip_does_not_strip_leading_trailing_whitespace(self) -> None:
        payload = "  padded  "
        record = make_bitemporal_record(_fixed_event_time(), payload)
        assert record.content == payload

    def test_round_trip_does_not_normalise_line_endings(self) -> None:
        crlf = "line1\r\nline2\r\n"
        record = make_bitemporal_record(_fixed_event_time(), crlf)
        assert record.content == crlf

    def test_round_trip_independent_of_event_time(self) -> None:
        """Changing event_time must not affect the content stored."""
        payload = "generic payload"
        t1 = datetime(2020, 1, 1, tzinfo=UTC)
        t2 = datetime(2030, 12, 31, tzinfo=UTC)
        r1 = make_bitemporal_record(t1, payload)
        r2 = make_bitemporal_record(t2, payload)
        assert r1.content == payload
        assert r2.content == payload
        assert r1.content == r2.content

    def test_round_trip_independent_of_ingestion_override(self) -> None:
        """Supplying an ingestion_time override must not touch the content."""
        payload = "unchanged payload"
        override = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        record = make_bitemporal_record(
            _fixed_event_time(), payload, ingestion_time=override
        )
        assert record.content == payload

    def test_two_records_with_different_payloads_do_not_share_content(self) -> None:
        r1 = make_bitemporal_record(_fixed_event_time(), "alpha")
        r2 = make_bitemporal_record(_fixed_event_time(), "beta")
        assert r1.content == "alpha"
        assert r2.content == "beta"
        assert r1.content != r2.content


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_bitemporal_record(self) -> None:
        record = make_bitemporal_record(_fixed_event_time(), "payload")
        assert isinstance(record, BiTemporalRecord)

    def test_each_call_returns_new_instance(self) -> None:
        r1 = make_bitemporal_record(_fixed_event_time(), "payload")
        r2 = make_bitemporal_record(_fixed_event_time(), "payload")
        assert r1 is not r2


# ---------------------------------------------------------------------------
# Explicit ingestion_time override
# ---------------------------------------------------------------------------

class TestIngestionTimeOverride:
    """When ingestion_time is supplied it must be used verbatim as ingested_at."""

    _OVERRIDE = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    def test_explicit_ingestion_time_is_used(self) -> None:
        record = make_bitemporal_record(
            _fixed_event_time(), "payload", ingestion_time=self._OVERRIDE
        )
        assert record.ingested_at == self._OVERRIDE

    def test_explicit_ingestion_time_not_replaced_by_now(self) -> None:
        """Factory must not overwrite the override with datetime.now()."""
        override = datetime(2020, 1, 1, tzinfo=UTC)
        record = make_bitemporal_record(
            _fixed_event_time(), "payload", ingestion_time=override
        )
        # now() is years after 2020 — they cannot be equal
        assert record.ingested_at == override
        assert record.ingested_at != datetime.now(UTC)

    def test_override_preserves_microseconds(self) -> None:
        override = datetime(2025, 11, 3, 7, 30, 45, 987654, tzinfo=UTC)
        record = make_bitemporal_record(
            _fixed_event_time(), "payload", ingestion_time=override
        )
        assert record.ingested_at == override

    def test_override_can_be_in_the_future(self) -> None:
        future = datetime(2099, 12, 31, 23, 59, 59, tzinfo=UTC)
        record = make_bitemporal_record(
            _fixed_event_time(), "payload", ingestion_time=future
        )
        assert record.ingested_at == future

    def test_override_equal_to_event_time_allowed(self) -> None:
        """Both axes may share the same instant — factory must not prevent this."""
        same = datetime(2026, 3, 31, 0, 0, 0, tzinfo=UTC)
        record = make_bitemporal_record(same, "payload", ingestion_time=same)
        assert record.ingested_at == same
        assert record.event_time == same

    def test_override_none_falls_back_to_now(self) -> None:
        """Passing ingestion_time=None explicitly must behave like the default."""
        before = datetime.now(UTC)
        record = make_bitemporal_record(
            _fixed_event_time(), "payload", ingestion_time=None
        )
        after = datetime.now(UTC)
        assert before <= record.ingested_at <= after + timedelta(milliseconds=1)

    def test_event_time_unaffected_by_override(self) -> None:
        event = datetime(2023, 5, 20, tzinfo=UTC)
        override = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        record = make_bitemporal_record(event, "payload", ingestion_time=override)
        assert record.event_time == event

    def test_content_unaffected_by_override(self) -> None:
        record = make_bitemporal_record(
            _fixed_event_time(), "important payload", ingestion_time=self._OVERRIDE
        )
        assert record.content == "important payload"

    @pytest.mark.parametrize("override", [
        datetime(2000, 1, 1, tzinfo=UTC),
        datetime(2026, 3, 31, 0, 0, 0, tzinfo=UTC),
        datetime(2099, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    ])
    def test_override_parametrized(self, override: datetime) -> None:
        record = make_bitemporal_record(
            _fixed_event_time(), "payload", ingestion_time=override
        )
        assert record.ingested_at == override


# ---------------------------------------------------------------------------
# event_time is required
# ---------------------------------------------------------------------------

class TestEventTimeRequired:
    """event_time must be supplied; omitting it is a TypeError."""

    def test_no_arguments_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            make_bitemporal_record()  # type: ignore[call-arg]

    def test_content_only_kwarg_raises_type_error(self) -> None:
        """Passing only content (without event_time) must fail."""
        with pytest.raises(TypeError):
            make_bitemporal_record(content="hello")  # type: ignore[call-arg]

    def test_error_message_names_event_time(self) -> None:
        """The TypeError should identify event_time as the missing argument."""
        with pytest.raises(TypeError, match="event_time"):
            make_bitemporal_record(content="hello")  # type: ignore[call-arg]

    def test_content_alone_is_not_sufficient(self) -> None:
        """Even with ingestion_time supplied, event_time is still required."""
        override = datetime(2026, 1, 1, tzinfo=UTC)
        with pytest.raises(TypeError):
            make_bitemporal_record(  # type: ignore[call-arg]
                content="payload", ingestion_time=override
            )
