"""Round-trip serialization tests for MemoryType.

Each variant must survive the cycle:
    variant → str() → from_string() → variant
and the symmetric cycle:
    variant → .value → from_string() → variant
"""

import pytest

from shad.vault.shadow_index import MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_VARIANTS = list(MemoryType)


# ---------------------------------------------------------------------------
# str() round-trip  (variant → str(variant) → from_string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_str_round_trip(variant: MemoryType) -> None:
    """str(variant) produces the value string, which from_string parses back."""
    serialized = str(variant)
    restored = MemoryType.from_string(serialized)
    assert restored is variant


# ---------------------------------------------------------------------------
# .value round-trip  (variant → variant.value → from_string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_value_round_trip(variant: MemoryType) -> None:
    """variant.value is a valid from_string input that reconstructs the same variant."""
    restored = MemoryType.from_string(variant.value)
    assert restored is variant


# ---------------------------------------------------------------------------
# .name round-trip  (variant → variant.name → from_string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_name_round_trip(variant: MemoryType) -> None:
    """variant.name (upper-case identifier) is accepted by from_string."""
    restored = MemoryType.from_string(variant.name)
    assert restored is variant


# ---------------------------------------------------------------------------
# Case-insensitive name round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_lower_name_round_trip(variant: MemoryType) -> None:
    """Lower-cased member name also parses correctly."""
    restored = MemoryType.from_string(variant.name.lower())
    assert restored is variant


# ---------------------------------------------------------------------------
# str() and .value agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_str_equals_value(variant: MemoryType) -> None:
    """str(variant) must equal variant.value — both represent the serialised form."""
    assert str(variant) == variant.value


# ---------------------------------------------------------------------------
# Concrete expectations: value strings are stable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant,expected_value", [
    (MemoryType.WORKING,    "working_memory"),
    (MemoryType.SEMANTIC,   "semantic_memory"),
    (MemoryType.EPISODIC,   "episodic_memory"),
    (MemoryType.PROCEDURAL, "procedural_memory"),
])
def test_value_strings_are_stable(variant: MemoryType, expected_value: str) -> None:
    """The serialised string for each variant must not drift over time."""
    assert variant.value == expected_value
    assert str(variant) == expected_value


# ---------------------------------------------------------------------------
# All variants covered
# ---------------------------------------------------------------------------

def test_all_variants_have_unique_values() -> None:
    """No two variants share the same serialised string."""
    values = [v.value for v in MemoryType]
    assert len(values) == len(set(values))


def test_from_string_covers_all_variants() -> None:
    """from_string(v.value) is defined for every declared variant."""
    for variant in MemoryType:
        assert MemoryType.from_string(variant.value) is variant


# ---------------------------------------------------------------------------
# Invalid inputs raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "unknown_memory",
    "UNKNOWN",
    "",
    "   ",
    "working memory", # space instead of underscore
])
def test_invalid_string_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid MemoryType"):
        MemoryType.from_string(bad)
