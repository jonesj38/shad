"""Tests for the memory taxonomy — MemoryType enum, MEMORY_TAG_TAXONOMY,
MemoryState, and TransitionType."""

import pytest

from shad.vault.shadow_index import MEMORY_TAG_TAXONOMY, MemoryType
from shad.vault.taxonomy import MemoryState, MemoryTypeEnum, StateMachine, TransitionType


# ---------------------------------------------------------------------------
# MemoryType enum
# ---------------------------------------------------------------------------

class TestMemoryTypeEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in MemoryType}
        assert names == {"WORKING", "SEMANTIC", "EPISODIC", "PROCEDURAL"}

    def test_values_are_snake_case_strings(self) -> None:
        for member in MemoryType:
            assert "_memory" in member.value

    def test_str_returns_value(self) -> None:
        assert str(MemoryType.SEMANTIC) == "semantic_memory"
        assert str(MemoryType.EPISODIC) == "episodic_memory"
        assert str(MemoryType.WORKING) == "working_memory"
        assert str(MemoryType.PROCEDURAL) == "procedural_memory"

    @pytest.mark.parametrize("value,expected", [
        ("semantic_memory", MemoryType.SEMANTIC),
        ("episodic_memory", MemoryType.EPISODIC),
        ("working_memory", MemoryType.WORKING),
        ("procedural_memory", MemoryType.PROCEDURAL),
        # name-form (case-insensitive)
        ("SEMANTIC", MemoryType.SEMANTIC),
        ("episodic", MemoryType.EPISODIC),
    ])
    def test_from_string_roundtrip(self, value: str, expected: MemoryType) -> None:
        assert MemoryType.from_string(value) is expected

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid MemoryType"):
            MemoryType.from_string("unknown_memory")

    @pytest.mark.parametrize("bad", ["", "   ", "\t", "\n  \t"])
    def test_from_string_empty_or_whitespace_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid MemoryType"):
            MemoryType.from_string(bad)

    def test_from_string_none_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid MemoryType"):
            MemoryType.from_string(None)  # type: ignore[arg-type]

    def test_all_types_returns_all_variants(self) -> None:
        all_types = MemoryType.all_types()
        assert set(all_types) == set(MemoryType)
        assert len(all_types) == len(MemoryType)


# ---------------------------------------------------------------------------
# MemoryTypeEnum (taxonomy.py)
# ---------------------------------------------------------------------------


class TestMemoryTypeEnumTaxonomy:
    _EXPECTED_NAMES = {"WORKING", "EPISODIC", "SEMANTIC", "PROCEDURAL"}
    _EXPECTED_VALUES = {"working", "episodic", "semantic", "procedural"}

    def test_contains_exactly_four_variants(self) -> None:
        assert len(MemoryTypeEnum) == 4

    def test_member_names_are_exactly_expected(self) -> None:
        assert {m.name for m in MemoryTypeEnum} == self._EXPECTED_NAMES

    def test_member_values_are_exactly_expected(self) -> None:
        assert {m.value for m in MemoryTypeEnum} == self._EXPECTED_VALUES

    def test_working_present(self) -> None:
        assert MemoryTypeEnum.WORKING.value == "working"

    def test_episodic_present(self) -> None:
        assert MemoryTypeEnum.EPISODIC.value == "episodic"

    def test_semantic_present(self) -> None:
        assert MemoryTypeEnum.SEMANTIC.value == "semantic"

    def test_procedural_present(self) -> None:
        assert MemoryTypeEnum.PROCEDURAL.value == "procedural"

    def test_no_unexpected_variants(self) -> None:
        extra = {m.name for m in MemoryTypeEnum} - self._EXPECTED_NAMES
        assert extra == set(), f"Unexpected variants: {extra}"


# ---------------------------------------------------------------------------
# MEMORY_TAG_TAXONOMY frozenset
# ---------------------------------------------------------------------------

class TestMemoryTagTaxonomy:
    def test_is_frozenset(self) -> None:
        assert isinstance(MEMORY_TAG_TAXONOMY, frozenset)

    def test_contains_all_memory_type_values(self) -> None:
        for member in MemoryType:
            assert member.value in MEMORY_TAG_TAXONOMY, (
                f"Expected {member.value!r} in MEMORY_TAG_TAXONOMY"
            )

    def test_contains_known_source_types(self) -> None:
        for source in ("github", "url", "folder", "feed", "sessions"):
            assert source in MEMORY_TAG_TAXONOMY

    def test_contains_feature_area_labels(self) -> None:
        for area in ("retrieval", "sandbox", "engine", "cli", "verification"):
            assert area in MEMORY_TAG_TAXONOMY

    def test_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            MEMORY_TAG_TAXONOMY.add("new_tag")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MemoryState enum (from taxonomy.py)
# ---------------------------------------------------------------------------


class TestMemoryStateEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in MemoryState}
        assert names == {"EPISODIC", "SEMANTIC", "PROCEDURAL", "ARCHIVAL", "DORMANT"}

    def test_str_returns_value(self) -> None:
        assert str(MemoryState.EPISODIC) == "episodic"
        assert str(MemoryState.SEMANTIC) == "semantic"
        assert str(MemoryState.PROCEDURAL) == "procedural"
        assert str(MemoryState.ARCHIVAL) == "archival"
        assert str(MemoryState.DORMANT) == "dormant"

    @pytest.mark.parametrize("value,expected", [
        ("episodic", MemoryState.EPISODIC),
        ("semantic", MemoryState.SEMANTIC),
        ("procedural", MemoryState.PROCEDURAL),
        ("archival", MemoryState.ARCHIVAL),
        ("dormant", MemoryState.DORMANT),
        # name-form (case-insensitive)
        ("EPISODIC", MemoryState.EPISODIC),
        ("Semantic", MemoryState.SEMANTIC),
    ])
    def test_from_string_roundtrip(self, value: str, expected: MemoryState) -> None:
        assert MemoryState.from_string(value) is expected

    def test_from_string_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid MemoryState"):
            MemoryState.from_string("unknown_state")

    def test_from_string_error_lists_valid_values(self) -> None:
        with pytest.raises(ValueError, match='"episodic"'):
            MemoryState.from_string("bad")

    @pytest.mark.parametrize("bad", ["", "   ", "\t", "\n  \t"])
    def test_from_string_empty_or_whitespace_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid MemoryState"):
            MemoryState.from_string(bad)

    def test_from_string_none_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid MemoryState"):
            MemoryState.from_string(None)  # type: ignore[arg-type]

    def test_active_states_returns_retrieval_participants(self) -> None:
        active = MemoryState.active_states()
        assert set(active) == {MemoryState.EPISODIC, MemoryState.SEMANTIC, MemoryState.PROCEDURAL}

    def test_active_states_excludes_archival_and_dormant(self) -> None:
        active = MemoryState.active_states()
        assert MemoryState.ARCHIVAL not in active
        assert MemoryState.DORMANT not in active

    def test_values_are_lowercase_strings(self) -> None:
        for member in MemoryState:
            assert member.value == member.value.lower()

    def test_from_string_strips_whitespace(self) -> None:
        assert MemoryState.from_string("  episodic  ") is MemoryState.EPISODIC


# ---------------------------------------------------------------------------
# TransitionType enum (from taxonomy.py)
# ---------------------------------------------------------------------------


class TestTransitionTypeEnum:
    def test_all_variants_present(self) -> None:
        names = {m.name for m in TransitionType}
        assert names == {"PROMOTE", "DEMOTE", "MERGE", "SPLIT"}

    def test_str_returns_value(self) -> None:
        assert str(TransitionType.PROMOTE) == "promote"
        assert str(TransitionType.DEMOTE) == "demote"
        assert str(TransitionType.MERGE) == "merge"
        assert str(TransitionType.SPLIT) == "split"

    @pytest.mark.parametrize("value,expected", [
        ("promote", TransitionType.PROMOTE),
        ("demote", TransitionType.DEMOTE),
        ("merge", TransitionType.MERGE),
        ("split", TransitionType.SPLIT),
        # name-form (case-insensitive)
        ("PROMOTE", TransitionType.PROMOTE),
        ("Demote", TransitionType.DEMOTE),
    ])
    def test_from_string_roundtrip(self, value: str, expected: TransitionType) -> None:
        assert TransitionType.from_string(value) is expected

    def test_from_string_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid TransitionType"):
            TransitionType.from_string("elevate")

    def test_from_string_error_lists_valid_values(self) -> None:
        with pytest.raises(ValueError, match='"promote"'):
            TransitionType.from_string("bad")

    def test_from_string_strips_whitespace(self) -> None:
        assert TransitionType.from_string("  merge  ") is TransitionType.MERGE

    @pytest.mark.parametrize("bad", ["", "   ", "\t", "\n  \t"])
    def test_from_string_empty_or_whitespace_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid TransitionType"):
            TransitionType.from_string(bad)

    def test_from_string_none_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid TransitionType"):
            TransitionType.from_string(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_procedural_cannot_be_promoted(self) -> None:
        """PROCEDURAL is the highest active state; promote must return failure."""
        sm = StateMachine()
        # No valid promote target exists for PROCEDURAL — try every possible target.
        for target in MemoryState:
            result = sm.validate(MemoryState.PROCEDURAL, TransitionType.PROMOTE, target)
            assert not result.ok, (
                f"Expected promote from PROCEDURAL to {target.value!r} to fail, but got ok=True"
            )
            assert result.reason != "ok"
