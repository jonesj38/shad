"""Tests for verification layer.

Per SPEC.md Section 2.6:
- Import resolution: All imports resolve to existing files/symbols
- Syntax/parse: Code is syntactically valid
- Manifest integrity: No path traversal, no duplicates
- Type check: TypeScript errors via tsc --noEmit
- Unit tests: Tests pass
- Lint: Style conformance

Verification levels:
- off: No checks
- basic (default): Imports + syntax + manifest blocking
- build: Basic + typecheck blocking
- strict: Build + tests blocking
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.output.manifest import FileEntry, FileManifest
from shad.verification.layer import (
    CheckResult,
    CheckStatus,
    ErrorClassification,
    RepairAction,
    VerificationCheck,
    VerificationConfig,
    VerificationLayer,
    VerificationLevel,
    VerificationResult,
)


class TestVerificationConfig:
    """Tests for verification configuration."""

    def test_default_config(self) -> None:
        """Test default verification config."""
        config = VerificationConfig()
        assert config.level == VerificationLevel.BASIC

    def test_off_level(self) -> None:
        """Test off verification level."""
        config = VerificationConfig(level=VerificationLevel.OFF)
        assert config.should_run("syntax") is False
        assert config.should_run("import_resolution") is False

    def test_basic_level(self) -> None:
        """Test basic verification level."""
        config = VerificationConfig(level=VerificationLevel.BASIC)
        assert config.should_run("syntax") is True
        assert config.should_run("import_resolution") is True
        assert config.should_run("manifest_integrity") is True
        # Typecheck is advisory at basic level
        assert config.is_blocking("typecheck") is False

    def test_build_level(self) -> None:
        """Test build verification level."""
        config = VerificationConfig(level=VerificationLevel.BUILD)
        assert config.should_run("typecheck") is True
        assert config.is_blocking("typecheck") is True

    def test_strict_level(self) -> None:
        """Test strict verification level."""
        config = VerificationConfig(level=VerificationLevel.STRICT)
        assert config.is_blocking("unit_tests") is True
        assert config.is_blocking("lint") is True

    def test_per_check_override(self) -> None:
        """Test per-check configuration override."""
        config = VerificationConfig(
            level=VerificationLevel.BASIC,
            block_on=["typecheck"],  # Make typecheck blocking
            warn_only=["syntax"],  # Make syntax advisory
        )
        assert config.is_blocking("typecheck") is True
        assert config.is_blocking("syntax") is False


class TestVerificationCheck:
    """Tests for individual verification checks."""

    def test_check_result_structure(self) -> None:
        """Test check result structure."""
        result = CheckResult(
            check_name="syntax",
            status=CheckStatus.PASSED,
            message="All files are syntactically valid",
        )
        assert result.check_name == "syntax"
        assert result.status == CheckStatus.PASSED

    def test_check_result_with_errors(self) -> None:
        """Test check result with errors."""
        result = CheckResult(
            check_name="syntax",
            status=CheckStatus.FAILED,
            message="Syntax errors found",
            errors=["src/api.ts:10: Unexpected token"],
        )
        assert result.status == CheckStatus.FAILED
        assert len(result.errors) == 1


class TestVerificationLayer:
    """Tests for the verification layer."""

    @pytest.fixture
    def valid_manifest(self) -> FileManifest:
        """Create a valid manifest for testing."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User { id: string; name: string; }",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/api.ts",
            content="""import { User } from './types';
export function getUser(): User { return { id: '1', name: 'Test' }; }""",
            language="ts",
        ))
        return manifest

    @pytest.fixture
    def layer(self) -> VerificationLayer:
        """Create verification layer."""
        return VerificationLayer()

    @pytest.mark.asyncio
    async def test_verify_valid_manifest(
        self, layer: VerificationLayer, valid_manifest: FileManifest
    ) -> None:
        """Test verifying a valid manifest."""
        config = VerificationConfig(level=VerificationLevel.BASIC)
        result = await layer.verify(valid_manifest, config)

        assert result.passed is True
        assert len(result.blocking_failures) == 0

    @pytest.mark.asyncio
    async def test_verify_syntax_check(
        self, layer: VerificationLayer
    ) -> None:
        """Test syntax verification."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/api.ts",
            content="export function broken( { return; }",  # Invalid syntax
            language="ts",
        ))

        config = VerificationConfig(level=VerificationLevel.BASIC)
        result = await layer.verify(manifest, config)

        # Should have syntax error
        assert result.passed is False
        assert any("syntax" in c.check_name for c in result.failed_checks)

    @pytest.mark.asyncio
    async def test_verify_import_resolution(
        self, layer: VerificationLayer
    ) -> None:
        """Test import resolution verification."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/api.ts",
            content="""import { User } from './types';  // types.ts doesn't exist
export function getUser(): User {}""",
            language="ts",
        ))

        config = VerificationConfig(level=VerificationLevel.BASIC)
        result = await layer.verify(manifest, config)

        assert result.passed is False
        assert any("import" in c.check_name for c in result.failed_checks)

    @pytest.mark.asyncio
    async def test_verify_manifest_integrity(
        self, layer: VerificationLayer
    ) -> None:
        """Test manifest integrity verification."""
        manifest = FileManifest(run_id="test-123")
        # Add duplicate paths manually (bypassing add_file)
        manifest.files = [
            FileEntry(path="src/index.ts", content="1", language="ts"),
            FileEntry(path="src/index.ts", content="2", language="ts"),
        ]

        config = VerificationConfig(level=VerificationLevel.BASIC)
        result = await layer.verify(manifest, config)

        assert result.passed is False
        assert any("manifest" in c.check_name or "integrity" in c.check_name
                   for c in result.failed_checks)

    @pytest.mark.asyncio
    async def test_off_level_skips_all(
        self, layer: VerificationLayer
    ) -> None:
        """Test that off level skips all checks."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/broken.ts",
            content="this is not valid typescript {{{{",
            language="ts",
        ))

        config = VerificationConfig(level=VerificationLevel.OFF)
        result = await layer.verify(manifest, config)

        # Should pass because no checks run
        assert result.passed is True
        assert len(result.checks) == 0

    @pytest.mark.asyncio
    async def test_advisory_failures_dont_block(
        self, layer: VerificationLayer, valid_manifest: FileManifest
    ) -> None:
        """Test that advisory failures don't block verification."""
        # Add file with potential lint issues but valid syntax/imports
        valid_manifest.add_file(FileEntry(
            path="src/messy.ts",
            content="var x=1;  // var instead of const",
            language="ts",
        ))

        # With basic level, lint is advisory
        config = VerificationConfig(level=VerificationLevel.BASIC)
        result = await layer.verify(valid_manifest, config)

        # Should still pass (lint is advisory at basic)
        assert result.passed is True


class TestErrorClassification:
    """Tests for error classification."""

    def test_classify_syntax_error(self) -> None:
        """Test classifying syntax errors."""
        classification = ErrorClassification.classify(
            check_name="syntax",
            error_message="Unexpected token",
        )
        assert classification == ErrorClassification.SYNTAX

    def test_classify_type_error(self) -> None:
        """Test classifying type errors."""
        classification = ErrorClassification.classify(
            check_name="typecheck",
            error_message="Type 'string' is not assignable to type 'number'",
        )
        assert classification == ErrorClassification.TYPE_ERROR

    def test_classify_import_error(self) -> None:
        """Test classifying import errors."""
        classification = ErrorClassification.classify(
            check_name="import_resolution",
            error_message="Module not found",
        )
        assert classification == ErrorClassification.IMPORT_ERROR


class TestRepairAction:
    """Tests for repair actions."""

    def test_repair_action_for_syntax(self) -> None:
        """Test repair action for syntax errors."""
        action = RepairAction.for_classification(ErrorClassification.SYNTAX)
        assert action.scope == "local"
        assert action.max_retries == 2

    def test_repair_action_for_type_error(self) -> None:
        """Test repair action for type errors."""
        action = RepairAction.for_classification(ErrorClassification.TYPE_ERROR)
        assert action.scope == "local"
        assert action.needs_sibling_context is True

    def test_repair_action_for_integration(self) -> None:
        """Test repair action for integration errors."""
        action = RepairAction.for_classification(ErrorClassification.INTEGRATION)
        assert action.scope == "escalate"


class TestVerificationResult:
    """Tests for verification results."""

    def test_result_structure(self) -> None:
        """Test verification result structure."""
        result = VerificationResult(
            manifest_id="test-123",
            passed=True,
            checks=[
                CheckResult("syntax", CheckStatus.PASSED, "OK"),
                CheckResult("import_resolution", CheckStatus.PASSED, "OK"),
            ],
        )

        assert result.passed is True
        assert len(result.checks) == 2

    def test_result_failed_checks(self) -> None:
        """Test getting failed checks from result."""
        result = VerificationResult(
            manifest_id="test-123",
            passed=False,
            checks=[
                CheckResult("syntax", CheckStatus.PASSED, "OK"),
                CheckResult("import_resolution", CheckStatus.FAILED, "Missing import"),
            ],
        )

        failed = result.failed_checks
        assert len(failed) == 1
        assert failed[0].check_name == "import_resolution"

    def test_result_blocking_failures(self) -> None:
        """Test identifying blocking failures."""
        result = VerificationResult(
            manifest_id="test-123",
            passed=False,
            checks=[
                CheckResult("syntax", CheckStatus.FAILED, "Error", is_blocking=True),
                CheckResult("lint", CheckStatus.FAILED, "Warning", is_blocking=False),
            ],
        )

        blocking = result.blocking_failures
        assert len(blocking) == 1
        assert blocking[0].check_name == "syntax"


class TestVerificationIntegration:
    """Integration tests for verification."""

    @pytest.mark.asyncio
    async def test_full_verification_workflow(self) -> None:
        """Test complete verification workflow."""
        layer = VerificationLayer()

        # Create valid manifest
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User { id: string; }",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/service.ts",
            content="""import { User } from './types';
export class UserService {
    getUser(id: string): User {
        return { id };
    }
}""",
            language="ts",
        ))

        # Verify with basic level
        config = VerificationConfig(level=VerificationLevel.BASIC)
        result = await layer.verify(manifest, config)

        assert result.passed is True
        assert len(result.blocking_failures) == 0
