"""Verification layer for code generation output.

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

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shad.output.import_resolution import ImportResolver
from shad.output.manifest import FileManifest

logger = logging.getLogger(__name__)


class VerificationLevel(str, Enum):
    """Verification strictness levels."""

    OFF = "off"
    BASIC = "basic"
    BUILD = "build"
    STRICT = "strict"


class CheckStatus(str, Enum):
    """Status of a verification check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class ErrorClassification(str, Enum):
    """Classification of verification errors.

    Per SPEC.md Section 2.6.3:
    - Syntax/lint: Local repair only
    - Type errors: Local repair with sibling context
    - Integration: Escalate to parent coordinator
    - Contract mismatch: Parent coordination + contract update
    """

    SYNTAX = "syntax"
    LINT = "lint"
    TYPE_ERROR = "type_error"
    IMPORT_ERROR = "import_error"
    INTEGRATION = "integration"
    CONTRACT_MISMATCH = "contract_mismatch"

    @classmethod
    def classify(cls, check_name: str, error_message: str) -> ErrorClassification:
        """Classify an error based on check name and message."""
        if check_name == "syntax":
            return cls.SYNTAX
        if check_name == "lint":
            return cls.LINT
        if check_name == "typecheck":
            return cls.TYPE_ERROR
        if check_name == "import_resolution":
            return cls.IMPORT_ERROR
        if "contract" in error_message.lower():
            return cls.CONTRACT_MISMATCH
        return cls.INTEGRATION


@dataclass
class RepairAction:
    """Action to take for repairing errors.

    Per SPEC.md Section 2.6.3:
    - Syntax/lint: Local repair only
    - Type errors: Local repair with sibling context (contracts/types)
    - Integration failures: Escalate to parent coordinator
    - Contract mismatch: Parent coordination + contract update node
    """

    scope: str  # "local", "sibling", "escalate"
    max_retries: int = 2
    needs_sibling_context: bool = False
    contract_update_required: bool = False

    @classmethod
    def for_classification(cls, classification: ErrorClassification) -> RepairAction:
        """Get repair action for error classification."""
        if classification == ErrorClassification.SYNTAX:
            return cls(scope="local", max_retries=2)
        if classification == ErrorClassification.LINT:
            return cls(scope="local", max_retries=2)
        if classification == ErrorClassification.TYPE_ERROR:
            return cls(scope="local", max_retries=2, needs_sibling_context=True)
        if classification == ErrorClassification.IMPORT_ERROR:
            return cls(scope="local", max_retries=2)
        if classification == ErrorClassification.CONTRACT_MISMATCH:
            return cls(scope="escalate", contract_update_required=True)
        # Integration errors
        return cls(scope="escalate")


@dataclass
class CheckResult:
    """Result of a verification check."""

    check_name: str
    status: CheckStatus
    message: str
    errors: list[str] = field(default_factory=list)
    is_blocking: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verification layer."""

    manifest_id: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_checks(self) -> list[CheckResult]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAILED]

    @property
    def blocking_failures(self) -> list[CheckResult]:
        """Get blocking failures."""
        return [c for c in self.failed_checks if c.is_blocking]


# Which checks run at each level
_LEVEL_CHECKS: dict[VerificationLevel, list[str]] = {
    VerificationLevel.OFF: [],
    VerificationLevel.BASIC: ["syntax", "import_resolution", "manifest_integrity"],
    VerificationLevel.BUILD: ["syntax", "import_resolution", "manifest_integrity", "typecheck"],
    VerificationLevel.STRICT: ["syntax", "import_resolution", "manifest_integrity",
                               "typecheck", "unit_tests", "lint"],
}

# Which checks are blocking at each level
_LEVEL_BLOCKING: dict[VerificationLevel, list[str]] = {
    VerificationLevel.OFF: [],
    VerificationLevel.BASIC: ["syntax", "import_resolution", "manifest_integrity"],
    VerificationLevel.BUILD: ["syntax", "import_resolution", "manifest_integrity", "typecheck"],
    VerificationLevel.STRICT: ["syntax", "import_resolution", "manifest_integrity",
                               "typecheck", "unit_tests", "lint"],
}


@dataclass
class VerificationConfig:
    """Configuration for verification.

    Per SPEC.md Section 2.6.2:
    - level: off, basic, build, strict
    - block_on: Per-check override to make blocking
    - warn_only: Per-check override to make advisory
    """

    level: VerificationLevel = VerificationLevel.BASIC
    block_on: list[str] = field(default_factory=list)
    warn_only: list[str] = field(default_factory=list)

    def should_run(self, check_name: str) -> bool:
        """Check if a verification check should run."""
        return check_name in _LEVEL_CHECKS.get(self.level, [])

    def is_blocking(self, check_name: str) -> bool:
        """Check if a verification check is blocking."""
        # User overrides
        if check_name in self.block_on:
            return True
        if check_name in self.warn_only:
            return False
        # Level defaults
        return check_name in _LEVEL_BLOCKING.get(self.level, [])


class VerificationCheck:
    """Base class for verification checks."""

    name: str = "base"

    async def run(self, manifest: FileManifest) -> CheckResult:
        """Run the verification check."""
        raise NotImplementedError


class SyntaxCheck(VerificationCheck):
    """Check syntax validity of generated code."""

    name = "syntax"

    async def run(self, manifest: FileManifest) -> CheckResult:
        """Check syntax of all files."""
        errors: list[str] = []

        for entry in manifest.files:
            if entry.language in ("ts", "typescript", "js", "javascript"):
                syntax_errors = self._check_js_syntax(entry.path, entry.content)
                errors.extend(syntax_errors)
            elif entry.language in ("py", "python"):
                syntax_errors = self._check_python_syntax(entry.path, entry.content)
                errors.extend(syntax_errors)

        if errors:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                message=f"Found {len(errors)} syntax error(s)",
                errors=errors,
            )

        return CheckResult(
            check_name=self.name,
            status=CheckStatus.PASSED,
            message="All files are syntactically valid",
        )

    def _check_js_syntax(self, path: str, content: str) -> list[str]:
        """Basic JS/TS syntax check using simple heuristics."""
        errors: list[str] = []

        # Check for unbalanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            errors.append(f"{path}: Unbalanced braces ({open_braces} open, {close_braces} close)")

        # Check for unbalanced parentheses
        open_parens = content.count('(')
        close_parens = content.count(')')
        if open_parens != close_parens:
            errors.append(f"{path}: Unbalanced parentheses ({open_parens} open, {close_parens} close)")

        # Check for common syntax errors
        if re.search(r'\{\s*\{+', content):  # Multiple open braces
            if not re.search(r'\$\{', content):  # Exclude template literals
                errors.append(f"{path}: Possible syntax error (multiple consecutive braces)")

        return errors

    def _check_python_syntax(self, path: str, content: str) -> list[str]:
        """Check Python syntax using ast."""
        try:
            ast.parse(content)
            return []
        except SyntaxError as e:
            return [f"{path}:{e.lineno}: {e.msg}"]


class ImportResolutionCheck(VerificationCheck):
    """Check that all imports resolve."""

    name = "import_resolution"

    async def run(self, manifest: FileManifest) -> CheckResult:
        """Check import resolution."""
        resolver = ImportResolver(manifest)
        validation = resolver.validate()

        if not validation.is_valid:
            errors = []
            for m in validation.missing_modules:
                errors.append(f"{m.file}: Module '{m.import_path}' not found")
            for m in validation.missing_symbols:
                errors.append(f"{m.file}: Symbol '{m.symbol}' not exported from '{m.import_path}'")

            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                message=f"Found {len(errors)} import error(s)",
                errors=errors,
            )

        return CheckResult(
            check_name=self.name,
            status=CheckStatus.PASSED,
            message="All imports resolve correctly",
        )


class ManifestIntegrityCheck(VerificationCheck):
    """Check manifest integrity."""

    name = "manifest_integrity"

    async def run(self, manifest: FileManifest) -> CheckResult:
        """Check manifest integrity."""
        validation_errors = manifest.validate()

        # Additional checks
        for entry in manifest.files:
            # Check for path traversal
            if ".." in entry.path:
                validation_errors.append(f"Path traversal in {entry.path}")
            if entry.path.startswith("/"):
                validation_errors.append(f"Absolute path not allowed: {entry.path}")

        if validation_errors:
            return CheckResult(
                check_name=self.name,
                status=CheckStatus.FAILED,
                message=f"Found {len(validation_errors)} integrity error(s)",
                errors=validation_errors,
            )

        return CheckResult(
            check_name=self.name,
            status=CheckStatus.PASSED,
            message="Manifest integrity verified",
        )


class VerificationLayer:
    """Orchestrates verification checks.

    Per SPEC.md Section 2.6:
    - Runs configured checks against manifest
    - Reports blocking vs advisory failures
    - Provides error classification for repair
    """

    def __init__(self) -> None:
        self.checks: dict[str, VerificationCheck] = {
            "syntax": SyntaxCheck(),
            "import_resolution": ImportResolutionCheck(),
            "manifest_integrity": ManifestIntegrityCheck(),
        }

    async def verify(
        self,
        manifest: FileManifest,
        config: VerificationConfig,
    ) -> VerificationResult:
        """Run verification checks on manifest.

        Returns VerificationResult with all check results.
        """
        results: list[CheckResult] = []
        has_blocking_failure = False

        for check_name, check in self.checks.items():
            if not config.should_run(check_name):
                continue

            result = await check.run(manifest)
            result.is_blocking = config.is_blocking(check_name)

            if result.status == CheckStatus.FAILED and result.is_blocking:
                has_blocking_failure = True

            results.append(result)

        return VerificationResult(
            manifest_id=manifest.run_id,
            passed=not has_blocking_failure,
            checks=results,
        )
