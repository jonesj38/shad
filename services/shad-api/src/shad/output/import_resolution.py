"""Two-pass import resolution for code generation.

Per SPEC.md Section 2.7.2:
- Pass 1: Build export index (symbol → file mapping) via early contracts node
- Pass 2: Generate implementations using export index as ground truth
- Post-generation validation: Check all imports resolve to existing files/symbols
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shad.output.manifest import FileManifest, build_export_index

logger = logging.getLogger(__name__)


@dataclass
class MissingImport:
    """A missing import detected during validation."""

    file: str
    import_path: str
    symbol: str | None = None
    reason: str = ""


@dataclass
class ImportValidation:
    """Result of import validation.

    Per SPEC.md Section 2.7.2:
    - missing_modules: Imports that don't resolve to files in manifest
    - missing_symbols: Symbols not exported from their modules
    """

    is_valid: bool
    missing_modules: list[MissingImport] = field(default_factory=list)
    missing_symbols: list[MissingImport] = field(default_factory=list)

    def to_report(self) -> str:
        """Convert to human-readable report."""
        if self.is_valid:
            return "All imports resolve correctly."

        lines = ["Import Resolution Errors:"]

        if self.missing_modules:
            lines.append("\nMissing Modules:")
            for m in self.missing_modules:
                lines.append(f"  - {m.file}: import from '{m.import_path}' - {m.reason}")

        if self.missing_symbols:
            lines.append("\nMissing Symbols:")
            for m in self.missing_symbols:
                lines.append(f"  - {m.file}: '{m.symbol}' from '{m.import_path}' - {m.reason}")

        return "\n".join(lines)


class ImportResolver:
    """Resolves and validates imports in a file manifest.

    Per SPEC.md Section 2.7.2:
    - Validates all imports resolve to existing files/symbols
    - Handles relative imports and path aliases
    - Reports missing modules and symbols
    """

    def __init__(
        self,
        manifest: FileManifest,
        path_aliases: dict[str, str] | None = None,
    ) -> None:
        self.manifest = manifest
        self.path_aliases = path_aliases or {}

        # Build export index
        self.export_index = build_export_index(manifest)

        # Build file path index
        self.file_paths = {f.path for f in manifest.files}
        # Also include paths without extension
        self.file_paths_no_ext = {
            p.rsplit('.', 1)[0] if '.' in p else p
            for p in self.file_paths
        }

    def validate(self) -> ImportValidation:
        """Validate all imports in the manifest.

        Returns ImportValidation with any errors found.
        """
        missing_modules: list[MissingImport] = []
        missing_symbols: list[MissingImport] = []

        for file_entry in self.manifest.files:
            if file_entry.language not in ("ts", "typescript", "js", "javascript"):
                continue

            imports = self._extract_imports(file_entry.content)

            for import_path, symbols in imports:
                # Resolve import path
                resolved = self.resolve_import_path(file_entry.path, import_path)

                # Check if module exists
                if not self._module_exists(resolved):
                    missing_modules.append(MissingImport(
                        file=file_entry.path,
                        import_path=import_path,
                        reason="Module not found in manifest",
                    ))
                    continue

                # Check if symbols exist
                for symbol in symbols:
                    if not self._symbol_exists(symbol, resolved):
                        missing_symbols.append(MissingImport(
                            file=file_entry.path,
                            import_path=import_path,
                            symbol=symbol,
                            reason="Symbol not exported from module",
                        ))

        is_valid = len(missing_modules) == 0 and len(missing_symbols) == 0

        return ImportValidation(
            is_valid=is_valid,
            missing_modules=missing_modules,
            missing_symbols=missing_symbols,
        )

    def resolve_import_path(self, from_file: str, import_path: str) -> str:
        """Resolve an import path to a file path.

        Handles:
        - Relative imports (./foo, ../foo)
        - Path aliases (@/foo)
        - Implicit extensions (.ts, .js)
        """
        # Apply path aliases
        for alias, replacement in self.path_aliases.items():
            if import_path.startswith(alias):
                import_path = replacement + import_path[len(alias):]
                # Now treat it as absolute from src root
                return self._normalize_path(import_path)

        # Handle relative imports
        if import_path.startswith('.'):
            from_dir = Path(from_file).parent
            # Use pathlib to resolve .. properly
            parts = list(from_dir.parts) + import_path.split('/')
            resolved_parts: list[str] = []
            for part in parts:
                if part == '..':
                    if resolved_parts:
                        resolved_parts.pop()
                elif part != '.':
                    resolved_parts.append(part)
            resolved = '/'.join(resolved_parts)
            return self._normalize_path(resolved)

        # Non-relative import (node_modules, etc.)
        return import_path

    def _normalize_path(self, path: str) -> str:
        """Normalize path to match manifest format."""
        # Remove leading ./
        if path.startswith('./'):
            path = path[2:]

        # Try with extensions
        for ext in ['', '.ts', '.tsx', '.js', '.jsx']:
            candidate = f"{path}{ext}"
            if candidate in self.file_paths:
                return candidate

        return path

    def _module_exists(self, resolved_path: str) -> bool:
        """Check if a module exists in the manifest."""
        # Check exact path
        if resolved_path in self.file_paths:
            return True

        # Check without extension
        base_path = resolved_path.rsplit('.', 1)[0] if '.' in resolved_path else resolved_path
        if base_path in self.file_paths_no_ext:
            return True

        # Check with common extensions
        for ext in ['.ts', '.tsx', '.js', '.jsx']:
            if f"{base_path}{ext}" in self.file_paths:
                return True

        return False

    def _symbol_exists(self, symbol: str, module_path: str) -> bool:
        """Check if a symbol is exported from a module."""
        # Check export index
        if symbol in self.export_index.exports:
            return True

        # Check in the actual file
        base_path = module_path.rsplit('.', 1)[0] if '.' in module_path else module_path
        for file_entry in self.manifest.files:
            file_base = file_entry.path.rsplit('.', 1)[0]
            if file_base == base_path or file_entry.path == module_path:
                # Check if symbol is exported
                if self._is_exported(symbol, file_entry.content):
                    return True

        return False

    def _is_exported(self, symbol: str, content: str) -> bool:
        """Check if a symbol is exported in the content."""
        # Check for named export
        patterns = [
            rf'export\s+(?:interface|type|class|enum|const|let|var|function|async\s+function)\s+{re.escape(symbol)}\b',
            rf'export\s+\{{\s*[^}}]*\b{re.escape(symbol)}\b[^}}]*\}}',
        ]

        for pattern in patterns:
            if re.search(pattern, content):
                return True

        return False

    def _extract_imports(self, content: str) -> list[tuple[str, list[str]]]:
        """Extract imports from TypeScript/JavaScript content.

        Returns list of (import_path, symbols) tuples.
        """
        imports: list[tuple[str, list[str]]] = []

        # Match named imports: import { A, B } from 'path'
        named_pattern = r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(named_pattern, content):
            symbols_str, path = match.groups()
            symbols = [s.strip().split(' as ')[0].strip()
                       for s in symbols_str.split(',')]
            symbols = [s for s in symbols if s]
            imports.append((path, symbols))

        # Match default imports: import Name from 'path'
        default_pattern = r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(default_pattern, content):
            name, path = match.groups()
            # Skip if it's part of a named import we already caught
            if not any(p == path for p, _ in imports):
                imports.append((path, [name]))

        # Match combined: import Name, { A, B } from 'path'
        combined_pattern = r"import\s+(\w+)\s*,\s*\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(combined_pattern, content):
            default_name, named_str, path = match.groups()
            named = [s.strip().split(' as ')[0].strip()
                     for s in named_str.split(',')]
            symbols = [default_name] + named
            # Update existing or add new
            existing = next((i for i, (p, _) in enumerate(imports) if p == path), None)
            if existing is not None:
                imports[existing] = (path, list(set(imports[existing][1] + symbols)))
            else:
                imports.append((path, symbols))

        return imports


class TypeContractsBuilder:
    """Builds type contracts from type definitions.

    Used in Pass 1 to establish the contract/type foundation
    before implementation generation.
    """

    def __init__(self) -> None:
        self._contracts: dict[str, dict[str, Any]] = {}
        self._import_hints: dict[str, str] = {}

    def add_types(self, path: str, content: str) -> None:
        """Add type definitions from a file.

        Extracts interfaces, types, classes, enums, and function signatures.
        """
        # Extract interfaces
        interfaces = re.findall(
            r'export\s+interface\s+(\w+)\s*(?:extends\s+[\w,\s]+)?\s*\{[^}]*\}',
            content,
            re.DOTALL,
        )
        for name in interfaces:
            self._contracts[name] = {"path": path, "kind": "interface"}
            self._import_hints[name] = path

        # Extract types
        types = re.findall(
            r'export\s+type\s+(\w+)\s*=',
            content,
        )
        for name in types:
            self._contracts[name] = {"path": path, "kind": "type"}
            self._import_hints[name] = path

        # Extract classes
        classes = re.findall(
            r'export\s+(?:abstract\s+)?class\s+(\w+)',
            content,
        )
        for name in classes:
            self._contracts[name] = {"path": path, "kind": "class"}
            self._import_hints[name] = path

        # Extract enums
        enums = re.findall(
            r'export\s+(?:const\s+)?enum\s+(\w+)',
            content,
        )
        for name in enums:
            self._contracts[name] = {"path": path, "kind": "enum"}
            self._import_hints[name] = path

        # Extract function signatures
        functions = re.findall(
            r'export\s+(?:async\s+)?function\s+(\w+)',
            content,
        )
        for name in functions:
            self._contracts[name] = {"path": path, "kind": "function"}
            self._import_hints[name] = path

    def get_contracts(self) -> dict[str, dict[str, Any]]:
        """Get all contracts."""
        return self._contracts.copy()

    def get_import_hints(self) -> dict[str, str]:
        """Get import hints (symbol → file mapping)."""
        return self._import_hints.copy()
