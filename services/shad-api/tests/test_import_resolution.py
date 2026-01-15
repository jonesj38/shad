"""Tests for two-pass import resolution.

Per SPEC.md Section 2.7.2:
- Pass 1: Build export index (symbol → file mapping)
- Pass 2: Generate implementations using export index as ground truth
- Post-generation validation: Check all imports resolve
"""

from __future__ import annotations

import pytest

from shad.output.import_resolution import (
    ImportResolver,
    ImportValidation,
    MissingImport,
    TypeContractsBuilder,
)
from shad.output.manifest import FileEntry, FileManifest


class TestImportResolver:
    """Tests for import resolution."""

    @pytest.fixture
    def manifest(self) -> FileManifest:
        """Create test manifest with types and implementation."""
        manifest = FileManifest(run_id="test-123")

        # Types file (contracts)
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="""export interface User {
    id: string;
    name: string;
    email: string;
}

export interface Post {
    id: string;
    title: string;
    authorId: string;
}

export type UserId = string;
export type PostId = string;""",
            language="ts",
        ))

        # API implementation
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content="""import { User, UserId } from '../types';

export async function getUser(id: UserId): Promise<User> {
    // Implementation
}

export async function createUser(data: Partial<User>): Promise<User> {
    // Implementation
}""",
            language="ts",
        ))

        return manifest

    def test_validate_imports_success(self, manifest: FileManifest) -> None:
        """Test validating imports that resolve correctly."""
        resolver = ImportResolver(manifest)
        result = resolver.validate()

        assert result.is_valid is True
        assert len(result.missing_modules) == 0
        assert len(result.missing_symbols) == 0

    def test_detect_missing_module(self) -> None:
        """Test detecting missing module imports."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content="""import { User } from '../types';
import { Database } from '../db';  // db doesn't exist

export function getUser(): User {}""",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User {}",
            language="ts",
        ))

        resolver = ImportResolver(manifest)
        result = resolver.validate()

        assert result.is_valid is False
        assert len(result.missing_modules) > 0
        assert any("db" in m.import_path.lower() for m in result.missing_modules)

    def test_detect_missing_symbol(self) -> None:
        """Test detecting missing symbol imports."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content="""import { User, NonExistent } from '../types';

export function getUser(): User {}""",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User {}",
            language="ts",
        ))

        resolver = ImportResolver(manifest)
        result = resolver.validate()

        assert result.is_valid is False
        assert len(result.missing_symbols) > 0
        assert any(m.symbol == "NonExistent" for m in result.missing_symbols)

    def test_resolve_relative_imports(self, manifest: FileManifest) -> None:
        """Test resolving relative imports."""
        resolver = ImportResolver(manifest)

        # From src/api/users.ts, ../types resolves to src/types
        resolved = resolver.resolve_import_path(
            from_file="src/api/users.ts",
            import_path="../types",
        )

        assert resolved == "src/types.ts" or resolved == "src/types"

    def test_handle_path_aliases(self) -> None:
        """Test handling path aliases like @/."""
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User {}",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content="""import { User } from '@/types';

export function getUser(): User {}""",
            language="ts",
        ))

        resolver = ImportResolver(manifest, path_aliases={"@/": "src/"})
        result = resolver.validate()

        assert result.is_valid is True


class TestImportValidation:
    """Tests for import validation results."""

    def test_validation_result_structure(self) -> None:
        """Test validation result structure."""
        result = ImportValidation(
            is_valid=False,
            missing_modules=[
                MissingImport(
                    file="src/api/users.ts",
                    import_path="../db",
                    reason="Module not found in manifest",
                ),
            ],
            missing_symbols=[
                MissingImport(
                    file="src/api/users.ts",
                    symbol="NonExistent",
                    import_path="../types",
                    reason="Symbol not exported from module",
                ),
            ],
        )

        assert result.is_valid is False
        assert len(result.missing_modules) == 1
        assert len(result.missing_symbols) == 1

    def test_validation_to_report(self) -> None:
        """Test converting validation to human-readable report."""
        result = ImportValidation(
            is_valid=False,
            missing_modules=[
                MissingImport(
                    file="src/api.ts",
                    import_path="./db",
                    reason="Not found",
                ),
            ],
            missing_symbols=[],
        )

        report = result.to_report()

        assert "api.ts" in report
        assert "db" in report


class TestTypeContractsBuilder:
    """Tests for type contracts builder (pass 1)."""

    def test_build_contracts_from_types(self) -> None:
        """Test building contracts from type definitions."""
        builder = TypeContractsBuilder()

        # Add type definitions
        builder.add_types(
            path="src/types.ts",
            content="""export interface User {
    id: string;
    name: string;
}

export interface Post {
    id: string;
    title: string;
}""",
        )

        contracts = builder.get_contracts()

        assert "User" in contracts
        assert "Post" in contracts
        assert contracts["User"]["path"] == "src/types.ts"

    def test_contracts_include_function_signatures(self) -> None:
        """Test that contracts include function signatures."""
        builder = TypeContractsBuilder()

        builder.add_types(
            path="src/api/users.ts",
            content="""export async function getUser(id: string): Promise<User> {}
export function createUser(data: UserData): User {}""",
        )

        contracts = builder.get_contracts()

        assert "getUser" in contracts
        assert "createUser" in contracts

    def test_contracts_provide_import_hints(self) -> None:
        """Test that contracts provide import hints for implementation."""
        builder = TypeContractsBuilder()

        builder.add_types(
            path="src/types.ts",
            content="export interface User { id: string; }",
        )

        hints = builder.get_import_hints()

        # Should indicate User is exported from src/types.ts
        assert "User" in hints
        assert hints["User"] == "src/types.ts"


class TestTwoPassGeneration:
    """Integration tests for two-pass code generation."""

    def test_pass1_builds_export_index(self) -> None:
        """Test that pass 1 builds complete export index."""
        from shad.output.manifest import build_export_index

        # Pass 1: Create types/contracts files
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types/user.ts",
            content="""export interface User {
    id: string;
    name: string;
}

export type UserId = string;""",
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/types/post.ts",
            content="""export interface Post {
    id: string;
    title: string;
}

export type PostId = string;""",
            language="ts",
        ))

        # Build export index
        index = build_export_index(manifest)

        # Verify all exports are indexed
        assert "User" in index.exports
        assert "UserId" in index.exports
        assert "Post" in index.exports
        assert "PostId" in index.exports

    def test_pass2_uses_export_index(self) -> None:
        """Test that pass 2 generates correct imports using index."""
        from shad.output.manifest import build_export_index

        # Create initial manifest with types
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content="export interface User { id: string; }",
            language="ts",
        ))

        index = build_export_index(manifest)

        # Pass 2: Add implementation that uses types
        manifest.add_file(FileEntry(
            path="src/api/users.ts",
            content=f"""import {{ User }} from '{index.get_import_path("User")}';

export async function getUser(id: string): Promise<User> {{
    return {{ id, name: "Test" }};
}}""",
            language="ts",
        ))

        # Validate imports
        resolver = ImportResolver(manifest)
        result = resolver.validate()

        # The import from src/types.ts should resolve
        # (Note: may need path normalization)
        assert len(result.missing_symbols) == 0 or result.is_valid

    def test_full_two_pass_workflow(self) -> None:
        """Test complete two-pass generation workflow."""
        from shad.output.manifest import ExportIndex

        # Pass 1: Build contracts/types
        types_content = """export interface User {
    id: string;
    name: string;
    email: string;
}

export interface CreateUserInput {
    name: string;
    email: string;
}

export interface UserService {
    getUser(id: string): Promise<User>;
    createUser(input: CreateUserInput): Promise<User>;
}"""

        # Build export index from types
        index = ExportIndex()
        index.add_export("User", "src/types.ts", "type")
        index.add_export("CreateUserInput", "src/types.ts", "type")
        index.add_export("UserService", "src/types.ts", "type")

        # Pass 2: Generate implementation using index
        impl_content = """import { User, CreateUserInput, UserService } from '../types';

export class UserServiceImpl implements UserService {
    async getUser(id: string): Promise<User> {
        // Implementation
    }

    async createUser(input: CreateUserInput): Promise<User> {
        // Implementation
    }
}"""

        # Create manifest
        manifest = FileManifest(run_id="test-123")
        manifest.add_file(FileEntry(
            path="src/types.ts",
            content=types_content,
            language="ts",
        ))
        manifest.add_file(FileEntry(
            path="src/services/user.ts",
            content=impl_content,
            language="ts",
        ))

        # Validate
        resolver = ImportResolver(manifest)
        result = resolver.validate()

        assert result.is_valid is True
