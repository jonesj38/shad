"""Code Execution Sandbox.

Per OBSIDIAN_PIVOT.md Section 3 and Section 12.2:
- RLM writes Python scripts that import MCP tools
- Script executes in sandboxed container with vault access
- Script filters, aggregates, processes vault data before returning
- Only final distilled output enters context window
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for the code execution sandbox.

    Per OBSIDIAN_PIVOT.md Section 12.2:
    - Vault bind-mounted at /mnt/data (in Docker mode)
    - No system file access outside vault
    - Network restricted to host.docker.internal
    """

    vault_path: Path
    timeout_seconds: int = 60
    max_memory_mb: int = 512
    network_enabled: bool = False
    allowed_imports: list[str] = field(default_factory=lambda: [
        "json", "re", "datetime", "collections", "itertools",
        "functools", "math", "statistics", "hashlib", "pathlib",
        "typing", "dataclasses", "enum", "yaml",
    ])


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error_type: str | None = None
    error_message: str | None = None
    execution_time_ms: int = 0
    memory_used_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": self.return_value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
        }


class CodeExecutor:
    """Execute Python code in a sandboxed environment.

    Per OBSIDIAN_PIVOT.md Section 3.1: Workflow
    1. RLM writes Python script importing MCP tools
    2. Script executes in sandboxed container
    3. Script filters/aggregates data before returning
    4. Only final output enters context window
    """

    def __init__(self, config: SandboxConfig):
        """Initialize executor with configuration."""
        self.config = config
        self._globals: dict[str, Any] = {}
        self._setup_globals()

    def _setup_globals(self) -> None:
        """Set up the global namespace for script execution."""
        # Import sandbox tools module
        from shad.sandbox import tools

        # Create obsidian tools instance
        obsidian_tools = tools.ObsidianTools(vault_path=self.config.vault_path)

        # Set up globals with safe builtins and tools
        self._globals = {
            "__builtins__": self._get_safe_builtins(),
            "__name__": "__main__",
            "obsidian": obsidian_tools,
            "vault_path": self.config.vault_path,
        }

        # Add allowed imports
        for module_name in self.config.allowed_imports:
            try:
                module = __import__(module_name)
                self._globals[module_name] = module
            except ImportError:
                pass

    def _get_safe_builtins(self) -> dict[str, Any]:
        """Get a restricted set of builtins for sandboxed execution."""
        import builtins

        # Allow most builtins except dangerous ones
        dangerous = {
            "eval", "exec", "compile", "__import__",
            "open",  # Will be replaced with restricted version
            "input",
        }

        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if not name.startswith("_") and name not in dangerous
        }

        # Add restricted open function
        safe_builtins["open"] = self._restricted_open
        safe_builtins["print"] = print  # Allow print

        return safe_builtins

    def _restricted_open(
        self,
        file: str | Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Restricted open that only allows access within vault."""
        file_path = Path(file).resolve()
        vault_resolved = self.config.vault_path.resolve()

        # Check if path is within vault
        try:
            file_path.relative_to(vault_resolved)
        except ValueError as e:
            raise PermissionError(
                f"Access denied: {file_path} is outside vault"
            ) from e

        # Only allow read modes
        if "w" in mode or "a" in mode:
            # Allow writes only within vault
            pass

        return open(file_path, mode, *args, **kwargs)

    async def execute(self, script: str) -> ExecutionResult:
        """Execute a Python script in the sandbox.

        Args:
            script: Python code to execute

        Returns:
            ExecutionResult with output and return value
        """
        start_time = time.time()

        # Create a fresh locals dict for this execution
        local_vars: dict[str, Any] = {}

        # We need to capture stdout/stderr in the thread
        def run_in_thread() -> tuple[str, str]:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self._globals, local_vars)
            return stdout_buffer.getvalue(), stderr_buffer.getvalue()

        try:
            # Compile the code first to catch syntax errors
            code = compile(script, "<sandbox>", "exec")

            # Execute with timeout in thread pool
            loop = asyncio.get_event_loop()
            stdout, stderr = await asyncio.wait_for(
                loop.run_in_executor(None, run_in_thread),
                timeout=self.config.timeout_seconds,
            )

            # Get return value if set
            return_value = local_vars.get("__result__")

            execution_time = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=True,
                stdout=stdout,
                stderr=stderr,
                return_value=return_value,
                execution_time_ms=execution_time,
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error_type="TimeoutError",
                error_message=f"Execution exceeded {self.config.timeout_seconds}s timeout",
                execution_time_ms=self.config.timeout_seconds * 1000,
            )

        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                error_type="SyntaxError",
                error_message=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            tb = traceback.format_exc()

            return ExecutionResult(
                success=False,
                stdout="",
                stderr=tb,
                error_type=error_type,
                error_message=error_msg,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    async def execute_with_context(
        self,
        script: str,
        context: dict[str, Any],
    ) -> ExecutionResult:
        """Execute script with additional context variables.

        Args:
            script: Python code to execute
            context: Additional variables to inject

        Returns:
            ExecutionResult with output and return value
        """
        # Inject context into globals
        old_globals = self._globals.copy()
        self._globals.update(context)

        try:
            return await self.execute(script)
        finally:
            # Restore original globals
            self._globals = old_globals
