"""qmd retrieval backend.

Wraps the qmd CLI for hybrid search over markdown knowledge bases.
See: https://github.com/tobi/qmd
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from shad.retrieval.layer import RetrievalResult

logger = logging.getLogger(__name__)


class QmdRetriever:
    """Retrieval backend using qmd for hybrid search.

    qmd provides:
    - BM25 full-text search (fast, keyword-based)
    - Vector semantic search (recall-focused)
    - Hybrid search with LLM re-ranking (best quality)

    Requires qmd to be installed: bun install -g https://github.com/tobi/qmd
    """

    # Map search modes to qmd commands
    MODE_COMMANDS = {
        "bm25": "search",
        "vector": "vsearch",
        "hybrid": "query",
    }

    def __init__(self) -> None:
        """Initialize the qmd retriever."""
        self._qmd_path: str | None = None

    @property
    def available(self) -> bool:
        """Check if qmd is available on the system."""
        if self._qmd_path is None:
            self._qmd_path = shutil.which("qmd")
        return self._qmd_path is not None

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        collections: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Search for documents using qmd.

        Args:
            query: Search query string
            mode: Search mode - "bm25", "vector", or "hybrid" (default)
            collections: Optional list of collection names to search
            limit: Maximum number of results
            min_score: Minimum relevance score (0.0 to 1.0)

        Returns:
            List of RetrievalResult objects
        """
        if not self.available:
            logger.warning("qmd not available, returning empty results")
            return []

        # Get the qmd command for this mode
        cmd = self.MODE_COMMANDS.get(mode, "query")

        # Build command arguments
        args = [
            "qmd",
            cmd,
            query,
            "-n", str(limit),
            "--json",
        ]

        # Add collection filter if specified
        if collections:
            args.extend(["-c", ",".join(collections)])

        # Add minimum score filter
        if min_score > 0:
            args.extend(["--min-score", str(min_score)])

        try:
            logger.debug(f"Running qmd: {' '.join(args)}")
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=60.0,  # 60 second timeout for large vaults
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.warning("qmd search timed out after 60s")
                return []

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"qmd search failed: {error_msg}")
                return []

            if not stdout:
                return []

            # Handle potential non-JSON output
            output = stdout.decode().strip()
            if not output or not output.startswith(("[", "{")):
                logger.warning(f"qmd returned non-JSON output: {output[:100]}")
                return []

            # Parse JSON output
            data = json.loads(output)
            return self._parse_results(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse qmd output: {e}")
            return []
        except Exception as e:
            logger.error(f"qmd search error: {e}")
            return []

    async def get(
        self,
        path: str,
        collection: str | None = None,
    ) -> str | None:
        """Retrieve a single document by path.

        Args:
            path: Relative path or docid (prefixed with #)
            collection: Optional collection name

        Returns:
            Document content or None if not found
        """
        if not self.available:
            logger.warning("qmd not available")
            return None

        args = ["qmd", "get", path, "--full"]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return None

            return stdout.decode() if stdout else None

        except Exception as e:
            logger.error(f"qmd get error: {e}")
            return None

    async def status(self) -> dict[str, Any]:
        """Get qmd status information.

        Returns:
            Dictionary with collections, index health, etc.
        """
        if not self.available:
            return {"available": False, "error": "qmd not installed"}

        try:
            process = await asyncio.create_subprocess_exec(
                "qmd", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return {"available": True, "error": stderr.decode().strip()}

            data = json.loads(stdout.decode())
            return {"available": True, **data}

        except json.JSONDecodeError:
            # qmd status might not output JSON, try plain output
            return {"available": True, "status": "ok"}
        except Exception as e:
            return {"available": True, "error": str(e)}

    async def add_collection(
        self,
        path: str,
        name: str | None = None,
        mask: str = "**/*.md",
    ) -> bool:
        """Add a directory as a qmd collection.

        Args:
            path: Path to the directory
            name: Collection name (defaults to directory name)
            mask: Glob pattern for files to index

        Returns:
            True if successful
        """
        if not self.available:
            return False

        args = ["qmd", "collection", "add", path]
        if name:
            args.extend(["--name", name])
        args.extend(["--mask", mask])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to add collection: {stderr.decode()}")
                return False

            return True

        except Exception as e:
            logger.error(f"qmd collection add error: {e}")
            return False

    async def embed(self, force: bool = False) -> bool:
        """Generate embeddings for all collections.

        Args:
            force: Force re-embedding even if already done

        Returns:
            True if successful
        """
        if not self.available:
            return False

        args = ["qmd", "embed"]
        if force:
            args.append("-f")

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"qmd embed failed: {stderr.decode()}")
                return False

            return True

        except Exception as e:
            logger.error(f"qmd embed error: {e}")
            return False

    async def list_collections(self) -> list[dict[str, Any]]:
        """List all qmd collections.

        Returns:
            List of collection info dictionaries
        """
        if not self.available:
            return []

        try:
            process = await asyncio.create_subprocess_exec(
                "qmd", "collection", "list", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return []

            return json.loads(stdout.decode()) if stdout else []

        except Exception:
            return []

    def _parse_results(self, data: list[dict[str, Any]] | dict[str, Any]) -> list[RetrievalResult]:
        """Parse qmd JSON output into RetrievalResult objects.

        Args:
            data: Raw JSON data from qmd

        Returns:
            List of RetrievalResult objects
        """
        # Handle both list and dict formats
        results_data = data if isinstance(data, list) else data.get("results", [])

        results = []
        for item in results_data:
            # qmd uses different field names depending on output format
            result = RetrievalResult(
                path=item.get("path", item.get("filepath", "")),
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
                snippet=item.get("snippet", item.get("context", None)),
                collection=item.get("collection", ""),
                docid=item.get("docid", item.get("id", "")),
                matched_line=item.get("matched_line", None),
                metadata={
                    "title": item.get("title", ""),
                    "context": item.get("context", ""),
                },
            )
            results.append(result)

        return results
