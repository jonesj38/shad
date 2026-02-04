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

    def __init__(
        self,
        collection_names: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the qmd retriever.

        Args:
            collection_names: Optional mapping of collection name -> path.
                              Used to validate and resolve collection names.
        """
        self._qmd_path: str | None = None
        self._collection_names = collection_names or {}
        # Cache for qmd-registered collection names mapped by path
        self._qmd_collections_by_path: dict[str, str] | None = None

    @property
    def available(self) -> bool:
        """Check if qmd is available on the system."""
        if self._qmd_path is None:
            self._qmd_path = shutil.which("qmd")
        return self._qmd_path is not None

    async def _load_qmd_collections(self) -> dict[str, str]:
        """Load qmd-registered collections and map by path.

        Returns:
            Dict mapping normalized path -> collection name
        """
        if self._qmd_collections_by_path is not None:
            return self._qmd_collections_by_path

        self._qmd_collections_by_path = {}
        collections = await self.list_collections()

        for coll in collections:
            # qmd returns collection info with 'name' and 'path' fields
            name = coll.get("name", "")
            path = coll.get("path", "")
            if name and path:
                # Normalize path for comparison
                from pathlib import Path
                normalized = str(Path(path).resolve())
                self._qmd_collections_by_path[normalized] = name
                logger.debug(f"Mapped qmd collection: {normalized} -> {name}")

        return self._qmd_collections_by_path

    async def resolve_collection_names(
        self,
        requested_names: list[str],
    ) -> list[str]:
        """Resolve collection names, mapping by path if needed.

        If a requested collection name doesn't exist in qmd but we have
        a path mapping for it, look up the qmd-registered name for that path.

        Args:
            requested_names: List of collection names to resolve

        Returns:
            List of resolved collection names that exist in qmd
        """
        if not requested_names:
            return []

        # Load qmd collections
        qmd_by_path = await self._load_qmd_collections()
        qmd_names = set(qmd_by_path.values())

        resolved = []
        for name in requested_names:
            # If name exists in qmd, use it directly
            if name in qmd_names:
                resolved.append(name)
                continue

            # Try to resolve by path
            if name in self._collection_names:
                from pathlib import Path
                path = self._collection_names[name]
                normalized = str(Path(path).resolve())
                if normalized in qmd_by_path:
                    qmd_name = qmd_by_path[normalized]
                    logger.info(f"Resolved collection '{name}' -> '{qmd_name}' by path")
                    resolved.append(qmd_name)
                    continue

            # Fall back to using the name as-is (may fail if not in qmd)
            logger.debug(f"Collection '{name}' not found in qmd, using as-is")
            resolved.append(name)

        return resolved

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
            # Resolve collection names (handles path-based lookup)
            resolved_collections = await self.resolve_collection_names(collections)
            if resolved_collections:
                args.extend(["-c", ",".join(resolved_collections)])

        # Add minimum score filter
        if min_score > 0:
            args.extend(["--min-score", str(min_score)])

        try:
            logger.debug(f"Running qmd: {' '.join(args)}")
            # Pass environment with QMD_OPENAI=1 for fast query expansion
            import os
            env = os.environ.copy()
            env["QMD_OPENAI"] = "1"
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
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
                # "No results found." is valid, not an error
                if not output or "no results" in output.lower():
                    return []
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
            # qmd uses "file" field for the path
            # Fall back to snippet/context if content is empty (common with qmd search results)
            content = item.get("content", "") or item.get("snippet", item.get("context", ""))
            result = RetrievalResult(
                path=item.get("file", item.get("path", item.get("filepath", ""))),
                content=content,
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
