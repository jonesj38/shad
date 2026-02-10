"""qmd retrieval backend.

Wraps the qmd CLI for hybrid search over markdown knowledge bases.
See: https://github.com/tobi/qmd
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
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

    # Stop words for keyword extraction (prevents long queries hanging qmd)
    STOP_WORDS = frozenset({
        # Standard stop words
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'do', 'does', 'did', 'have', 'has', 'had', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'we', 'how', 'what',
        'which', 'who', 'when', 'where', 'why', 'and', 'or', 'not',
        'it', 'its', 'this', 'that', 'these', 'those', 'can', 'could',
        'would', 'should', 'will', 'may', 'might', 'must', 'shall',
        'into', 'onto', 'upon', 'about', 'after', 'before', 'between',
        'through', 'during', 'under', 'over', 'above', 'below',
        'all', 'any', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'than', 'too', 'very', 'just', 'also',
        # Task/goal meta-words
        'goal', 'goals', 'task', 'tasks', 'run', 'runs', 'statement',
        'build', 'create', 'make', 'implement', 'add', 'update', 'fix',
        'write', 'develop', 'design', 'setup', 'configure', 'enable',
        'use', 'using', 'need', 'needs', 'want', 'wants', 'please',
        'help', 'ensure', 'provide', 'include', 'support', 'allow',
        'following', 'based', 'like', 'new', 'existing', 'current',
    })

    def _extract_keywords(self, query: str, max_keywords: int = 15) -> str:
        """Extract search keywords from a query string.

        Removes stop words and short words to prevent qmd hanging on
        long queries during query expansion.

        Args:
            query: Raw query string (possibly long/verbose)
            max_keywords: Maximum keywords to extract

        Returns:
            Space-separated keywords for search
        """
        words = re.split(r'\W+', query.lower())
        seen: set[str] = set()
        keywords: list[str] = []
        for w in words:
            if w and len(w) > 2 and w not in self.STOP_WORDS and w not in seen:
                seen.add(w)
                keywords.append(w)
                if len(keywords) >= max_keywords:
                    break
        return " ".join(keywords)

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

        # Extract keywords for long queries to prevent qmd hanging during expansion
        if len(query) > 100 or '\n' in query:
            search_query = self._extract_keywords(query)
            logger.debug(f"Extracted keywords from long query: {search_query}")
        else:
            search_query = query

        if not search_query:
            logger.warning("No search query after keyword extraction")
            return []

        # Get the qmd command for this mode
        cmd = self.MODE_COMMANDS.get(mode, "query")

        # Build command arguments
        args = [
            "qmd",
            cmd,
            search_query,
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

            # Use synchronous subprocess in thread pool to avoid asyncio subprocess issues
            # (asyncio subprocess can hang due to event loop/signal handler interference)
            def _run_qmd() -> subprocess.CompletedProcess[bytes]:
                env = os.environ.copy()
                env["QMD_OPENAI"] = "1"
                return subprocess.run(
                    args,
                    capture_output=True,
                    timeout=60,
                    env=env,
                )

            loop = asyncio.get_running_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _run_qmd),
                    timeout=65.0,  # Slightly higher than subprocess timeout
                )
            except asyncio.TimeoutError:
                logger.warning("qmd search timed out after 65s")
                return []
            except subprocess.TimeoutExpired:
                logger.warning("qmd search timed out after 60s")
                return []

            if result.returncode != 0:
                error_msg = result.stderr.decode().strip() if result.stderr else "Unknown error"
                logger.error(f"qmd search failed: {error_msg}")
                return []

            if not result.stdout:
                return []

            # Handle potential non-JSON output
            output = result.stdout.decode().strip()
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

    async def ensure_collections(
        self,
        collection_names: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Ensure vault paths are registered as qmd collections with up-to-date indexes.

        For each vault path:
        1. Check if it's already a qmd collection
        2. If not, add it via `qmd collection add`
        3. Run `qmd update` to refresh the BM25 index for any new/changed files
        4. Run `qmd embed` for any chunks missing embeddings

        Args:
            collection_names: Mapping of collection name -> path. Falls back
                              to self._collection_names if not provided.

        Returns:
            Dict mapping collection name -> True if provisioned/updated successfully
        """
        if not self.available:
            return {}

        names = collection_names or self._collection_names
        if not names:
            return {}

        # Get existing qmd collections
        existing = await self.list_collections()
        existing_by_path: dict[str, str] = {}
        existing_names: set[str] = set()
        for coll in existing:
            cname = coll.get("name", "")
            cpath = coll.get("path", "")
            if cname and cpath:
                from pathlib import Path
                existing_by_path[str(Path(cpath).resolve())] = cname
                existing_names.add(cname)

        results: dict[str, bool] = {}
        added_any = False

        for name, path in names.items():
            from pathlib import Path
            resolved = str(Path(str(path)).resolve())

            if resolved in existing_by_path:
                logger.info(f"[QMD] Collection '{existing_by_path[resolved]}' already exists for {resolved}")
                results[name] = True
            elif name in existing_names:
                logger.info(f"[QMD] Collection '{name}' exists (different path)")
                results[name] = True
            else:
                logger.info(f"[QMD] Auto-provisioning collection '{name}' from {resolved}")
                success = await self.add_collection(resolved, name=name)
                if success:
                    added_any = True
                    results[name] = True
                    logger.info(f"[QMD] Added collection '{name}'")
                else:
                    # May already exist under a slightly different path — treat as OK
                    logger.warning(f"[QMD] Could not add collection '{name}' (may already exist)")
                    results[name] = True

        # Refresh BM25 index for new/changed files
        await self.update_index()

        # Embed any chunks that are missing embeddings (best-effort, don't block)
        try:
            if added_any:
                logger.info("[QMD] Running embeddings for new collections (best-effort)...")
                await asyncio.wait_for(self.embed(), timeout=30)
            else:
                # Incremental embed — only embeds chunks without vectors
                await asyncio.wait_for(self.embed_incremental(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("[QMD] Embedding timed out (30s) — continuing without full embeddings. BM25 search still works.")
        except Exception as e:
            logger.warning(f"[QMD] Embedding failed (non-fatal): {e}")

        # Invalidate collection cache so next search picks up new collections
        self._qmd_collections_by_path = None

        return results

    async def update_index(self) -> bool:
        """Update the BM25 full-text index for all collections.

        Runs `qmd update` which scans for new/modified/deleted files
        and refreshes the SQLite FTS5 index.

        Returns:
            True if successful
        """
        if not self.available:
            return False

        try:
            def _run() -> subprocess.CompletedProcess[bytes]:
                return subprocess.run(
                    ["qmd", "update"],
                    capture_output=True,
                    timeout=120,
                )

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _run)

            if result.returncode != 0:
                stderr = result.stderr.decode().strip() if result.stderr else ""
                logger.warning(f"qmd update returned non-zero: {stderr}")
                return False

            logger.info("[QMD] Index updated successfully")
            return True

        except Exception as e:
            logger.error(f"qmd update error: {e}")
            return False

    async def embed_incremental(self) -> bool:
        """Embed only chunks that are missing embeddings.

        Unlike `embed()` which re-embeds everything, this only processes
        new content that doesn't have vectors yet.

        Returns:
            True if successful
        """
        if not self.available:
            return False

        try:
            def _run() -> subprocess.CompletedProcess[bytes]:
                env = os.environ.copy()
                env["QMD_OPENAI"] = "1"
                return subprocess.run(
                    ["qmd", "embed"],
                    capture_output=True,
                    timeout=300,  # Embedding can take a while
                    env=env,
                )

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _run)

            if result.returncode != 0:
                stderr = result.stderr.decode().strip() if result.stderr else ""
                logger.warning(f"qmd embed returned non-zero: {stderr}")
                return False

            logger.info("[QMD] Incremental embedding complete")
            return True

        except Exception as e:
            logger.error(f"qmd embed error: {e}")
            return False

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
