"""Open Notebook API Client.

Connects to the Open Notebook service (lfnovo/open-notebook) for
knowledge management and retrieval.

API docs available at: http://localhost:5055/docs
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from shad.models.notebook import (
    Note,
    RetrievalResult,
    Source,
)
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class OpenNotebookClient:
    """
    Client for Open Notebook REST API.

    Open Notebook provides:
    - Notebook management (collections of sources)
    - Source ingestion (PDFs, web pages, text, etc.)
    - Note creation and management
    - Full-text and vector search
    - Chat/query interface
    """

    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = base_url or getattr(settings, "open_notebook_url", "http://localhost:5055")
        self._client: httpx.AsyncClient | None = None
        self._connected = False

    async def connect(self) -> bool:
        """Initialize HTTP client and verify connection."""
        if self._connected:
            return True

        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
            )

            # Test connection
            response = await self._client.get("/health")
            if response.status_code == 200:
                self._connected = True
                logger.info(f"Connected to Open Notebook at {self.base_url}")
                return True
            else:
                logger.warning(f"Open Notebook health check failed: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"Failed to connect to Open Notebook: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None

    # ==================== Notebook Operations ====================

    async def list_notebooks(self) -> list[dict[str, Any]]:
        """List all notebooks."""
        if not self.is_connected:
            return []

        try:
            response = await self._client.get("/api/notebooks")
            if response.status_code == 200:
                return response.json()
            logger.warning(f"Failed to list notebooks: {response.status_code}")
        except Exception as e:
            logger.error(f"Error listing notebooks: {e}")

        return []

    async def get_notebook(self, notebook_id: str) -> dict[str, Any] | None:
        """Get a notebook by ID."""
        if not self.is_connected:
            return None

        try:
            response = await self._client.get(f"/api/notebooks/{notebook_id}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting notebook: {e}")

        return None

    async def create_notebook(
        self,
        name: str,
        description: str = "",
    ) -> dict[str, Any] | None:
        """Create a new notebook."""
        if not self.is_connected:
            return None

        try:
            response = await self._client.post(
                "/api/notebooks",
                json={"name": name, "description": description},
            )
            if response.status_code in (200, 201):
                logger.info(f"Created notebook: {name}")
                return response.json()
        except Exception as e:
            logger.error(f"Error creating notebook: {e}")

        return None

    # ==================== Source Operations ====================

    async def list_sources(self, notebook_id: str) -> list[dict[str, Any]]:
        """List sources in a notebook."""
        if not self.is_connected:
            return []

        try:
            response = await self._client.get(
                f"/api/notebooks/{notebook_id}/sources"
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error listing sources: {e}")

        return []

    async def add_source(
        self,
        notebook_id: str,
        content: str,
        title: str = "",
        source_type: str = "text",
        url: str | None = None,
    ) -> dict[str, Any] | None:
        """Add a source to a notebook."""
        if not self.is_connected:
            return None

        try:
            payload = {
                "content": content,
                "title": title,
                "type": source_type,
            }
            if url:
                payload["url"] = url

            response = await self._client.post(
                f"/api/notebooks/{notebook_id}/sources",
                json=payload,
            )
            if response.status_code in (200, 201):
                logger.info(f"Added source to notebook {notebook_id}: {title}")
                return response.json()
        except Exception as e:
            logger.error(f"Error adding source: {e}")

        return None

    async def add_source_from_url(
        self,
        notebook_id: str,
        url: str,
    ) -> dict[str, Any] | None:
        """Add a source from a URL (web page, PDF, etc.)."""
        if not self.is_connected:
            return None

        try:
            response = await self._client.post(
                f"/api/notebooks/{notebook_id}/sources/url",
                json={"url": url},
            )
            if response.status_code in (200, 201):
                logger.info(f"Added URL source to notebook {notebook_id}: {url}")
                return response.json()
        except Exception as e:
            logger.error(f"Error adding URL source: {e}")

        return None

    # ==================== Search & Retrieval ====================

    async def search(
        self,
        notebook_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for content in a notebook."""
        if not self.is_connected:
            return []

        try:
            response = await self._client.post(
                f"/api/notebooks/{notebook_id}/search",
                json={"query": query, "limit": limit},
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error searching notebook: {e}")

        return []

    async def retrieve(
        self,
        notebook_id: str,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve relevant content from a notebook.

        This is the main interface used by RLMEngine for context retrieval.
        Converts Open Notebook search results to Shad's RetrievalResult format.
        """
        if not self.is_connected:
            return RetrievalResult(query=query)

        try:
            # Search the notebook
            results = await self.search(notebook_id, query, limit)

            nodes = []
            scores = {}

            for result in results:
                # Convert to Shad's Source/Note format
                node_type = result.get("type", "source")
                node_id = result.get("id", "")

                if node_type == "note":
                    node = Note(
                        node_id=node_id,
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        author=result.get("author", "system"),
                    )
                else:
                    node = Source(
                        node_id=node_id,
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        extracted_text=result.get("content", ""),
                        source_type=result.get("source_type", "text"),
                        url=result.get("url"),
                    )

                nodes.append(node)
                scores[node_id] = result.get("score", 0.5)

            return RetrievalResult(
                nodes=nodes,
                scores=scores,
                total_nodes=len(nodes),
                query=query,
            )

        except Exception as e:
            logger.error(f"Error retrieving from notebook: {e}")
            return RetrievalResult(query=query)

    # ==================== Chat/Query Interface ====================

    async def chat(
        self,
        notebook_id: str,
        message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Send a chat message to query a notebook."""
        if not self.is_connected:
            return None

        try:
            payload = {"message": message}
            if conversation_id:
                payload["conversation_id"] = conversation_id

            response = await self._client.post(
                f"/api/notebooks/{notebook_id}/chat",
                json=payload,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error in notebook chat: {e}")

        return None

    # ==================== Notes Operations ====================

    async def list_notes(self, notebook_id: str) -> list[dict[str, Any]]:
        """List notes in a notebook."""
        if not self.is_connected:
            return []

        try:
            response = await self._client.get(
                f"/api/notebooks/{notebook_id}/notes"
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error listing notes: {e}")

        return []

    async def create_note(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Create a note in a notebook."""
        if not self.is_connected:
            return None

        try:
            payload = {
                "title": title,
                "content": content,
            }
            if source_ids:
                payload["source_ids"] = source_ids

            response = await self._client.post(
                f"/api/notebooks/{notebook_id}/notes",
                json=payload,
            )
            if response.status_code in (200, 201):
                logger.info(f"Created note in notebook {notebook_id}: {title}")
                return response.json()
        except Exception as e:
            logger.error(f"Error creating note: {e}")

        return None

    # ==================== Utility Methods ====================

    async def get_stats(self) -> dict[str, Any]:
        """Get Open Notebook statistics."""
        if not self.is_connected:
            return {"connected": False}

        try:
            notebooks = await self.list_notebooks()
            return {
                "connected": True,
                "base_url": self.base_url,
                "notebook_count": len(notebooks),
            }
        except Exception as e:
            return {"connected": True, "error": str(e)}
