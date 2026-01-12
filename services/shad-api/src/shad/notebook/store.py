"""NotebookStore - File-based implementation of OpenNotebookLM.

Per SPEC.md Section 3:
- Notebook, Source, and Note are nodes in a graph
- Relationships are typed edges
- Read-only during reasoning, write-only during persistence

This implementation uses JSON files for storage, which can be
upgraded to a proper graph database later.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from shad.models.notebook import (
    Edge,
    EdgeType,
    Node,
    Note,
    Notebook,
    RetrievalResult,
    Source,
)
from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


class NotebookStore:
    """
    File-based knowledge graph store.

    Provides:
    - CRUD operations for notebooks, sources, notes
    - Graph traversal and retrieval
    - Text-based search (vector search can be added later)
    """

    def __init__(self, storage_path: Path | None = None):
        settings = get_settings()
        self.storage_path = storage_path or settings.notebooks_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage files
        self._notebooks_file = self.storage_path / "notebooks.json"
        self._sources_file = self.storage_path / "sources.json"
        self._notes_file = self.storage_path / "notes.json"
        self._edges_file = self.storage_path / "edges.json"

        # In-memory cache
        self._notebooks: dict[str, Notebook] = {}
        self._sources: dict[str, Source] = {}
        self._notes: dict[str, Note] = {}
        self._edges: dict[str, Edge] = {}

        self._load()

    def _load(self) -> None:
        """Load all data from disk."""
        self._notebooks = self._load_file(self._notebooks_file, Notebook)
        self._sources = self._load_file(self._sources_file, Source)
        self._notes = self._load_file(self._notes_file, Note)
        self._edges = self._load_edges()

    def _load_file(self, path: Path, model_class: type) -> dict[str, Any]:
        """Load a JSON file into a dict of models."""
        if not path.exists():
            return {}

        try:
            with path.open() as f:
                data = json.load(f)
                return {
                    item["node_id"]: model_class(**item)
                    for item in data
                }
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return {}

    def _load_edges(self) -> dict[str, Edge]:
        """Load edges from disk."""
        if not self._edges_file.exists():
            return {}

        try:
            with self._edges_file.open() as f:
                data = json.load(f)
                return {
                    item["edge_id"]: Edge(**item)
                    for item in data
                }
        except Exception as e:
            logger.warning(f"Failed to load edges: {e}")
            return {}

    def _save(self) -> None:
        """Save all data to disk."""
        self._save_file(self._notebooks_file, list(self._notebooks.values()))
        self._save_file(self._sources_file, list(self._sources.values()))
        self._save_file(self._notes_file, list(self._notes.values()))
        self._save_file(self._edges_file, list(self._edges.values()))

    def _save_file(self, path: Path, items: list) -> None:
        """Save a list of models to JSON."""
        try:
            with path.open("w") as f:
                json.dump(
                    [item.model_dump(mode="json") for item in items],
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")

    # ==================== Notebook Operations ====================

    def create_notebook(
        self,
        title: str,
        description: str = "",
        policies: dict[str, Any] | None = None,
    ) -> Notebook:
        """Create a new notebook."""
        notebook = Notebook(
            title=title,
            description=description,
            policies=policies or {},
        )
        self._notebooks[notebook.node_id] = notebook
        self._save()
        logger.info(f"Created notebook: {notebook.node_id} - {title}")
        return notebook

    def get_notebook(self, notebook_id: str) -> Notebook | None:
        """Get a notebook by ID."""
        return self._notebooks.get(notebook_id)

    def list_notebooks(self) -> list[Notebook]:
        """List all notebooks."""
        return list(self._notebooks.values())

    def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook and its edges."""
        if notebook_id not in self._notebooks:
            return False

        del self._notebooks[notebook_id]

        # Remove related edges
        self._edges = {
            eid: e for eid, e in self._edges.items()
            if e.source_id != notebook_id and e.target_id != notebook_id
        }

        self._save()
        return True

    # ==================== Source Operations ====================

    def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "text",
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source | None:
        """Add a source to a notebook."""
        if notebook_id not in self._notebooks:
            logger.warning(f"Notebook {notebook_id} not found")
            return None

        source = Source(
            title=title,
            content=content,
            source_type=source_type,
            url=url,
            extracted_text=content,  # For text sources, content is the text
            metadata=metadata or {},
        )

        self._sources[source.node_id] = source

        # Create edge: Source INCLUDED_IN Notebook
        edge = Edge(
            source_id=source.node_id,
            target_id=notebook_id,
            edge_type=EdgeType.INCLUDED_IN,
        )
        self._edges[edge.edge_id] = edge

        self._save()
        logger.info(f"Added source: {source.node_id} to notebook {notebook_id}")
        return source

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID."""
        return self._sources.get(source_id)

    def list_sources(self, notebook_id: str | None = None) -> list[Source]:
        """List sources, optionally filtered by notebook."""
        if notebook_id is None:
            return list(self._sources.values())

        # Find sources in this notebook
        source_ids = {
            e.source_id for e in self._edges.values()
            if e.target_id == notebook_id and e.edge_type == EdgeType.INCLUDED_IN
        }
        return [s for s in self._sources.values() if s.node_id in source_ids]

    # ==================== Note Operations ====================

    def add_note(
        self,
        notebook_id: str,
        title: str,
        content: str,
        author: str = "system",
        source_refs: list[str] | None = None,
        confidence: float = 1.0,
        provisional: bool = False,
    ) -> Note | None:
        """Add a note to a notebook."""
        if notebook_id not in self._notebooks:
            logger.warning(f"Notebook {notebook_id} not found")
            return None

        note = Note(
            title=title,
            content=content,
            author=author,
            source_refs=source_refs or [],
            confidence=confidence,
            provisional=provisional,
        )

        self._notes[note.node_id] = note

        # Create edge: Note PART_OF Notebook
        edge = Edge(
            source_id=note.node_id,
            target_id=notebook_id,
            edge_type=EdgeType.PART_OF,
        )
        self._edges[edge.edge_id] = edge

        # Create DERIVED_FROM edges for source refs
        for source_id in source_refs or []:
            if source_id in self._sources:
                ref_edge = Edge(
                    source_id=note.node_id,
                    target_id=source_id,
                    edge_type=EdgeType.DERIVED_FROM,
                )
                self._edges[ref_edge.edge_id] = ref_edge

        self._save()
        logger.info(f"Added note: {note.node_id} to notebook {notebook_id}")
        return note

    def get_note(self, note_id: str) -> Note | None:
        """Get a note by ID."""
        return self._notes.get(note_id)

    def list_notes(self, notebook_id: str | None = None) -> list[Note]:
        """List notes, optionally filtered by notebook."""
        if notebook_id is None:
            return list(self._notes.values())

        # Find notes in this notebook
        note_ids = {
            e.source_id for e in self._edges.values()
            if e.target_id == notebook_id and e.edge_type == EdgeType.PART_OF
        }
        return [n for n in self._notes.values() if n.node_id in note_ids]

    # ==================== Edge Operations ====================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """Add an edge between two nodes."""
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            metadata=metadata or {},
        )
        self._edges[edge.edge_id] = edge
        self._save()
        return edge

    def get_edges(
        self,
        node_id: str,
        direction: str = "both",
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Get edges connected to a node."""
        edges = []
        for edge in self._edges.values():
            if edge_type and edge.edge_type != edge_type:
                continue

            if direction in ("out", "both") and edge.source_id == node_id:
                edges.append(edge)
            elif direction in ("in", "both") and edge.target_id == node_id:
                edges.append(edge)

        return edges

    # ==================== Retrieval ====================

    async def retrieve(
        self,
        notebook_id: str,
        query: str,
        limit: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant nodes from a notebook based on query.

        This is a simple text-matching implementation.
        Can be upgraded to vector search later.
        """
        if notebook_id not in self._notebooks:
            return RetrievalResult(query=query)

        nodes: list[Node] = []
        scores: dict[str, float] = {}
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Search sources
        if include_sources:
            for source in self.list_sources(notebook_id):
                score = self._compute_relevance(source, query_lower, query_words)
                if score > 0:
                    nodes.append(source)
                    scores[source.node_id] = score

        # Search notes
        if include_notes:
            for note in self.list_notes(notebook_id):
                score = self._compute_relevance(note, query_lower, query_words)
                if score > 0:
                    nodes.append(note)
                    scores[note.node_id] = score

        # Sort by score and limit
        nodes.sort(key=lambda n: scores.get(n.node_id, 0), reverse=True)
        nodes = nodes[:limit]

        # Get related edges
        node_ids = {n.node_id for n in nodes}
        edges = [
            e for e in self._edges.values()
            if e.source_id in node_ids or e.target_id in node_ids
        ]

        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            scores=scores,
            total_nodes=len(nodes),
            query=query,
        )

    def _compute_relevance(
        self,
        node: Node,
        query_lower: str,
        query_words: set[str],
    ) -> float:
        """Compute relevance score for a node."""
        score = 0.0

        # Check title
        title_lower = node.title.lower()
        if query_lower in title_lower:
            score += 2.0
        else:
            title_words = set(title_lower.split())
            overlap = len(query_words & title_words)
            score += overlap * 0.5

        # Check content
        content_lower = node.content.lower()
        if query_lower in content_lower:
            score += 1.0
        else:
            # Word overlap (sample first 500 words)
            content_words = set(content_lower.split()[:500])
            overlap = len(query_words & content_words)
            score += overlap * 0.2

        return score

    # ==================== Convenience Methods ====================

    def get_node(self, node_id: str) -> Node | None:
        """Get any node by ID."""
        return (
            self._notebooks.get(node_id)
            or self._sources.get(node_id)
            or self._notes.get(node_id)
        )

    def get_stats(self) -> dict[str, int]:
        """Get store statistics."""
        return {
            "notebooks": len(self._notebooks),
            "sources": len(self._sources),
            "notes": len(self._notes),
            "edges": len(self._edges),
        }

    def import_text_file(
        self,
        notebook_id: str,
        file_path: Path,
        title: str | None = None,
    ) -> Source | None:
        """Import a text file as a source."""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        content = file_path.read_text()
        return self.add_source(
            notebook_id=notebook_id,
            title=title or file_path.name,
            content=content,
            source_type="text",
            metadata={"file_path": str(file_path)},
        )

    def import_directory(
        self,
        notebook_id: str,
        dir_path: Path,
        extensions: list[str] | None = None,
    ) -> list[Source]:
        """Import all files from a directory as sources."""
        extensions = extensions or [".txt", ".md", ".py", ".js", ".json"]
        sources = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                source = self.import_text_file(notebook_id, file_path)
                if source:
                    sources.append(source)

        return sources
