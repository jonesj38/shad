"""OpenNotebookLM data models - Graph-based knowledge substrate."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    NOTEBOOK = "notebook"
    SOURCE = "source"
    NOTE = "note"


class EdgeType(str, Enum):
    """Types of edges between nodes."""

    DERIVED_FROM = "derived_from"
    SUMMARIZES = "summarizes"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    REFERENCES = "references"
    PART_OF = "part_of"
    INCLUDED_IN = "included_in"


class Edge(BaseModel):
    """An edge in the knowledge graph."""

    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Node(BaseModel):
    """Base class for all nodes in the knowledge graph."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    title: str
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None


class Notebook(Node):
    """A contextual container / lens / view over the graph."""

    node_type: NodeType = NodeType.NOTEBOOK
    description: str = ""
    policies: dict[str, Any] = Field(default_factory=dict)


class Source(Node):
    """An external or primary artifact (PDF, web page, transcript, etc.)."""

    node_type: NodeType = NodeType.SOURCE
    source_type: str = "text"  # text, pdf, web, transcript, code, etc.
    url: str | None = None
    checksum: str | None = None
    extracted_text: str = ""


class Note(Node):
    """A derived, human- or AI-authored artifact."""

    node_type: NodeType = NodeType.NOTE
    author: str = "system"  # system, human, or model name
    confidence: float = 1.0
    provisional: bool = False
    source_refs: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result from a retrieval query."""

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    total_nodes: int = 0
    query: str = ""
