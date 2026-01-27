"""Retrieval layer for Shad.

This package provides a unified interface for searching across knowledge bases,
with qmd as the primary backend and filesystem search as a fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from shad.retrieval.filesystem import FilesystemRetriever
from shad.retrieval.layer import RetrievalLayer, RetrievalResult
from shad.retrieval.qmd import QmdRetriever

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalLayer",
    "RetrievalResult",
    "QmdRetriever",
    "FilesystemRetriever",
    "get_retriever",
]


def get_retriever(
    paths: list[Path] | None = None,
    collection_names: dict[str, Path] | None = None,
    prefer: str = "auto",
) -> RetrievalLayer:
    """Get the best available retrieval backend.

    Args:
        paths: List of paths to search (for filesystem fallback)
        collection_names: Mapping of collection name -> path
        prefer: Preferred backend - "auto", "qmd", or "filesystem"

    Returns:
        A RetrievalLayer implementation
    """
    if prefer == "filesystem":
        logger.info("Using filesystem retriever (requested)")
        return FilesystemRetriever(paths=paths, collection_names=collection_names)

    # Try qmd first
    qmd = QmdRetriever()
    if qmd.available and prefer != "filesystem":
        logger.info("Using qmd retriever")
        return qmd

    # Fall back to filesystem
    if prefer == "qmd":
        logger.warning("qmd requested but not available, falling back to filesystem")
    else:
        logger.info("qmd not available, using filesystem retriever")

    return FilesystemRetriever(paths=paths, collection_names=collection_names)
