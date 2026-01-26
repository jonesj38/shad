"""Model registry and resolution for Shad."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from shad.utils.config import get_settings

logger = logging.getLogger(__name__)

# Shorthand aliases for common models
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-20250514",
    "sonnet": "sonnet",  # Will be resolved from API or cache
    "haiku": "haiku",  # Will be resolved from API or cache
}

# Built-in defaults (used only when no cache and no API key)
BUILTIN_MODELS: list[dict[str, Any]] = [
    {"id": "claude-opus-4-20250514", "shorthand": "opus", "display_name": "Claude Opus 4"},
    {"id": "claude-sonnet-4-20250514", "shorthand": "sonnet", "display_name": "Claude Sonnet 4"},
    {"id": "claude-haiku-4-20250514", "shorthand": "haiku", "display_name": "Claude Haiku 4"},
    {"id": "claude-3-5-sonnet-20241022", "shorthand": None, "display_name": "Claude 3.5 Sonnet"},
]

# Cache settings
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _get_shad_home() -> Path:
    """Get the Shad home directory."""
    return Path(os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad")))


def _get_cache_path() -> Path:
    """Get path to models cache file."""
    return _get_shad_home() / "models_cache.json"


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    shorthand: str | None = None
    display_name: str | None = None
    created: str | None = None
    context_window: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-assign shorthand for known model families (only Claude 4 models)
        # Check for "-4-" pattern to identify Claude 4 models specifically
        if self.shorthand is None:
            model_lower = self.id.lower()
            # Only assign shorthands to Claude 4 models (e.g., claude-opus-4-20250514)
            if "-4-" in self.id:
                if "opus" in model_lower:
                    self.shorthand = "opus"
                elif "sonnet" in model_lower:
                    self.shorthand = "sonnet"
                elif "haiku" in model_lower:
                    self.shorthand = "haiku"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        """Create ModelInfo from a dictionary."""
        return cls(
            id=data.get("id", ""),
            shorthand=data.get("shorthand"),
            display_name=data.get("display_name"),
            created=data.get("created"),
            context_window=data.get("context_window"),
            metadata=data.get("metadata", {}),
        )


def _get_shorthand_models() -> list[ModelInfo]:
    """Helper to get models with shorthands (for deduplication)."""
    return []


@dataclass
class ModelsCache:
    """Cached models data."""

    models: list[dict[str, Any]]
    fetched_at: float

    def is_valid(self) -> bool:
        """Check if cache is still valid (within TTL)."""
        return (time.time() - self.fetched_at) < CACHE_TTL_SECONDS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "models": self.models,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelsCache:
        """Create from dictionary."""
        return cls(
            models=data.get("models", []),
            fetched_at=data.get("fetched_at", 0),
        )


def _load_cache() -> ModelsCache | None:
    """Load models cache from disk."""
    cache_path = _get_cache_path()
    if not cache_path.exists():
        return None
    try:
        with cache_path.open() as f:
            data = json.load(f)
        return ModelsCache.from_dict(data)
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.debug(f"Failed to load models cache: {e}")
        return None


def _save_cache(cache: ModelsCache) -> None:
    """Save models cache to disk."""
    cache_path = _get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("w") as f:
            json.dump(cache.to_dict(), f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to save models cache: {e}")


def _fetch_models_from_api() -> list[dict[str, Any]] | None:
    """Fetch available models from Anthropic API."""
    settings = get_settings()
    api_key = settings.anthropic_api_key

    if not api_key:
        logger.debug("No ANTHROPIC_API_KEY set, cannot fetch models from API")
        return None

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            response.raise_for_status()
            data = response.json()

        models: list[dict[str, Any]] = []
        for model_data in data.get("data", []):
            model_id = model_data.get("id", "")
            # Skip non-Claude models
            if not model_id.startswith("claude"):
                continue

            # Determine shorthand
            shorthand: str | None = None
            model_lower = model_id.lower()
            if "opus" in model_lower and "4" in model_id:
                shorthand = "opus"
            elif "sonnet" in model_lower and "4" in model_id:
                shorthand = "sonnet"
            elif "haiku" in model_lower and "4" in model_id:
                shorthand = "haiku"

            models.append({
                "id": model_id,
                "shorthand": shorthand,
                "display_name": model_data.get("display_name", model_id),
                "created": model_data.get("created_at"),
            })

        logger.info(f"Fetched {len(models)} models from Anthropic API")
        return models

    except httpx.HTTPStatusError as e:
        logger.warning(f"API error fetching models: {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error fetching models: {e}")
        return None


def get_available_models(force_refresh: bool = False) -> list[ModelInfo]:
    """Get available models from cache, API, or built-in defaults.

    Priority order:
    1. If force_refresh and API key available: fetch from API
    2. Valid cache: use cached models
    3. API available: fetch and cache
    4. Built-in defaults: fallback

    Args:
        force_refresh: Force refresh from API even if cache is valid

    Returns:
        List of ModelInfo objects
    """
    settings = get_settings()

    # Check cache first (unless forcing refresh)
    if not force_refresh:
        cache = _load_cache()
        if cache and cache.is_valid():
            logger.debug("Using cached models")
            return [ModelInfo.from_dict(m) for m in cache.models]

    # Try to fetch from API
    if settings.anthropic_api_key:
        api_models = _fetch_models_from_api()
        if api_models:
            # Save to cache
            cache = ModelsCache(models=api_models, fetched_at=time.time())
            _save_cache(cache)
            return [ModelInfo.from_dict(m) for m in api_models]

    # Fall back to cache even if expired
    cache = _load_cache()
    if cache:
        logger.debug("Using expired cache (API unavailable)")
        return [ModelInfo.from_dict(m) for m in cache.models]

    # Last resort: built-in defaults
    logger.debug("Using built-in model defaults")
    return [ModelInfo.from_dict(m) for m in BUILTIN_MODELS]


def get_model_by_shorthand(shorthand: str) -> str | None:
    """Resolve a shorthand to a full model ID.

    Args:
        shorthand: Shorthand like 'opus', 'sonnet', 'haiku'

    Returns:
        Full model ID or None if not found
    """
    models = get_available_models()
    for model in models:
        if model.shorthand == shorthand:
            return model.id
    return None


def normalize_model_name(name: str) -> str:
    """Normalize a model name (shorthand or full ID) to a full model ID.

    Accepts:
    - Full model IDs: returned as-is
    - Shorthands: resolved to full model ID

    Args:
        name: Model name (shorthand or full ID)

    Returns:
        Full model ID

    Raises:
        ValueError: If shorthand cannot be resolved
    """
    # Check if it's already a full model ID
    if name.startswith("claude-"):
        return name

    # Try to resolve as shorthand
    resolved = get_model_by_shorthand(name.lower())
    if resolved:
        return resolved

    # Check built-in aliases as fallback
    for shorthand, full_id in MODEL_ALIASES.items():
        if name.lower() == shorthand:
            # For dynamic shorthands (sonnet, haiku), check settings
            if full_id == shorthand:
                settings = get_settings()
                if shorthand == "sonnet":
                    return settings.worker_model
                elif shorthand == "haiku":
                    return settings.leaf_model
            return full_id

    raise ValueError(f"Unknown model: {name}. Use 'shad models' to see available models.")


def get_default_models() -> dict[str, str]:
    """Get the default models for each tier from settings.

    Returns:
        Dict with keys: orchestrator, worker, leaf
    """
    settings = get_settings()
    return {
        "orchestrator": settings.orchestrator_model,
        "worker": settings.worker_model,
        "leaf": settings.leaf_model,
    }
