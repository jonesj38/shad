"""Redis-based subtree caching for RLM engine.

Implements hierarchical key scheme per SPEC.md:
- (goal_type, intent, entities, key_slots...) → stable key
- Fallback: exact string hash
- Main cache + staging cache for provisional results

Per OBSIDIAN_PIVOT.md Section 6: Caching Strategy
- Hash validation: Cache keys include context_hash derived from file content/mtime
- Before cache lookup: query MCP server for current file hash
- If hash mismatch: cache miss, re-compute reasoning
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import redis.asyncio as redis
from pydantic import BaseModel

from shad.utils.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheKey:
    """Hierarchical cache key components."""

    goal_type: str
    intent: str
    entities: tuple[str, ...]
    context_hash: str
    extra_slots: tuple[str, ...] = ()

    def to_string(self) -> str:
        """Convert to string key for Redis."""
        parts = [
            self.goal_type,
            self.intent,
            ":".join(sorted(self.entities)) if self.entities else "_",
            self.context_hash[:12],
        ]
        if self.extra_slots:
            parts.extend(self.extra_slots)
        return "shad:cache:" + ":".join(parts)

    @classmethod
    def from_task(
        cls,
        task: str,
        context: str = "",
        intent: str = "general",
        entities: list[str] | None = None,
    ) -> CacheKey:
        """Create a cache key from task components."""
        # Simple goal type detection
        task_lower = task.lower()
        if any(w in task_lower for w in ["compare", "difference", "vs"]):
            goal_type = "comparison"
        elif any(w in task_lower for w in ["explain", "what", "how", "why"]):
            goal_type = "explanation"
        elif any(w in task_lower for w in ["list", "enumerate", "find all"]):
            goal_type = "enumeration"
        elif any(w in task_lower for w in ["summarize", "summary"]):
            goal_type = "summary"
        else:
            goal_type = "general"

        # Hash context for key
        context_hash = hashlib.sha256(context.encode()).hexdigest()

        return cls(
            goal_type=goal_type,
            intent=intent,
            entities=tuple(entities) if entities else (),
            context_hash=context_hash,
        )


class CacheEntry(BaseModel):
    """A cached result entry."""

    key: str
    value: str
    tokens_used: int = 0
    created_at: str
    ttl_seconds: int | None = None
    provisional: bool = False
    metadata: dict[str, Any] = {}


class RedisCache:
    """
    Redis-based cache for RLM subtree results.

    Features:
    - Hierarchical key scheme for stable cache keys
    - Main cache for verified results
    - Staging cache for provisional results awaiting review
    - TTL support for cache expiration
    - Atomic operations for cache updates
    """

    # Cache namespaces
    MAIN_PREFIX = "shad:cache:main:"
    STAGING_PREFIX = "shad:cache:staging:"

    # Default TTLs
    DEFAULT_MAIN_TTL = timedelta(days=30)
    DEFAULT_STAGING_TTL = timedelta(hours=24)

    def __init__(self, redis_url: str | None = None):
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self._client: redis.Redis | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._client = None
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False

    async def get(self, key: str | CacheKey) -> str | None:
        """
        Get a cached result from main cache.

        Args:
            key: Cache key (string or CacheKey object)

        Returns:
            Cached result or None if not found
        """
        if not self._connected or not self._client:
            return None

        str_key = key.to_string() if isinstance(key, CacheKey) else key
        full_key = self.MAIN_PREFIX + str_key

        try:
            data = await self._client.get(full_key)
            if data:
                entry = json.loads(data)
                logger.debug(f"Cache hit for {str_key}")
                return entry.get("value")
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        return None

    async def set(
        self,
        key: str | CacheKey,
        value: str,
        tokens_used: int = 0,
        ttl: timedelta | None = None,
        provisional: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Set a cached result.

        Args:
            key: Cache key
            value: Result to cache
            tokens_used: Tokens used to generate this result
            ttl: Time-to-live (uses default if None)
            provisional: If True, store in staging cache
            metadata: Additional metadata to store

        Returns:
            True if cached successfully
        """
        if not self._connected or not self._client:
            return False

        str_key = key.to_string() if isinstance(key, CacheKey) else key
        prefix = self.STAGING_PREFIX if provisional else self.MAIN_PREFIX
        full_key = prefix + str_key

        if ttl is None:
            ttl = self.DEFAULT_STAGING_TTL if provisional else self.DEFAULT_MAIN_TTL

        from datetime import datetime

        entry = {
            "key": str_key,
            "value": value,
            "tokens_used": tokens_used,
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": int(ttl.total_seconds()),
            "provisional": provisional,
            "metadata": metadata or {},
        }

        try:
            await self._client.setex(
                full_key,
                ttl,
                json.dumps(entry),
            )
            logger.debug(f"Cached result for {str_key} (provisional={provisional})")
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def get_staging(self, key: str | CacheKey) -> str | None:
        """Get a result from staging cache."""
        if not self._connected or not self._client:
            return None

        str_key = key.to_string() if isinstance(key, CacheKey) else key
        full_key = self.STAGING_PREFIX + str_key

        try:
            data = await self._client.get(full_key)
            if data:
                entry = json.loads(data)
                return entry.get("value")
        except Exception as e:
            logger.warning(f"Staging cache get error: {e}")

        return None

    async def promote(self, key: str | CacheKey) -> bool:
        """
        Promote a result from staging to main cache.

        Used after HITL review approves a provisional result.
        """
        if not self._connected or not self._client:
            return False

        str_key = key.to_string() if isinstance(key, CacheKey) else key
        staging_key = self.STAGING_PREFIX + str_key
        main_key = self.MAIN_PREFIX + str_key

        try:
            # Get from staging
            data = await self._client.get(staging_key)
            if not data:
                return False

            entry = json.loads(data)
            entry["provisional"] = False
            entry["promoted_at"] = __import__("datetime").datetime.utcnow().isoformat()

            # Set in main cache with longer TTL
            await self._client.setex(
                main_key,
                self.DEFAULT_MAIN_TTL,
                json.dumps(entry),
            )

            # Remove from staging
            await self._client.delete(staging_key)

            logger.info(f"Promoted cache entry {str_key} to main cache")
            return True
        except Exception as e:
            logger.warning(f"Cache promote error: {e}")
            return False

    async def invalidate(self, key: str | CacheKey) -> bool:
        """Invalidate a cache entry."""
        if not self._connected or not self._client:
            return False

        str_key = key.to_string() if isinstance(key, CacheKey) else key

        try:
            # Delete from both caches
            deleted = await self._client.delete(
                self.MAIN_PREFIX + str_key,
                self.STAGING_PREFIX + str_key,
            )
            return deleted > 0
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self._connected or not self._client:
            return {"connected": False}

        try:
            info = await self._client.info("stats")
            main_keys = await self._client.keys(self.MAIN_PREFIX + "*")
            staging_keys = await self._client.keys(self.STAGING_PREFIX + "*")

            return {
                "connected": True,
                "main_entries": len(main_keys),
                "staging_entries": len(staging_keys),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"connected": True, "error": str(e)}

    async def clear_staging(self) -> int:
        """Clear all staging cache entries."""
        if not self._connected or not self._client:
            return 0

        try:
            keys = await self._client.keys(self.STAGING_PREFIX + "*")
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear staging error: {e}")
            return 0

    # Helper for simple string-based caching (backwards compatible)
    def make_simple_key(self, task: str, context: str = "") -> str:
        """Generate a simple hash-based cache key."""
        key_data = f"{task}::{context[:500] if context else ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def get_with_hash_validation(
        self,
        key: str | CacheKey,
        current_hash: str,
    ) -> str | None:
        """Get cached result with hash validation.

        Per OBSIDIAN_PIVOT.md Section 6.2: Hash Validation
        - Before cache lookup: query for current file hash
        - If hash mismatch: cache miss, re-compute

        Args:
            key: Cache key
            current_hash: Current content hash to validate against

        Returns:
            Cached result if hash matches, None otherwise
        """
        if not self._connected or not self._client:
            return None

        str_key = key.to_string() if isinstance(key, CacheKey) else key
        full_key = self.MAIN_PREFIX + str_key

        try:
            data = await self._client.get(full_key)
            if data:
                entry = json.loads(data)
                stored_hash = entry.get("metadata", {}).get("context_hash")

                # Validate hash
                if stored_hash and stored_hash != current_hash:
                    logger.debug(f"Cache invalidated by hash mismatch for {str_key}")
                    # Hash mismatch - treat as cache miss
                    await self.invalidate(key)
                    return None

                logger.debug(f"Cache hit with valid hash for {str_key}")
                return entry.get("value")
        except Exception as e:
            logger.warning(f"Cache get with hash validation error: {e}")

        return None

    async def set_with_hash(
        self,
        key: str | CacheKey,
        value: str,
        context_hash: str,
        tokens_used: int = 0,
        ttl: timedelta | None = None,
        provisional: bool = False,
    ) -> bool:
        """Set cached result with context hash for validation.

        Per OBSIDIAN_PIVOT.md Section 6.2:
        Cache keys include context_hash derived from file content/mtime.

        Args:
            key: Cache key
            value: Result to cache
            context_hash: Hash of context for validation
            tokens_used: Tokens used
            ttl: Time-to-live
            provisional: If True, store in staging cache

        Returns:
            True if cached successfully
        """
        return await self.set(
            key=key,
            value=value,
            tokens_used=tokens_used,
            ttl=ttl,
            provisional=provisional,
            metadata={"context_hash": context_hash},
        )

    # ==================== Budget Ledger Operations ====================
    # Per OBSIDIAN_PIVOT.md Section 7.2: Central Ledger (Redis)

    BUDGET_PREFIX = "shad:budget:"

    async def init_budget(self, run_id: str, token_budget: int) -> bool:
        """Initialize budget counter for a run.

        Per OBSIDIAN_PIVOT.md Section 7.2:
        Sub-agents decrement shared Redis counter atomically.
        """
        if not self._connected or not self._client:
            return False

        key = f"{self.BUDGET_PREFIX}{run_id}:tokens"
        try:
            await self._client.set(key, token_budget)
            await self._client.expire(key, timedelta(hours=24))
            return True
        except Exception as e:
            logger.error(f"Failed to init budget: {e}")
            return False

    async def deduct_budget(self, run_id: str, tokens: int) -> int | None:
        """Atomically deduct from budget and return remaining.

        Per OBSIDIAN_PIVOT.md Section 7.2:
        Atomic check-and-deduct before each LLM call.

        Returns:
            Remaining budget, or None if error
            Negative value means budget exhausted
        """
        if not self._connected or not self._client:
            return None

        key = f"{self.BUDGET_PREFIX}{run_id}:tokens"
        try:
            remaining = await self._client.decrby(key, tokens)
            return remaining
        except Exception as e:
            logger.error(f"Failed to deduct budget: {e}")
            return None

    async def get_remaining_budget(self, run_id: str) -> int | None:
        """Get remaining budget for a run."""
        if not self._connected or not self._client:
            return None

        key = f"{self.BUDGET_PREFIX}{run_id}:tokens"
        try:
            value = await self._client.get(key)
            return int(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get budget: {e}")
            return None
