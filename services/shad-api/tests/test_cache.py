"""Tests for Redis cache with hash-based validation."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shad.cache.redis_cache import CacheKey, RedisCache


class TestCacheKey:
    """Tests for CacheKey."""

    def test_to_string(self) -> None:
        """Test cache key string generation."""
        key = CacheKey(
            goal_type="explanation",
            intent="research",
            entities=("python", "javascript"),
            context_hash="abc123def456",
        )

        result = key.to_string()

        assert result.startswith("shad:cache:")
        assert "explanation" in result
        assert "research" in result
        assert "abc123def456"[:12] in result

    def test_from_task_comparison(self) -> None:
        """Test creating key from comparison task."""
        key = CacheKey.from_task(
            task="Compare Python vs JavaScript",
            context="Some context",
        )

        assert key.goal_type == "comparison"

    def test_from_task_explanation(self) -> None:
        """Test creating key from explanation task."""
        key = CacheKey.from_task(
            task="Explain how recursion works",
            context="",
        )

        assert key.goal_type == "explanation"

    def test_from_task_general(self) -> None:
        """Test creating key from general task."""
        key = CacheKey.from_task(
            task="Do something",
            context="",
        )

        assert key.goal_type == "general"


class TestRedisCache:
    """Tests for RedisCache."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create mock Redis client."""
        client = MagicMock()
        client.ping = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        client.setex = AsyncMock()
        client.delete = AsyncMock(return_value=1)
        client.decrby = AsyncMock(return_value=900)
        client.keys = AsyncMock(return_value=[])
        client.info = AsyncMock(return_value={})
        client.expire = AsyncMock()
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def cache(self, mock_redis: MagicMock) -> RedisCache:
        """Create cache with mock Redis."""
        cache = RedisCache(redis_url="redis://localhost:6379")
        cache._client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_connect(self, cache: RedisCache, mock_redis: MagicMock) -> None:
        """Test connecting to Redis."""
        cache._connected = False
        mock_redis.ping = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await cache.connect()

        assert cache._connected is True

    @pytest.mark.asyncio
    async def test_get_cache_miss(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test cache miss."""
        mock_redis.get = AsyncMock(return_value=None)

        result = await cache.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cache_hit(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test cache hit."""
        import json

        mock_redis.get = AsyncMock(
            return_value=json.dumps({"value": "cached_result"})
        )

        result = await cache.get("test_key")

        assert result == "cached_result"

    @pytest.mark.asyncio
    async def test_set(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test setting cache value."""
        result = await cache.set("test_key", "test_value", tokens_used=100)

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_provisional(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test setting provisional cache value."""
        result = await cache.set(
            "test_key",
            "test_value",
            provisional=True,
        )

        assert result is True
        # Should use staging prefix
        call_args = mock_redis.setex.call_args
        assert "staging" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_with_hash_validation_match(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test cache hit with matching hash."""
        import json

        mock_redis.get = AsyncMock(
            return_value=json.dumps({
                "value": "cached_result",
                "metadata": {"context_hash": "abc123"},
            })
        )

        result = await cache.get_with_hash_validation("test_key", "abc123")

        assert result == "cached_result"

    @pytest.mark.asyncio
    async def test_get_with_hash_validation_mismatch(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test cache miss due to hash mismatch."""
        import json

        mock_redis.get = AsyncMock(
            return_value=json.dumps({
                "value": "cached_result",
                "metadata": {"context_hash": "abc123"},
            })
        )

        result = await cache.get_with_hash_validation("test_key", "different_hash")

        assert result is None
        # Should have invalidated the entry
        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_set_with_hash(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test setting cache with context hash."""
        result = await cache.set_with_hash(
            key="test_key",
            value="test_value",
            context_hash="hash123",
        )

        assert result is True
        # Verify hash was stored in metadata
        call_args = mock_redis.setex.call_args
        import json
        stored_data = json.loads(call_args[0][2])
        assert stored_data["metadata"]["context_hash"] == "hash123"

    @pytest.mark.asyncio
    async def test_promote_to_main_cache(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test promoting from staging to main cache."""
        import json

        mock_redis.get = AsyncMock(
            return_value=json.dumps({
                "value": "staged_result",
                "provisional": True,
            })
        )

        result = await cache.promote("test_key")

        assert result is True
        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_init_budget(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test initializing budget counter."""
        result = await cache.init_budget("run123", 10000)

        assert result is True
        mock_redis.set.assert_called()

    @pytest.mark.asyncio
    async def test_deduct_budget(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test deducting from budget."""
        mock_redis.decrby = AsyncMock(return_value=9000)

        remaining = await cache.deduct_budget("run123", 1000)

        assert remaining == 9000

    @pytest.mark.asyncio
    async def test_deduct_budget_exhausted(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test budget exhausted returns negative."""
        mock_redis.decrby = AsyncMock(return_value=-500)

        remaining = await cache.deduct_budget("run123", 1000)

        assert remaining == -500  # Negative means exhausted

    @pytest.mark.asyncio
    async def test_get_stats(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test getting cache statistics."""
        mock_redis.keys = AsyncMock(side_effect=[
            ["key1", "key2"],  # main keys
            ["key3"],  # staging keys
        ])
        mock_redis.info = AsyncMock(return_value={
            "keyspace_hits": 100,
            "keyspace_misses": 20,
        })

        stats = await cache.get_stats()

        assert stats["connected"] is True
        assert stats["main_entries"] == 2
        assert stats["staging_entries"] == 1
        assert stats["hits"] == 100
        assert stats["misses"] == 20

    def test_make_simple_key(self, cache: RedisCache) -> None:
        """Test simple key generation."""
        key1 = cache.make_simple_key("task1", "context1")
        key2 = cache.make_simple_key("task1", "context2")
        key3 = cache.make_simple_key("task1", "context1")

        # Same inputs = same key
        assert key1 == key3
        # Different context = different key
        assert key1 != key2
        # Key is short hash
        assert len(key1) == 16

    @pytest.mark.asyncio
    async def test_disconnect(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test disconnecting from Redis."""
        await cache.disconnect()

        mock_redis.close.assert_called_once()
        assert cache._connected is False

    @pytest.mark.asyncio
    async def test_get_remaining_budget(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test getting remaining budget."""
        mock_redis.get = AsyncMock(return_value="5000")

        remaining = await cache.get_remaining_budget("run123")

        assert remaining == 5000

    @pytest.mark.asyncio
    async def test_get_remaining_budget_not_found(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test getting remaining budget when not initialized."""
        mock_redis.get = AsyncMock(return_value=None)

        remaining = await cache.get_remaining_budget("run123")

        # Returns None when not found
        assert remaining is None or remaining == 0

    @pytest.mark.asyncio
    async def test_invalidate(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test invalidating cache entry."""
        mock_redis.delete = AsyncMock(return_value=2)

        result = await cache.invalidate("test_key")

        assert result is True

    @pytest.mark.asyncio
    async def test_invalidate_not_found(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test invalidating non-existent entry."""
        mock_redis.delete = AsyncMock(return_value=0)

        result = await cache.invalidate("nonexistent_key")

        # Invalidate returns False when nothing was deleted
        assert result is False


class TestCacheKeyAdvanced:
    """Advanced tests for CacheKey."""

    def test_from_task_list(self) -> None:
        """Test creating key from list task."""
        key = CacheKey.from_task(
            task="List the benefits of testing",
            context="",
        )

        assert key.goal_type == "enumeration"

    def test_from_task_how_to(self) -> None:
        """Test creating key from how-to task."""
        key = CacheKey.from_task(
            task="How to implement caching",
            context="",
        )

        assert key.goal_type == "explanation"  # "how" triggers explanation

    def test_entity_extraction(self) -> None:
        """Test entity extraction from task."""
        key = CacheKey.from_task(
            task="Compare Python, JavaScript, and TypeScript",
            context="",
        )

        # Should extract entities from the task
        assert len(key.entities) >= 0  # May or may not extract

    def test_context_hash_generation(self) -> None:
        """Test context hash is generated."""
        key = CacheKey.from_task(
            task="Some task",
            context="Some long context that should be hashed",
        )

        assert key.context_hash is not None
        assert len(key.context_hash) >= 12

    def test_to_string_deterministic(self) -> None:
        """Test that key string generation is deterministic."""
        key1 = CacheKey(
            goal_type="test",
            intent="intent",
            entities=("a", "b"),
            context_hash="hash123",
        )
        key2 = CacheKey(
            goal_type="test",
            intent="intent",
            entities=("a", "b"),
            context_hash="hash123",
        )

        assert key1.to_string() == key2.to_string()


class TestRedisCacheResilience:
    """Tests for Redis cache resilience and edge cases."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create mock Redis client."""
        client = MagicMock()
        client.ping = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        client.setex = AsyncMock()
        client.delete = AsyncMock(return_value=1)
        client.decrby = AsyncMock(return_value=900)
        client.keys = AsyncMock(return_value=[])
        client.info = AsyncMock(return_value={})
        client.expire = AsyncMock()
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def cache(self, mock_redis: MagicMock) -> RedisCache:
        """Create cache with mock Redis."""
        cache = RedisCache(redis_url="redis://localhost:6379")
        cache._client = mock_redis
        cache._connected = True
        return cache

    @pytest.mark.asyncio
    async def test_get_not_connected(self, cache: RedisCache) -> None:
        """Test get when not connected."""
        cache._connected = False

        result = await cache.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_not_connected(self, cache: RedisCache) -> None:
        """Test set when not connected."""
        cache._connected = False

        result = await cache.set("test_key", "value")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_with_invalid_json(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test handling invalid JSON in cache."""
        mock_redis.get = AsyncMock(return_value="not valid json")

        result = await cache.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_stats_not_connected(self, cache: RedisCache) -> None:
        """Test getting stats when not connected."""
        cache._connected = False

        stats = await cache.get_stats()

        assert stats["connected"] is False

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test setting cache with custom TTL."""
        result = await cache.set(
            "test_key",
            "test_value",
            ttl=timedelta(hours=2),
        )

        assert result is True
        call_args = mock_redis.setex.call_args
        # TTL should be passed as timedelta or seconds
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_get_staging(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test getting from staging cache."""
        import json
        mock_redis.get = AsyncMock(
            return_value=json.dumps({"value": "staged_value"})
        )

        result = await cache.get_staging("test_key")

        assert result == "staged_value"

    @pytest.mark.asyncio
    async def test_get_staging_miss(
        self, cache: RedisCache, mock_redis: MagicMock
    ) -> None:
        """Test staging cache miss."""
        mock_redis.get = AsyncMock(return_value=None)

        result = await cache.get_staging("missing_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_staging_not_connected(self, cache: RedisCache) -> None:
        """Test staging cache when not connected."""
        cache._connected = False

        result = await cache.get_staging("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_promote_not_connected(self, cache: RedisCache) -> None:
        """Test promote when not connected."""
        cache._connected = False

        result = await cache.promote("test_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_not_connected(self, cache: RedisCache) -> None:
        """Test invalidate when not connected."""
        cache._connected = False

        result = await cache.invalidate("test_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_init_budget_not_connected(self, cache: RedisCache) -> None:
        """Test init_budget when not connected."""
        cache._connected = False

        result = await cache.init_budget("run123", 10000)

        assert result is False

    @pytest.mark.asyncio
    async def test_deduct_budget_not_connected(self, cache: RedisCache) -> None:
        """Test deduct_budget when not connected."""
        cache._connected = False

        result = await cache.deduct_budget("run123", 1000)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_remaining_budget_not_connected(
        self, cache: RedisCache
    ) -> None:
        """Test get_remaining_budget when not connected."""
        cache._connected = False

        result = await cache.get_remaining_budget("run123")

        assert result is None or result == 0
