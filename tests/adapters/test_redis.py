"""Tests for RedisStorageAdapter.

Skipped if redis-py is not installed or no Redis instance is reachable.
"""

import os
from datetime import datetime, timezone

import pytest

from kemi.models import LifecycleState, MemoryObject, MemorySource


def _redis_available() -> tuple[bool, str]:
    try:
        import redis
    except ImportError:
        return False, "redis not installed"
    try:
        r = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        r.ping()
        r.close()
    except Exception as e:
        return False, f"Redis not reachable: {e}"
    return True, ""


_redis_ok, _redis_reason = _redis_available()
pytestmark = pytest.mark.skipif(not _redis_ok, reason=_redis_reason)


@pytest.fixture
def redis_adapter():
    from kemi.adapters.storage.redis import RedisStorageAdapter

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    adapter = RedisStorageAdapter(url=url)
    # Flush test keys (only the prefix namespace)
    keys = adapter._redis.keys(f"{adapter._prefix}:*")
    if keys:
        adapter._redis.delete(*keys)
    yield adapter
    keys = adapter._redis.keys(f"{adapter._prefix}:*")
    if keys:
        adapter._redis.delete(*keys)
    adapter.close()


def _make_mem(**overrides: object) -> MemoryObject:
    defaults = {
        "memory_id": "test-id",
        "user_id": "user1",
        "content": "test content",
        "embedding": [0.1] * 64,
        "score": 0.0,
        "created_at": datetime.now(timezone.utc),
        "last_accessed_at": datetime.now(timezone.utc),
        "source": MemorySource.USER_STATED,
        "importance": 0.5,
        "lifecycle_state": LifecycleState.ACTIVE,
        "metadata": {},
        "embedding_dim": 64,
        "tags": [],
        "confidence": 1.0,
        "memory_type": "episodic",
        "session_id": None,
        "namespace": "default",
        "version": 1,
        "agent_id": None,
        "run_id": None,
        "app_id": None,
    }
    defaults.update(overrides)
    return MemoryObject(**defaults)


class TestRedisStorageAdapter:
    def test_store_and_get(self, redis_adapter) -> None:
        mem = _make_mem(
            memory_id="r1",
            content="I am vegetarian",
            source=MemorySource.AGENT_INFERRED,
            importance=0.7,
        )
        redis_adapter.store(mem)
        result = redis_adapter.get("r1")
        assert result is not None
        assert result.memory_id == "r1"
        assert result.content == "I am vegetarian"
        assert result.source == MemorySource.AGENT_INFERRED
        assert result.importance == 0.7
        assert result.embedding == pytest.approx([0.1] * 64)

    def test_search(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="s1", embedding=[1.0] * 64))
        redis_adapter.store(_make_mem(memory_id="s2", embedding=[0.0] * 64))
        results = redis_adapter.search("user1", [1.0] * 64, top_k=10)
        assert len(results) == 2

    def test_search_lifecycle(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="a1"))
        redis_adapter.store(_make_mem(memory_id="d1", lifecycle_state=LifecycleState.DELETED))
        results = redis_adapter.search("user1", [1.0] * 64, top_k=10)
        assert all(m.lifecycle_state != LifecycleState.DELETED for m in results)

    def test_delete_by_id(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="x1"))
        assert redis_adapter.delete_by_id("x1") is True
        assert redis_adapter.get("x1") is None
        assert redis_adapter.delete_by_id("x1") is False

    def test_delete_by_user(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="u1", user_id="u1"))
        redis_adapter.store(_make_mem(memory_id="u2", user_id="u1"))
        count = redis_adapter.delete_by_user("u1")
        assert count == 2
        assert redis_adapter.get("u1") is None
        assert redis_adapter.get("u2") is None

    def test_count(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="c1"))
        assert redis_adapter.count("user1") == 1

    def test_embedding_roundtrip(self, redis_adapter) -> None:
        emb = [0.1 * i for i in range(64)]
        redis_adapter.store(_make_mem(memory_id="e1", embedding=emb, embedding_dim=64))
        result = redis_adapter.get("e1")
        assert result is not None
        assert result.embedding is not None
        assert result.embedding == pytest.approx(emb)

    def test_get_all_by_user(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="g1"))
        redis_adapter.store(_make_mem(memory_id="g2", lifecycle_state=LifecycleState.DECAYING))
        results = redis_adapter.get_all_by_user("user1")
        assert len(results) == 2

    def test_get_all(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="a1", user_id="u1"))
        redis_adapter.store(_make_mem(memory_id="a2", user_id="u2"))
        results = redis_adapter.get_all()
        assert len(results) == 2

    def test_get_all_users(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="uu1", user_id="alice"))
        redis_adapter.store(_make_mem(memory_id="uu2", user_id="bob"))
        users = redis_adapter.get_all_users()
        assert "alice" in users
        assert "bob" in users

    def test_get_by_tag(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="t1", tags=["cat", "pet"]))
        redis_adapter.store(_make_mem(memory_id="t2", tags=["dog", "pet"]))
        results = redis_adapter.get_by_tag("user1", "cat")
        assert len(results) == 1
        assert results[0].memory_id == "t1"

    def test_search_by_content(self, redis_adapter) -> None:
        redis_adapter.store(_make_mem(memory_id="sc1", content="I love pizza"))
        redis_adapter.store(_make_mem(memory_id="sc2", content="I love sushi"))
        results = redis_adapter.search_by_content("user1", "pizza")
        assert len(results) == 1
        assert results[0].memory_id == "sc1"

    def test_store_many(self, redis_adapter) -> None:
        mems = [_make_mem(memory_id=f"b{i}") for i in range(5)]
        count = redis_adapter.store_many(mems)
        assert count == 5
        assert redis_adapter.get("b0") is not None
        assert redis_adapter.get("b4") is not None

    def test_context_manager(self) -> None:
        from kemi.adapters.storage.redis import RedisStorageAdapter

        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        with RedisStorageAdapter(url=url) as adapter:
            adapter.store(_make_mem(memory_id="ctx1", content="ctx test"))
            result = adapter.get("ctx1")
            assert result is not None
            assert result.content == "ctx test"

    def test_upgrade_schema(self, redis_adapter) -> None:
        redis_adapter.upgrade_schema(0, 1)

    def test_namespace_filter(self, redis_adapter) -> None:
        redis_adapter.store(
            _make_mem(memory_id="ns1", namespace="team-space", embedding=[1.0] * 64)
        )
        redis_adapter.store(_make_mem(memory_id="ns2", namespace="default", embedding=[1.0] * 64))
        results = redis_adapter.search("user1", [1.0] * 64, namespace="team-space")
        assert len(results) == 1
        assert results[0].memory_id == "ns1"

    def test_get_nonexistent(self, redis_adapter) -> None:
        assert redis_adapter.get("nonexistent") is None

    def test_delete_by_id_twice(self, redis_adapter) -> None:
        """Deleting an already-deleted id should return False."""
        redis_adapter.store(_make_mem(memory_id="d-twice"))
        assert redis_adapter.delete_by_id("d-twice") is True
        # Second delete is a no-op
        assert redis_adapter.delete_by_id("d-twice") is False
        assert redis_adapter.delete_by_id("d-twice") is False

    def test_delete_by_id_cleans_tag_index(self, redis_adapter) -> None:
        """After delete, the memory id should be removed from tag indices."""
        redis_adapter.store(_make_mem(memory_id="tag-clean", tags=["vegan"]))
        # Pre-condition: tag index has the memory
        pre = redis_adapter._redis.smembers(
            redis_adapter._tag_index_key("user1", "vegan")
        )
        assert "tag-clean" in pre

        redis_adapter.delete_by_id("tag-clean")
        post = redis_adapter._redis.smembers(
            redis_adapter._tag_index_key("user1", "vegan")
        )
        assert "tag-clean" not in post

    def test_delete_by_user_no_memories(self, redis_adapter) -> None:
        """delete_by_user on a user with no memories returns 0."""
        assert redis_adapter.delete_by_user("ghost-user") == 0

    def test_close_idempotent(self) -> None:
        """Calling close() twice should not raise."""
        from kemi.adapters.storage.redis import RedisStorageAdapter

        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        adapter = RedisStorageAdapter(url=url)
        adapter.close()
        # Second close should be a no-op
        adapter.close()

    def test_update_memory(self, redis_adapter) -> None:
        """update() should modify fields of an existing memory."""
        redis_adapter.store(
            _make_mem(memory_id="upd1", content="before", importance=0.5)
        )
        result = redis_adapter.get("upd1")
        assert result is not None
        result.content = "after"
        result.importance = 0.9
        assert redis_adapter.update(result) is True
        reloaded = redis_adapter.get("upd1")
        assert reloaded is not None
        assert reloaded.content == "after"
        assert reloaded.importance == 0.9

    def test_update_nonexistent_returns_false(self, redis_adapter) -> None:
        """update() on a non-existent memory should return False."""
        result = _make_mem(memory_id="nope")
        assert redis_adapter.update(result) is False

    def test_get_all_by_user_with_namespace(self, redis_adapter) -> None:
        """get_all_by_user should respect the namespace argument."""
        redis_adapter.store(_make_mem(memory_id="g-ns1", namespace="alpha"))
        redis_adapter.store(_make_mem(memory_id="g-ns2", namespace="beta"))
        results = redis_adapter.get_all_by_user("user1", namespace="alpha")
        assert len(results) == 1
        assert results[0].memory_id == "g-ns1"

    def test_count_per_user_isolated(self, redis_adapter) -> None:
        """count() should be per-user, not global."""
        redis_adapter.store(_make_mem(memory_id="c-u1", user_id="u1"))
        redis_adapter.store(_make_mem(memory_id="c-u2a", user_id="u2"))
        redis_adapter.store(_make_mem(memory_id="c-u2b", user_id="u2"))
        assert redis_adapter.count("u1") == 1
        assert redis_adapter.count("u2") == 2

    def test_search_top_k(self, redis_adapter) -> None:
        """search() should respect the top_k limit."""
        for i in range(10):
            redis_adapter.store(_make_mem(memory_id=f"k{i}"))
        results = redis_adapter.search("user1", [1.0] * 64, top_k=3)
        assert len(results) == 3

    def test_init_without_redis_raises(self, monkeypatch) -> None:
        """Importing the module without redis installed should fail at init."""
        import sys
        from kemi.adapters.storage import redis as redis_mod

        monkeypatch.setattr(redis_mod, "_REDIS_AVAILABLE", False)
        with pytest.raises(ImportError, match="redis"):
            redis_mod.RedisStorageAdapter(url="redis://localhost:6379/0")

    def test_get_by_tag_no_matches(self, redis_adapter) -> None:
        """get_by_tag() with no matches returns empty list."""
        assert redis_adapter.get_by_tag("user1", "nonexistent-tag") == []

    def test_ttl_field_roundtrip(self, redis_adapter) -> None:
        """expires_at should roundtrip through the hash field."""
        from datetime import timedelta

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        mem = _make_mem(memory_id="ttl-r", expires_at=future)
        redis_adapter.store(mem)
        result = redis_adapter.get("ttl-r")
        assert result is not None
        assert result.expires_at is not None
        # Approximate equality (microsecond loss from serialization)
        diff = abs((result.expires_at - future).total_seconds())
        assert diff < 1
