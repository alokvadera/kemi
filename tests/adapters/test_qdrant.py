"""Tests for the Qdrant storage adapter.

Uses Qdrant's built-in in-memory mode (``location=":memory:"``)
so no external Qdrant server is required.
"""

from datetime import datetime, timezone

import pytest

from kemi.adapters.storage.qdrant import QdrantStorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource, MemoryType


@pytest.fixture
def qdrant_adapter() -> QdrantStorageAdapter:
    return QdrantStorageAdapter(
        location=":memory:",
        collection_name="test_kemi",
        embedding_dim=64,
    )


def _make_memory(
    memory_id: str,
    user_id: str = "user1",
    content: str = "test memory",
    embedding: list[float] | None = None,
    importance: float = 0.5,
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    namespace: str = "default",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    session_id: str | None = None,
) -> MemoryObject:
    if embedding is None:
        embedding = [0.1] * 64
    return MemoryObject(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=embedding,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=importance,
        lifecycle_state=lifecycle_state,
        metadata=metadata or {},
        embedding_dim=len(embedding),
        tags=tags or [],
        confidence=1.0,
        memory_type=MemoryType.EPISODIC,
        session_id=session_id,
        namespace=namespace,
        version=1,
    )


def test_store_and_get(qdrant_adapter) -> None:
    mem = _make_memory("test-id", content="I am vegetarian", importance=0.7)
    mem.source = MemorySource.AGENT_INFERRED

    qdrant_adapter.store(mem)
    result = qdrant_adapter.get("test-id")

    assert result is not None
    assert result.memory_id == "test-id"
    assert result.user_id == "user1"
    assert result.content == "I am vegetarian"
    assert result.source == MemorySource.AGENT_INFERRED
    assert result.importance == 0.7


def test_search_returns_results(qdrant_adapter) -> None:
    # Use non-colinear vectors so similarity scores are distinct
    mem1 = _make_memory("id1", content="I am vegetarian", embedding=[1.0, 0.0] * 32)
    mem2 = _make_memory("id2", content="I live in Mumbai", embedding=[0.0, 1.0] * 32)

    qdrant_adapter.store(mem1)
    qdrant_adapter.store(mem2)

    query = [1.0, 0.0] * 32
    results = qdrant_adapter.search("user1", query, top_k=10)

    assert len(results) == 2
    assert results[0].memory_id == "id1"


def test_search_lifecycle_filter(qdrant_adapter) -> None:
    active = _make_memory("id1", content="active", lifecycle_state=LifecycleState.ACTIVE)
    deleted = _make_memory("id2", content="deleted", lifecycle_state=LifecycleState.DELETED)

    qdrant_adapter.store(active)
    qdrant_adapter.store(deleted)

    query = [1.0] * 64
    results = qdrant_adapter.search("user1", query, top_k=10)

    assert all(m.lifecycle_state != LifecycleState.DELETED for m in results)


def test_search_with_session_filter(qdrant_adapter) -> None:
    mem1 = _make_memory("id1", content="session1 mem", embedding=[1.0] * 64, session_id="sess-1")
    mem2 = _make_memory("id2", content="session2 mem", embedding=[1.0] * 64, session_id="sess-2")

    qdrant_adapter.store(mem1)
    qdrant_adapter.store(mem2)

    query = [1.0] * 64
    results = qdrant_adapter.search("user1", query, top_k=10, session_id="sess-1")

    assert len(results) == 1
    assert results[0].memory_id == "id1"


def test_delete_by_id(qdrant_adapter) -> None:
    mem = _make_memory("test-id")
    qdrant_adapter.store(mem)

    result = qdrant_adapter.delete_by_id("test-id")
    assert result is True

    get_result = qdrant_adapter.get("test-id")
    assert get_result is None


def test_delete_by_user(qdrant_adapter) -> None:
    mem1 = _make_memory("id1")
    mem2 = _make_memory("id2")

    qdrant_adapter.store(mem1)
    qdrant_adapter.store(mem2)

    count = qdrant_adapter.delete_by_user("user1")
    assert count == 2


def test_count(qdrant_adapter) -> None:
    mem = _make_memory("id1")
    qdrant_adapter.store(mem)

    count = qdrant_adapter.count("user1")
    assert count == 1


def test_get_all_by_user(qdrant_adapter) -> None:
    mem1 = _make_memory("id1", content="test1")
    mem2 = _make_memory("id2", content="test2", lifecycle_state=LifecycleState.DECAYING)

    qdrant_adapter.store(mem1)
    qdrant_adapter.store(mem2)

    results = qdrant_adapter.get_all_by_user("user1")
    assert len(results) == 2


def test_get_all_users(qdrant_adapter) -> None:
    mem1 = _make_memory("id1", user_id="user1")
    mem2 = _make_memory("id2", user_id="user2")

    qdrant_adapter.store(mem1)
    qdrant_adapter.store(mem2)

    users = qdrant_adapter.get_all_users()
    assert sorted(users) == ["user1", "user2"]


def test_update(qdrant_adapter) -> None:
    mem = _make_memory("test-id", content="original")
    qdrant_adapter.store(mem)

    mem.content = "updated"
    qdrant_adapter.update(mem)

    result = qdrant_adapter.get("test-id")
    assert result is not None
    assert result.content == "updated"


def test_get_nonexistent(qdrant_adapter) -> None:
    result = qdrant_adapter.get("non-existent")
    assert result is None


def test_search_by_content(qdrant_adapter) -> None:
    mem1 = _make_memory("id1", content="I love pizza")
    mem2 = _make_memory("id2", content="I love sushi")

    qdrant_adapter.store(mem1)
    qdrant_adapter.store(mem2)

    results = qdrant_adapter.search_by_content("user1", "pizza", top_k=10)
    assert len(results) >= 1


def test_store_many(qdrant_adapter) -> None:
    mem1 = _make_memory("id1")
    mem2 = _make_memory("id2")

    count = qdrant_adapter.store_many([mem1, mem2])
    assert count == 2

    assert qdrant_adapter.count("user1") == 2


def test_context_manager() -> None:
    with QdrantStorageAdapter(
        location=":memory:",
        collection_name="test_ctx",
        embedding_dim=64,
    ) as adapter:
        mem = _make_memory("ctx-1")
        adapter.store(mem)
        result = adapter.get("ctx-1")
        assert result is not None
        assert result.content == "test memory"


def test_metadata_roundtrip(qdrant_adapter) -> None:
    mem = _make_memory("meta-1", metadata={"key": "value", "nested": {"a": 1}})
    qdrant_adapter.store(mem)

    result = qdrant_adapter.get("meta-1")
    assert result is not None
    assert result.metadata["key"] == "value"
    assert result.metadata["nested"]["a"] == 1


def test_tags_roundtrip(qdrant_adapter) -> None:
    mem = _make_memory("tag-1", tags=["hello", "world"])
    qdrant_adapter.store(mem)

    result = qdrant_adapter.get("tag-1")
    assert result is not None
    assert "hello" in result.tags
    assert "world" in result.tags


def test_upgrade_schema_noop(qdrant_adapter) -> None:
    qdrant_adapter.upgrade_schema(1, 2)
