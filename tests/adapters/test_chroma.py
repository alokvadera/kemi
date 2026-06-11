"""Tests for the Chroma storage adapter.

Uses a temporary directory for the Chroma persistent client
so no external service is required.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from kemi.adapters.storage.chroma import ChromaStorageAdapter
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType


@pytest.fixture
def chroma_adapter(tmp_path: Any) -> ChromaStorageAdapter:
    adapter = ChromaStorageAdapter(
        path=str(tmp_path / "chroma_test"),
        collection_name="test_kemi",
    )
    yield adapter


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


def test_store_and_get(chroma_adapter) -> None:
    mem = _make_memory("test-id", content="I am vegetarian", importance=0.7)
    mem.source = MemorySource.AGENT_INFERRED

    chroma_adapter.store(mem)
    result = chroma_adapter.get("test-id")

    assert result is not None
    assert result.memory_id == "test-id"
    assert result.user_id == "user1"
    assert result.content == "I am vegetarian"
    assert result.source == MemorySource.AGENT_INFERRED
    assert result.importance == 0.7


def test_search_returns_results(chroma_adapter) -> None:
    mem1 = _make_memory("id1", content="I am vegetarian", embedding=[1.0] * 64)
    mem2 = _make_memory("id2", content="I live in Mumbai", embedding=[0.1] * 64)

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    query = [1.0] * 64
    results = chroma_adapter.search("user1", query, top_k=10)

    assert len(results) == 2


def test_search_lifecycle_filter(chroma_adapter) -> None:
    active = _make_memory("id1", content="active", lifecycle_state=LifecycleState.ACTIVE)
    deleted = _make_memory("id2", content="deleted", lifecycle_state=LifecycleState.DELETED)

    chroma_adapter.store(active)
    chroma_adapter.store(deleted)

    query = [1.0] * 64
    results = chroma_adapter.search("user1", query, top_k=10)

    assert all(m.lifecycle_state != LifecycleState.DELETED for m in results)


def test_delete_by_id(chroma_adapter) -> None:
    mem = _make_memory("test-id")
    chroma_adapter.store(mem)

    result = chroma_adapter.delete_by_id("test-id")
    assert result is True

    get_result = chroma_adapter.get("test-id")
    assert get_result is None


def test_delete_nonexistent(chroma_adapter) -> None:
    result = chroma_adapter.delete_by_id("non-existent")
    assert result is False


def test_delete_by_user(chroma_adapter) -> None:
    mem1 = _make_memory("id1")
    mem2 = _make_memory("id2")

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    count = chroma_adapter.delete_by_user("user1")
    assert count == 2


def test_count(chroma_adapter) -> None:
    mem = _make_memory("id1")
    chroma_adapter.store(mem)

    count = chroma_adapter.count("user1")
    assert count == 1


def test_get_all_by_user(chroma_adapter) -> None:
    mem1 = _make_memory("id1", content="test1")
    mem2 = _make_memory("id2", content="test2", lifecycle_state=LifecycleState.DECAYING)

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    results = chroma_adapter.get_all_by_user("user1")
    assert len(results) == 2


def test_get_all_users(chroma_adapter) -> None:
    mem1 = _make_memory("id1", user_id="user1")
    mem2 = _make_memory("id2", user_id="user2")

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    users = chroma_adapter.get_all_users()
    assert sorted(users) == ["user1", "user2"]


def test_update(chroma_adapter) -> None:
    mem = _make_memory("test-id", content="original")
    chroma_adapter.store(mem)

    mem.content = "updated"
    chroma_adapter.update(mem)

    result = chroma_adapter.get("test-id")
    assert result is not None
    assert result.content == "updated"


def test_get_nonexistent(chroma_adapter) -> None:
    result = chroma_adapter.get("non-existent")
    assert result is None


def test_search_by_content(chroma_adapter) -> None:
    mem1 = _make_memory("id1", content="I love pizza")
    mem2 = _make_memory("id2", content="I love sushi")

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    results = chroma_adapter.search_by_content("user1", "pizza", top_k=10)
    assert len(results) >= 1


def test_store_many(chroma_adapter) -> None:
    mem1 = _make_memory("id1")
    mem2 = _make_memory("id2")

    count = chroma_adapter.store_many([mem1, mem2])
    assert count == 2

    assert chroma_adapter.count("user1") == 2


def test_context_manager() -> None:
    import tempfile

    with ChromaStorageAdapter(
        path=tempfile.mkdtemp(),
        collection_name="test_ctx",
    ) as adapter:
        mem = _make_memory("ctx-1")
        adapter.store(mem)
        result = adapter.get("ctx-1")
        assert result is not None


def test_metadata_roundtrip(chroma_adapter) -> None:
    mem = _make_memory("meta-1", metadata={"key": "value", "nested": {"a": 1}})
    chroma_adapter.store(mem)

    result = chroma_adapter.get("meta-1")
    assert result is not None
    assert result.metadata["key"] == "value"
    assert result.metadata["nested"]["a"] == 1


def test_tags_roundtrip(chroma_adapter) -> None:
    mem = _make_memory("tag-1", tags=["hello", "world"])
    chroma_adapter.store(mem)

    result = chroma_adapter.get("tag-1")
    assert result is not None
    assert "hello" in result.tags
    assert "world" in result.tags


def test_session_id_filtering(chroma_adapter) -> None:
    mem1 = _make_memory("id1", session_id="sess-1")
    mem2 = _make_memory("id2", session_id="sess-2")

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    results = chroma_adapter.get_all_by_user("user1", session_id="sess-1")
    assert len(results) == 1
    assert results[0].memory_id == "id1"


def test_search_with_session_filter(chroma_adapter) -> None:
    mem1 = _make_memory("id1", content="session1 mem", embedding=[1.0] * 64, session_id="sess-1")
    mem2 = _make_memory("id2", content="session2 mem", embedding=[1.0] * 64, session_id="sess-2")

    chroma_adapter.store(mem1)
    chroma_adapter.store(mem2)

    query = [1.0] * 64
    results = chroma_adapter.search("user1", query, top_k=10, session_id="sess-1")

    assert len(results) == 1
    assert results[0].memory_id == "id1"


def test_upgrade_schema_noop(chroma_adapter) -> None:
    chroma_adapter.upgrade_schema(1, 2)
