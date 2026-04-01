import pytest

from datetime import datetime

from kemi.adapters.storage.json import JSONStorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource


@pytest.fixture
def json_adapter(tmp_path) -> JSONStorageAdapter:
    return JSONStorageAdapter(path=str(tmp_path / "test.json"))


def test_store_and_get(json_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="I am vegetarian",
        embedding=[0.1] * 64,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.AGENT_INFERRED,
        importance=0.7,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={"key": "value"},
        embedding_dim=64,
    )

    json_adapter.store(mem)
    result = json_adapter.get("test-id")

    assert result is not None
    assert result.memory_id == "test-id"
    assert result.content == "I am vegetarian"
    assert result.source == MemorySource.AGENT_INFERRED


def test_search_returns_results(json_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    mem2 = MemoryObject(
        memory_id="id2",
        user_id="user1",
        content="I live in Mumbai",
        embedding=[0.1] * 64,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    json_adapter.store(mem1)
    json_adapter.store(mem2)

    query = [1.0] * 64
    results = json_adapter.search("user1", query, top_k=10)

    assert len(results) == 2


def test_search_lifecycle_filter(json_adapter) -> None:
    active_mem = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="active memory",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    deleted_mem = MemoryObject(
        memory_id="id2",
        user_id="user1",
        content="deleted memory",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.DELETED,
        metadata={},
        embedding_dim=64,
    )

    json_adapter.store(active_mem)
    json_adapter.store(deleted_mem)

    query = [1.0] * 64
    results = json_adapter.search("user1", query, top_k=10)

    assert all(m.lifecycle_state != LifecycleState.DELETED for m in results)


def test_delete_by_id(json_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    json_adapter.store(mem)
    result = json_adapter.delete_by_id("test-id")
    assert result is True

    get_result = json_adapter.get("test-id")
    assert get_result is None

    not_found = json_adapter.delete_by_id("non-existent")
    assert not_found is False


def test_delete_by_user(json_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test1",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    mem2 = MemoryObject(
        memory_id="id2",
        user_id="user1",
        content="test2",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    json_adapter.store(mem1)
    json_adapter.store(mem2)

    count = json_adapter.delete_by_user("user1")
    assert count == 2


def test_count(json_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    json_adapter.store(mem1)
    count = json_adapter.count("user1")
    assert count == 1


def test_source_roundtrip(json_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.AGENT_INFERRED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    json_adapter.store(mem)
    result = json_adapter.get("test-id")

    assert result is not None
    assert result.source == MemorySource.AGENT_INFERRED


def test_get_all_by_user(json_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test1",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    mem2 = MemoryObject(
        memory_id="id2",
        user_id="user1",
        content="test2",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    json_adapter.store(mem1)
    json_adapter.store(mem2)

    results = json_adapter.get_all_by_user("user1")
    assert len(results) == 2


def test_update(json_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="original",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    json_adapter.store(mem)

    mem.content = "updated"
    json_adapter.update(mem)

    result = json_adapter.get("test-id")
    assert result.content == "updated"


def test_upgrade_schema(json_adapter) -> None:
    json_adapter.upgrade_schema(1, 2)
    assert json_adapter._data["schema_version"] == 2


def test_search_with_embedding(json_adapter) -> None:
    mem = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test",
        embedding=[1.0, 0.0] * 32,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    json_adapter.store(mem)
    query = [1.0, 0.0] * 32
    results = json_adapter.search("user1", query, top_k=10)
    assert len(results) == 1
