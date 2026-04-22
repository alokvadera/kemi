from datetime import datetime, timezone

import pytest

from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource


@pytest.fixture
def sqlite_adapter() -> SQLiteStorageAdapter:
    return SQLiteStorageAdapter(db_path=":memory:")


def test_store_and_get(sqlite_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="I am vegetarian",
        embedding=[0.1] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.AGENT_INFERRED,
        importance=0.7,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={"key": "value"},
        embedding_dim=64,
    )

    sqlite_adapter.store(mem)
    result = sqlite_adapter.get("test-id")

    assert result is not None
    assert result.memory_id == "test-id"
    assert result.user_id == "user1"
    assert result.content == "I am vegetarian"
    assert result.source == MemorySource.AGENT_INFERRED
    assert result.importance == 0.7
    assert result.embedding == pytest.approx([0.1] * 64)


def test_search_returns_results(sqlite_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
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
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    sqlite_adapter.store(mem1)
    sqlite_adapter.store(mem2)

    query = [1.0] * 64
    results = sqlite_adapter.search("user1", query, top_k=10)

    assert len(results) == 2


def test_search_lifecycle_filter(sqlite_adapter) -> None:
    active_mem = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="active memory",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
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
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.DELETED,
        metadata={},
        embedding_dim=64,
    )

    sqlite_adapter.store(active_mem)
    sqlite_adapter.store(deleted_mem)

    query = [1.0] * 64
    results = sqlite_adapter.search("user1", query, top_k=10)

    assert all(m.lifecycle_state != LifecycleState.DELETED for m in results)


def test_delete_by_id(sqlite_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    sqlite_adapter.store(mem)
    result = sqlite_adapter.delete_by_id("test-id")
    assert result is True

    get_result = sqlite_adapter.get("test-id")
    assert get_result is None


def test_delete_by_user(sqlite_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test1",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
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
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    sqlite_adapter.store(mem1)
    sqlite_adapter.store(mem2)

    count = sqlite_adapter.delete_by_user("user1")
    assert count == 2


def test_count(sqlite_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    sqlite_adapter.store(mem1)
    count = sqlite_adapter.count("user1")
    assert count == 1


def test_source_roundtrip(sqlite_adapter) -> None:
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.AGENT_INFERRED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    sqlite_adapter.store(mem)
    result = sqlite_adapter.get("test-id")

    assert result is not None
    assert result.source == MemorySource.AGENT_INFERRED


def test_embedding_roundtrip(sqlite_adapter) -> None:
    embedding = [0.1 * i for i in range(64)]
    mem = MemoryObject(
        memory_id="test-id",
        user_id="user1",
        content="test",
        embedding=embedding,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    sqlite_adapter.store(mem)
    result = sqlite_adapter.get("test-id")

    assert result is not None
    assert result.embedding is not None
    assert len(result.embedding) == 64
    assert result.embedding == pytest.approx(embedding)


def test_search_empty_query_embedding(sqlite_adapter) -> None:
    mem = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test",
        embedding=[0.0] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    sqlite_adapter.store(mem)
    results = sqlite_adapter.search("user1", [0.0] * 64, top_k=10)
    assert len(results) == 1


def test_get_all_by_user_sqlite(sqlite_adapter) -> None:
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="test1",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
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
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.DECAYING,
        metadata={},
        embedding_dim=None,
    )

    sqlite_adapter.store(mem1)
    sqlite_adapter.store(mem2)

    results = sqlite_adapter.get_all_by_user("user1")
    assert len(results) == 2


def test_get_by_tag_exact_match_no_false_positives(sqlite_adapter) -> None:
    """Test that searching for 'cat' doesn't match 'category'."""
    mem1 = MemoryObject(
        memory_id="id1",
        user_id="user1",
        content="I have a pet cat",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
        tags=["pet", "cat"],
    )

    mem2 = MemoryObject(
        memory_id="id2",
        user_id="user1",
        content="I work in the category industry",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
        tags=["work", "category"],
    )

    sqlite_adapter.store(mem1)
    sqlite_adapter.store(mem2)

    results = sqlite_adapter.get_by_tag("user1", "cat")

    assert len(results) == 1
    assert results[0].memory_id == "id1"
    assert "cat" in results[0].tags
    assert "category" not in results[0].tags


def test_migration_creates_schema_version_table(sqlite_adapter) -> None:
    cursor = sqlite_adapter._get_connection().execute("SELECT version FROM schema_version LIMIT 1")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == 2


def test_migration_idempotent(sqlite_adapter) -> None:
    sqlite_adapter.upgrade_schema(1, 2)
    sqlite_adapter.upgrade_schema(1, 2)

    cursor = sqlite_adapter._get_connection().execute("SELECT version FROM schema_version LIMIT 1")
    row = cursor.fetchone()
    assert row[0] == 2


def test_sqlite_close() -> None:
    adapter = SQLiteStorageAdapter(db_path=":memory:")
    adapter.close()
    assert adapter._shared_conn is None


def test_get_connection_creates_new_conn() -> None:
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        adapter = SQLiteStorageAdapter(db_path=db_path)

        conn = adapter._get_connection()
        cursor = conn.execute("SELECT 1")
        row = cursor.fetchone()
        assert row[0] == 1
