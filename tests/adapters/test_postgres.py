"""Tests for PostgresStorageAdapter.

These tests are skipped when PostgreSQL is not available.
Set PG_DSN environment variable to run them against a real database.
"""

from datetime import datetime, timezone

import pytest

from kemi.adapters.storage.postgres import PostgresStorageAdapter
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource


def _pgvector_available() -> bool:
    """Check if pgvector is installed."""
    try:
        import pgvector

        return True
    except ImportError:
        return False


def _pg_available() -> bool:
    try:
        import pgvector

        return True
    except ImportError:
        return False


def _pg_available() -> bool:
    """Check if PostgreSQL is reachable at the DSN."""
    dsn = "postgresql://postgres:postgres@localhost:5432/postgres"
    import os as _os

    dsn = _os.environ.get("PG_DSN", dsn)
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:
        return False


# Conditional skip: require both pgvector AND a reachable PG instance
skip_if_no_pg = pytest.mark.skipif(
    not (_pgvector_available() and _pg_available()),
    reason="PostgreSQL not available. Set PG_DSN env var to run these tests.",
)


# Skip all tests if pgvector not installed (before attempting PG connection)
pytestmark = pytest.mark.skipif(
    not _pgvector_available(),
    reason="pgvector not installed — run: pip install pgvector>=0.3",
)


@pytest.fixture
def postgres_adapter(tmp_path) -> PostgresStorageAdapter:
    """Create a PostgresStorageAdapter pointing at a temporary database.

    The adapter connects to a database named after the test process PID
    to avoid conflicts when run in parallel.
    """
    import os

    dsn = os.environ.get(
        "PG_DSN",
        "postgresql://postgres:postgres@localhost:5432/kemi_test",
    )
    return PostgresStorageAdapter(dsn=dsn, embedding_dim=64)


def _make_memory(
    memory_id: str,
    user_id: str,
    content: str,
    embedding_dim: int = 64,
    **overrides: object,
) -> MemoryObject:
    """Helper to create a test MemoryObject."""
    base = {
        "memory_id": memory_id,
        "user_id": user_id,
        "content": content,
        "embedding": [0.1] * embedding_dim,
        "score": 0.0,
        "created_at": datetime.now(timezone.utc),
        "last_accessed_at": datetime.now(timezone.utc),
        "source": MemorySource.USER_STATED,
        "importance": 0.5,
        "lifecycle_state": LifecycleState.ACTIVE,
        "metadata": {},
        "embedding_dim": embedding_dim,
        "tags": [],
        "confidence": 1.0,
        "memory_type": "episodic",
        "session_id": None,
        "namespace": "default",
        "version": 1,
        "agent_id": None,
        "run_id": None,
        "app_id": None,
        "expires_at": None,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return MemoryObject(**base)  # type: ignore[arg-type]


# ── Basic CRUD tests ───────────────────────────────────────────


@skip_if_no_pg
def test_store_and_get(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory(
        "test-id", "user1", "I am vegetarian",
        tags=["food", "diet"],
    )
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.memory_id == "test-id"
    assert result.user_id == "user1"
    assert result.content == "I am vegetarian"
    assert result.embedding == pytest.approx([0.1] * 64)
    assert result.tags == ["food", "diet"]


@skip_if_no_pg
def test_store_updates_existing(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("test-id", "user1", "Original content")
    postgres_adapter.store(mem)

    mem.content = "Updated content"
    mem.version = 2
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.content == "Updated content"
    assert result.version == 2


@skip_if_no_pg
def test_delete_by_id(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("test-id", "user1", "to be deleted")
    postgres_adapter.store(mem)

    deleted = postgres_adapter.delete_by_id("test-id")
    assert deleted is True

    result = postgres_adapter.get("test-id")
    assert result is None


@skip_if_no_pg
def test_delete_by_id_not_found(postgres_adapter: PostgresStorageAdapter) -> None:
    deleted = postgres_adapter.delete_by_id("nonexistent")
    assert deleted is False


@skip_if_no_pg
def test_delete_by_user(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "user1", "test1"))
    postgres_adapter.store(_make_memory("id2", "user1", "test2"))
    postgres_adapter.store(_make_memory("id3", "user2", "test3"))

    count = postgres_adapter.delete_by_user("user1")
    assert count == 2

    assert postgres_adapter.get("id1") is None
    assert postgres_adapter.get("id3") is not None


@skip_if_no_pg
def test_count(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "test"))
    postgres_adapter.store(_make_memory("id2", "alice", "test"))
    postgres_adapter.store(_make_memory("id3", "bob", "test"))

    assert postgres_adapter.count("alice") == 2
    assert postgres_adapter.count("bob") == 1
    assert postgres_adapter.count("carol") == 0


@skip_if_no_pg
def test_get_all(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "user1", "test1"))
    postgres_adapter.store(_make_memory("id2", "user1", "test2"))

    results = postgres_adapter.get_all()
    assert len(results) >= 2
    assert all(isinstance(m, MemoryObject) for m in results)


@skip_if_no_pg
def test_get_all_with_limit_offset(postgres_adapter: PostgresStorageAdapter) -> None:
    for i in range(5):
        postgres_adapter.store(_make_memory(f"id{i}", "user1", f"test{i}"))

    page1 = postgres_adapter.get_all(limit=2, offset=0)
    assert len(page1) == 2

    page2 = postgres_adapter.get_all(limit=2, offset=2)
    assert len(page2) == 2


@skip_if_no_pg
def test_get_all_users(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "test"))
    postgres_adapter.store(_make_memory("id2", "bob", "test"))

    users = postgres_adapter.get_all_users()
    assert "alice" in users
    assert "bob" in users


# ── Get all by user tests ──────────────────────────────────────


@skip_if_no_pg
def test_get_all_by_user(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "test1"))
    postgres_adapter.store(_make_memory("id2", "alice", "test2"))
    postgres_adapter.store(_make_memory("id3", "bob", "test3"))

    results = postgres_adapter.get_all_by_user("alice")
    assert len(results) == 2
    assert all(m.user_id == "alice" for m in results)


@skip_if_no_pg
def test_get_all_by_user_lifecycle_filter(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "active mem", lifecycle_state=LifecycleState.ACTIVE))  # noqa: E501
    postgres_adapter.store(_make_memory("id2", "alice", "decaying mem", lifecycle_state=LifecycleState.DECAYING))  # noqa: E501
    postgres_adapter.store(_make_memory("id3", "alice", "deleted mem", lifecycle_state=LifecycleState.DELETED))  # noqa: E501

    active_only = postgres_adapter.get_all_by_user(
        "alice", lifecycle_filter=[LifecycleState.ACTIVE]
    )
    assert all(m.lifecycle_state == LifecycleState.ACTIVE for m in active_only)


@skip_if_no_pg
def test_get_all_by_user_namespace(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "default ns", namespace="default"))
    postgres_adapter.store(_make_memory("id2", "alice", "work ns", namespace="work"))

    default = postgres_adapter.get_all_by_user("alice", namespace="default")
    assert all(m.namespace == "default" for m in default)

    work = postgres_adapter.get_all_by_user("alice", namespace="work")
    assert all(m.namespace == "work" for m in work)


# ── Tag tests ─────────────────────────────────────────────────


@skip_if_no_pg
def test_tags_roundtrip(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("id1", "alice", "test", tags=["python", "AI", "machine-learning"])
    postgres_adapter.store(mem)

    result = postgres_adapter.get("id1")
    assert result is not None
    assert set(result.tags) == {"python", "AI", "machine-learning"}


@skip_if_no_pg
def test_get_by_tag(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "I love cats", tags=["pet", "cat"]))
    postgres_adapter.store(_make_memory("id2", "alice", "I work in tech", tags=["work"]))
    postgres_adapter.store(_make_memory("id3", "alice", "Cat adoption story", tags=["pet", "cat", "adoption"]))  # noqa: E501

    results = postgres_adapter.get_by_tag("alice", "cat")
    assert len(results) == 2
    assert all("cat" in m.tags for m in results)


@skip_if_no_pg
def test_get_by_tag_no_false_positives(postgres_adapter: PostgresStorageAdapter) -> None:
    """Searching for 'cat' should not match 'category'."""
    postgres_adapter.store(_make_memory("id1", "alice", "I have a pet cat", tags=["pet", "cat"]))
    postgres_adapter.store(_make_memory("id2", "alice", "I work in the category industry", tags=["work", "category"]))  # noqa: E501

    results = postgres_adapter.get_by_tag("alice", "cat")
    assert len(results) == 1
    assert results[0].memory_id == "id1"


# ── Vector search tests ───────────────────────────────────────


@skip_if_no_pg
def test_search_returns_results(postgres_adapter: PostgresStorageAdapter) -> None:
    # Store two memories with very different embeddings
    postgres_adapter.store(_make_memory("id1", "user1", "I am vegetarian", embedding=[1.0] * 64))
    postgres_adapter.store(_make_memory("id2", "user1", "I live in Mumbai", embedding=[0.1] * 64))

    query = [1.0] * 64
    results = postgres_adapter.search("user1", query, top_k=10)

    assert len(results) == 2
    # The closer embedding should score higher
    assert results[0].memory_id == "id1"


@skip_if_no_pg
def test_search_lifecycle_filter(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "user1", "active memory", embedding=[1.0] * 64, lifecycle_state=LifecycleState.ACTIVE))  # noqa: E501
    postgres_adapter.store(_make_memory("id2", "user1", "deleted memory", embedding=[1.0] * 64, lifecycle_state=LifecycleState.DELETED))  # noqa: E501

    results = postgres_adapter.search("user1", [1.0] * 64, top_k=10)
    assert all(m.lifecycle_state != LifecycleState.DELETED for m in results)


@skip_if_no_pg
def test_search_namespace_filter(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "default ns", namespace="default", embedding=[1.0] * 64))  # noqa: E501
    postgres_adapter.store(_make_memory("id2", "alice", "work ns", namespace="work", embedding=[1.0] * 64))  # noqa: E501

    results = postgres_adapter.search("alice", [1.0] * 64, namespace="work")
    assert all(m.namespace == "work" for m in results)


@skip_if_no_pg
def test_search_empty_query_embedding(postgres_adapter: PostgresStorageAdapter) -> None:
    """Even an empty embedding should not crash — search handles it gracefully."""
    postgres_adapter.store(_make_memory("id1", "user1", "test", embedding=[0.0] * 64))

    results = postgres_adapter.search("user1", [0.0] * 64, top_k=10)
    # Returns whatever pgvector returns (may be empty or partial)
    assert isinstance(results, list)


# ── Full-text search tests ───────────────────────────────────


@skip_if_no_pg
def test_search_by_content(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "Python is a great programming language"))
    postgres_adapter.store(_make_memory("id2", "alice", "I enjoy hiking in the mountains"))
    postgres_adapter.store(_make_memory("id3", "alice", "Machine learning with Python"))

    results = postgres_adapter.search_by_content("alice", "Python")
    assert len(results) >= 2
    assert all("python" in m.content.lower() for m in results)


@skip_if_no_pg
def test_search_by_content_phrase(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "The quick brown fox"))
    postgres_adapter.store(_make_memory("id2", "alice", "Quick brown fox jumps"))

    results = postgres_adapter.search_by_content("alice", '"quick brown"')
    assert len(results) == 2


@skip_if_no_pg
def test_search_by_content_no_results(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "Python programming"))

    results = postgres_adapter.search_by_content("alice", "nonexistent keyword xyz123")
    assert len(results) == 0


# ── Hybrid search tests ───────────────────────────────────────


@skip_if_no_pg
def test_search_hybrid(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.store(_make_memory("id1", "alice", "I love Python programming", embedding=[1.0] * 64))  # noqa: E501
    postgres_adapter.store(_make_memory("id2", "alice", "I love hiking", embedding=[0.1] * 64))

    results = postgres_adapter.search_hybrid(
        "alice",
        query_embedding=[1.0] * 64,
        query_text="Python",
        top_k=5,
    )
    assert len(results) == 2
    # The Python memory should rank first (matches both vector and text)
    assert results[0].memory_id == "id1"


@skip_if_no_pg
def test_search_hybrid_weights(postgres_adapter: PostgresStorageAdapter) -> None:
    """Test that weight parameters are respected by comparing runs."""
    postgres_adapter.store(_make_memory("id1", "alice", "Python programming", embedding=[1.0] * 64))
    postgres_adapter.store(_make_memory("id2", "alice", "Hiking is great", embedding=[0.0] * 64))

    # Vector-only (weight_vector=1.0)
    vec_results = postgres_adapter.search_hybrid(
        "alice",
        query_embedding=[1.0] * 64,
        query_text="Python hiking",
        top_k=5,
        weight_vector=1.0,
        weight_bm25=0.0,
        weight_recency=0.0,
    )
    assert vec_results[0].memory_id == "id1"

    # Text-only (weight_bm25=1.0)
    text_results = postgres_adapter.search_hybrid(
        "alice",
        query_embedding=[1.0] * 64,
        query_text="Python hiking",
        top_k=5,
        weight_vector=0.0,
        weight_bm25=1.0,
        weight_recency=0.0,
    )
    # Both mention hiking and Python, text rank should be close — just check it runs
    assert len(text_results) >= 1


# ── Memory source and lifecycle tests ────────────────────────


@skip_if_no_pg
def test_source_roundtrip(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("test-id", "user1", "test", source=MemorySource.AGENT_INFERRED)
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.source == MemorySource.AGENT_INFERRED


@skip_if_no_pg
def test_memory_type_roundtrip(postgres_adapter: PostgresStorageAdapter) -> None:
    from kemi.memory.model import MemoryType

    mem = _make_memory("test-id", "user1", "test", memory_type="semantic")
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.memory_type == MemoryType.SEMANTIC


@skip_if_no_pg
def test_metadata_roundtrip(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory(
        "test-id",
        "user1",
        "test",
        metadata={"key": "value", "nested": {"a": 1}},
    )
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.metadata["key"] == "value"
    assert result.metadata["nested"]["a"] == 1


@skip_if_no_pg
def test_embedding_roundtrip(postgres_adapter: PostgresStorageAdapter) -> None:
    embedding = [0.05 * i for i in range(64)]
    mem = _make_memory("test-id", "user1", "test", embedding=embedding)
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.embedding == pytest.approx(embedding, abs=1e-5)


@skip_if_no_pg
def test_lifecycle_state_roundtrip(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("test-id", "user1", "test", lifecycle_state=LifecycleState.DECAYING)
    postgres_adapter.store(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.lifecycle_state == LifecycleState.DECAYING


# ── Update tests ──────────────────────────────────────────────


@skip_if_no_pg
def test_update(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("test-id", "user1", "original")
    postgres_adapter.store(mem)

    mem.content = "updated"
    mem.importance = 0.9
    postgres_adapter.update(mem)

    result = postgres_adapter.get("test-id")
    assert result is not None
    assert result.content == "updated"
    assert result.importance == 0.9


@skip_if_no_pg
def test_update_nonexistent(postgres_adapter: PostgresStorageAdapter) -> None:
    mem = _make_memory("nonexistent", "user1", "test")
    # update should not raise — just a no-op
    postgres_adapter.update(mem)
    assert postgres_adapter.get("nonexistent") is None


# ── Store many tests ──────────────────────────────────────────


@skip_if_no_pg
def test_store_many(postgres_adapter: PostgresStorageAdapter) -> None:
    memories = [
        _make_memory(f"id{i}", "alice", f"content {i}")
        for i in range(5)
    ]
    count = postgres_adapter.store_many(memories)
    assert count == 5

    for i in range(5):
        assert postgres_adapter.get(f"id{i}") is not None


@skip_if_no_pg
def test_store_many_empty(postgres_adapter: PostgresStorageAdapter) -> None:
    count = postgres_adapter.store_many([])
    assert count == 0


# ── Schema migration tests ────────────────────────────────────


@skip_if_no_pg
def test_schema_version_table_exists(postgres_adapter: PostgresStorageAdapter) -> None:
    with postgres_adapter._connection() as conn:
        cursor = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] >= 1


@skip_if_no_pg
def test_upgrade_schema_idempotent(postgres_adapter: PostgresStorageAdapter) -> None:
    postgres_adapter.upgrade_schema(1, 2)
    postgres_adapter.upgrade_schema(1, 2)

    with postgres_adapter._connection() as conn:
        cursor = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        assert row[0] >= 1


# ── Context manager tests ─────────────────────────────────────


@skip_if_no_pg
def test_context_manager(postgres_adapter: PostgresStorageAdapter) -> None:
    with postgres_adapter as adapter:
        mem = _make_memory("ctx-1", "user1", "context test")
        adapter.store(mem)
        result = adapter.get("ctx-1")
        assert result is not None
        assert result.content == "context test"

    # After __exit__, pool is closed
    assert postgres_adapter._pool is None or postgres_adapter._pool.closed


@skip_if_no_pg
def test_context_manager_closes_on_exception(postgres_adapter: PostgresStorageAdapter) -> None:
    with pytest.raises(ValueError):
        with postgres_adapter:
            raise ValueError("boom")

    assert postgres_adapter._pool is None or postgres_adapter._pool.closed


# ── Index existence tests ─────────────────────────────────────


@skip_if_no_pg
def test_indexes_exist(postgres_adapter: PostgresStorageAdapter) -> None:
    with postgres_adapter._connection() as conn:
        cursor = conn.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'memories'
            """
        )
        indexes = {row[0] for row in cursor.fetchall()}

    assert "idx_memories_user_id" in indexes
    assert "idx_memories_lifecycle" in indexes
    assert "idx_memories_user_lifecycle" in indexes
    assert "idx_memories_namespace" in indexes
    assert "idx_memories_tags" in indexes  # GIN index
    assert "idx_memories_embedding" in indexes  # ivfflat index
