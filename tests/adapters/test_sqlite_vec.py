from datetime import datetime, timezone

import pytest

from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource

pytestmark = pytest.mark.slow


@pytest.fixture
def vec_adapter() -> SQLiteVecStorageAdapter:
    return SQLiteVecStorageAdapter(db_path=":memory:", embedding_dim=64)


# ── Availability & basic properties ───────────────────────────────


def test_is_vec_available() -> None:
    result = SQLiteVecStorageAdapter.is_vec_available()
    assert isinstance(result, bool)


def test_is_lazy_default() -> None:
    adapter = SQLiteVecStorageAdapter(db_path=":memory:", lazy=False)
    assert adapter.is_lazy() is False


def test_is_lazy_true() -> None:
    adapter = SQLiteVecStorageAdapter(db_path=":memory:", lazy=True)
    assert adapter.is_lazy() is True


# ── Store and get (fallback to parent when vec unavailable) ─────────


def test_store_and_get_fallback(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="v1",
        user_id="user1",
        content="vector test",
        embedding=[0.5] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )
    vec_adapter.store(mem)
    result = vec_adapter.get("v1")
    assert result is not None
    assert result.memory_id == "v1"
    assert result.content == "vector test"


# ── Search fallback to parent ─────────────────────────────────────────


def test_search_fallback_no_embedding(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="v2",
        user_id="user1",
        content="search fallback",
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
    vec_adapter.store(mem)
    results = vec_adapter.search("user1", [1.0] * 64, top_k=10)
    assert len(results) == 1
    assert results[0].memory_id == "v2"


def test_search_fallback_when_vec_not_loaded(vec_adapter, monkeypatch) -> None:
    """If _vec_loaded is False, search falls back to parent brute-force."""
    monkeypatch.setattr(vec_adapter, "_vec_loaded", False)
    mem = MemoryObject(
        memory_id="v3",
        user_id="user1",
        content="fallback brute force",
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
    vec_adapter.store(mem)
    results = vec_adapter.search("user1", [1.0] * 64, top_k=10)
    assert len(results) == 1


def test_search_fallback_empty_embedding(vec_adapter) -> None:
    """If query_embedding is empty, fallback to parent."""
    mem = MemoryObject(
        memory_id="v4",
        user_id="user1",
        content="empty embedding fallback",
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
    vec_adapter.store(mem)
    vec_adapter.search("user1", [], top_k=10)
    # Empty embedding triggers fallback; parent search won't match because
    # cosine_similarity with empty list may not work. This test mainly
    # verifies it doesn't crash.


# ── Delete ─────────────────────────────────────────────────────────


def test_delete_by_id_fallback(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="vdel1",
        user_id="user1",
        content="delete me",
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
    vec_adapter.store(mem)
    assert vec_adapter.delete_by_id("vdel1") is True
    assert vec_adapter.get("vdel1") is None


def test_delete_by_user_fallback(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="vdel2",
        user_id="user1",
        content="delete all",
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
    vec_adapter.store(mem)
    count = vec_adapter.delete_by_user("user1")
    assert count == 1


# ── Update ─────────────────────────────────────────────────────────


def test_update_fallback(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="vupd",
        user_id="user1",
        content="original",
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
    vec_adapter.store(mem)
    mem.content = "updated"
    vec_adapter.update(mem)
    result = vec_adapter.get("vupd")
    assert result is not None
    assert result.content == "updated"


# ── Count / get_all_by_user ──────────────────────────────────────────


def test_count_fallback(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="vcnt",
        user_id="user1",
        content="count me",
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
    vec_adapter.store(mem)
    assert vec_adapter.count("user1") == 1


# ── Lazy mode basics (when vec unavailable, lazy flag still works) ──


def test_lazy_mode_no_crash(vec_adapter) -> None:
    """Lazy adapter should store without crashing even if vec is unavailable."""
    adapter = SQLiteVecStorageAdapter(db_path=":memory:", embedding_dim=64, lazy=True)
    mem = MemoryObject(
        memory_id="vlazy",
        user_id="user1",
        content="lazy test",
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
    adapter.store(mem)
    assert adapter.get("vlazy") is not None


# ── _memory_to_row includes vec_rowid ───────────────────────────────


def test_memory_to_row_vec_rowid(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="vrow",
        user_id="user1",
        content="vec rowid test",
        embedding=[0.1] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={"_vec_rowid": 42},
        embedding_dim=64,
    )
    row = vec_adapter._memory_to_row(mem)
    assert row["vec_rowid"] == 42


# ── _row_to_memory includes vec_rowid ───────────────────────────────


def test_row_to_memory_vec_rowid(vec_adapter) -> None:
    mem = MemoryObject(
        memory_id="vrow2",
        user_id="user1",
        content="row to mem",
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
    vec_adapter.store(mem)
    # Manually set vec_rowid in DB
    conn = vec_adapter._get_connection()
    conn.execute("UPDATE memories SET vec_rowid = ? WHERE memory_id = ?", (99, "vrow2"))
    conn.commit()
    result = vec_adapter.get("vrow2")
    assert result is not None
    assert result.metadata.get("_vec_rowid") == 99


# ── _init_vec_table no-op when unavailable ──────────────────────────


def test_init_vec_table_no_op() -> None:
    """_init_vec_table should be a no-op when sqlite-vec is unavailable."""
    adapter = SQLiteVecStorageAdapter(db_path=":memory:", embedding_dim=64)
    conn = adapter._get_connection()
    # Should not raise even though sqlite-vec is unavailable
    adapter._init_vec_table(conn)


# ── Migration idempotent ────────────────────────────────────────────


def test_migration_idempotent(vec_adapter) -> None:
    vec_adapter.upgrade_schema(1, vec_adapter.CURRENT_VERSION)
    cursor = vec_adapter._get_connection().execute(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
    )
    row = cursor.fetchone()
    assert row[0] == vec_adapter.CURRENT_VERSION


# ── Store with embedding but vec not loaded ─────────────────────────


def test_store_embedding_vec_not_loaded(vec_adapter, monkeypatch) -> None:
    """If _vec_loaded is False, store should skip vec index."""
    monkeypatch.setattr(vec_adapter, "_vec_loaded", False)
    mem = MemoryObject(
        memory_id="vskip",
        user_id="user1",
        content="skip vec",
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
    # Should not crash
    vec_adapter.store(mem)
    assert vec_adapter.get("vskip") is not None
