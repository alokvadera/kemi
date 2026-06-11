"""Comprehensive tests for SQLiteVecStorageAdapter.

Covers: lazy mode, flush pending, migrations, delete-by-id/user,
search via vec0, row helpers, extension loading, and edge cases.
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from kemi.adapters.storage.sqlite_vec import (
    _SQLITE_VEC_AVAILABLE,
    SQLiteVecStorageAdapter,
    _embedding_to_json,
)
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource
from tests._helpers.factories import make_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    memory_id: str = "test-1",
    user_id: str = "user1",
    content: str = "User prefers dark mode",
    embedding: list[float] | None = None,
    importance: float = 0.9,
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    tags: list[str] | None = None,
) -> MemoryObject:
    """Create a MemoryObject with sensible defaults."""
    if embedding is None:
        embedding = [0.1] * 384
    return make_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=embedding,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=importance,
        lifecycle_state=lifecycle_state,
        metadata={},
        embedding_dim=384,
        tags=tags or [],
    )


def _tmp_db() -> str:
    return tempfile.mktemp(suffix=".db")


def _make_lazy_adapter(tmp: str) -> SQLiteVecStorageAdapter:
    """Create a lazy adapter with _vec_loaded forced True.

    This allows store() to reach _store_pending_on_conn even without
    sqlite-vec installed, because the pending table is a regular table.
    """
    adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
    adapter._vec_loaded = True  # force so store() reaches pending path
    return adapter


# ---------------------------------------------------------------------------
# Basic store / search / delete (original tests, kept)
# ---------------------------------------------------------------------------


class TestBasicOperations:
    def test_store_search_delete(self):
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            assert adapter._vec_loaded == _SQLITE_VEC_AVAILABLE

            mem = _make_memory()
            adapter.store(mem)
            if _SQLITE_VEC_AVAILABLE:
                assert "_vec_rowid" in mem.metadata

            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 1
            assert results[0].content == "User prefers dark mode"

            # Second memory
            mem2 = _make_memory(
                memory_id="test-2",
                content="User hates bright colors",
                embedding=[0.9] * 384,
                importance=0.8,
            )
            adapter.store(mem2)

            results2 = adapter.search("user1", [0.12] * 384, top_k=2)
            assert len(results2) == 2
            assert results2[0].memory_id == "test-1"

            # Lifecycle filter
            results3 = adapter.search(
                "user1",
                [0.1] * 384,
                top_k=5,
                lifecycle_filter=[LifecycleState.DECAYING],
            )
            assert len(results3) == 0

            fetched = adapter.get("test-1")
            assert fetched is not None

            adapter.delete_by_id("test-1")
            assert adapter.get("test-1") is None
            assert adapter.count("user1") == 1

            adapter.delete_by_user("user1")
            assert adapter.count("user1") == 0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_fallback_without_sqlite_vec(self):
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(memory_id="test-fallback", content="Test fallback")
            adapter.store(mem)
            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_update_same_memory_id(self):
        """Storing the same memory_id twice exercises the update path."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(memory_id="test-3")
            adapter.store(mem)
            adapter.store(mem)  # update
            assert adapter.get("test-3") is not None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# is_vec_available / is_lazy / _embedding_to_json
# ---------------------------------------------------------------------------


class TestUtility:
    def test_is_vec_available_returns_bool(self):
        assert isinstance(SQLiteVecStorageAdapter.is_vec_available(), bool)
        assert SQLiteVecStorageAdapter.is_vec_available() == _SQLITE_VEC_AVAILABLE

    def test_is_lazy_default_false(self):
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            assert adapter.is_lazy() is False
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_is_lazy_true(self):
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
            assert adapter.is_lazy() is True
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_embedding_to_json(self):
        emb = [0.1, 0.2, 0.3]
        result = _embedding_to_json(emb)
        assert json.loads(result) == emb


# ---------------------------------------------------------------------------
# Lazy mode: pending table
# ---------------------------------------------------------------------------


class TestLazyMode:
    def test_store_to_pending_table(self):
        """With lazy=True, embeddings go to memories_vec_pending."""
        tmp = _tmp_db()
        try:
            adapter = _make_lazy_adapter(tmp)
            mem = _make_memory()
            adapter.store(mem)

            # Should be in pending table
            assert adapter._has_pending()
            assert adapter._count_pending() == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_count_pending_cached(self):
        """_count_pending caches the result in _pending_count."""
        tmp = _tmp_db()
        try:
            adapter = _make_lazy_adapter(tmp)
            mem = _make_memory()
            adapter.store(mem)

            # First call computes and caches
            count1 = adapter._count_pending()
            assert count1 == 1

            # Second call uses cache
            count2 = adapter._count_pending()
            assert count2 == 1

            # Cache is invalidated on store
            adapter.store(_make_memory(memory_id="m2"))
            assert adapter._count_pending() == 2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_flush_pending_inserts_to_vec0(self):
        """_flush_pending moves entries from pending → vec0 index."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0 table")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
            mem1 = _make_memory()
            mem2 = _make_memory(memory_id="test-2", content="Second", embedding=[0.9] * 384)
            adapter.store(mem1)
            adapter.store(mem2)
            assert adapter._count_pending() == 2

            adapter._flush_pending()
            assert adapter._count_pending() == 0

            # Now search should work via vec0
            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) >= 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_flush_when_vec_not_loaded(self):
        """_flush_pending is a no-op when _vec_loaded is False."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
            adapter._vec_loaded = False
            # Should not raise
            adapter._flush_pending()
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_flush_no_pending(self):
        """_flush_pending is a no-op when there are no pending entries."""
        tmp = _tmp_db()
        try:
            adapter = _make_lazy_adapter(tmp)
            # No stores → no pending → flush is no-op
            adapter._flush_pending()
            assert adapter._count_pending() == 0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_flush_re_flush_updates_existing(self):
        """Re-flushing a memory with existing vec_rowid updates, not duplicates."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0 table")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
            mem = _make_memory()
            adapter.store(mem)
            adapter._flush_pending()
            assert adapter._count_pending() == 0

            # Store again with lazy=True (should go to pending)
            adapter._pending_count = None  # invalidate cache
            mem.embedding = [0.5] * 384
            adapter.store(mem)
            assert adapter._has_pending()

            # Flush again — pending is cleared, no duplicate vec0 entries
            adapter._flush_pending()
            assert adapter._count_pending() == 0

            # Verify exactly 1 vec0 entry (no duplicates)
            conn = adapter._get_connection()
            vec_count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
            assert vec_count == 1

            # Search returns exactly 1 result
            results = adapter.search("user1", [0.5] * 384, top_k=5)
            assert len(results) == 1
            assert results[0].content == "User prefers dark mode"
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_triggers_flush(self):
        """search() auto-flushes pending vectors before searching."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0 table")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
            mem = _make_memory()
            adapter.store(mem)
            assert adapter._has_pending()

            # search triggers flush
            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 1
            assert adapter._has_pending() is False
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_no_flush_when_no_pending(self):
        """search() does not attempt flush when there are no pending entries."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=True)
            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_pending_count_invalidation(self):
        """_pending_count cache is invalidated on store (set to None)."""
        tmp = _tmp_db()
        try:
            adapter = _make_lazy_adapter(tmp)
            adapter.store(_make_memory(memory_id="m1"))
            assert adapter._count_pending() == 1

            # Store another → _pending_count set to None
            adapter.store(_make_memory(memory_id="m2"))
            # Next _count_pending recomputes from DB
            assert adapter._count_pending() == 2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Search via vec0
# ---------------------------------------------------------------------------


class TestSearchVec:
    def test_search_vec_returns_ranked_results(self):
        """_search_vec returns results ordered by distance."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem1 = _make_memory(embedding=[0.1] * 384)
            mem2 = _make_memory(
                memory_id="test-2",
                content="Different",
                embedding=[0.9] * 384,
            )
            adapter.store(mem1)
            adapter.store(mem2)

            results = adapter._search_vec("user1", [0.1] * 384, top_k=2, states_list=["active"])
            assert len(results) == 2
            assert results[0].memory_id == "test-1"
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_vec_empty_result(self):
        """_search_vec returns [] when no matching memories exist."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            results = adapter._search_vec(
                "nonexistent",
                [0.1] * 384,
                top_k=5,
                states_list=["active"],
            )
            assert results == []
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_vec_lifecycle_filter(self):
        """_search_vec respects lifecycle_state filter."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(lifecycle_state=LifecycleState.ACTIVE)
            adapter.store(mem)

            results_decaying = adapter._search_vec(
                "user1",
                [0.1] * 384,
                top_k=5,
                states_list=["decaying"],
            )
            assert len(results_decaying) == 0

            results_active = adapter._search_vec(
                "user1",
                [0.1] * 384,
                top_k=5,
                states_list=["active"],
            )
            assert len(results_active) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_vec_score_conversion(self):
        """vec0 distance is converted to a score in [0, 1]."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(embedding=[0.1] * 384)
            adapter.store(mem)

            results = adapter._search_vec("user1", [0.1] * 384, top_k=1, states_list=["active"])
            assert len(results) == 1
            assert 0.0 <= results[0].score <= 1.0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_fallback_to_bruteforce(self):
        """When vec0 is not loaded, search falls back to parent brute-force."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter._vec_loaded = False
            mem = _make_memory()
            adapter.store(mem)
            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_vec_multiple_lifecycle_states(self):
        """_search_vec with multiple lifecycle states in filter."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(memory_id="m1", lifecycle_state=LifecycleState.ACTIVE))
            adapter.store(
                _make_memory(
                    memory_id="m2",
                    embedding=[0.2] * 384,
                    lifecycle_state=LifecycleState.DECAYING,
                )
            )

            results = adapter._search_vec(
                "user1",
                [0.1] * 384,
                top_k=5,
                states_list=["active", "decaying"],
            )
            assert len(results) == 2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Delete paths
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_id_with_vec_rowid(self):
        """delete_by_id removes from both memories and memories_vec."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            adapter.store(mem)
            assert "_vec_rowid" in mem.metadata

            deleted = adapter.delete_by_id("test-1")
            assert deleted is True
            assert adapter.get("test-1") is None

            # Verify vec0 entry is also gone
            with adapter._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM memories_vec WHERE memory_id = ?", ("test-1",)
                ).fetchone()
                assert row is None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_by_id_no_vec_rowid(self):
        """delete_by_id works when memory has no vec_rowid."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            adapter.store(mem)

            deleted = adapter.delete_by_id("test-1")
            assert deleted is True
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_by_id_nonexistent(self):
        """delete_by_id returns False for non-existent memory."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            deleted = adapter.delete_by_id("nonexistent")
            assert deleted is False
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_by_user_with_vec_rowids(self):
        """delete_by_user removes vec0 entries for all user memories."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(memory_id="m1"))
            adapter.store(_make_memory(memory_id="m2", embedding=[0.2] * 384))
            adapter.store(
                _make_memory(
                    memory_id="m3",
                    user_id="user2",
                    embedding=[0.3] * 384,
                )
            )

            count = adapter.delete_by_user("user1")
            assert count == 2
            assert adapter.count("user1") == 0
            assert adapter.get("m3") is not None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_by_user_no_vec_rowids(self):
        """delete_by_user works when no memories have vec_rowids."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter._vec_loaded = False
            adapter.store(_make_memory(memory_id="m1"))
            adapter.store(_make_memory(memory_id="m2"))

            count = adapter.delete_by_user("user1")
            assert count == 2
            assert adapter.count("user1") == 0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_by_user_cleans_pending(self):
        """delete_by_user also removes pending entries."""
        tmp = _tmp_db()
        try:
            adapter = _make_lazy_adapter(tmp)
            adapter.store(_make_memory(memory_id="m1"))
            adapter.store(_make_memory(memory_id="m2", embedding=[0.2] * 384))
            assert adapter._count_pending() == 2

            count = adapter.delete_by_user("user1")
            assert count == 2
            assert adapter._count_pending() == 0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_by_id_cleans_pending(self):
        """delete_by_id also removes pending entries."""
        tmp = _tmp_db()
        try:
            adapter = _make_lazy_adapter(tmp)
            adapter.store(_make_memory(memory_id="m1"))
            assert adapter._count_pending() == 1

            adapter.delete_by_id("m1")
            assert adapter._count_pending() == 0
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_delete_vec_not_loaded(self):
        """Delete works even when vec0 is not loaded (no vec0 cleanup needed)."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter._vec_loaded = False
            adapter.store(_make_memory())

            deleted = adapter.delete_by_id("test-1")
            assert deleted is True

            count = adapter.delete_by_user("user1")
            assert count == 0  # already deleted
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Store: direct vec path (update existing vec_rowid)
# ---------------------------------------------------------------------------


class TestStoreVecDirect:
    def test_store_updates_existing_vec_entry(self):
        """Storing a memory with an existing vec_rowid updates the vec0 entry."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(embedding=[0.1] * 384)
            adapter.store(mem)
            original_rowid = mem.metadata.get("_vec_rowid")
            assert original_rowid is not None

            # Store again with updated embedding — should UPDATE, not INSERT
            mem.embedding = [0.5] * 384
            adapter.store(mem)
            assert mem.metadata.get("_vec_rowid") == original_rowid

            results = adapter.search("user1", [0.5] * 384, top_k=1)
            assert len(results) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_store_with_none_embedding(self):
        """Storing a memory with None embedding skips vec0 insertion."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            mem.embedding = None
            adapter.store(mem)
            fetched = adapter.get("test-1")
            assert fetched is not None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_store_vec_not_loaded(self):
        """When _vec_loaded is False, store skips vec0 insertion."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter._vec_loaded = False
            mem = _make_memory()
            adapter.store(mem)
            fetched = adapter.get("test-1")
            assert fetched is not None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_store_no_lazy_direct_vec_insert(self):
        """Non-lazy mode inserts directly into vec0."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384, lazy=False)
            mem = _make_memory()
            adapter.store(mem)
            assert "_vec_rowid" in mem.metadata
            assert adapter._has_pending() is False
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


class TestRowHelpers:
    def test_memory_to_row_includes_vec_rowid(self):
        """_memory_to_row includes vec_rowid from metadata."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            mem.metadata["_vec_rowid"] = 42
            row = adapter._memory_to_row(mem)
            assert row["vec_rowid"] == 42
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_memory_to_row_no_vec_rowid(self):
        """_memory_to_row sets vec_rowid to None when not in metadata."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            row = adapter._memory_to_row(mem)
            assert row["vec_rowid"] is None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_memory_to_row_no_metadata(self):
        """_memory_to_row handles memory with None metadata."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            mem.metadata = None  # type: ignore[assignment]
            row = adapter._memory_to_row(mem)
            assert row["vec_rowid"] is None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_row_to_memory_with_vec_rowid(self):
        """_row_to_memory extracts vec_rowid into metadata."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required for vec0")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            adapter.store(mem)
            assert "_vec_rowid" in mem.metadata

            with adapter._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", ("test-1",)
                ).fetchone()
            parsed = adapter._row_to_memory(row)
            assert "_vec_rowid" in parsed.metadata
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_row_to_memory_no_vec_rowid(self):
        """_row_to_memory handles missing vec_rowid gracefully."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            adapter.store(mem)
            # Manually set vec_rowid to NULL
            with adapter._get_connection() as conn:
                conn.execute(
                    "UPDATE memories SET vec_rowid = NULL WHERE memory_id = ?",
                    ("test-1",),
                )
            with adapter._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", ("test-1",)
                ).fetchone()
            parsed = adapter._row_to_memory(row)
            assert "_vec_rowid" not in parsed.metadata
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_update_delegates_to_store(self):
        """update() calls store() internally."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            adapter.store(mem)

            # Update content via update()
            mem.content = "Updated content"
            adapter.update(mem)

            fetched = adapter.get("test-1")
            assert fetched is not None
            assert fetched.content == "Updated content"
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Schema / Migrations
# ---------------------------------------------------------------------------


class TestSchema:
    def test_init_schema_creates_all_tables(self):
        """_init_schema creates memories, schema_version, memories_vec_pending tables."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            with adapter._get_connection() as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = {t[0] for t in tables}
                assert "memories" in table_names
                assert "schema_version" in table_names
                assert "memories_vec_pending" in table_names
                if _SQLITE_VEC_AVAILABLE:
                    assert "memories_vec" in table_names
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_init_schema_indexes(self):
        """_init_schema creates expected indexes."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            with adapter._get_connection() as conn:
                indexes = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
                index_names = {t[0] for t in indexes}
                assert "idx_memories_user_id" in index_names
                assert "idx_memories_lifecycle" in index_names
                assert "idx_memories_user_lifecycle" in index_names
                assert "idx_memories_tags" in index_names
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_init_vec_table_skips_when_not_available(self):
        """_init_vec_table is a no-op when sqlite-vec is not installed."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter._vec_loaded = False
            with adapter._get_connection() as conn:
                adapter._init_vec_table(conn)
            # Should not crash, _vec_loaded stays False (without vec)
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_run_migrations_idempotent(self):
        """Running migrations twice does not cause errors."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            with adapter._get_connection() as conn:
                adapter._run_migrations(conn)
                adapter._run_migrations(conn)
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_schema_version_table_exists(self):
        """schema_version table is created during init."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            with adapter._get_connection() as conn:
                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                assert row is not None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Connection loading
# ---------------------------------------------------------------------------


class TestConnection:
    def test_get_connection_loads_vec_extension(self):
        """_get_connection loads the vec0 extension when available."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            conn = adapter._get_connection()
            # Second call should not re-load
            conn2 = adapter._get_connection()
            if adapter._shared_conn is not None:
                assert conn is conn2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_get_connection_loads_extension_when_available(self):
        """When sqlite-vec is available, the adapter marks _vec_loaded."""
        if not _SQLITE_VEC_AVAILABLE:
            pytest.skip("sqlite-vec required")

        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            # _init_vec_table sets _vec_loaded on the adapter
            assert adapter._vec_loaded is True
            # Connection should also be usable for vec0 operations
            conn = adapter._get_connection()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'"
            ).fetchone()
            assert row is not None
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_get_connection_returns_same_shared_conn(self):
        """_get_connection always returns the same shared connection."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            conn1 = adapter._get_connection()
            conn2 = adapter._get_connection()
            assert conn1 is conn2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_context_manager_basic(self):
        """SQLiteVecStorageAdapter works as a context manager via inherited __enter__/__exit__."""
        tmp = _tmp_db()
        try:
            with SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384) as adapter:
                assert adapter._shared_conn is not None
                mem = _make_memory()
                adapter.store(mem)
                result = adapter.get("test-1")
                assert result is not None
                assert result.content == "User prefers dark mode"
            # Connection closed after context exit
            assert adapter._shared_conn is None
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_context_manager_in_memory(self):
        """SQLiteVecStorageAdapter context manager works with :memory: database."""
        with SQLiteVecStorageAdapter(db_path=":memory:", embedding_dim=384) as adapter:
            assert adapter._shared_conn is not None
            mem = _make_memory()
            adapter.store(mem)
            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 1
        assert adapter._shared_conn is None

    def test_context_manager_closes_on_exception(self):
        """SQLiteVecStorageAdapter context manager closes connection on exception."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            with pytest.raises(ValueError):
                with adapter:
                    raise ValueError("boom")
            assert adapter._shared_conn is None
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_store_multiple_users(self):
        """Store and search across multiple users."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(user_id="alice"))
            adapter.store(_make_memory(memory_id="m2", user_id="bob", embedding=[0.2] * 384))

            results_alice = adapter.search("alice", [0.1] * 384, top_k=5)
            assert len(results_alice) == 1
            assert results_alice[0].user_id == "alice"

            results_bob = adapter.search("bob", [0.2] * 384, top_k=5)
            assert len(results_bob) == 1
            assert results_bob[0].user_id == "bob"
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_get_all_by_user(self):
        """get_all_by_user returns all memories for a user."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(memory_id="m1"))
            adapter.store(_make_memory(memory_id="m2", embedding=[0.2] * 384))
            all_mems = adapter.get_all_by_user("user1")
            assert len(all_mems) == 2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_get_all(self):
        """get_all returns all memories across users."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(user_id="alice"))
            adapter.store(_make_memory(memory_id="m2", user_id="bob", embedding=[0.2] * 384))
            all_mems = adapter.get_all()
            assert len(all_mems) == 2
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_get_all_users(self):
        """get_all_users returns unique user IDs."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(user_id="alice"))
            adapter.store(_make_memory(memory_id="m2", user_id="bob", embedding=[0.2] * 384))
            users = adapter.get_all_users()
            assert set(users) == {"alice", "bob"}
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_upgrade_schema(self):
        """upgrade_schema runs without errors."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.upgrade_schema(1, 4)
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_default_lifecycle_filter(self):
        """search with no lifecycle_filter uses ACTIVE + DECAYING."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(lifecycle_state=LifecycleState.ACTIVE)
            adapter.store(mem)

            results = adapter.search("user1", [0.1] * 384, top_k=5)
            assert len(results) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_store_with_tags(self):
        """Tags are preserved through store and retrieve."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory(tags=["food", "preference"])
            adapter.store(mem)

            fetched = adapter.get("test-1")
            assert fetched is not None
            assert "food" in fetched.tags
            assert "preference" in fetched.tags
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_get_by_tag(self):
        """get_by_tag returns matching memories."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory(tags=["food"]))
            adapter.store(_make_memory(memory_id="m2", tags=["work"], embedding=[0.2] * 384))

            results = adapter.get_by_tag("user1", "food")
            assert len(results) == 1
            assert results[0].memory_id == "test-1"
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_empty_query_embedding(self):
        """Empty query_embedding falls back to brute-force."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            mem = _make_memory()
            adapter.store(mem)
            results = adapter.search("user1", [], top_k=5)
            assert len(results) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_search_top_k_exceeds_results(self):
        """top_k larger than available results returns all results."""
        tmp = _tmp_db()
        try:
            adapter = SQLiteVecStorageAdapter(db_path=tmp, embedding_dim=384)
            adapter.store(_make_memory())
            results = adapter.search("user1", [0.1] * 384, top_k=100)
            assert len(results) == 1
            adapter.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_in_memory_db(self):
        """:memory: DB works with vec adapter."""
        adapter = SQLiteVecStorageAdapter(db_path=":memory:", embedding_dim=384)
        adapter.store(_make_memory())
        results = adapter.search("user1", [0.1] * 384, top_k=5)
        assert len(results) == 1
        adapter.close()
