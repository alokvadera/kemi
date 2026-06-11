"""Tests for TTL (time-to-live) on memories."""

from datetime import datetime, timedelta, timezone

import pytest

from kemi import Memory
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType
from tests._helpers.factories import make_memory

pytestmark = pytest.mark.slow


def _make_mem_with_ttl(**overrides: object) -> MemoryObject:
    """Build a MemoryObject with a given expires_at offset (seconds from now)."""
    defaults = {
        "memory_id": "ttl-id",
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
        "memory_type": MemoryType.EPISODIC,
        "session_id": None,
        "namespace": "default",
        "version": 1,
        "expires_at": None,
    }
    defaults.update(overrides)
    return make_memory(**defaults)


class TestMemoryObjectTTL:
    def test_default_expires_at_is_none(self) -> None:
        mem = _make_mem_with_ttl()
        assert mem.expires_at is None

    def test_expires_at_can_be_set(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        mem = _make_mem_with_ttl(expires_at=future)
        assert mem.expires_at == future


class TestRememberTTL:
    def test_remember_with_ttl_sets_expires_at(self, real_db_memory: Memory) -> None:
        memory_id = real_db_memory.remember(
            "user1", "Will expire soon", ttl_seconds=3600
        )
        mem = real_db_memory._store.get(memory_id)
        assert mem is not None
        assert mem.expires_at is not None
        # Should be approximately 1 hour from now
        delta = (mem.expires_at - datetime.now(timezone.utc)).total_seconds()
        assert 3500 < delta <= 3600

    def test_remember_without_ttl_no_expires_at(self, real_db_memory: Memory) -> None:
        memory_id = real_db_memory.remember("user1", "No TTL on this")
        mem = real_db_memory._store.get(memory_id)
        assert mem is not None
        assert mem.expires_at is None

    def test_remember_invalid_ttl_raises(self, real_db_memory: Memory) -> None:
        with pytest.raises(ValueError):
            real_db_memory.remember("user1", "Bad TTL", ttl_seconds=0)
        with pytest.raises(ValueError):
            real_db_memory.remember("user1", "Bad TTL", ttl_seconds=-10)
        with pytest.raises(ValueError):
            real_db_memory.remember("user1", "Bad TTL", ttl_seconds="not an int")  # type: ignore[arg-type]

    def test_remember_many_with_ttl(self, real_db_memory: Memory) -> None:
        ids = real_db_memory.remember_many(
            "user1", ["a", "b", "c"], ttl_seconds=600
        )
        for mid in ids:
            mem = real_db_memory._store.get(mid)
            assert mem is not None
            assert mem.expires_at is not None


class TestPruneExpired:
    def test_prune_expired_deletes_expired(self, real_db_memory: Memory) -> None:
        # Manually create a memory that's already expired
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        mem = _make_mem_with_ttl(memory_id="expired-1", expires_at=past)
        real_db_memory._store.store(mem)

        # Create a non-expired memory
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        mem2 = _make_mem_with_ttl(memory_id="fresh-1", expires_at=future)
        real_db_memory._store.store(mem2)

        deleted = real_db_memory.prune_expired(user_id="user1")
        assert deleted == 1
        assert real_db_memory._store.get("expired-1") is None
        assert real_db_memory._store.get("fresh-1") is not None

    def test_prune_expired_no_ttl_memories_ignored(
        self, real_db_memory: Memory
    ) -> None:
        # Memory with no expires_at should never be pruned by prune_expired
        mem = _make_mem_with_ttl(memory_id="no-ttl")
        assert mem.expires_at is None
        real_db_memory._store.store(mem)

        deleted = real_db_memory.prune_expired(user_id="user1")
        assert deleted == 0
        assert real_db_memory._store.get("no-ttl") is not None

    def test_prune_expired_all_users(self, real_db_memory: Memory) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(memory_id="u1-exp", user_id="alice", expires_at=past)
        )
        real_db_memory._store.store(
            _make_mem_with_ttl(memory_id="u2-exp", user_id="bob", expires_at=past)
        )

        deleted = real_db_memory.prune_expired()
        assert deleted == 2

    def test_prune_expired_with_ttl_remember(
        self, real_db_memory: Memory
    ) -> None:
        # Use remember() with a very short TTL — simulate elapsed time by
        # directly storing a memory with past expires_at
        real_db_memory.remember("user1", "test ttl", ttl_seconds=3600)
        # Manually backdate the stored memory
        mems = real_db_memory._store.get_all_by_user("user1")
        assert len(mems) == 1
        mems[0].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.update(mems[0])

        deleted = real_db_memory.prune_expired(user_id="user1")
        assert deleted == 1
        assert real_db_memory._store.get(mems[0].memory_id) is None


class TestRunMaintenanceWithTTL:
    def test_run_maintenance_includes_expired(self, real_db_memory: Memory) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(memory_id="exp-mem", expires_at=past)
        )

        result = real_db_memory.run_maintenance(
            user_id="user1",
            auto_prune=False,
            auto_consolidate=False,
            auto_prune_expired=True,
        )
        assert result["expired"] == 1
        assert real_db_memory._store.get("exp-mem") is None

    def test_run_maintenance_disable_expired(self, real_db_memory: Memory) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(memory_id="exp-mem-2", expires_at=past)
        )

        result = real_db_memory.run_maintenance(
            user_id="user1",
            auto_prune=False,
            auto_consolidate=False,
            auto_prune_expired=False,
        )
        assert result["expired"] == 0
        # Memory still exists
        assert real_db_memory._store.get("exp-mem-2") is not None

    def test_prune_expired_with_namespace_filter(
        self, real_db_memory: Memory
    ) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(
                memory_id="ns1-exp", namespace="alpha", expires_at=past
            )
        )
        real_db_memory._store.store(
            _make_mem_with_ttl(
                memory_id="ns2-exp", namespace="beta", expires_at=past
            )
        )

        # Only sweep namespace "alpha"
        deleted = real_db_memory.prune_expired(
            user_id="user1", namespace="alpha"
        )
        assert deleted == 1
        assert real_db_memory._store.get("ns1-exp") is None
        assert real_db_memory._store.get("ns2-exp") is not None

    def test_prune_expired_returns_zero_when_none_expired(
        self, real_db_memory: Memory
    ) -> None:
        # No memories at all
        assert real_db_memory.prune_expired(user_id="user1") == 0

        # All memories in the future
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(memory_id="future-1", expires_at=future)
        )
        assert real_db_memory.prune_expired(user_id="user1") == 0

    def test_prune_expired_sweeps_all_namespaces(
        self, real_db_memory: Memory
    ) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(
                memory_id="alpha-exp", namespace="alpha", expires_at=past
            )
        )
        real_db_memory._store.store(
            _make_mem_with_ttl(
                memory_id="beta-exp", namespace="beta", expires_at=past
            )
        )
        real_db_memory._store.store(
            _make_mem_with_ttl(
                memory_id="gamma-exp", namespace="gamma", expires_at=past
            )
        )

        # No namespace filter should sweep ALL namespaces
        deleted = real_db_memory.prune_expired(user_id="user1")
        assert deleted == 3
        assert real_db_memory._store.get("alpha-exp") is None
        assert real_db_memory._store.get("beta-exp") is None
        assert real_db_memory._store.get("gamma-exp") is None

    def test_run_maintenance_combined_prune_and_prune_expired(
        self, real_db_memory: Memory
    ) -> None:
        # An ACTIVE memory that will be pruned by lifecycle (not by TTL)
        real_db_memory.remember("user1", "to be lifecycle-pruned")
        # An ACTIVE memory that's already TTL-expired
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        real_db_memory._store.store(
            _make_mem_with_ttl(memory_id="ttl-expired", expires_at=past)
        )

        result = real_db_memory.run_maintenance(
            user_id="user1",
            auto_prune=True,
            auto_consolidate=False,
            auto_prune_expired=True,
        )
        # Should contain both 'pruned' and 'expired' keys
        assert "pruned" in result
        assert "expired" in result
        assert result["expired"] >= 1
        assert real_db_memory._store.get("ttl-expired") is None
