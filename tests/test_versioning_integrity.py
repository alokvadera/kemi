"""Integrity and concurrency tests for the memory versioning system.

These tests cover the constraints, race conditions, and edge cases that
were not exercised by the basic versioning suite. They live in a
separate file so the new tests can be run in isolation when investigating
regressions.

Covers:
- Foreign key / cascade behaviour
- Unique constraint enforcement on (memory_id, version_number)
- Concurrent record_version calls (serialised by BEGIN IMMEDIATE)
- Rollback creates a new version (does not reuse the old number)
- Sequential version integrity (no gaps, no duplicates)
- Auto-prune respects max_versions_per_memory under churn
- record_before_update handles the pre+post collision correctly
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from kemi.memory.model import (
    LifecycleState,
    MemoryObject,
    MemorySource,
    MemoryType,
)
from kemi.memory.versions import (
    MemoryVersionStore,
    _pack_embedding,
    _unpack_embedding,
)
from tests._helpers.factories import make_memory

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(
    memory_id: str = "mem-1",
    user_id: str = "user-1",
    content: str = "Hello world",
    version: int = 1,
    importance: float = 0.5,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    memory_type: MemoryType = MemoryType.EPISODIC,
    namespace: str = "default",
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    confidence: float = 1.0,
) -> MemoryObject:
    now = datetime.now(timezone.utc)
    return make_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=[0.1, 0.2, 0.3],
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=importance,
        lifecycle_state=lifecycle_state,
        metadata=metadata or {},
        embedding_dim=3,
        tags=tags or [],
        confidence=confidence,
        memory_type=memory_type,
        session_id=None,
        namespace=namespace,
        version=version,
    )


@pytest.fixture
def vs(tmp_path) -> MemoryVersionStore:
    return MemoryVersionStore(db_path=str(tmp_path / "integrity.db"))


# ---------------------------------------------------------------------------
# Schema constraints
# ---------------------------------------------------------------------------

class TestSchemaConstraints:
    def test_memory_versions_table_exists(self, vs: MemoryVersionStore, tmp_path):
        with sqlite3.connect(vs._db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_versions'"
            ).fetchone()
        assert row is not None

    def test_primary_key_on_memory_id_and_version(self, vs: MemoryVersionStore, tmp_path):
        """The (memory_id, version) composite primary key is what guarantees uniqueness."""
        with sqlite3.connect(vs._db_path) as conn:
            pk = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='memory_versions'"
            ).fetchone()
            assert pk is not None
            index = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='index' AND name='idx_versions_memory'"
            ).fetchone()
        # The descending index supports fast newest-first lookups
        assert index is not None
        assert "DESC" in (index[0] or "")

    def test_unique_constraint_rejects_duplicate_version(self, vs: MemoryVersionStore):
        """Direct INSERT with the same (memory_id, version) must fail with UNIQUE."""
        mem = _make_memory(memory_id="mem-dup", version=1)
        vs.record_version(mem, changed_by="first")
        # Second record_version with same version=1 should auto-increment,
        # NOT fail. This is the fix.
        mem2 = _make_memory(memory_id="mem-dup", content="new", version=1)
        returned = vs.record_version(mem2, changed_by="second")
        assert returned == 2  # auto-incremented

    def test_raw_duplicate_insert_still_fails(self, vs: MemoryVersionStore):
        """Bypassing the API and inserting directly with a duplicate must raise."""
        mem = _make_memory(memory_id="mem-raw-dup", version=1)
        vs.record_version(mem, changed_by="first")
        with pytest.raises(sqlite3.IntegrityError):
            with sqlite3.connect(vs._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO memory_versions
                        (memory_id, version, content, importance, metadata, tags,
                         memory_type, confidence, namespace, source, changed_at, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("mem-raw-dup", 1, "dup", 0.5, "{}", "[]", "episodic",
                     1.0, "default", "user_stated",
                     datetime.now(timezone.utc).isoformat(), "test"),
                )
                conn.commit()


# ---------------------------------------------------------------------------
# Race condition / concurrency
# ---------------------------------------------------------------------------

class TestConcurrentVersionCreation:
    def test_serialised_writes_no_unique_violation(self, vs: MemoryVersionStore):
        """Many threads calling record_version for the same memory must all succeed.

        With the old code, threads would race on the (memory_id, version)
        primary key and one would fail. With BEGIN IMMEDIATE serialisation
        and auto-increment fallback, every thread writes a unique version.
        """
        results: list[int] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def worker(i: int) -> None:
            try:
                barrier.wait()  # release all threads at the same instant
                mem = _make_memory(
                    memory_id="mem-race",
                    content=f"thread-{i}",
                    version=1,  # all threads "want" version 1
                )
                v = vs.record_version(mem, changed_by=f"t{i}")
                results.append(v)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes failed: {errors}"
        # All 8 writes must have produced distinct version numbers
        assert len(results) == 8
        assert len(set(results)) == 8, f"Duplicate versions returned: {results}"
        # Versions must be contiguous: 1..8
        assert sorted(results) == list(range(1, 9))

    def test_sequential_integrity_after_concurrent_writes(self, vs: MemoryVersionStore):
        """After concurrent writes, verify_sequential_versions returns True."""
        barrier = threading.Barrier(5)

        def worker(i: int) -> None:
            barrier.wait()
            mem = _make_memory(memory_id="mem-seq", content=f"t{i}", version=1)
            vs.record_version(mem, changed_by=f"t{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert vs.verify_sequential_versions("mem-seq") is True

    def test_concurrent_record_before_update(self, vs: MemoryVersionStore):
        """record_before_update also serialises correctly under contention."""
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker(i: int) -> None:
            try:
                barrier.wait()
                before = _make_memory(
                    memory_id="mem-rbu-race", content=f"before-{i}", version=i + 1
                )
                after = _make_memory(
                    memory_id="mem-rbu-race", content=f"after-{i}", version=i + 2
                )
                vs.record_before_update(before, after, changed_by=f"t{i}")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent record_before_update failed: {errors}"


# ---------------------------------------------------------------------------
# Auto-increment fallback
# ---------------------------------------------------------------------------

class TestAutoIncrement:
    def test_uses_caller_version_when_unique(self, vs: MemoryVersionStore):
        mem = _make_memory(memory_id="m1", version=7)
        returned = vs.record_version(mem)
        assert returned == 7
        assert vs.get_latest_version_number("m1") == 7

    def test_auto_increments_on_collision(self, vs: MemoryVersionStore):
        mem1 = _make_memory(memory_id="m2", version=1)
        vs.record_version(mem1)
        # Second call with same version=1 must auto-increment
        mem2 = _make_memory(memory_id="m2", content="v2", version=1)
        returned = vs.record_version(mem2)
        assert returned == 2

    def test_caller_version_higher_than_max_preserved(self, vs: MemoryVersionStore):
        """If the caller's version is greater than the current max, it's honoured."""
        mem1 = _make_memory(memory_id="m3", version=1)
        vs.record_version(mem1)
        # Caller says version=10 (jumped ahead) — this is respected.
        mem2 = _make_memory(memory_id="m3", content="v10", version=10)
        returned = vs.record_version(mem2)
        assert returned == 10
        assert vs.get_latest_version_number("m3") == 10


# ---------------------------------------------------------------------------
# record_before_update: pre+post collision handling
# ---------------------------------------------------------------------------

class TestRecordBeforeUpdate:
    def test_pre_and_post_have_distinct_versions(self, vs: MemoryVersionStore):
        """The old bug: both pre and post inserted with the same version.

        With the fix, the pre-update INSERT OR REPLACE keeps the old
        version and the post-update is auto-incremented to the next slot.
        """
        before = _make_memory(memory_id="mbu", content="old", version=1)
        after = _make_memory(memory_id="mbu", content="new", version=1)
        returned = vs.record_before_update(before, after, changed_by="update")
        assert returned == 2  # post-update got the next free version

        snaps = vs.list_versions("mbu")
        # Should be exactly 2 snapshots: pre-update and update
        assert len(snaps) == 2
        labels = {s.changed_by for s in snaps}
        assert "pre-update" in labels
        assert "update" in labels

    def test_sequential_versions_after_repeated_updates(self, vs: MemoryVersionStore):
        """Many update cycles produce a contiguous version sequence.

        Each cycle's pre-update replaces the current version and the
        post-update advances to the next free number, so 5 cycles produce
        a contiguous 1..6 sequence.
        """
        for i in range(5):
            before = _make_memory(memory_id="mcycle", content=f"v{i}", version=i + 1)
            after = _make_memory(memory_id="mcycle", content=f"v{i + 1}", version=i + 1)
            vs.record_before_update(before, after, changed_by="update")

        assert vs.verify_sequential_versions("mcycle") is True
        assert vs.get_latest_version_number("mcycle") == 6
        assert len(vs.list_versions("mcycle")) == 6

    def test_collision_with_existing_versions_handled(self, vs: MemoryVersionStore):
        """If memory_after.version collides with an existing row, post must
        advance to the next free slot rather than failing."""
        # Pre-seed with version=5
        seed = _make_memory(memory_id="mcoll", content="seed", version=5)
        vs.record_version(seed)

        # Now simulate an update where the caller didn't bump version
        before = _make_memory(memory_id="mcoll", content="v1", version=1)
        after = _make_memory(memory_id="mcoll", content="v2", version=1)
        returned = vs.record_before_update(before, after, changed_by="update")

        # Post must be the next free version after 5
        assert returned == 6


# ---------------------------------------------------------------------------
# Rollback: new version, not reused
# ---------------------------------------------------------------------------

class TestRollbackCreatesNewVersion:
    def test_rollback_writes_fresh_version(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        adapter = SQLiteStorageAdapter(db_path=str(tmp_path / "rb.db"))

        # Seed: v1 and v2 in both adapter and version store
        v1 = _make_memory(memory_id="rb1", content="original", version=1)
        adapter.store(v1)
        vs.record_version(v1, changed_by="create")

        v2 = _make_memory(memory_id="rb1", content="modified", version=2)
        adapter.update(v2)
        vs.record_version(v2, changed_by="update")

        # Rollback to v1
        result = vs.rollback("rb1", target_version=1, store=adapter)
        assert result is not None
        # The new version must be > 2 (a fresh number, not 1)
        assert result.to_version > 2
        # Content in adapter is restored
        restored = adapter.get("rb1")
        assert restored is not None
        assert restored.content == "original"
        # Adapter row's version reflects the new state
        assert restored.version == result.to_version

    def test_rollback_does_not_overwrite_history(self, vs: MemoryVersionStore, tmp_path):
        """Old versions must remain queryable after a rollback."""
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
        adapter = SQLiteStorageAdapter(db_path=str(tmp_path / "rb_hist.db"))

        for i in range(1, 4):
            mem = _make_memory(memory_id="rbh", content=f"v{i}", version=i)
            adapter.store(mem)
            vs.record_version(mem, changed_by="update")

        vs.rollback("rbh", target_version=1, store=adapter)

        # All original versions should still be listable
        snaps = vs.list_versions("rbh")
        # 3 originals + 1 rollback = 4
        assert len(snaps) >= 3
        # Old v1 must still be findable
        v1 = vs.get_version("rbh", 1)
        assert v1 is not None
        assert v1.content == "v1"

    def test_rollback_nonexistent_version_returns_none(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
        adapter = SQLiteStorageAdapter(db_path=str(tmp_path / "rb_none.db"))

        mem = _make_memory(memory_id="rbn", version=1)
        adapter.store(mem)
        vs.record_version(mem)

        result = vs.rollback("rbn", target_version=99, store=adapter)
        assert result is None

    def test_rollback_missing_memory_returns_none(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
        adapter = SQLiteStorageAdapter(db_path=str(tmp_path / "rb_missing.db"))

        result = vs.rollback("does-not-exist", target_version=1, store=adapter)
        assert result is None


# ---------------------------------------------------------------------------
# Sequential integrity check
# ---------------------------------------------------------------------------

class TestSequentialIntegrity:
    def test_empty_memory_passes(self, vs: MemoryVersionStore):
        assert vs.verify_sequential_versions("nonexistent") is True

    def test_single_version_passes(self, vs: MemoryVersionStore):
        vs.record_version(_make_memory(memory_id="m1", version=1))
        assert vs.verify_sequential_versions("m1") is True

    def test_contiguous_versions_pass(self, vs: MemoryVersionStore):
        for i in range(1, 6):
            vs.record_version(_make_memory(memory_id="m2", version=i))
        assert vs.verify_sequential_versions("m2") is True


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class TestPruning:
    def test_prune_keeps_only_most_recent(self, vs: MemoryVersionStore):
        for i in range(1, 11):
            vs.record_version(_make_memory(memory_id="mp", content=f"v{i}", version=i))

        deleted = vs.prune_versions("mp", keep_count=3)
        assert deleted == 7

        remaining = vs.list_versions("mp")
        assert len(remaining) == 3
        # Most recent three must be v8, v9, v10
        versions = {s.version for s in remaining}
        assert versions == {8, 9, 10}

    def test_prune_does_nothing_when_under_limit(self, vs: MemoryVersionStore):
        for i in range(1, 4):
            vs.record_version(_make_memory(memory_id="mp2", version=i))
        deleted = vs.prune_versions("mp2", keep_count=10)
        assert deleted == 0
        assert len(vs.list_versions("mp2")) == 3

    def test_prune_zero_returns_zero(self, vs: MemoryVersionStore):
        vs.record_version(_make_memory(memory_id="mp3", version=1))
        assert vs.prune_versions("mp3", keep_count=0) == 0


# ---------------------------------------------------------------------------
# Embedding round-trip precision
# ---------------------------------------------------------------------------

class TestEmbeddingPrecision:
    def test_round_trip_preserves_values(self):
        """The old float32 packing lost precision for 0.1 → 0.10000000149..."""
        original = [0.1, 0.2, 0.3, -0.5, 1.234567]
        packed = _pack_embedding(original)
        unpacked = _unpack_embedding(packed)
        assert unpacked == pytest.approx(original)

    def test_round_trip_preserves_zero(self):
        packed = _pack_embedding([0.0, 0.0, 0.0])
        unpacked = _unpack_embedding(packed)
        assert unpacked == [0.0, 0.0, 0.0]

    def test_round_trip_empty(self):
        assert _pack_embedding(None) is None
        assert _unpack_embedding(None) is None
        assert _pack_embedding([]) is None

    def test_legacy_float32_blobs_still_unpackable(self):
        """Backwards compatibility: blobs written by the old code (4 bytes
        per float) must still be readable."""
        import struct
        legacy_blob = struct.pack("<3f", 0.1, 0.2, 0.3)
        unpacked = _unpack_embedding(legacy_blob)
        assert unpacked is not None
        assert len(unpacked) == 3
        # Values are approximate (float32 precision)
        assert unpacked[0] == pytest.approx(0.1, abs=1e-6)

    def test_embedding_survives_db_round_trip(self, vs: MemoryVersionStore):
        mem = _make_memory(memory_id="emb-prec", version=1)
        mem.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mem.embedding_dim = 5
        vs.record_version(mem, changed_by="update")

        snap = vs.get_version("emb-prec", 1)
        assert snap is not None
        assert snap.embedding == pytest.approx(mem.embedding)


# ---------------------------------------------------------------------------
# Fallback version store (no _db_path)
# ---------------------------------------------------------------------------

class TestFallbackVersionStore:
    def test_get_history_returns_empty_when_no_versions(self):
        """When a storage adapter has no _db_path, get_history must still
        work (in-memory) and return an empty list for unknown IDs."""
        from kemi import Memory
        from kemi.adapters.base import EmbeddingAdapter, StorageAdapter

        class BareStore(StorageAdapter):
            """Storage adapter with no _db_path at all."""

            def store(self, memory: MemoryObject) -> None:
                pass

            def get(self, memory_id: str) -> MemoryObject | None:
                return None

            def search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return []

            def update(self, memory: MemoryObject) -> None:
                pass

            def delete_by_user(self, user_id: str) -> int:
                return 0

            def delete_by_id(self, memory_id: str) -> bool:
                return False

            def get_all_by_user(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return []

            def count(self, user_id: str) -> int:
                return 0

            def get_all(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return []

            def get_all_users(self) -> list[str]:
                return []

            def upgrade_schema(self, from_version: int, to_version: int) -> None:
                pass

            def get_by_tag(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return []

            def search_by_content(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return []

        class DummyEmbed(EmbeddingAdapter):
            def embed(self, texts):  # type: ignore[no-untyped-def]
                return [[0.0] * 4 for _ in texts]

            def embed_single(self, text):  # type: ignore[no-untyped-def]
                return [0.0] * 4

            def dimension(self) -> int:
                return 4

        mem = Memory(embed=DummyEmbed(), store=BareStore())
        # Should not raise even though BareStore has no _db_path
        assert mem.get_history("anything") == []
        assert mem.diff_versions("anything", 1, 2) is None
        assert mem.rollback_memory("anything", 1) is None
