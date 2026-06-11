"""Reproduction tests for versioning race conditions (R1, R2, R4).

These tests are expected to FAIL against the current (unfixed) code.
The main agent will apply fixes and then re-run this file.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

import pytest

from kemi import Memory
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.memory.model import (
    LifecycleState,
    MemoryObject,
    MemorySource,
    MemoryType,
)
from kemi.memory.versions import MemoryVersionStore
from tests._helpers.factories import make_memory

pytestmark = pytest.mark.slow



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]

    def embed_single(self, text: str) -> list[float]:
        return [0.1] * 64

    def dimension(self) -> int:
        return 64


def _make_memory(
    memory_id: str = "mem-1",
    content: str = "hello",
    version: int = 1,
) -> MemoryObject:
    now = datetime.now(timezone.utc)
    return make_memory(
        memory_id=memory_id,
        user_id="user-1",
        content=content,
        embedding=[0.1] * 64,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        tags=[],
        memory_type=MemoryType.EPISODIC,
        session_id=None,
        version=version,
    )


def _make_memory_with_adapter(
    tmp_path,
    db_name: str = "memories.db",
) -> tuple[Memory, str]:
    db_path = str(tmp_path / db_name)
    store = SQLiteStorageAdapter(db_path=db_path)
    embed = _MockEmbedding()
    memory = Memory(embed=embed, store=store)
    return memory, db_path


# ---------------------------------------------------------------------------
# R1 — concurrent pre-snapshot clobber
# ---------------------------------------------------------------------------

class TestR1ConcurrentPreSnapshot:
    def test_r1_concurrent_updates_preserve_both_pre_snapshots(self, tmp_path):
        """Two threads updating the same memory must both leave their
        pre-update snapshot in memory_versions (no INSERT OR REPLACE clobber)."""
        db_path = str(tmp_path / "r1.db")
        vs = MemoryVersionStore(db_path=db_path)

        # Seed the version store with the original memory at v=1
        seed = _make_memory(memory_id="mem-r1", content="seed", version=1)
        vs.record_version(seed, changed_by="create")

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def updater(content: str) -> None:
            try:
                barrier.wait()
                before = _make_memory(memory_id="mem-r1", content="seed", version=1)
                after = _make_memory(memory_id="mem-r1", content=content, version=1)
                vs.record_before_update(before, after, changed_by="update")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=updater, args=("A",))
        t2 = threading.Thread(target=updater, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent record_before_update raised: {errors}"

        snaps = vs.list_versions("mem-r1")
        # With the fix (INSERT OR IGNORE for the pre-snapshot) both threads
        # serialise through BEGIN IMMEDIATE: each gets a distinct post-version
        # and the pre-snapshot is preserved. Both threads build the *same*
        # pre-snapshot from the seed state, so the result is 1 pre + 2 post.
        versions = sorted((s.version, s.changed_by) for s in snaps)
        assert versions == [(1, "create"), (2, "update"), (3, "update")], (
            f"Expected seed v=1 + two distinct post snapshots, got {versions}"
        )


# ---------------------------------------------------------------------------
# R2 — store row version drifts from version-store version
# ---------------------------------------------------------------------------

class TestR2VersionDrift:
    def test_r2_update_does_not_drift_version_number(self, tmp_path):
        """memory_versions.version must match the store row's version after update()."""
        memory, db_path = _make_memory_with_adapter(tmp_path, "r2.db")
        memory.configure_versioning(
            db_path=db_path,
            max_versions_per_memory=50,
            auto_prune_versions=False,
        )

        mid = memory.remember(user_id="alice", content="original")
        memory.update(mid, content="updated once")

        store_row = memory._store.get(mid)
        assert store_row is not None

        # Access the version store directly
        vs = memory._version_store
        assert vs is not None
        versions = vs.list_versions(mid)
        max_vs_version = max((v.version for v in versions), default=0)

        # The store row version should equal the highest version in the
        # version store.  With the bug, _io.py does memory.version += 1
        # after record_before_update already bumped it, so the store row
        # ends up one ahead.
        assert store_row.version == max_vs_version, (
            f"Store row version {store_row.version} != max version-store version {max_vs_version}"
        )


# ---------------------------------------------------------------------------
# R4 — concurrent rollbacks collide on new_version
# ---------------------------------------------------------------------------

class TestR4ConcurrentRollback:
    def test_r4_concurrent_rollbacks_do_not_crash(self, tmp_path):
        """Two threads calling rollback() concurrently must both record a
        distinct rollback snapshot; the second thread must not overwrite
        the first via INSERT OR REPLACE."""
        db_path = str(tmp_path / "r4.db")
        vs = MemoryVersionStore(db_path=db_path)
        store = SQLiteStorageAdapter(db_path=db_path)

        # Seed v=1, v=2, v=3 in the version store.
        for i in range(1, 4):
            mem = _make_memory(memory_id="mem-r4", content=f"v{i}", version=i)
            store.store(mem)
            vs.record_version(mem, changed_by="update")

        barrier = threading.Barrier(2)
        results: dict[str, Any] = {}

        def rollbacker(name: str) -> None:
            try:
                barrier.wait()
                result = vs.rollback("mem-r4", target_version=1, store=store)
                results[name] = result
            except Exception as e:
                results[name] = e

        t1 = threading.Thread(target=rollbacker, args=("A",))
        t2 = threading.Thread(target=rollbacker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        for name, value in results.items():
            assert not isinstance(
                value, sqlite3.IntegrityError
            ), f"Thread {name} raised sqlite3.IntegrityError: {value}"

        history = vs.list_versions("mem-r4")
        rollback_snaps = [s for s in history if s.changed_by == "rollback"]
        # With the bug both threads compute new_version=4 and the second
        # INSERT OR REPLACE overwrites the first → only 1 rollback snapshot.
        # The fix must produce two distinct rollback snapshots.
        assert len(rollback_snaps) == 2, (
            f"Expected 2 rollback snapshots, got {len(rollback_snaps)}: "
            f"{[(s.version, s.changed_by) for s in history]}"
        )

        assert vs.verify_sequential_versions("mem-r4") is True


# ---------------------------------------------------------------------------
# R3 — auto-prune is inside the version-store transaction
# ---------------------------------------------------------------------------

class TestR3AutoPruneInsideTransaction:
    def test_r3_record_and_update_prunes_inside_same_transaction(self, tmp_path):
        """``record_and_update`` must do the prune step under ``BEGIN IMMEDIATE``
        so a concurrent reader cannot observe versions that are mid-delete."""
        from kemi.memory.versions import MemoryVersionStore

        db_path = str(tmp_path / "r3.db")
        vs = MemoryVersionStore(db_path=db_path)

        _make_memory(memory_id="mem-r3", content="seed", version=1)
        before = _make_memory(memory_id="mem-r3", content="seed", version=1)
        after = _make_memory(memory_id="mem-r3", content="updated", version=1)
        # Drive enough updates to exceed keep_count and trigger pruning.
        keep = 3
        for _i in range(keep + 2):
            vs.record_and_update(
                before, after, store=_NoopStore(), changed_by="update", keep_count=keep,
            )
            after.version = 1

        history = vs.list_versions("mem-r3")
        # Pruning kept the most-recent ``keep`` rows.
        assert len(history) == keep, (
            f"Expected {keep} versions after prune, got {len(history)}: "
            f"{[(s.version, s.changed_by) for s in history]}"
        )
        versions_kept = sorted(s.version for s in history)
        expected = list(range(versions_kept[0], versions_kept[0] + keep))
        assert versions_kept == expected, (
            f"Surviving versions {versions_kept} are not contiguous; expected {expected}"
        )

    def test_r3_concurrent_writers_keep_versions_contiguous(self, tmp_path):
        """Two threads doing ``record_and_update`` with ``keep_count`` set
        must never leave a gap in the surviving version sequence."""
        from kemi.memory.versions import MemoryVersionStore

        db_path = str(tmp_path / "r3b.db")
        vs = MemoryVersionStore(db_path=db_path)

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def writer(label: str) -> None:
            try:
                for i in range(5):
                    barrier.wait(timeout=2)
                    before = _make_memory(
                        memory_id="mem-r3b", content=f"{label}-pre", version=1,
                    )
                    after = _make_memory(
                        memory_id="mem-r3b", content=f"{label}-{i}", version=1,
                    )
                    vs.record_and_update(
                        before, after, store=_NoopStore(),
                        changed_by="update", keep_count=3,
                    )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=("A",))
        t2 = threading.Thread(target=writer, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent record_and_update raised: {errors}"
        history = vs.list_versions("mem-r3b")
        versions_kept = sorted(s.version for s in history)
        # The surviving versions are the most-recent ``keep`` contiguous rows.
        assert len(versions_kept) == 3
        assert versions_kept == list(range(versions_kept[0], versions_kept[0] + 3)), (
            f"Surviving versions {versions_kept} are not contiguous"
        )


class _NoopStore:
    """A storage adapter stub that does nothing on ``update``."""

    def update(self, memory: Any) -> None:  # pragma: no cover - stub
        return None


# ---------------------------------------------------------------------------
# R6 — verify_sequential_versions returns a stable answer under load
# ---------------------------------------------------------------------------

class TestR6ReadStability:
    def test_r6_verify_sequential_versions_is_stable_under_concurrent_writes(
        self, tmp_path,
    ):
        """While a writer is recording versions, a concurrent reader that
        calls ``verify_sequential_versions`` must always see a contiguous
        sequence (the reader sees a transactional snapshot)."""
        from kemi.memory.versions import MemoryVersionStore

        db_path = str(tmp_path / "r6.db")
        vs = MemoryVersionStore(db_path=db_path)

        seed = _make_memory(memory_id="mem-r6", content="seed", version=1)
        vs.record_version(seed, changed_by="create")

        stop = threading.Event()
        seen_not_contiguous: list[bool] = []
        lock = threading.Lock()

        def writer() -> None:
            i = 2
            while not stop.is_set():
                _make_memory(memory_id="mem-r6", content="seed", version=1)
                after = _make_memory(
                    memory_id="mem-r6", content=f"v{i}", version=1,
                )
                vs.record_version(after, changed_by="update")
                i += 1

        def reader() -> None:
            for _ in range(50):
                contiguous = vs.verify_sequential_versions("mem-r6")
                with lock:
                    seen_not_contiguous.append(not contiguous)
                if stop.is_set():
                    return

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_read.join()
        stop.set()
        t_write.join()

        assert not any(seen_not_contiguous), (
            "verify_sequential_versions returned False under concurrent writes"
        )


# ---------------------------------------------------------------------------
# R7 — list_versions returns a consistent snapshot
# ---------------------------------------------------------------------------

class TestR7ListSnapshot:
    def test_r7_list_versions_is_self_consistent(self, tmp_path):
        """A ``list_versions`` call must return a snapshot of the version
        table that is internally consistent: no two snapshots with the
        same version, and the row count equals the count from a fresh
        second connection taken at the same instant."""
        from kemi.memory.versions import MemoryVersionStore

        db_path = str(tmp_path / "r7.db")
        vs = MemoryVersionStore(db_path=db_path)

        seed = _make_memory(memory_id="mem-r7", content="seed", version=1)
        vs.record_version(seed, changed_by="create")

        stop = threading.Event()
        list_dups: list[int] = []
        lock = threading.Lock()

        def writer() -> None:
            i = 2
            while not stop.is_set():
                mem = _make_memory(
                    memory_id="mem-r7", content=f"v{i}", version=1,
                )
                vs.record_version(mem, changed_by="update")
                i += 1

        def reader() -> None:
            for _ in range(30):
                snaps = vs.list_versions("mem-r7")
                versions = [s.version for s in snaps]
                dups = len(versions) - len(set(versions))
                with lock:
                    list_dups.append(dups)
                if stop.is_set():
                    return

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_read.join()
        stop.set()
        t_write.join()

        assert max(list_dups) == 0, (
            f"list_versions returned duplicate versions under load: {list_dups}"
        )
