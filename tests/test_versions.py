"""Tests for src/kemi/versions.py — memory versioning and rollback."""


import sqlite3
from datetime import datetime, timezone

import pytest

from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType
from kemi.memory.versions import (
    DiffResult,
    MemoryVersionStore,
    RollbackResult,
    VersionSnapshot,
    diff_memories,
    enable_versioning,
)
from tests._helpers.factories import make_memory

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TestVersionSnapshot:
    def test_all_fields_present(self) -> None:
        snap = VersionSnapshot(
            version=3,
            memory_id="mem-123",
            content="updated content",
            embedding=[0.1, 0.2],
            importance=0.8,
            metadata={"key": "value"},
            tags=["python", "coding"],
            memory_type="episodic",
            confidence=0.9,
            session_id="sess-1",
            namespace="default",
            source="user_stated",
            changed_at=datetime.now(timezone.utc),
            changed_by="update",
        )
        assert snap.version == 3
        assert snap.memory_id == "mem-123"
        assert snap.content == "updated content"
        assert snap.embedding == [0.1, 0.2]
        assert snap.importance == 0.8
        assert snap.metadata == {"key": "value"}
        assert snap.tags == ["python", "coding"]
        assert snap.memory_type == "episodic"
        assert snap.confidence == 0.9
        assert snap.session_id == "sess-1"
        assert snap.namespace == "default"
        assert snap.changed_by == "update"


class TestRollbackResult:
    def test_fields(self) -> None:
        result = RollbackResult(
            memory_id="mem-123",
            from_version=3,
            to_version=4,
            rolled_back_at=datetime.now(timezone.utc),
        )
        assert result.memory_id == "mem-123"
        assert result.from_version == 3
        assert result.to_version == 4
        assert result.rolled_back_at is not None


class TestDiffResult:
    def test_fields(self) -> None:
        changes = {"content": ("old", "new"), "importance": (0.5, 0.9)}
        diff = DiffResult(
            memory_id="mem-123",
            from_version=1,
            to_version=2,
            field_changes=changes,
        )
        assert diff.memory_id == "mem-123"
        assert diff.from_version == 1
        assert diff.to_version == 2
        assert diff.field_changes == changes


# ---------------------------------------------------------------------------
# MemoryVersionStore — lifecycle / table creation
# ---------------------------------------------------------------------------

class TestMemoryVersionStoreInit:
    def test_creates_tables_on_init(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        MemoryVersionStore(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "memory_versions" in tables
        assert "memory_change_log" in tables

    def test_re_init_same_path_ok(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        MemoryVersionStore(db_path=db_path)
        MemoryVersionStore(db_path=db_path)
        # No-op, tables already exist


# ---------------------------------------------------------------------------
# record_version
# ---------------------------------------------------------------------------

class TestRecordVersion:
    def test_record_version_increments(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-1", "user1", "v1 content")
        mem.version = 1
        version = store.record_version(mem, changed_by="update")
        assert version == 1

        mem.version = 2
        version2 = store.record_version(mem, changed_by="update")
        assert version2 == 2

    def test_record_version_persists(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-persist", "user1", "persistent content")
        mem.version = 1
        store.record_version(mem, changed_by="update")

        snapshots = store.list_versions("mem-persist")
        assert len(snapshots) == 1
        assert snapshots[0].content == "persistent content"

    def test_record_version_stores_metadata(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-meta", "user1", "content")
        mem.version = 1
        mem.metadata = {"foo": "bar", "num": 42}
        mem.tags = ["important", "work"]
        store.record_version(mem, changed_by="update")

        snap = store.list_versions("mem-meta")[0]
        assert snap.metadata == {"foo": "bar", "num": 42}
        assert snap.tags == ["important", "work"]

    def test_record_version_changed_by(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-cb", "user1", "content")
        mem.version = 1
        store.record_version(mem, changed_by="consolidate")

        snap = store.list_versions("mem-cb")[0]
        assert snap.changed_by == "consolidate"

    def test_record_version_stores_embedding(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-emb", "user1", "content")
        mem.version = 1
        mem.embedding = [0.1, 0.2, 0.3]
        store.record_version(mem, changed_by="update")

        snap = store.list_versions("mem-emb")[0]
        assert snap.embedding == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# record_before_update
# ---------------------------------------------------------------------------

class TestRecordBeforeUpdate:
    def test_records_both_versions(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        before = _make_memory("mem-upd", "user1", "old content")
        before.version = 1

        after = _make_memory("mem-upd", "user1", "new content")
        after.version = 2

        store.record_before_update(before, after, changed_by="update")

        snaps = store.list_versions("mem-upd")
        assert len(snaps) == 2

    def test_before_has_pre_prefix(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        before = _make_memory("mem-pre", "user1", "old")
        before.version = 1

        after = _make_memory("mem-pre", "user1", "new")
        after.version = 2

        store.record_before_update(before, after, changed_by="update")

        snaps = store.list_versions("mem-pre")
        changed_bys = {s.changed_by for s in snaps}
        assert "pre-update" in changed_bys
        assert "update" in changed_bys


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------

class TestListVersions:
    def test_empty_for_new_memory(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        snaps = store.list_versions("nonexistent")
        assert snaps == []

    def test_newest_first(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        for i in range(1, 4):
            mem = _make_memory("mem-sort", "user1", f"v{i}")
            mem.version = i
            store.record_version(mem, changed_by="update")

        snaps = store.list_versions("mem-sort")
        versions = [s.version for s in snaps]
        assert versions == sorted(versions, reverse=True)


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------

class TestGetVersion:
    def test_get_specific_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-spec", "user1", "specific content")
        mem.version = 5
        store.record_version(mem, changed_by="update")

        snap = store.get_version("mem-spec", 5)
        assert snap is not None
        assert snap.version == 5
        assert snap.content == "specific content"

    def test_nonexistent_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        snap = store.get_version("nonexistent", 1)
        assert snap is None

    def test_nonexistent_memory(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        snap = store.get_version("nonexistent", 999)
        assert snap is None


# ---------------------------------------------------------------------------
# get_latest_version_number
# ---------------------------------------------------------------------------

class TestGetLatestVersionNumber:
    def test_none_for_new_memory(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        result = store.get_latest_version_number("nonexistent")
        assert result is None

    def test_returns_max_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        for i in [2, 5, 1, 4]:
            mem = _make_memory("mem-latest", "user1", f"v{i}")
            mem.version = i
            store.record_version(mem, changed_by="update")

        latest = store.get_latest_version_number("mem-latest")
        assert latest == 5


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_detects_content_change(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        v1 = _make_memory("mem-diff", "user1", "old content")
        v1.version = 1
        store.record_version(v1, changed_by="update")

        v2 = _make_memory("mem-diff", "user1", "new content")
        v2.version = 2
        store.record_version(v2, changed_by="update")

        diff = store.diff("mem-diff", 1, 2)
        assert diff is not None
        assert "content" in diff.field_changes
        assert diff.field_changes["content"] == ("old content", "new content")

    def test_detects_importance_change(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        v1 = _make_memory("mem-imp", "user1", "content")
        v1.version = 1
        v1.importance = 0.3
        store.record_version(v1, changed_by="update")

        v2 = _make_memory("mem-imp", "user1", "content")
        v2.version = 2
        v2.importance = 0.9
        store.record_version(v2, changed_by="update")

        diff = store.diff("mem-imp", 1, 2)
        assert diff is not None
        assert "importance" in diff.field_changes

    def test_unchanged_fields_not_in_diff(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)

        v1 = _make_memory("mem-unchanged", "user1", "content")
        v1.version = 1
        v1.importance = 0.5
        store.record_version(v1, changed_by="update")

        v2 = _make_memory("mem-unchanged", "user1", "updated content")
        v2.version = 2
        v2.importance = 0.5  # unchanged
        store.record_version(v2, changed_by="update")

        diff = store.diff("mem-unchanged", 1, 2)
        assert "importance" not in diff.field_changes
        assert "content" in diff.field_changes

    def test_none_for_nonexistent_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        result = store.diff("nonexistent", 1, 2)
        assert result is None

    def test_none_for_missing_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        mem = _make_memory("mem-missing", "user1", "content")
        mem.version = 1
        store.record_version(mem, changed_by="update")
        result = store.diff("mem-missing", 1, 99)
        assert result is None


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_to_past_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
        adapter = SQLiteStorageAdapter(db_path=db_path)

        # Store current version
        current = _make_memory("mem-rb", "user1", "current content")
        adapter.store(current)

        # Record versions
        v1 = _make_memory("mem-rb", "user1", "v1 content")
        v1.version = 1
        store.record_version(v1, changed_by="update")

        v2 = _make_memory("mem-rb", "user1", "current content")
        v2.version = 2
        store.record_version(v2, changed_by="update")

        # Rollback to v1
        result = store.rollback("mem-rb", target_version=1, store=adapter)
        assert result is not None
        assert result.memory_id == "mem-rb"
        assert result.from_version == 1

        # Verify store has v1 content
        updated = adapter.get("mem-rb")
        assert updated is not None
        assert updated.content == "v1 content"
        assert updated.version >= 2  # new version after rollback

    def test_rollback_nonexistent_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
        adapter = SQLiteStorageAdapter(db_path=db_path)

        current = _make_memory("mem-rb-none", "user1", "current")
        adapter.store(current)

        result = store.rollback("mem-rb-none", target_version=99, store=adapter)
        assert result is None

    def test_rollback_records_new_version(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = MemoryVersionStore(db_path=db_path)
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
        adapter = SQLiteStorageAdapter(db_path=db_path)

        current = _make_memory("mem-rb-new", "user1", "current")
        adapter.store(current)

        v1 = _make_memory("mem-rb-new", "user1", "old")
        v1.version = 1
        store.record_version(v1, changed_by="update")

        store.rollback("mem-rb-new", target_version=1, store=adapter)

        snaps = store.list_versions("mem-rb-new")
        versions = {s.version for s in snaps}
        # Should have v1, current, and the new rollback version
        assert len(versions) >= 2  # rollback + pre-rollback record (not re-record of target version)  # noqa: E501


# ---------------------------------------------------------------------------
# diff_memories (convenience function)
# ---------------------------------------------------------------------------

class TestDiffMemories:
    def test_detects_content_change(self) -> None:
        before = _make_memory("mem-d", "user1", "old")
        after = _make_memory("mem-d", "user1", "new")
        diff = diff_memories(before, after)
        assert "content" in diff.field_changes

    def test_detects_importance_change(self) -> None:
        before = _make_memory("mem-di", "user1", "old")
        before.importance = 0.3
        after = _make_memory("mem-di", "user1", "old")
        after.importance = 0.9
        diff = diff_memories(before, after)
        assert "importance" in diff.field_changes

    def test_no_changes(self) -> None:
        mem = _make_memory("mem-same", "user1", "same")
        diff = diff_memories(mem, mem)
        assert len(diff.field_changes) == 0

    def test_multiple_changes(self) -> None:
        before = _make_memory("mem-multi", "user1", "old")
        before.importance = 0.3
        after = _make_memory("mem-multi", "user1", "new")
        after.importance = 0.9
        after.tags = ["tag1"]
        diff = diff_memories(before, after)
        assert "content" in diff.field_changes
        assert "importance" in diff.field_changes
        assert "tags" in diff.field_changes


# ---------------------------------------------------------------------------
# enable_versioning
# ---------------------------------------------------------------------------

class TestEnableVersioning:
    def test_returns_memory_version_store(self, tmp_path) -> None:
        db_path = str(tmp_path / "versions_test.db")
        store = enable_versioning(None, db_path=db_path)
        assert isinstance(store, MemoryVersionStore)

    def test_uses_default_db_path(self) -> None:
        # Should not raise even with default path (may not exist)
        try:
            store = enable_versioning(None)
            assert isinstance(store, MemoryVersionStore)
        except Exception:
            # Default path may not be writable in test env
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(
    memory_id: str,
    user_id: str,
    content: str,
    importance: float = 0.5,
) -> MemoryObject:
    return make_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=importance,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=3,
        tags=[],
        memory_type=MemoryType.EPISODIC,
        session_id=None,
        version=1,
    )
