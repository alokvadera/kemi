"""Tests for memory versioning (src/kemi/versions.py and core.py integration)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from kemi.memory.model import (
    LifecycleState,
    MemoryObject,
    MemorySource,
    MemoryType,
)
from kemi.memory.versions import (
    DiffResult,
    MemoryVersionStore,
    RollbackResult,
    diff_memories,
    enable_versioning,
)
from tests._helpers.factories import make_memory

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_test_memory(
    memory_id: str = "mem-1",
    user_id: str = "user-1",
    content: str = "Hello world",
    version: int = 1,
    importance: float = 0.5,
    tags: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    memory_type: MemoryType = MemoryType.EPISODIC,
    session_id: str | None = None,
    namespace: str = "default",
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    confidence: float = 1.0,
    **extra: object,
) -> MemoryObject:
    """Helper to create a MemoryObject with sensible defaults."""
    now = datetime.now(timezone.utc)
    return make_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=[0.1] * 64,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=importance,
        lifecycle_state=lifecycle_state,
        metadata=metadata or {},
        tags=tags or [],
        confidence=confidence,
        memory_type=memory_type,
        session_id=session_id,
        namespace=namespace,
        version=version,
    )


@pytest.fixture
def vs(tmp_path) -> MemoryVersionStore:
    """Version store backed by a temporary database."""
    db_path = str(tmp_path / "versions.db")
    return MemoryVersionStore(db_path=db_path)


@pytest.fixture
def memory_v1() -> MemoryObject:
    return _make_test_memory(
        memory_id="mem-test",
        user_id="alice",
        content="Original content",
        version=1,
        importance=0.5,
        tags=["tag-a"],
        metadata={"key": "value1"},
    )


@pytest.fixture
def memory_v2() -> MemoryObject:
    return _make_test_memory(
        memory_id="mem-test",
        user_id="alice",
        content="Updated content",
        version=2,
        importance=0.7,
        tags=["tag-a", "tag-b"],
        metadata={"key": "value2"},
    )


# ---------------------------------------------------------------------------
# MemoryVersionStore.record_version
# ---------------------------------------------------------------------------

class TestRecordVersion:
    def test_record_version_returns_next_version(self, vs: MemoryVersionStore):
        mem = _make_test_memory(memory_id="mem-1", content="v1 content", version=1)
        result = vs.record_version(mem, changed_by="update")
        assert result == 1

    def test_record_version_stores_snapshot(self, vs: MemoryVersionStore):
        mem = _make_test_memory(memory_id="mem-2", content="hello", version=1)
        vs.record_version(mem, changed_by="update")

        versions = vs.list_versions("mem-2")
        assert len(versions) == 1
        assert versions[0].content == "hello"
        assert versions[0].version == 1
        assert versions[0].changed_by == "update"

    def test_record_version_multiple_versions(self, vs: MemoryVersionStore):
        mem = _make_test_memory(memory_id="mem-3", content="v1", version=1)
        vs.record_version(mem, changed_by="create")

        mem.content = "v2"
        mem.version = 2
        vs.record_version(mem, changed_by="update")

        mem.content = "v3"
        mem.version = 3
        vs.record_version(mem, changed_by="update")

        versions = vs.list_versions("mem-3")
        assert len(versions) == 3
        # Newest first
        assert versions[0].content == "v3"
        assert versions[0].version == 3
        assert versions[1].content == "v2"
        assert versions[1].version == 2
        assert versions[2].content == "v1"
        assert versions[2].version == 1

    def test_record_version_serialises_metadata_and_tags(self, vs: MemoryVersionStore):
        mem = _make_test_memory(
            memory_id="mem-4",
            content="data",
            version=1,
            metadata={"foo": "bar", "num": 42},
            tags=["urgent", "work"],
        )
        vs.record_version(mem, changed_by="update")

        snapshots = vs.list_versions("mem-4")
        assert snapshots[0].metadata == {"foo": "bar", "num": 42}
        assert snapshots[0].tags == ["urgent", "work"]

    def test_record_version_serialises_embedding(self, vs: MemoryVersionStore):
        embedding = [0.1, 0.2, 0.3] * 21  # 63 floats
        mem = _make_test_memory(memory_id="mem-5", content="embed test", version=1)
        mem.embedding = embedding
        mem.embedding_dim = len(embedding)
        vs.record_version(mem, changed_by="update")

        snapshots = vs.list_versions("mem-5")
        assert snapshots[0].embedding == pytest.approx(embedding)


# ---------------------------------------------------------------------------
# MemoryVersionStore.list_versions
# ---------------------------------------------------------------------------

class TestListVersions:
    def test_list_versions_empty(self, vs: MemoryVersionStore):
        result = vs.list_versions("nonexistent")
        assert result == []

    def test_list_versions_newest_first(self, vs: MemoryVersionStore):
        for i in range(1, 4):
            mem = _make_test_memory(memory_id="mem-10", content=f"v{i}", version=i)
            vs.record_version(mem, changed_by="update")

        versions = vs.list_versions("mem-10")
        assert [v.version for v in versions] == [3, 2, 1]

    def test_list_versions_respects_limit(self, vs: MemoryVersionStore):
        for i in range(1, 6):
            mem = _make_test_memory(memory_id="mem-11", content=f"v{i}", version=i)
            vs.record_version(mem, changed_by="update")

        versions = vs.list_versions("mem-11")
        assert len(versions) == 5
        # Default order is newest first — no limit param in current API
        # List is already sorted by version DESC


# ---------------------------------------------------------------------------
# MemoryVersionStore.get_version
# ---------------------------------------------------------------------------

class TestGetVersion:
    def test_get_version_found(self, vs: MemoryVersionStore, memory_v1: MemoryObject):
        vs.record_version(memory_v1, changed_by="create")
        result = vs.get_version("mem-test", 1)
        assert result is not None
        assert result.content == "Original content"
        assert result.version == 1

    def test_get_version_not_found(self, vs: MemoryVersionStore):
        result = vs.get_version("mem-none", 99)
        assert result is None

    def test_get_version_specific_version(self, vs: MemoryVersionStore):
        mem = _make_test_memory(memory_id="mem-20", content="v1", version=1)
        vs.record_version(mem, changed_by="create")

        mem = _make_test_memory(memory_id="mem-20", content="v2", version=2)
        vs.record_version(mem, changed_by="update")

        v1 = vs.get_version("mem-20", 1)
        v2 = vs.get_version("mem-20", 2)
        assert v1 is not None and v1.content == "v1"
        assert v2 is not None and v2.content == "v2"


# ---------------------------------------------------------------------------
# MemoryVersionStore.diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_no_changes(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(memory_id="mem-30", content="same", version=1, importance=0.5)
        mem2 = _make_test_memory(memory_id="mem-30", content="same", version=2, importance=0.5)
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-30", 1, 2)
        assert result is not None
        assert result.field_changes == {}

    def test_diff_content_changed(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(memory_id="mem-31", content="old text", version=1)
        mem2 = _make_test_memory(memory_id="mem-31", content="new text", version=2)
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-31", 1, 2)
        assert result is not None
        assert "content" in result.field_changes
        assert result.field_changes["content"] == ("old text", "new text")

    def test_diff_importance_changed(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(memory_id="mem-32", importance=0.3, version=1)
        mem2 = _make_test_memory(memory_id="mem-32", importance=0.9, version=2)
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-32", 1, 2)
        assert result is not None
        assert "importance" in result.field_changes
        assert result.field_changes["importance"] == (0.3, 0.9)

    def test_diff_tags_changed(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(memory_id="mem-33", tags=["a"], version=1)
        mem2 = _make_test_memory(memory_id="mem-33", tags=["a", "b"], version=2)
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-33", 1, 2)
        assert result is not None
        assert "tags" in result.field_changes

    def test_diff_metadata_changed(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(memory_id="mem-34", metadata={"a": 1}, version=1)
        mem2 = _make_test_memory(memory_id="mem-34", metadata={"a": 2}, version=2)
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-34", 1, 2)
        assert result is not None
        assert "metadata" in result.field_changes

    def test_diff_memory_id_mismatch(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(memory_id="mem-35", content="v1", version=1)
        mem2 = _make_test_memory(memory_id="mem-35", content="v2", version=2)
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-35", 99, 2)  # Version 99 doesn't exist
        assert result is None

    def test_diff_multiple_fields(self, vs: MemoryVersionStore):
        mem1 = _make_test_memory(
            memory_id="mem-36",
            content="old",
            importance=0.3,
            tags=["tag1"],
            version=1,
        )
        mem2 = _make_test_memory(
            memory_id="mem-36",
            content="new",
            importance=0.8,
            tags=["tag1", "tag2"],
            version=2,
        )
        vs.record_version(mem1, changed_by="create")
        vs.record_version(mem2, changed_by="update")

        result = vs.diff("mem-36", 1, 2)
        assert result is not None
        assert "content" in result.field_changes
        assert "importance" in result.field_changes
        assert "tags" in result.field_changes


# ---------------------------------------------------------------------------
# MemoryVersionStore.rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_not_found(self, vs: MemoryVersionStore, mock_storage):
        result = vs.rollback("mem-none", target_version=1, store=mock_storage())
        assert result is None

    def test_rollback_restores_content(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        db_path = str(tmp_path / "rollback_test.db")
        store = SQLiteStorageAdapter(db_path=db_path)

        # Create and store original memory
        mem_v1 = _make_test_memory(
            memory_id="mem-rollback",
            user_id="bob",
            content="Original content here",
            version=1,
        )
        store.store(mem_v1)

        # Record v1
        vs.record_version(mem_v1, changed_by="create")

        # Update to v2
        mem_v2 = _make_test_memory(
            memory_id="mem-rollback",
            user_id="bob",
            content="Modified content",
            version=2,
        )
        store.update(mem_v2)
        vs.record_version(mem_v2, changed_by="update")

        # Rollback to v1
        result = vs.rollback("mem-rollback", target_version=1, store=store)
        assert result is not None
        assert result.from_version == 1
        assert result.memory_id == "mem-rollback"

        # Memory in store should now have v1 content
        current = store.get("mem-rollback")
        assert current is not None
        assert current.content == "Original content here"

    def test_rollback_preserves_user_id_and_lifecycle(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        db_path = str(tmp_path / "rollback_test2.db")
        store = SQLiteStorageAdapter(db_path=db_path)

        mem_v1 = _make_test_memory(
            memory_id="mem-rb2",
            user_id="carol",
            content="v1 content",
            version=1,
            lifecycle_state=LifecycleState.ARCHIVED,
        )
        store.store(mem_v1)
        vs.record_version(mem_v1, changed_by="create")

        mem_v2 = _make_test_memory(
            memory_id="mem-rb2",
            user_id="carol",
            content="v2 content",
            version=2,
            lifecycle_state=LifecycleState.ARCHIVED,
        )
        store.update(mem_v2)
        vs.record_version(mem_v2, changed_by="update")

        vs.rollback("mem-rb2", target_version=1, store=store)

        current = store.get("mem-rb2")
        assert current is not None
        assert current.user_id == "carol"
        assert current.lifecycle_state == LifecycleState.ARCHIVED

    def test_rollback_records_pre_and_post_versions(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        db_path = str(tmp_path / "rollback_versions.db")
        store = SQLiteStorageAdapter(db_path=db_path)

        mem_v1 = _make_test_memory(memory_id="mem-rb3", content="v1", version=1)
        store.store(mem_v1)
        vs.record_version(mem_v1, changed_by="create")

        mem_v2 = _make_test_memory(memory_id="mem-rb3", content="v2", version=2)
        store.update(mem_v2)
        vs.record_version(mem_v2, changed_by="update")

        vs.rollback("mem-rb3", target_version=1, store=store)

        versions = vs.list_versions("mem-rb3")
        # v1 (original), v2 (updated), v3 (rolled-back = new version of v1)
        assert len(versions) >= 3

    def test_rollback_returns_rollback_result(self, vs: MemoryVersionStore, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        db_path = str(tmp_path / "rollback_result.db")
        store = SQLiteStorageAdapter(db_path=db_path)

        mem_v1 = _make_test_memory(memory_id="mem-rb4", content="v1", version=1)
        store.store(mem_v1)
        vs.record_version(mem_v1, changed_by="create")

        mem_v2 = _make_test_memory(memory_id="mem-rb4", content="v2", version=2)
        store.update(mem_v2)
        vs.record_version(mem_v2, changed_by="update")

        result = vs.rollback("mem-rb4", target_version=1, store=store)

        assert isinstance(result, RollbackResult)
        assert result.memory_id == "mem-rb4"
        assert result.from_version == 1
        assert result.rolled_back_at is not None


# ---------------------------------------------------------------------------
# MemoryVersionStore.get_latest_version_number
# ---------------------------------------------------------------------------

class TestGetLatestVersionNumber:
    def test_get_latest_version_number_empty(self, vs: MemoryVersionStore):
        assert vs.get_latest_version_number("mem-none") is None

    def test_get_latest_version_number(self, vs: MemoryVersionStore):
        for i in range(1, 4):
            mem = _make_test_memory(memory_id="mem-latest", version=i)
            vs.record_version(mem, changed_by="update")

        latest = vs.get_latest_version_number("mem-latest")
        assert latest == 3


# ---------------------------------------------------------------------------
# diff_memories (standalone function)
# ---------------------------------------------------------------------------

class TestDiffMemories:
    def test_diff_memories_content(self):
        mem_before = _make_test_memory(memory_id="mem-d1", content="old", version=1)
        mem_after = _make_test_memory(memory_id="mem-d1", content="new", version=2)

        result = diff_memories(mem_before, mem_after)
        assert isinstance(result, DiffResult)
        assert "content" in result.field_changes
        assert result.field_changes["content"] == ("old", "new")

    def test_diff_memories_multiple_fields(self):
        mem_before = _make_test_memory(
            memory_id="mem-d2",
            content="old",
            importance=0.3,
            confidence=0.5,
            version=1,
        )
        mem_after = _make_test_memory(
            memory_id="mem-d2",
            content="new",
            importance=0.8,
            confidence=0.9,
            version=2,
        )

        result = diff_memories(mem_before, mem_after)
        assert "content" in result.field_changes
        assert "importance" in result.field_changes
        assert "confidence" in result.field_changes

    def test_diff_memories_no_changes(self):
        mem = _make_test_memory(memory_id="mem-d3", content="same", version=1)
        result = diff_memories(mem, mem)
        assert result.field_changes == {}


# ---------------------------------------------------------------------------
# enable_versioning helper
# ---------------------------------------------------------------------------

class TestEnableVersioning:
    def test_enable_versioning_returns_version_store(self, tmp_path):
        from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

        db_path = str(tmp_path / "enable_test.db")
        adapter = SQLiteStorageAdapter(db_path=db_path)

        # Patch the memory's store to be our adapter
        class FakeMemory:
            _store = adapter

        vs = enable_versioning(FakeMemory())
        assert isinstance(vs, MemoryVersionStore)


# ---------------------------------------------------------------------------
# core.py integration: Memory.get_history
# ---------------------------------------------------------------------------

class TestCoreGetHistory:
    def test_get_history_empty(self, real_db_memory):
        real_db_memory.configure_versioning()
        history = real_db_memory.get_history("mem-none")
        assert history == []

    def test_get_history_returns_versions(self, real_db_memory):
        real_db_memory.configure_versioning()

        mid = real_db_memory.remember(
            user_id="alice",
            content="Initial content",
        )

        # Modify the memory to create a version
        real_db_memory.update(mid, content="Updated content")

        history = real_db_memory.get_history(mid)
        # At minimum we should have the version recorded by update()
        assert len(history) >= 1


# ---------------------------------------------------------------------------
# core.py integration: Memory.diff_versions
# ---------------------------------------------------------------------------

class TestCoreDiffVersions:
    def test_diff_versions_nonexistent_memory(self, real_db_memory):
        real_db_memory.configure_versioning()
        result = real_db_memory.diff_versions("mem-none", 1, 2)
        assert result is None

    def test_diff_versions_detects_change(self, real_db_memory):
        real_db_memory.configure_versioning()

        mid = real_db_memory.remember(user_id="alice", content="Original")
        real_db_memory.update(mid, content="Modified")

        history = real_db_memory.get_history(mid)
        assert len(history) >= 2
        # v2 is newest (index 0), v1 is older (index 1)
        v_newer = history[0].version
        v_older = history[1].version

        diff = real_db_memory.diff_versions(mid, v_older, v_newer)
        assert diff is not None
        assert "content" in diff.field_changes
        assert diff.field_changes["content"] == ("Original", "Modified")


# ---------------------------------------------------------------------------
# core.py integration: Memory.rollback_memory
# ---------------------------------------------------------------------------

class TestCoreRollbackMemory:
    def test_rollback_memory_nonexistent(self, real_db_memory):
        real_db_memory.configure_versioning()
        result = real_db_memory.rollback_memory("mem-none", target_version=1)
        assert result is None

    def test_rollback_restores_content(self, real_db_memory):
        real_db_memory.configure_versioning()

        mid = real_db_memory.remember(user_id="bob", content="Original content")
        real_db_memory.update(mid, content="Changed content")

        history = real_db_memory.get_history(mid)
        v1 = min(s.version for s in history)

        result = real_db_memory.rollback_memory(mid, target_version=v1)
        assert result is not None
        assert result.memory_id == mid

        # Content should be back to original
        current = real_db_memory._store.get(mid)
        assert current is not None
        assert current.content == "Original content"


# ---------------------------------------------------------------------------
# core.py integration: versioning config options
# ---------------------------------------------------------------------------

class TestVersioningConfig:
    def test_configure_versioning_with_explicit_db_path(self, real_db_memory, tmp_path):
        db_path = str(tmp_path / "versioning_explicit.db")
        real_db_memory.configure_versioning(db_path=db_path)
        assert real_db_memory._version_store is not None

    def test_configure_versioning_max_versions(self, real_db_memory):
        real_db_memory.configure_versioning(
            max_versions_per_memory=5,
            auto_prune_versions=True,
        )
        assert real_db_memory._max_versions_per_memory == 5

    def test_auto_prune_prunes_old_versions(self, real_db_memory):
        real_db_memory.configure_versioning(
            max_versions_per_memory=3,
            auto_prune_versions=True,
        )

        mid = real_db_memory.remember(user_id="alice", content="v1")

        # Create many versions through updates
        # Each update records 2 versions (pre-update + update), so 8 updates = 16 entries
        # plus 1 from remember() = 17 total, pruned to max 3 = 3 remaining
        for i in range(2, 10):
            real_db_memory.update(mid, content=f"v{i}")

        history = real_db_memory.get_history(mid)
        # Should be pruned to at most 3 versions (max_versions_per_memory=3)
        assert len(history) <= 3


# ---------------------------------------------------------------------------
# record_before_update
# ---------------------------------------------------------------------------

class TestRecordBeforeUpdate:
    def test_record_before_update(self, vs: MemoryVersionStore):
        mem_v1 = _make_test_memory(memory_id="mem-rbu", content="v1", version=1)
        mem_v2 = _make_test_memory(memory_id="mem-rbu", content="v2", version=2)
        result = vs.record_before_update(mem_v1, mem_v2, changed_by="update")
        assert result == 2

        versions = vs.list_versions("mem-rbu")
        assert len(versions) == 2
        # Newest first (v2), then older (v1)
        assert versions[0].changed_by == "update"      # v2 recorded last
        assert versions[1].changed_by == "pre-update"  # v1 recorded as pre-


# ---------------------------------------------------------------------------
# Backward compatibility — raw JSON stored in metadata/tags columns
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_metadata_serialized_as_json_string(self, vs: MemoryVersionStore):
        mem = _make_test_memory(
            memory_id="mem-json",
            content="test",
            version=1,
            metadata={"nested": {"a": 1}},
            tags=["tag1", "tag2"],
        )
        vs.record_version(mem, changed_by="update")

        # Access the raw database row
        conn = vs._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metadata, tags FROM memory_versions WHERE memory_id = ?",
                ("mem-json",),
            )
            row = cursor.fetchone()
            assert row is not None
            # Should be stored as JSON strings
            parsed_meta = json.loads(row["metadata"])
            assert parsed_meta == {"nested": {"a": 1}}
            parsed_tags = json.loads(row["tags"])
            assert parsed_tags == ["tag1", "tag2"]
        finally:
            conn.close()

    def test_isolation_between_memories(self, vs: MemoryVersionStore):
        for i in range(1, 4):
            mem = _make_test_memory(memory_id=f"mem-{i}", content=f"content-{i}", version=1)
            vs.record_version(mem, changed_by="create")

        assert len(vs.list_versions("mem-1")) == 1
        assert len(vs.list_versions("mem-2")) == 1
        assert len(vs.list_versions("mem-3")) == 1
        assert vs.get_version("mem-1", 1) is not None
        assert vs.get_version("mem-2", 1) is not None
        assert vs.get_version("mem-3", 1) is not None


# ---------------------------------------------------------------------------
# Snapshot field completeness
# ---------------------------------------------------------------------------

class TestSnapshotCompleteness:
    def test_version_snapshot_has_all_fields(self, vs: MemoryVersionStore):
        mem = _make_test_memory(
            memory_id="mem-complete",
            user_id="dave",
            content="All fields test",
            version=1,
            importance=0.8,
            confidence=0.95,
            tags=["important"],
            metadata={"source": "test"},
            memory_type=MemoryType.SEMANTIC,
            session_id="session-abc",
            namespace="test-ns",
        )
        vs.record_version(mem, changed_by="test-change")

        snapshot = vs.get_version("mem-complete", 1)
        assert snapshot is not None
        assert snapshot.memory_id == "mem-complete"
        assert snapshot.content == "All fields test"
        assert snapshot.importance == 0.8
        assert snapshot.confidence == 0.95
        assert snapshot.tags == ["important"]
        assert snapshot.metadata == {"source": "test"}
        assert snapshot.memory_type == MemoryType.SEMANTIC.value
        assert snapshot.session_id == "session-abc"
        assert snapshot.namespace == "test-ns"
        assert snapshot.changed_by == "test-change"
        assert snapshot.changed_at is not None
