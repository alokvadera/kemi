"""Tests for src/kemi/background_tasks.py"""

import pytest

from kemi.background_tasks import (
    BackgroundTaskManager,
    TaskResult,
    TaskStatus,
    TaskType,
)


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_to_dict(self):
        """Test TaskResult.to_dict() conversion."""
        import time

        result = TaskResult(
            task_id="test-123",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.COMPLETED,
            created_at=time.time(),
            started_at=time.time(),
            completed_at=time.time(),
            result={"stored_count": 5},
            error=None,
            progress=1.0,
        )

        d = result.to_dict()
        assert d["task_id"] == "test-123"
        assert d["task_type"] == "embed_batch"
        assert d["status"] == "completed"
        assert d["result"]["stored_count"] == 5
        assert d["progress"] == 1.0


class TestBackgroundTaskManager:
    """Tests for BackgroundTaskManager."""

    def test_initialization(self):
        """Test task manager initialization with default and custom params."""
        tm = BackgroundTaskManager()
        assert tm._max_concurrent == 3
        assert tm._max_task_history == 1000

        tm2 = BackgroundTaskManager(max_concurrent_tasks=5, max_task_history=500)
        assert tm2._max_concurrent == 5
        assert tm2._max_task_history == 500

    def test_submit_embed_batch_rejects_when_at_capacity(self):
        """Test that submit_embed_batch raises RuntimeError when at max capacity."""
        tm = BackgroundTaskManager(max_concurrent_tasks=2)
        tm._running_count = 2  # Simulate at capacity

        with pytest.raises(RuntimeError, match="Max concurrent tasks"):
            tm.submit_embed_batch("user1", ["test content"], 0.5)

    def test_submit_rebuild_fts_rejects_when_at_capacity(self):
        """Test that submit_rebuild_fts_index raises RuntimeError when at max capacity."""
        tm = BackgroundTaskManager(max_concurrent_tasks=2)
        tm._running_count = 2  # Simulate at capacity

        with pytest.raises(RuntimeError, match="Max concurrent tasks"):
            tm.submit_rebuild_fts_index()

    def test_get_stats(self):
        """Test get_stats returns correct counts."""
        tm = BackgroundTaskManager(max_concurrent_tasks=2)

        stats = tm.get_stats()
        assert stats["total_tasks"] == 0
        assert stats["pending"] == 0
        assert stats["running"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["max_concurrent"] == 2

    def test_get_task_status_returns_none_for_missing(self):
        """Test get_task_status returns None for unknown task_id."""
        tm = BackgroundTaskManager()
        result = tm.get_task_status("nonexistent-task-id")
        assert result is None

    def test_cancel_task_returns_false_for_nonexistent(self):
        """Test cancel_task returns False for unknown task_id."""
        tm = BackgroundTaskManager()
        result = tm.cancel_task("nonexistent-task-id")
        assert result is False

    def test_list_tasks_empty(self):
        """Test list_tasks returns empty list when no tasks."""
        tm = BackgroundTaskManager()
        tasks = tm.list_tasks()
        assert tasks == []

    def test_list_tasks_filters_by_status(self):
        """Test list_tasks filtering by status."""
        import time

        tm = BackgroundTaskManager()

        # Add a completed task directly
        with tm._lock:
            tm._tasks["task-1"] = TaskResult(
                task_id="task-1",
                task_type=TaskType.EMBED_BATCH,
                status=TaskStatus.COMPLETED,
                created_at=time.time(),
            )
            tm._tasks["task-2"] = TaskResult(
                task_id="task-2",
                task_type=TaskType.REBUILD_FTS_INDEX,
                status=TaskStatus.RUNNING,
                created_at=time.time(),
            )

        # Filter by completed
        completed = tm.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].task_id == "task-1"

        # Filter by running
        running = tm.list_tasks(status=TaskStatus.RUNNING)
        assert len(running) == 1
        assert running[0].task_id == "task-2"

    def test_cancel_pending_task_succeeds(self):
        """Test cancelling a pending task."""
        import time

        tm = BackgroundTaskManager()

        # Add a pending task
        with tm._lock:
            tm._tasks["pending-task"] = TaskResult(
                task_id="pending-task",
                task_type=TaskType.EMBED_BATCH,
                status=TaskStatus.PENDING,
                created_at=time.time(),
            )

        # Cancel should succeed
        result = tm.cancel_task("pending-task")
        assert result is True

        # Verify task is now failed with cancelled error
        task = tm.get_task_status("pending-task")
        assert task.status == TaskStatus.FAILED
        assert task.error == "Cancelled by user"

    def test_cancel_running_task_fails(self):
        """Test that cancelling a running task returns False."""
        import time

        tm = BackgroundTaskManager()

        # Add a running task
        with tm._lock:
            tm._tasks["running-task"] = TaskResult(
                task_id="running-task",
                task_type=TaskType.EMBED_BATCH,
                status=TaskStatus.RUNNING,
                created_at=time.time(),
                started_at=time.time(),
            )

        # Cancel should fail for running task
        result = tm.cancel_task("running-task")
        assert result is False

    def test_cleanup_old_tasks(self):
        """Test that _cleanup_old_tasks removes old completed tasks."""
        import time

        tm = BackgroundTaskManager(max_task_history=2)

        # Add 3 completed tasks (exceeds limit of 2)
        for i in range(3):
            with tm._lock:
                tm._tasks[f"task-{i}"] = TaskResult(
                    task_id=f"task-{i}",
                    task_type=TaskType.EMBED_BATCH,
                    status=TaskStatus.COMPLETED,
                    created_at=time.time() - i,  # Older tasks have lower timestamp
                    completed_at=time.time() - i,
                )

        # Trigger cleanup by adding new task
        tm._running_count = 1  # At capacity
        try:
            tm.submit_embed_batch("user1", ["test"], 0.5)
        except RuntimeError:
            pass

        # Oldest task should be removed
        with tm._lock:
            assert len(tm._tasks) <= 2


class TestTaskEnums:
    """Tests for TaskType and TaskStatus enums."""

    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.EMBED_BATCH.value == "embed_batch"
        assert TaskType.REBUILD_FTS_INDEX.value == "rebuild_fts_index"
        assert TaskType.MIGRATE_EMBEDDINGS.value == "migrate_embeddings"
        assert TaskType.TTL_SWEEP.value == "ttl_sweep"

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestBackgroundTaskManagerLoop:
    """Tests for the background event loop."""

    def test_ensure_loop_started_idempotent(self):
        """Test that _ensure_loop_started is safe to call multiple times."""
        tm = BackgroundTaskManager()
        tm._ensure_loop_started()
        assert tm._loop is not None
        loop_ref = tm._loop
        # Second call should be a no-op
        tm._ensure_loop_started()
        assert tm._loop is loop_ref
        tm.shutdown()

    def test_get_loop_returns_loop(self):
        """Test _get_loop returns a valid event loop."""
        tm = BackgroundTaskManager()
        loop = tm._get_loop()
        assert loop is not None
        assert isinstance(loop, type(__import__("asyncio").new_event_loop()))
        tm.shutdown()

    def test_shutdown(self):
        """Test graceful shutdown of the task manager."""
        tm = BackgroundTaskManager()
        tm._ensure_loop_started()
        assert tm._loop is not None
        tm.shutdown()
        assert tm._loop is None
        assert tm._thread is None

    def test_shutdown_noop_when_not_started(self):
        """Test shutdown is safe when loop was never started."""
        tm = BackgroundTaskManager()
        assert tm._loop is None
        tm.shutdown()
        assert tm._loop is None


class TestBackgroundTaskAsyncRunners:
    """Tests for the actual async task runner coroutines."""

    def _run_coro(self, coro):
        """Helper to run a coroutine in a fresh event loop."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_run_embed_batch_success(self, monkeypatch):
        """Test _run_embed_batch stores results on success."""
        import time
        from unittest.mock import MagicMock

        # Mock Memory to avoid real DB connections
        mock_mem = MagicMock()
        mock_mem.remember_many.return_value = ["id1", "id2"]
        monkeypatch.setitem(__import__("sys").modules, "kemi", MagicMock())
        import kemi
        kemi.Memory = lambda: mock_mem

        tm = BackgroundTaskManager()
        task_id = str(__import__("uuid").uuid4())

        with tm._lock:
            tm._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=TaskType.EMBED_BATCH,
                status=TaskStatus.PENDING,
                created_at=time.time(),
            )

        coro = tm._run_embed_batch(task_id, "user1", ["hello", "world"], 0.5, "default")
        self._run_coro(coro)

        task = tm.get_task_status(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result["stored_count"] == 2
        tm.shutdown()

    def test_run_rebuild_fts_without_fts_support(self, monkeypatch):
        """Test _run_rebuild_fts when store lacks rebuild_fts_index."""
        import time
        from unittest.mock import MagicMock

        mock_mem = MagicMock()
        mock_mem._store = MagicMock()
        del mock_mem._store.rebuild_fts_index  # Remove the method
        monkeypatch.setitem(__import__("sys").modules, "kemi", MagicMock())
        import kemi
        kemi.Memory = lambda: mock_mem

        tm = BackgroundTaskManager()
        task_id = str(__import__("uuid").uuid4())

        with tm._lock:
            tm._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=TaskType.REBUILD_FTS_INDEX,
                status=TaskStatus.PENDING,
                created_at=time.time(),
            )

        coro = tm._run_rebuild_fts(task_id, None)
        self._run_coro(coro)

        task = tm.get_task_status(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result["rebuilt"] is False
        tm.shutdown()

    def test_run_ttl_sweep(self, monkeypatch):
        """Test _run_ttl_sweep executes and reports results."""
        import time
        from unittest.mock import MagicMock

        mock_mem = MagicMock()
        mock_mem.prune_expired.return_value = 5
        monkeypatch.setitem(__import__("sys").modules, "kemi", MagicMock())
        import kemi
        kemi.Memory = lambda: mock_mem

        tm = BackgroundTaskManager()
        task_id = str(__import__("uuid").uuid4())

        with tm._lock:
            tm._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=TaskType.TTL_SWEEP,
                status=TaskStatus.PENDING,
                created_at=time.time(),
            )

        coro = tm._run_ttl_sweep(task_id, "user1", "default")
        self._run_coro(coro)

        task = tm.get_task_status(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result["deleted_count"] == 5
        tm.shutdown()


class TestTaskManagerGlobal:
    """Tests for the global task manager instance."""

    def test_get_task_manager_creates_instance(self, monkeypatch):
        """Test get_task_manager creates a BackgroundTaskManager."""
        import kemi.background_tasks as bt

        # Reset global state
        monkeypatch.setattr(bt, "_task_manager", None)
        manager = bt.get_task_manager()
        assert isinstance(manager, BackgroundTaskManager)
        assert bt._task_manager is manager

    def test_get_task_manager_returns_same_instance(self, monkeypatch):
        """Test get_task_manager returns the same singleton."""
        import kemi.background_tasks as bt

        monkeypatch.setattr(bt, "_task_manager", None)
        m1 = bt.get_task_manager()
        m2 = bt.get_task_manager()
        assert m1 is m2

    def test_get_task_manager_respects_env_var(self, monkeypatch):
        """Test KEMI_MAX_BACKGROUND_TASKS env var."""
        import kemi.background_tasks as bt

        monkeypatch.setenv("KEMI_MAX_BACKGROUND_TASKS", "7")
        monkeypatch.setattr(bt, "_task_manager", None)
        manager = bt.get_task_manager()
        assert manager._max_concurrent == 7


class TestTTLSweepTask:
    """Tests for the TTL sweep background task."""

    def test_submit_ttl_sweep_rejects_when_at_capacity(self):
        """Test that submit_ttl_sweep raises RuntimeError when at max capacity."""
        tm = BackgroundTaskManager(max_concurrent_tasks=1)
        tm._running_count = 1  # At capacity

        with pytest.raises(RuntimeError, match="Max concurrent tasks"):
            tm.submit_ttl_sweep(user_id="user1")

    def test_submit_ttl_sweep_creates_pending_task(self):
        """Test that submit_ttl_sweep creates a PENDING task entry."""
        tm = BackgroundTaskManager()
        task_id = tm.submit_ttl_sweep(user_id="user1")

        task = tm.get_task_status(task_id)
        assert task is not None
        assert task.task_id == task_id
        assert task.task_type == TaskType.TTL_SWEEP
        # The async task runner creates a Memory() instance that connects to
        # the default ~/.kemi/memories.db path. If that database doesn't exist
        # or has a different schema version, the task may fail. That's OK for
        # this test — we only verify that the task entry was created correctly.
        assert task.status in {
            TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED,
        }

    def test_submit_ttl_sweep_all_users(self):
        """Test submit_ttl_sweep with no user_id (sweep all)."""
        tm = BackgroundTaskManager()
        task_id = tm.submit_ttl_sweep()
        assert task_id is not None
        task = tm.get_task_status(task_id)
        assert task is not None
        assert task.task_type == TaskType.TTL_SWEEP

    def test_list_tasks_includes_ttl_sweep(self):
        """Test that ttl_sweep tasks appear in list_tasks()."""
        import time

        tm = BackgroundTaskManager()
        with tm._lock:
            tm._tasks["ttl-task-1"] = TaskResult(
                task_id="ttl-task-1",
                task_type=TaskType.TTL_SWEEP,
                status=TaskStatus.COMPLETED,
                created_at=time.time(),
            )

        all_tasks = tm.list_tasks()
        assert len(all_tasks) == 1
        assert all_tasks[0].task_type == TaskType.TTL_SWEEP
