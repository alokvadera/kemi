"""Tests for src/kemi/background_tasks.py — background task management."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from kemi.infra.background_tasks import (
    BackgroundTaskManager,
    TaskResult,
    TaskStatus,
    TaskType,
    get_task_manager,
)


def _drain_coro_threadsafe(coro, loop):
    """Replacement for ``asyncio.run_coroutine_threadsafe`` used in tests.

    The real function wraps a coroutine in a Future and schedules it on the
    target loop. In tests we don't want the coroutine to actually run, but
    we also need to avoid the "coroutine was never awaited" RuntimeWarning
    that fires when a coroutine object is GC'd without being awaited.
    Closing the coroutine consumes it and silences the warning.
    """
    if asyncio.iscoroutine(coro):
        coro.close()
    return MagicMock()


# Track managers that started a background loop so the autouse fixture
# below can shut them down at the end of each test. Without this the
# daemon threads leak across tests, and a fresh BackgroundTaskManager
# (or a previous test's still-running loop) can interfere with the next
# test's behaviour.
_started_managers: list[BackgroundTaskManager] = []
_original_ensure_loop_started = BackgroundTaskManager._ensure_loop_started


def _tracking_ensure_loop_started(self) -> None:
    """Wrap the real ``_ensure_loop_started`` to register managers in
    ``_started_managers`` so the autouse fixture can clean them up.
    """
    _original_ensure_loop_started(self)
    if self._loop is not None:
        _started_managers.append(self)


@pytest.fixture(autouse=True)
def _shutdown_started_managers(monkeypatch) -> None:
    """Shut down any BackgroundTaskManager that started a loop during a test.

    The real thread is daemon=True so it dies at interpreter shutdown, but
    keeping a reference to a still-running loop across tests is a latent
    flake source: a previous test's task could mutate shared module state
    (the global ``_task_manager`` singleton) or fire into a future test's
    patched methods.
    """
    monkeypatch.setattr(
        BackgroundTaskManager,
        "_ensure_loop_started",
        _tracking_ensure_loop_started,
    )
    _started_managers.clear()
    yield
    for mgr in _started_managers:
        try:
            mgr.shutdown()
        except Exception:
            pass
        if mgr._thread is not None:
            mgr._thread.join(timeout=0.5)
    _started_managers.clear()


# ---------------------------------------------------------------------------
# TaskType / TaskStatus enums
# ---------------------------------------------------------------------------

class TestTaskEnums:
    def test_task_type_values(self) -> None:
        assert TaskType.EMBED_BATCH.value == "embed_batch"
        assert TaskType.REBUILD_FTS_INDEX.value == "rebuild_fts_index"
        assert TaskType.MIGRATE_EMBEDDINGS.value == "migrate_embeddings"
        assert TaskType.TTL_SWEEP.value == "ttl_sweep"

    def test_task_status_values(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# TaskResult dataclass
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_task_result_defaults(self) -> None:
        result = TaskResult(
            task_id="test-123",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        assert result.started_at is None
        assert result.completed_at is None
        assert result.result is None
        assert result.error is None
        assert result.progress == 0.0

    def test_task_result_to_dict(self) -> None:
        result = TaskResult(
            task_id="test-123",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.COMPLETED,
            created_at=1234.0,
            started_at=1235.0,
            completed_at=1236.0,
            result={"count": 5},
            error=None,
            progress=1.0,
        )
        d = result.to_dict()
        assert d["task_id"] == "test-123"
        assert d["task_type"] == "embed_batch"
        assert d["status"] == "completed"
        assert d["created_at"] == 1234.0
        assert d["started_at"] == 1235.0
        assert d["completed_at"] == 1236.0
        assert d["result"] == {"count": 5}
        assert d["error"] is None
        assert d["progress"] == 1.0

    def test_task_result_to_dict_with_error(self) -> None:
        result = TaskResult(
            task_id="test-456",
            task_type=TaskType.TTL_SWEEP,
            status=TaskStatus.FAILED,
            created_at=100.0,
            error="Something broke",
        )
        d = result.to_dict()
        assert d["error"] == "Something broke"
        assert d["status"] == "failed"


# ---------------------------------------------------------------------------
# BackgroundTaskManager — empty state
# ---------------------------------------------------------------------------

class TestBackgroundTaskManagerEmpty:
    def test_init_defaults(self) -> None:
        mgr = BackgroundTaskManager()
        assert mgr._max_concurrent == 3
        assert mgr._max_task_history == 1000
        stats = mgr.get_stats()
        assert stats["total_tasks"] == 0
        assert stats["pending"] == 0
        assert stats["running"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0

    def test_init_custom_limits(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=5, max_task_history=200)
        assert mgr._max_concurrent == 5
        assert mgr._max_task_history == 200

    def test_get_task_status_missing(self) -> None:
        mgr = BackgroundTaskManager()
        assert mgr.get_task_status("nonexistent") is None

    def test_list_tasks_empty(self) -> None:
        mgr = BackgroundTaskManager()
        assert mgr.list_tasks() == []

    def test_list_tasks_with_limit(self) -> None:
        mgr = BackgroundTaskManager()
        assert mgr.list_tasks(limit=10) == []

    def test_cancel_nonexistent_task(self) -> None:
        mgr = BackgroundTaskManager()
        assert mgr.cancel_task("nonexistent") is False

    def test_cancel_pending_task(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        # Patch run_coroutine_threadsafe so the task never actually starts
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            task_id = mgr.submit_embed_batch("user1", ["content"])
        status = mgr.get_task_status(task_id)
        assert status is not None
        assert status.status == TaskStatus.PENDING
        assert mgr.cancel_task(task_id) is True
        updated = mgr.get_task_status(task_id)
        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "Cancelled by user"
        assert updated.completed_at is not None

    def test_cancel_running_task_with_future(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            task_id = mgr.submit_embed_batch("user1", ["content"])
        # Manually flip to RUNNING so cancel sees it as already running
        with mgr._lock:
            mgr._tasks[task_id].status = TaskStatus.RUNNING
        # MagicMock.cancel() returns a truthy MagicMock, so cancel succeeds
        assert mgr.cancel_task(task_id) is True
        updated = mgr.get_task_status(task_id)
        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "Cancelled by user"

    def test_cancel_running_task_when_future_cancel_returns_false(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            task_id = mgr.submit_embed_batch("user1", ["content"])
        # Replace the future with one whose cancel() returns False
        with mgr._lock:
            mgr._tasks[task_id].status = TaskStatus.RUNNING
            mgr._futures[task_id] = MagicMock()
            mgr._futures[task_id].cancel.return_value = False
        # Cooperative cancellation is initiated, so it returns True
        assert mgr.cancel_task(task_id) is True

    def test_shutdown_fresh_manager(self) -> None:
        mgr = BackgroundTaskManager()
        mgr.shutdown()
        assert mgr._loop is None
        assert mgr._thread is None

    def test_active_task_count(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid1 = mgr.submit_embed_batch("user1", ["a"])
            tid2 = mgr.submit_embed_batch("user1", ["b"])
        with mgr._lock:
            mgr._tasks[tid1].status = TaskStatus.RUNNING
        assert mgr._active_task_count() == 2
        with mgr._lock:
            mgr._tasks[tid1].status = TaskStatus.COMPLETED
        assert mgr._active_task_count() == 1
        with mgr._lock:
            mgr._tasks[tid2].status = TaskStatus.FAILED
        assert mgr._active_task_count() == 0

    def test_cleanup_old_tasks_no_op_when_under_limit(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10, max_task_history=100)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            for _ in range(5):
                mgr.submit_embed_batch("user1", ["content"])
        # Under limit, nothing removed
        assert len(mgr._tasks) == 5
        mgr._cleanup_old_tasks()
        assert len(mgr._tasks) == 5

    def test_cleanup_old_tasks_removes_oldest_completed(self) -> None:
        mgr = BackgroundTaskManager(max_task_history=2)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid1 = mgr.submit_embed_batch("user1", ["content"])
            tid2 = mgr.submit_embed_batch("user1", ["content"])
            tid3 = mgr.submit_embed_batch("user1", ["content"])
        # Mark two as completed so cleanup has candidates
        with mgr._lock:
            mgr._tasks[tid1].status = TaskStatus.COMPLETED
            mgr._tasks[tid1].completed_at = 100.0
            mgr._tasks[tid2].status = TaskStatus.COMPLETED
            mgr._tasks[tid2].completed_at = 200.0
            mgr._tasks[tid3].status = TaskStatus.PENDING
        mgr._cleanup_old_tasks()
        # Oldest completed (tid1) should be removed, tid2 and tid3 remain
        assert tid1 not in mgr._tasks
        assert tid2 in mgr._tasks
        assert tid3 in mgr._tasks

    def test_list_tasks_filter_by_status(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid1 = mgr.submit_embed_batch("user1", ["content"])
            tid2 = mgr.submit_embed_batch("user1", ["content"])
        with mgr._lock:
            mgr._tasks[tid1].status = TaskStatus.COMPLETED
            mgr._tasks[tid2].status = TaskStatus.PENDING
        completed = mgr.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].task_id == tid1
        pending = mgr.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == tid2

    def test_list_tasks_filter_no_match(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            mgr.submit_embed_batch("user1", ["content"])
        assert mgr.list_tasks(status=TaskStatus.FAILED) == []
        assert mgr.list_tasks(status=TaskStatus.RUNNING) == []

    def test_list_tasks_sorted_descending(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            mgr.submit_embed_batch("user1", ["content"])
            time.sleep(0.01)
            mgr.submit_embed_batch("user1", ["content"])
        tasks = mgr.list_tasks()
        assert len(tasks) == 2
        # Most recent first
        assert tasks[0].created_at >= tasks[1].created_at

    def test_list_tasks_respects_limit(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            for _ in range(5):
                mgr.submit_embed_batch("user1", ["content"])
                time.sleep(0.001)
        tasks = mgr.list_tasks(limit=2)
        assert len(tasks) == 2

    def test_stats_after_submission(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            mgr.submit_embed_batch("user1", ["content"])
        stats = mgr.get_stats()
        assert stats["total_tasks"] == 1
        assert stats["pending"] == 1
        assert stats["running"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["max_concurrent"] == 10

    def test_rebuild_fts_submission(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            task_id = mgr.submit_rebuild_fts_index("user1")
        status = mgr.get_task_status(task_id)
        assert status is not None
        assert status.task_type == TaskType.REBUILD_FTS_INDEX
        assert status.status == TaskStatus.PENDING

    def test_ttl_sweep_submission(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            task_id = mgr.submit_ttl_sweep("user1", namespace="ns1")
        status = mgr.get_task_status(task_id)
        assert status is not None
        assert status.task_type == TaskType.TTL_SWEEP
        assert status.status == TaskStatus.PENDING

    def test_max_concurrent_rejection(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=1)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            mgr.submit_embed_batch("user1", ["content"])
            # Manually bump running count so next submit is rejected
            with mgr._lock:
                mgr._running_count = 1
            with pytest.raises(RuntimeError, match="Max concurrent tasks"):
                mgr.submit_embed_batch("user1", ["content"])

    def test_ensure_loop_started(self) -> None:
        mgr = BackgroundTaskManager()
        assert mgr._loop is None
        mgr._ensure_loop_started()
        assert mgr._loop is not None
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        mgr.shutdown()
        # shutdown() now joins and closes the loop internally
        assert mgr._loop is None
        assert mgr._thread is None


# ---------------------------------------------------------------------------
# BackgroundTaskManager — cancel / shutdown edge cases
# ---------------------------------------------------------------------------

class TestBackgroundTaskManagerCancelShutdown:
    def test_shutdown_cancels_pending_tasks(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid = mgr.submit_embed_batch("user1", ["content"])
        mgr.shutdown()
        updated = mgr.get_task_status(tid)
        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "Cancelled by shutdown"
        assert updated.completed_at is not None

    def test_shutdown_cancels_running_tasks(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid = mgr.submit_embed_batch("user1", ["content"])
        with mgr._lock:
            mgr._tasks[tid].status = TaskStatus.RUNNING
        mgr.shutdown()
        updated = mgr.get_task_status(tid)
        assert updated is not None
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "Cancelled by shutdown"
        assert tid in mgr._cancelled_ids

    @pytest.mark.asyncio
    async def test_shutdown_preserves_cancelled_by_shutdown_message(self) -> None:
        """If shutdown() flags a running task and the coroutine later hits the
        checkpoint, the 'Cancelled by shutdown' message is preserved."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1"]

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.RUNNING,
            created_at=time.time(),
        )
        mgr._cancelled_ids.add("tid1")
        # Simulate what shutdown() does: set FAILED + "Cancelled by shutdown"
        mgr._tasks["tid1"].status = TaskStatus.FAILED
        mgr._tasks["tid1"].error = "Cancelled by shutdown"

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert status.error == "Cancelled by shutdown"

    def test_shutdown_clears_futures(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            mgr.submit_embed_batch("user1", ["content"])
        assert len(mgr._futures) == 1
        mgr.shutdown()
        assert len(mgr._futures) == 0

    def test_shutdown_idempotent_on_fresh_manager(self) -> None:
        mgr = BackgroundTaskManager()
        # Should not raise even when no loop was ever started
        mgr.shutdown()
        assert mgr._loop is None
        assert mgr._thread is None

    def test_cancel_task_already_completed(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid = mgr.submit_embed_batch("user1", ["content"])
        with mgr._lock:
            mgr._tasks[tid].status = TaskStatus.COMPLETED
        assert mgr.cancel_task(tid) is False
        assert mgr.get_task_status(tid).status == TaskStatus.COMPLETED

    def test_cancel_task_already_failed(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid = mgr.submit_embed_batch("user1", ["content"])
        with mgr._lock:
            mgr._tasks[tid].status = TaskStatus.FAILED
        assert mgr.cancel_task(tid) is False
        assert mgr.get_task_status(tid).status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancelled_id_checked_inside_coro(self) -> None:
        """A task whose ID is in _cancelled_ids is marked FAILED, not COMPLETED."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        mgr._cancelled_ids.add("tid1")

        await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        status = mgr.get_task_status("tid1")
        assert status is not None
        # CancelledError is explicitly caught and the task is marked FAILED
        assert status.status == TaskStatus.FAILED
        assert status.error == "Cancelled by user"
        assert status.completed_at is not None
        assert mgr._running_count == 0

    def test_cancel_running_task_future_cancel_false_returns_false(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid = mgr.submit_embed_batch("user1", ["content"])
        with mgr._lock:
            mgr._tasks[tid].status = TaskStatus.RUNNING
            # future.cancel() returns False → cooperative cancel path
            # still initiates cancellation and returns True.
            fake_future = MagicMock()
            fake_future.cancel.return_value = False
            mgr._futures[tid] = fake_future
        assert mgr.cancel_task(tid) is True
        assert tid in mgr._cancelled_ids

    def test_cancel_running_task_future_is_none(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        with patch("kemi.infra.background_tasks.asyncio.run_coroutine_threadsafe", side_effect=_drain_coro_threadsafe):  # noqa: E501
            tid = mgr.submit_embed_batch("user1", ["content"])
        with mgr._lock:
            mgr._tasks[tid].status = TaskStatus.RUNNING
            # The future was already popped from _futures
            mgr._futures.pop(tid, None)
        # Cooperative cancellation is still initiated even without a future
        assert mgr.cancel_task(tid) is True
        assert tid in mgr._cancelled_ids


# ---------------------------------------------------------------------------
# BackgroundTaskManager — async task lifecycle (with mocked Memory)
# ---------------------------------------------------------------------------

class TestBackgroundTaskManagerLifecycle:
    @pytest.mark.asyncio
    async def test_run_embed_batch_success(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1", "mem2"]

        # Pre-create task entry since _run_embed_batch expects it
        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a", "b"], 0.5, "ns")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.COMPLETED
        assert status.result == {"stored_count": 2, "user_id": "user1"}
        assert status.progress == 1.0
        assert status.started_at is not None
        assert status.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_embed_batch_failure(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.side_effect = RuntimeError("DB error")

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert "DB error" in status.error
        assert status.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_embed_batch_cancelled_after_work(self) -> None:
        """A task flagged as cancelled during work is marked FAILED, not COMPLETED."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1"]

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        # Flag the task as cancelled before the coroutine starts
        mgr._cancelled_ids.add("tid1")

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert status.error == "Cancelled by user"
        assert status.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_embed_batch_cancelled_during_work(self) -> None:
        """A task flagged as cancelled mid-work hits the post-work checkpoint."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1"]

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        # Simulate a cancel that arrives while remember_many is running:
        # patch _cancelled_ids so the post-work check sees the flag.
        def _flag_after_remember(*args, **kwargs):
            mgr._cancelled_ids.add("tid1")
            return ["mem1"]

        mock_memory.remember_many.side_effect = _flag_after_remember

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert status.error == "Cancelled by user"
        assert status.completed_at is not None
        assert status.result is None

    @pytest.mark.asyncio
    async def test_run_rebuild_fts_success(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_store = MagicMock()
        mock_store.rebuild_fts_index.return_value = 42
        mock_memory = MagicMock()
        mock_memory._store = mock_store

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.REBUILD_FTS_INDEX,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_rebuild_fts("tid1", "user1")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.COMPLETED
        assert status.result["rebuilt"] is True
        assert status.result["count"] == 42
        assert status.result["user_id"] == "user1"
        assert status.result["scope"] == "user"

    @pytest.mark.asyncio
    async def test_run_rebuild_fts_no_support(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        # _store exists but lacks rebuild_fts_index → triggers else branch
        mock_memory._store = MagicMock(spec=[])

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.REBUILD_FTS_INDEX,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_rebuild_fts("tid1", None)

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.COMPLETED
        assert status.result["rebuilt"] is False

    @pytest.mark.asyncio
    async def test_run_rebuild_fts_cancelled(self) -> None:
        """A rebuild task flagged as cancelled mid-work is marked FAILED."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_store = MagicMock()
        mock_store.rebuild_fts_index.return_value = 42
        mock_memory = MagicMock()
        mock_memory._store = mock_store

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.REBUILD_FTS_INDEX,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        # Flag the task as cancelled before the coroutine starts
        mgr._cancelled_ids.add("tid1")

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_rebuild_fts("tid1", "user1")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert status.error == "Cancelled by user"
        assert status.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_ttl_sweep_cancelled(self) -> None:
        """A TTL sweep flagged as cancelled mid-work is marked FAILED."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.prune_expired.return_value = 7

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.TTL_SWEEP,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        # Flag the task as cancelled before the coroutine starts
        mgr._cancelled_ids.add("tid1")

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_ttl_sweep("tid1", "user1", "ns1")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert status.error == "Cancelled by user"
        assert status.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_rebuild_fts_failure(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_store = MagicMock()
        mock_store.rebuild_fts_index.side_effect = Exception("fts fail")
        mock_memory = MagicMock()
        mock_memory._store = mock_store

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.REBUILD_FTS_INDEX,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_rebuild_fts("tid1", None)

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert "fts fail" in status.error

    @pytest.mark.asyncio
    async def test_run_ttl_sweep_success(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.prune_expired.return_value = 7

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.TTL_SWEEP,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_ttl_sweep("tid1", "user1", "ns1")

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.COMPLETED
        assert status.result == {"deleted_count": 7, "user_id": "user1", "namespace": "ns1"}

    @pytest.mark.asyncio
    async def test_run_ttl_sweep_failure(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.prune_expired.side_effect = Exception("prune fail")

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.TTL_SWEEP,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_ttl_sweep("tid1", None, None)

        status = mgr.get_task_status("tid1")
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert "prune fail" in status.error

    @pytest.mark.asyncio
    async def test_running_count_decremented_on_success(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = []

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        assert mgr._running_count == 0

    @pytest.mark.asyncio
    async def test_running_count_decremented_on_failure(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.side_effect = RuntimeError("fail")

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        assert mgr._running_count == 0

    @pytest.mark.asyncio
    async def test_run_embed_batch_closes_memory_on_success(self) -> None:
        """When no external memory is passed, the task must close the Memory it creates."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1"]

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        mock_memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_embed_batch_closes_memory_on_failure(self) -> None:
        """Memory is closed even when the task raises an exception."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.side_effect = RuntimeError("fail")

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        mock_memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_embed_batch_does_not_close_passed_memory(self) -> None:
        """When an external memory instance is passed, the task must NOT close it."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1"]

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        # Pass memory directly — task should not close it
        await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns", memory=mock_memory)

        mock_memory.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_rebuild_fts_closes_memory(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory._store = MagicMock(spec=["rebuild_fts_index"])
        mock_memory._store.rebuild_fts_index.return_value = 10

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.REBUILD_FTS_INDEX,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_rebuild_fts("tid1", "user1")

        mock_memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ttl_sweep_closes_memory(self) -> None:
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.prune_expired.return_value = 3

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.TTL_SWEEP,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_ttl_sweep("tid1", "user1", "ns1")

        mock_memory.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_embed_batch_no_close_when_cancelled_before_instantiation(self) -> None:
        """If cancelled before Memory() is called, no Memory exists to close."""
        mgr = BackgroundTaskManager(max_concurrent_tasks=10)
        mock_memory = MagicMock()
        mock_memory.remember_many.return_value = ["mem1"]

        mgr._tasks["tid1"] = TaskResult(
            task_id="tid1",
            task_type=TaskType.EMBED_BATCH,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        mgr._cancelled_ids.add("tid1")

        with patch("kemi.Memory", return_value=mock_memory):
            await mgr._run_embed_batch("tid1", "user1", ["a"], 0.5, "ns")

        # Memory() is never called because the pre-work CancelledError fires first
        mock_memory.close.assert_not_called()


# ---------------------------------------------------------------------------
# get_task_manager singleton
# ---------------------------------------------------------------------------

class TestGetTaskManager:
    def test_singleton_returns_same_instance(self) -> None:
        # Use fresh module state by manipulating the global
        import kemi.infra.background_tasks as bt
        orig = bt._task_manager
        bt._task_manager = None
        try:
            m1 = get_task_manager()
            m2 = get_task_manager()
            assert m1 is m2
        finally:
            bt._task_manager = orig

    def test_singleton_respects_env_var(self, monkeypatch) -> None:
        import kemi.infra.background_tasks as bt
        orig = bt._task_manager
        bt._task_manager = None
        monkeypatch.setenv("KEMI_MAX_BACKGROUND_TASKS", "7")
        try:
            m = get_task_manager()
            assert m._max_concurrent == 7
        finally:
            bt._task_manager = orig
