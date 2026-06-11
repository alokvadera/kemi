"""Background task management for async embedding and indexing operations.

This module provides a BackgroundTaskManager that handles long-running operations
like batch embedding and FTS index rebuilding as background tasks.
"""

import asyncio
import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# Constant used so task-body except handlers can preserve the shutdown
# message rather than overwriting it with "Cancelled by user".
_CANCELLED_BY_SHUTDOWN = "Cancelled by shutdown"


class TaskType(enum.Enum):
    """Types of background tasks supported."""

    EMBED_BATCH = "embed_batch"
    REBUILD_FTS_INDEX = "rebuild_fts_index"
    MIGRATE_EMBEDDINGS = "migrate_embeddings"
    TTL_SWEEP = "ttl_sweep"


class TaskStatus(enum.Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Result of a background task."""

    task_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: str | None = None
    progress: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }


class BackgroundTaskManager:
    """Manages background tasks for long-running operations.

    Tasks run in a dedicated event loop on a background thread.
    This allows non-blocking API responses while heavy operations
    like embedding batches or index rebuilding run asynchronously.

    Args:
        max_concurrent_tasks: Maximum number of tasks that can run simultaneously.
    """

    def __init__(self, max_concurrent_tasks: int = 3, max_task_history: int = 1000) -> None:
        self._max_concurrent = max_concurrent_tasks
        self._max_task_history = max_task_history
        self._tasks: dict[str, TaskResult] = {}
        self._futures: dict[str, Any] = {}
        self._lock = Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Any = None
        self._running_count = 0
        self._loop_ready = threading.Event()
        self._cancelled_ids: set[str] = set()

    def _ensure_loop_started(self) -> None:
        """Start the background event loop if not already running."""
        if self._loop is not None:
            return

        self._loop_ready.clear()

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._loop_ready.set()
            loop.run_forever()
            # Close the loop from the same thread after run_forever() returns
            if not loop.is_closed():
                loop.close()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready (avoids busy-wait)
        if not self._loop_ready.wait(timeout=5.0):
            logger.warning("Background event loop failed to start within 5 seconds")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the background event loop."""
        self._ensure_loop_started()
        assert self._loop is not None
        return self._loop

    def _active_task_count(self) -> int:
        """Return the number of tasks that are PENDING or RUNNING."""
        return sum(
            1
            for t in self._tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        )

    def _submit_guard(self, task_type: TaskType) -> str:
        """Atomic check + creation that we haven't exceeded max concurrent tasks.

        Creates the task entry under the same lock acquisition as the limit
        check, preventing the race where two threads both pass the guard
        before either creates a task.

        Raises:
            RuntimeError: If the active (PENDING + RUNNING) task count
                has reached ``self._max_concurrent``.

        Returns:
            The newly created task_id.
        """
        with self._lock:
            if self._active_task_count() >= self._max_concurrent:
                raise RuntimeError(
                    f"Max concurrent tasks ({self._max_concurrent}) reached. "
                    "Wait for a task to complete before submitting more."
                )
            task_id = str(uuid.uuid4())
            created_at = time.time()
            self._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                created_at=created_at,
            )
            self._cleanup_old_tasks()
            return task_id

    def submit_embed_batch(
        self,
        user_id: str,
        contents: list[str],
        importance: float = 0.5,
        namespace: str = "default",
        memory: Any = None,
    ) -> str:
        """Submit a batch embedding task to run in background.

        Args:
            user_id: User ID for the memories.
            contents: List of content strings to embed and store.
            importance: Importance value for all memories.
            namespace: Memory namespace.

        Returns:
            task_id that can be used to check status.

        Raises:
            RuntimeError: If max concurrent tasks limit reached.
        """
        task_id = self._submit_guard(TaskType.EMBED_BATCH)

        # Submit to event loop
        loop = self._get_loop()
        coro = self._run_embed_batch(task_id, user_id, contents, importance, namespace, memory)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        with self._lock:
            self._futures[task_id] = future

        logger.info(f"Submitted embed_batch task {task_id} with {len(contents)} items")
        return task_id

    async def _run_embed_batch(
        self,
        task_id: str,
        user_id: str,
        contents: list[str],
        importance: float,
        namespace: str,
        memory: Any = None,
    ) -> None:
        """Run the batch embedding task."""
        from kemi import Memory
        from kemi.memory.model import MemorySource, MemoryType

        with self._lock:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = time.time()
            self._running_count += 1

        mem = memory
        try:
            if task_id in self._cancelled_ids:
                raise asyncio.CancelledError()

            if mem is None:
                mem = Memory()
            # Run directly on background thread - no need for asyncio.to_thread
            total = len(contents)
            stored = mem.remember_many(
                user_id=user_id,
                contents=contents,
                importance=importance,
                namespace=namespace,
                source=MemorySource.USER_STATED,
                memory_type=MemoryType.EPISODIC,
            )

            # Cooperative checkpoint: abort if cancelled during the work
            if task_id in self._cancelled_ids:
                raise asyncio.CancelledError()

            result = {"stored_count": len(stored), "user_id": user_id}

            with self._lock:
                self._tasks[task_id].status = TaskStatus.COMPLETED
                self._tasks[task_id].result = result
                self._tasks[task_id].completed_at = time.time()
                self._tasks[task_id].progress = 1.0

            logger.info(f"Completed embed_batch task {task_id}: stored {total} memories")

        except asyncio.CancelledError:
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                # Preserve "Cancelled by shutdown" set by shutdown().
                if self._tasks[task_id].error != _CANCELLED_BY_SHUTDOWN:
                    self._tasks[task_id].error = "Cancelled by user"
                self._tasks[task_id].completed_at = time.time()

        except Exception as e:
            logger.error(f"Failed embed_batch task {task_id}: {e}")
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = time.time()

        finally:
            self._cleanup_task(task_id, mem, memory)

    def submit_rebuild_fts_index(self, user_id: str | None = None, memory: Any = None) -> str:
        """Submit an FTS index rebuild task to run in background.

        Args:
            user_id: Optional user ID to limit rebuild scope. If None, rebuilds all.

        Returns:
            task_id that can be used to check status.

        Raises:
            RuntimeError: If max concurrent tasks limit reached.
        """
        task_id = self._submit_guard(TaskType.REBUILD_FTS_INDEX)

        loop = self._get_loop()
        coro = self._run_rebuild_fts(task_id, user_id, memory)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        with self._lock:
            self._futures[task_id] = future

        scope = f"user {user_id}" if user_id else "all users"
        logger.info(f"Submitted rebuild_fts_index task {task_id} for {scope}")
        return task_id

    async def _run_rebuild_fts(self, task_id: str, user_id: str | None, memory: Any = None) -> None:
        """Run the FTS index rebuild task."""
        from kemi import Memory

        with self._lock:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = time.time()
            self._running_count += 1

        mem = memory
        try:
            if task_id in self._cancelled_ids:
                raise asyncio.CancelledError()

            if mem is None:
                mem = Memory()

            # Rebuild FTS index
            if hasattr(mem._store, "rebuild_fts_index"):
                # Pass user_id through so per-user rebuilds only touch that
                # user's FTS rows instead of rebuilding the whole index.
                count = await asyncio.to_thread(
                    mem._store.rebuild_fts_index,
                    user_id,
                )
            else:
                count = None

            # Cooperative checkpoint: abort if cancelled during the work
            if task_id in self._cancelled_ids:
                raise asyncio.CancelledError()

            if count is not None:
                result = {
                    "rebuilt": True,
                    "count": count,
                    "user_id": user_id,
                    "scope": "user" if user_id else "all",
                }
            else:
                result = {
                    "rebuilt": False,
                    "message": "Storage adapter does not support FTS rebuild",
                }

            with self._lock:
                self._tasks[task_id].status = TaskStatus.COMPLETED
                self._tasks[task_id].result = result
                self._tasks[task_id].completed_at = time.time()
                self._tasks[task_id].progress = 1.0

            logger.info(f"Completed rebuild_fts_index task {task_id}")

        except asyncio.CancelledError:
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                # Preserve "Cancelled by shutdown" set by shutdown().
                if self._tasks[task_id].error != _CANCELLED_BY_SHUTDOWN:
                    self._tasks[task_id].error = "Cancelled by user"
                self._tasks[task_id].completed_at = time.time()

        except Exception as e:
            logger.error(f"Failed rebuild_fts_index task {task_id}: {e}")
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = time.time()

        finally:
            if memory is None and mem is not None:
                try:
                    mem.close()
                except Exception:
                    pass
            with self._lock:
                self._running_count -= 1
                self._futures.pop(task_id, None)
                self._cancelled_ids.discard(task_id)

    def get_task_status(self, task_id: str) -> TaskResult | None:
        """Get the status of a task.

        Args:
            task_id: The task ID returned from submit_*.

        Returns:
            TaskResult if found, None otherwise.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[TaskResult]:
        """List all tasks, optionally filtered by status.

        Args:
            status: Optional filter by task status.
            limit: Maximum number of tasks to return.

        Returns:
            List of TaskResult objects.
        """
        with self._lock:
            tasks = list(self._tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get task manager statistics.

        Returns:
            Dict with counts of pending, running, completed, failed tasks.
        """
        with self._lock:
            tasks = list(self._tasks.values())

        pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
        running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)

        return {
            "total_tasks": len(tasks),
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "max_concurrent": self._max_concurrent,
        }

    def shutdown(self) -> None:
        """Gracefully shutdown the task manager.

        Cancels pending futures, signals running tasks to stop
        cooperatively, stops the background event loop, and waits for
        the background thread to finish.
        """
        with self._lock:
            # Signal running tasks to stop cooperatively. PENDING tasks
            # don't need the flag because their futures are cancelled
            # directly below and they never start.
            for _tid, task in self._tasks.items():
                if task.status == TaskStatus.RUNNING:
                    self._cancelled_ids.add(_tid)
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.status = TaskStatus.FAILED
                    task.error = _CANCELLED_BY_SHUTDOWN
                    task.completed_at = time.time()
            # Cancel pending futures
            for _tid, future in list(self._futures.items()):
                future.cancel()
            self._futures.clear()

        if self._loop is not None:
            loop = self._loop
            # Schedule loop stop from the background thread
            loop.call_soon_threadsafe(loop.stop)
            # Wait for the thread to finish before clearing references
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning(
                        "Background task thread did not stop within 5s; "
                        "some tasks may still be running."
                    )
            # The worker thread (run_loop) closes the loop after
            # run_forever() returns; calling close() here from a
            # different thread is unsafe.
            self._loop = None
            self._thread = None
            logger.info("Background task manager shutdown complete")

    def _cleanup_old_tasks(self) -> None:
        """Remove old completed/failed tasks if history limit exceeded."""
        if len(self._tasks) <= self._max_task_history:
            return

        # Get completed/failed tasks sorted by completion time
        old_tasks = [
            (tid, t.completed_at or t.created_at)
            for tid, t in self._tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        old_tasks.sort(key=lambda x: x[1])  # Sort by time ascending

        # Remove oldest until under limit
        tasks_to_remove = len(self._tasks) - self._max_task_history
        for tid, _ in old_tasks[:tasks_to_remove]:
            del self._tasks[tid]

    def _cleanup_task(self, task_id: str, mem: Any, caller_memory: Any) -> None:
        """Close an ephemeral Memory instance and remove task bookkeeping.

        Called from the ``finally`` block of every task body.  If
        *caller_memory* is None it means the Memory was created by the
        task itself and should be closed; otherwise it was supplied by
        the caller and must be left alone.
        """
        if caller_memory is None and mem is not None:
            try:
                mem.close()
            except Exception:
                pass
        with self._lock:
            self._running_count -= 1
            self._futures.pop(task_id, None)
            self._cancelled_ids.discard(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if the task is marked as cancelled (PENDING tasks are
            stopped immediately; RUNNING tasks are flagged and will stop
            at the next cooperative checkpoint).
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False

            future = self._futures.pop(task_id, None)
            cancelled = False
            if future is not None:
                cancelled = future.cancel()

            if cancelled or task.status == TaskStatus.PENDING:
                # Stopped before it started (or future was already done)
                task.status = TaskStatus.FAILED
                task.error = "Cancelled by user"
                task.completed_at = time.time()
                return True

            # Task is RUNNING and future.cancel() returned False.
            # asyncio cannot force-stop a running coroutine from outside
            # the event loop, so flag it for cooperative abort.
            self._cancelled_ids.add(task_id)
            return True

    def submit_ttl_sweep(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
        memory: Any = None,
    ) -> str:
        """Submit a TTL sweep task to delete expired memories in the background.

        Args:
            user_id: If provided, only sweep this user's expired memories.
                If None, sweep all users.
            namespace: If provided, only sweep this namespace.

        Returns:
            task_id that can be used to check status.

        Raises:
            RuntimeError: If max concurrent tasks limit reached.
        """
        task_id = self._submit_guard(TaskType.TTL_SWEEP)

        loop = self._get_loop()
        coro = self._run_ttl_sweep(task_id, user_id, namespace, memory)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        with self._lock:
            self._futures[task_id] = future

        scope = f"user {user_id}" if user_id else "all users"
        logger.info(f"Submitted ttl_sweep task {task_id} for {scope}")
        return task_id

    async def _run_ttl_sweep(
        self,
        task_id: str,
        user_id: str | None,
        namespace: str | None,
        memory: Any = None,
    ) -> None:
        """Run the TTL sweep task."""
        from kemi import Memory

        with self._lock:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = time.time()
            self._running_count += 1

        mem = memory
        try:
            if task_id in self._cancelled_ids:
                raise asyncio.CancelledError()

            if mem is None:
                mem = Memory()
            deleted = await asyncio.to_thread(
                mem.prune_expired,
                user_id,
                namespace,
            )

            # Cooperative checkpoint: abort if cancelled during the work
            if task_id in self._cancelled_ids:
                raise asyncio.CancelledError()

            result = {
                "deleted_count": deleted,
                "user_id": user_id,
                "namespace": namespace,
            }

            with self._lock:
                self._tasks[task_id].status = TaskStatus.COMPLETED
                self._tasks[task_id].result = result
                self._tasks[task_id].completed_at = time.time()
                self._tasks[task_id].progress = 1.0

            logger.info(
                f"Completed ttl_sweep task {task_id}: deleted {deleted} memories"
            )

        except asyncio.CancelledError:
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                # Preserve "Cancelled by shutdown" set by shutdown().
                if self._tasks[task_id].error != _CANCELLED_BY_SHUTDOWN:
                    self._tasks[task_id].error = "Cancelled by user"
                self._tasks[task_id].completed_at = time.time()

        except Exception as e:
            logger.error(f"Failed ttl_sweep task {task_id}: {e}")
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = time.time()

        finally:
            if memory is None and mem is not None:
                try:
                    mem.close()
                except Exception:
                    pass
            with self._lock:
                self._running_count -= 1
                self._futures.pop(task_id, None)
                self._cancelled_ids.discard(task_id)


# Global task manager instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get or create the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        max_concurrent = int(__import__("os").environ.get("KEMI_MAX_BACKGROUND_TASKS", "3"))
        _task_manager = BackgroundTaskManager(max_concurrent_tasks=max_concurrent)
    return _task_manager
