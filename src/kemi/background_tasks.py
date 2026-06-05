"""Background task management for async embedding and indexing operations.

This module provides a BackgroundTaskManager that handles long-running operations
like batch embedding and FTS index rebuilding as background tasks.
"""

import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


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
        self._lock = Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Any = None
        self._running_count = 0

    def _ensure_loop_started(self) -> None:
        """Start the background event loop if not already running."""
        if self._loop is not None:
            return

        def run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        import threading

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            time.sleep(0.01)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the background event loop."""
        self._ensure_loop_started()
        assert self._loop is not None
        return self._loop

    def submit_embed_batch(
        self,
        user_id: str,
        contents: list[str],
        importance: float = 0.5,
        namespace: str = "default",
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
        with self._lock:
            if self._running_count >= self._max_concurrent:
                raise RuntimeError(
                    f"Max concurrent tasks ({self._max_concurrent}) reached. "
                    "Wait for a task to complete before submitting more."
                )

        task_id = str(uuid.uuid4())
        created_at = time.time()

        with self._lock:
            self._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=TaskType.EMBED_BATCH,
                status=TaskStatus.PENDING,
                created_at=created_at,
            )
            self._cleanup_old_tasks()

        # Submit to event loop
        loop = self._get_loop()
        coro = self._run_embed_batch(task_id, user_id, contents, importance, namespace)
        asyncio.run_coroutine_threadsafe(coro, loop)

        logger.info(f"Submitted embed_batch task {task_id} with {len(contents)} items")
        return task_id

    async def _run_embed_batch(
        self,
        task_id: str,
        user_id: str,
        contents: list[str],
        importance: float,
        namespace: str,
    ) -> None:
        """Run the batch embedding task."""
        from kemi import Memory
        from kemi.models import MemorySource, MemoryType

        with self._lock:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = time.time()
            self._running_count += 1

        try:
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

            result = {"stored_count": len(stored), "user_id": user_id}

            with self._lock:
                self._tasks[task_id].status = TaskStatus.COMPLETED
                self._tasks[task_id].result = result
                self._tasks[task_id].completed_at = time.time()
                self._tasks[task_id].progress = 1.0

            logger.info(f"Completed embed_batch task {task_id}: stored {total} memories")

        except Exception as e:
            logger.error(f"Failed embed_batch task {task_id}: {e}")
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = time.time()

        finally:
            with self._lock:
                self._running_count -= 1

    def submit_rebuild_fts_index(self, user_id: str | None = None) -> str:
        """Submit an FTS index rebuild task to run in background.

        Args:
            user_id: Optional user ID to limit rebuild scope. If None, rebuilds all.

        Returns:
            task_id that can be used to check status.

        Raises:
            RuntimeError: If max concurrent tasks limit reached.
        """
        with self._lock:
            if self._running_count >= self._max_concurrent:
                raise RuntimeError(
                    f"Max concurrent tasks ({self._max_concurrent}) reached. "
                    "Wait for a task to complete before submitting more."
                )

        task_id = str(uuid.uuid4())
        created_at = time.time()

        with self._lock:
            self._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=TaskType.REBUILD_FTS_INDEX,
                status=TaskStatus.PENDING,
                created_at=created_at,
            )
            self._cleanup_old_tasks()

        loop = self._get_loop()
        coro = self._run_rebuild_fts(task_id, user_id)
        asyncio.run_coroutine_threadsafe(coro, loop)

        scope = f"user {user_id}" if user_id else "all users"
        logger.info(f"Submitted rebuild_fts_index task {task_id} for {scope}")
        return task_id

    async def _run_rebuild_fts(self, task_id: str, user_id: str | None) -> None:
        """Run the FTS index rebuild task."""
        from kemi import Memory

        with self._lock:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = time.time()
            self._running_count += 1

        try:
            mem = Memory()

            # Rebuild FTS index
            if hasattr(mem._store, "rebuild_fts_index"):
                # Pass user_id through so per-user rebuilds only touch that
                # user's FTS rows instead of rebuilding the whole index.
                count = await asyncio.to_thread(
                    mem._store.rebuild_fts_index,
                    user_id,
                )

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

        except Exception as e:
            logger.error(f"Failed rebuild_fts_index task {task_id}: {e}")
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = time.time()

        finally:
            with self._lock:
                self._running_count -= 1

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

        Stops accepting new tasks and closes the background event loop.
        """
        if self._loop is not None:
            # Schedule loop stop
            self._loop.call_soon_threadsafe(self._loop.stop)
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

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Note: Running tasks cannot be cancelled mid-execution.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if cancelled, False if not found or already running.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.FAILED
                task.error = "Cancelled by user"
                task.completed_at = time.time()
                return True
            return False

    def submit_ttl_sweep(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
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
        with self._lock:
            if self._running_count >= self._max_concurrent:
                raise RuntimeError(
                    f"Max concurrent tasks ({self._max_concurrent}) reached. "
                    "Wait for a task to complete before submitting more."
                )

        task_id = str(uuid.uuid4())
        created_at = time.time()

        with self._lock:
            self._tasks[task_id] = TaskResult(
                task_id=task_id,
                task_type=TaskType.TTL_SWEEP,
                status=TaskStatus.PENDING,
                created_at=created_at,
            )
            self._cleanup_old_tasks()

        loop = self._get_loop()
        coro = self._run_ttl_sweep(task_id, user_id, namespace)
        asyncio.run_coroutine_threadsafe(coro, loop)

        scope = f"user {user_id}" if user_id else "all users"
        logger.info(f"Submitted ttl_sweep task {task_id} for {scope}")
        return task_id

    async def _run_ttl_sweep(
        self,
        task_id: str,
        user_id: str | None,
        namespace: str | None,
    ) -> None:
        """Run the TTL sweep task."""
        from kemi import Memory

        with self._lock:
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = time.time()
            self._running_count += 1

        try:
            mem = Memory()
            deleted = await asyncio.to_thread(
                mem.prune_expired,
                user_id,
                namespace,
            )
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

        except Exception as e:
            logger.error(f"Failed ttl_sweep task {task_id}: {e}")
            with self._lock:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                self._tasks[task_id].completed_at = time.time()

        finally:
            with self._lock:
                self._running_count -= 1


# Global task manager instance
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get or create the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        max_concurrent = int(__import__("os").environ.get("KEMI_MAX_BACKGROUND_TASKS", "3"))
        _task_manager = BackgroundTaskManager(max_concurrent_tasks=max_concurrent)
    return _task_manager
