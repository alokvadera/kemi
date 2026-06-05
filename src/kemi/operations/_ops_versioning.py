"""Versioning operations: configure_versioning, get_history, diff_versions, rollback_memory.

These free functions are called by the corresponding ``Memory`` methods.
Public API is unchanged — the ``Memory`` class still exposes
``memory.configure_versioning()``, ``memory.get_history()``, etc.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from kemi.versions import DiffResult, MemoryVersionStore, RollbackResult, VersionSnapshot

if TYPE_CHECKING:
    from kemi._memory_impl import Memory

logger = logging.getLogger(__name__)


def configure(
    memory: "Memory",
    db_path: str | None,
    max_versions_per_memory: int,
    auto_prune_versions: bool,
) -> None:
    """Enable memory version history tracking."""
    if db_path is None:
        try:
            db_path = memory._store._db_path  # type: ignore[attr-defined]
        except AttributeError:
            logger.warning("Cannot determine database path for version store")
            return

    try:
        memory._version_store = MemoryVersionStore(db_path=db_path)
        memory._max_versions_per_memory = max_versions_per_memory
        memory._auto_prune_versions = auto_prune_versions
        logger.info(
            "Memory versioning enabled (max %d versions per memory)",
            max_versions_per_memory,
        )
    except (OSError, sqlite3.DatabaseError) as e:
        logger.warning("Failed to initialise version store: %s", e)


def get_store(memory: "Memory") -> MemoryVersionStore:
    """Get or lazily initialise the version store."""
    if memory._version_store is None:
        db_path: str | None
        try:
            db_path = memory._store._db_path  # type: ignore[attr-defined]
        except AttributeError:
            # No persistent path — use in-memory DB so in-process operations work.
            db_path = ":memory:"
        memory._version_store = MemoryVersionStore(db_path=db_path)
    return memory._version_store


def get_history(
    memory: "Memory",
    memory_id: str,
    limit: int = 100,
) -> list[VersionSnapshot]:
    """Return version history for a memory, newest first."""
    try:
        vs = get_store(memory)
        snapshots = vs.list_versions(memory_id)
        return snapshots[:limit]
    except (OSError, sqlite3.DatabaseError, AttributeError):
        logger.debug("get_history failed for %s", memory_id, exc_info=True)
        return []


def diff(
    memory: "Memory",
    memory_id: str,
    from_version: int,
    to_version: int,
) -> DiffResult | None:
    """Show field-level differences between two versions of a memory."""
    try:
        vs = get_store(memory)
        return vs.diff(memory_id, from_version, to_version)
    except (OSError, sqlite3.DatabaseError, AttributeError):
        logger.debug(
            "diff_versions failed for %s (%d -> %d)",
            memory_id,
            from_version,
            to_version,
            exc_info=True,
        )
        return None


def rollback(
    memory: "Memory",
    memory_id: str,
    target_version: int,
) -> RollbackResult | None:
    """Roll a memory back to a previous version."""
    try:
        vs = get_store(memory)
        return vs.rollback(
            memory_id=memory_id,
            target_version=target_version,
            store=memory._store,
        )
    except (OSError, sqlite3.DatabaseError, AttributeError, ValueError):
        logger.debug(
            "rollback_memory failed for %s to v%d",
            memory_id,
            target_version,
            exc_info=True,
        )
        return None


def auto_prune(memory: "Memory", memory_id: str) -> None:
    """Prune old versions for a memory if auto-prune is enabled."""
    if not memory._auto_prune_versions or memory._version_store is None:
        return
    try:
        vs = get_store(memory)
        snapshots = vs.list_versions(memory_id)
        if len(snapshots) > memory._max_versions_per_memory:
            version_ids = [
                s.version for s in snapshots[memory._max_versions_per_memory:]
            ]
            conn = vs._get_connection()
            try:
                for v in version_ids:
                    conn.execute(
                        "DELETE FROM memory_versions "
                        "WHERE memory_id = ? AND version = ?",
                        (memory_id, v),
                    )
                conn.commit()
            finally:
                conn.close()
            logger.info(
                "Pruned %d old versions for memory %s",
                len(version_ids),
                memory_id,
            )
    except (OSError, sqlite3.DatabaseError, AttributeError):
        logger.debug("Failed to prune versions for %s", memory_id, exc_info=True)
