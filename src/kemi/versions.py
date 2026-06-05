"""Memory versioning and undo: keep a history of memory edits and support rollback.

Every call to :meth:`Memory.update` can optionally be recorded in a versions
table. Users can then:
- List all past versions of a memory
- Preview any past version
- Roll back to a previous version
- Diff two versions to see what changed

This is useful for:
- Debugging: understand how a memory evolved over time
- Undo: revert to a known-good state after accidental edits
- Audit: track when and how memory content changed

Schema: a separate ``memory_versions`` table stores snapshots of memory
fields before each update. The current state is always in ``memory_versions``
with ``version = current_version``; older snapshots have ``version < current_version``.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kemi.models import MemoryObject, MemorySource, MemoryType

__all__ = [
    "MemoryVersionStore",
    "VersionSnapshot",
    "RollbackResult",
    "diff_memories",
]

logger = logging.getLogger(__name__)


def _pack_embedding(embedding: list[float] | None) -> bytes | None:
    """Pack a list of floats into 8 bytes per float for exact round-trip.

    Float32 is too imprecise for values like ``0.1`` (``0.10000000149...``),
    so we store as little-endian float64 instead.
    """
    if not embedding:
        return None
    return struct.pack(f"<{len(embedding)}d", *embedding)


def _unpack_embedding(blob: bytes | None) -> list[float] | None:
    """Unpack a float64 blob back into a list of Python floats."""
    if not blob:
        return None
    if len(blob) % 8 != 0:
        # Fall back to float32 in case an older row was written with the
        # original 4-byte-per-float encoding.
        if len(blob) % 4 == 0:
            return list(struct.unpack(f"<{len(blob) // 4}f", blob))
        return None
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class VersionSnapshot:
    """A snapshot of a memory at a point in time."""

    version: int  # version number (1 = original, increments per edit)
    memory_id: str
    content: str
    embedding: list[float] | None
    importance: float
    metadata: dict[str, Any]
    tags: list[str]
    memory_type: str
    confidence: float
    session_id: str | None
    namespace: str
    source: str
    changed_at: datetime
    changed_by: str | None  # "update", "import", "consolidate", etc.


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    memory_id: str
    from_version: int
    to_version: int
    rolled_back_at: datetime


@dataclass
class DiffResult:
    """Diff between two memory versions."""

    memory_id: str
    from_version: int
    to_version: int
    field_changes: dict[str, tuple[Any, Any]]  # field → (old, new)


# ---------------------------------------------------------------------------
# Version store (stored alongside the main SQLite adapter)
# ---------------------------------------------------------------------------

_VERSION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS memory_versions (
    memory_id      TEXT NOT NULL,
    version        INTEGER NOT NULL,
    content        TEXT NOT NULL,
    embedding      BLOB,
    importance     REAL NOT NULL DEFAULT 0.5,
    metadata       TEXT NOT NULL DEFAULT '{}',
    tags           TEXT NOT NULL DEFAULT '[]',
    memory_type    TEXT NOT NULL DEFAULT 'episodic',
    confidence     REAL NOT NULL DEFAULT 1.0,
    session_id     TEXT,
    namespace      TEXT NOT NULL DEFAULT 'default',
    source         TEXT NOT NULL DEFAULT 'user_stated',
    changed_at     TEXT NOT NULL,
    changed_by     TEXT,
    PRIMARY KEY (memory_id, version)
);
CREATE INDEX IF NOT EXISTS idx_versions_memory
    ON memory_versions(memory_id, version DESC);
"""

_CHANGE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS memory_change_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id      TEXT NOT NULL,
    from_version   INTEGER NOT NULL,
    to_version     INTEGER NOT NULL,
    changed_at     TEXT NOT NULL,
    changed_by     TEXT,
    field          TEXT NOT NULL,
    old_value      TEXT,
    new_value      TEXT
);
"""


class MemoryVersionStore:
    """Manages memory version history and rollback operations.

    Uses separate SQLite tables (``memory_versions`` and
    ``memory_change_log``) within the same database as the main
    memory store.

    Usage::

        vs = MemoryVersionStore(db_path="~/.kemi/memories.db")
        vs.record_version(memory_obj, changed_by="update")
        snapshots = vs.list_versions("mem-123")
        result = vs.rollback("mem-123", target_version=2)
    """

    def __init__(self, db_path: str | None = None) -> None:
        import os
        from pathlib import Path
        import sqlite3

        if db_path is None:
            db_path = os.path.join(os.path.expanduser("~"), ".kemi", "memories.db")
        self._db_path = str(Path(db_path).expanduser())
        self._ensure_tables()

    def _get_connection(self) -> Any:
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        conn = self._get_connection()
        try:
            conn.executescript(_VERSION_TABLE_DDL)
            conn.executescript(_CHANGE_TABLE_DDL)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def _next_version_number(
        self,
        conn: Any,
        memory_id: str,
        memory: MemoryObject,
    ) -> int:
        """Compute the next version number for a memory.

        Honours the caller's ``memory.version`` by default so non-contiguous,
        caller-specified version numbers (e.g. the rollback helper writing
        at a chosen position) are preserved. Falls back to
        ``MAX(version) + 1`` only when the supplied number would collide
        with an existing row, which prevents the ``UNIQUE`` constraint
        failure that occurs when concurrent writers or a caller that
        forgot to increment ``memory.version`` race the same
        ``(memory_id, version)`` primary key.
        """
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(version) FROM memory_versions WHERE memory_id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        current_max = row[0] if row and row[0] is not None else 0

        # If the caller's version is unused, respect it.
        cursor.execute(
            "SELECT 1 FROM memory_versions WHERE memory_id = ? AND version = ? LIMIT 1",
            (memory_id, memory.version),
        )
        if cursor.fetchone() is None:
            return memory.version

        # Otherwise advance to the next free number.
        return current_max + 1

    def record_version(
        self,
        memory: MemoryObject,
        *,
        changed_by: str = "update",
    ) -> int:
        """Record a new version snapshot of a memory.

        Uses the caller's ``memory.version`` when it advances the sequence.
        If the supplied version number would collide with an existing row
        (e.g. concurrent writes, or the caller hasn't incremented
        ``memory.version``), the next available version number is used
        automatically and written back to ``memory.version`` so subsequent
        calls see the correct value.

        Args:
            memory: The current MemoryObject to snapshot.
            changed_by: Label describing what operation triggered this snapshot
                (e.g., "update", "import", "consolidate").

        Returns:
            The version number that was written.
        """
        import json

        conn = self._get_connection()
        try:
            # BEGIN IMMEDIATE acquires a RESERVED lock for the duration of
            # the transaction. This serialises concurrent record_version
            # calls for the same memory_id, so the MAX(version)+1 read and
            # INSERT below cannot race with another writer.
            conn.execute("BEGIN IMMEDIATE")
            try:
                next_version = self._next_version_number(conn, memory.memory_id, memory)
                memory.version = next_version

                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO memory_versions
                        (memory_id, version, content, embedding, importance,
                         metadata, tags, memory_type, confidence, session_id,
                         namespace, source, changed_at, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.memory_id,
                        next_version,
                        memory.content,
                        _pack_embedding(memory.embedding),
                        memory.importance,
                        json.dumps(memory.metadata or {}),
                        json.dumps(memory.tags or []),
                        memory.memory_type.value
                        if hasattr(memory.memory_type, "value")
                        else str(memory.memory_type),
                        memory.confidence,
                        memory.session_id,
                        memory.namespace,
                        memory.source.value
                        if hasattr(memory.source, "value")
                        else str(memory.source),
                        datetime.now(timezone.utc).isoformat(),
                        changed_by,
                    ),
                )
                conn.commit()
                return next_version
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def record_before_update(
        self,
        memory_before: MemoryObject,
        memory_after: MemoryObject,
        *,
        changed_by: str = "update",
    ) -> int:
        """Record both the pre-update and post-update snapshots atomically.

        Records the pre-update snapshot at its current version number and
        the post-update snapshot at the next available version number.
        Both inserts run inside a single ``BEGIN IMMEDIATE`` transaction so
        they cannot interleave with another writer.

        Args:
            memory_before: State of memory before the update.
            memory_after: State of memory after the update.
            changed_by: Operation label.

        Returns:
            The new version number of memory_after.
        """
        import json

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.cursor()

                # Pre-update snapshot: store at its current version number
                # (which the caller typically set to the live version). If a
                # row already exists at that (memory_id, version) we still
                # upsert via INSERT OR REPLACE — re-recording a pre-update
                # snapshot for the same version is idempotent.
                pre_version = memory_before.version
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO memory_versions
                        (memory_id, version, content, embedding, importance,
                         metadata, tags, memory_type, confidence, session_id,
                         namespace, source, changed_at, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_before.memory_id,
                        pre_version,
                        memory_before.content,
                        _pack_embedding(memory_before.embedding),
                        memory_before.importance,
                        json.dumps(memory_before.metadata or {}),
                        json.dumps(memory_before.tags or []),
                        memory_before.memory_type.value
                        if hasattr(memory_before.memory_type, "value")
                        else str(memory_before.memory_type),
                        memory_before.confidence,
                        memory_before.session_id,
                        memory_before.namespace,
                        memory_before.source.value
                        if hasattr(memory_before.source, "value")
                        else str(memory_before.source),
                        datetime.now(timezone.utc).isoformat(),
                        "pre-" + changed_by,
                    ),
                )

                # Post-update snapshot: take the next available version
                # number so the primary key never collides.
                post_version = self._next_version_number(
                    conn, memory_after.memory_id, memory_after
                )
                memory_after.version = post_version
                cursor.execute(
                    """
                    INSERT INTO memory_versions
                        (memory_id, version, content, embedding, importance,
                         metadata, tags, memory_type, confidence, session_id,
                         namespace, source, changed_at, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_after.memory_id,
                        post_version,
                        memory_after.content,
                        _pack_embedding(memory_after.embedding),
                        memory_after.importance,
                        json.dumps(memory_after.metadata or {}),
                        json.dumps(memory_after.tags or []),
                        memory_after.memory_type.value
                        if hasattr(memory_after.memory_type, "value")
                        else str(memory_after.memory_type),
                        memory_after.confidence,
                        memory_after.session_id,
                        memory_after.namespace,
                        memory_after.source.value
                        if hasattr(memory_after.source, "value")
                        else str(memory_after.source),
                        datetime.now(timezone.utc).isoformat(),
                        changed_by,
                    ),
                )
                conn.commit()
                return post_version
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def prune_versions(self, memory_id: str, keep_count: int) -> int:
        """Prune old versions, keeping only the most recent N versions.

        Args:
            memory_id: Memory whose versions to prune.
            keep_count: Number of recent versions to keep.

        Returns:
            Number of versions deleted.
        """
        if keep_count <= 0:
            return 0
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT version FROM memory_versions
                    WHERE memory_id = ?
                    ORDER BY version DESC
                    """,
                    (memory_id,),
                )
                rows = cursor.fetchall()
                if len(rows) <= keep_count:
                    conn.commit()
                    return 0
                versions_to_delete = [r[0] for r in rows[keep_count:]]
                placeholders = ",".join("?" * len(versions_to_delete))
                cursor.execute(
                    f"DELETE FROM memory_versions "
                    f"WHERE memory_id = ? AND version IN ({placeholders})",
                    [memory_id] + versions_to_delete,
                )
                deleted = cursor.rowcount
                conn.commit()
                return deleted
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def verify_sequential_versions(self, memory_id: str) -> bool:
        """Verify that version numbers for a memory form a contiguous sequence.

        Returns True if versions are 1, 2, 3, ... with no gaps. Useful as an
        integrity check after concurrent writes.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM memory_versions WHERE memory_id = ? ORDER BY version ASC",
                (memory_id,),
            )
            versions = [r[0] for r in cursor.fetchall()]
            return versions == list(range(1, len(versions) + 1))
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def list_versions(self, memory_id: str) -> list[VersionSnapshot]:
        """Return all version snapshots for a memory, newest first.

        Args:
            memory_id: ID of the memory.

        Returns:
            List of VersionSnapshot objects, sorted by version descending.
        """
        import json

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT memory_id, version, content, embedding, importance,
                       metadata, tags, memory_type, confidence, session_id,
                       namespace, source, changed_at, changed_by
                FROM memory_versions
                WHERE memory_id = ?
                ORDER BY version DESC
                """,
                (memory_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_snapshot(row) for row in rows]
        finally:
            conn.close()

    def get_version(self, memory_id: str, version: int) -> VersionSnapshot | None:
        """Retrieve a specific version of a memory.

        Args:
            memory_id: ID of the memory.
            version: Version number to retrieve.

        Returns:
            VersionSnapshot if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT memory_id, version, content, embedding, importance,
                       metadata, tags, memory_type, confidence, session_id,
                       namespace, source, changed_at, changed_by
                FROM memory_versions
                WHERE memory_id = ? AND version = ?
                """,
                (memory_id, version),
            )
            row = cursor.fetchone()
            return self._row_to_snapshot(row) if row else None
        finally:
            conn.close()

    def get_latest_version_number(self, memory_id: str) -> int | None:
        """Return the highest version number for a memory, or None if no versions exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(version) FROM memory_versions WHERE memory_id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    def rollback(
        self,
        memory_id: str,
        target_version: int,
        store: Any,
        *,
        changed_by: str = "rollback",
    ) -> RollbackResult | None:
        """Roll a memory back to a specific version.

        Reconstructs the MemoryObject from the version snapshot and
        writes it back to the storage adapter. The new state is recorded
        as a fresh, monotonically-increasing version (MAX(version) + 1)
        rather than reusing the old version number, preserving the
        full audit trail.

        Args:
            memory_id: ID of the memory to roll back.
            target_version: Version number to roll back to.
            store: The StorageAdapter to write the rolled-back memory to.
            changed_by: Label for the rollback operation.

        Returns:
            RollbackResult if successful, None if target version not found.
        """
        snapshot = self.get_version(memory_id, target_version)
        if snapshot is None:
            return None

        # Fetch current memory to preserve user_id, lifecycle_state,
        # and to read the next available version number for the store row.
        current = store.get(memory_id)
        if current is None:
            return None

        # Reconstruct MemoryObject from snapshot
        rolled_back = MemoryObject(
            memory_id=memory_id,
            user_id=current.user_id,
            content=snapshot.content,
            embedding=snapshot.embedding,
            score=0.0,
            created_at=current.created_at,
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource(snapshot.source),
            importance=snapshot.importance,
            lifecycle_state=current.lifecycle_state,
            metadata=json.loads(snapshot.metadata) if isinstance(snapshot.metadata, str) else (snapshot.metadata or {}),
            embedding_dim=len(snapshot.embedding) if snapshot.embedding else None,
            tags=json.loads(snapshot.tags) if isinstance(snapshot.tags, str) else (snapshot.tags or []),
            confidence=snapshot.confidence,
            memory_type=MemoryType(snapshot.memory_type),
            session_id=snapshot.session_id,
            namespace=snapshot.namespace,
            version=current.version,
        )

        # Compute the new version number for the rollback row BEFORE writing
        # to the store. We use MAX(version) + 1 so the audit trail is
        # contiguous and the row's version field reflects the new state.
        new_version = self.get_latest_version_number(memory_id)
        if new_version is None:
            new_version = current.version
        else:
            new_version = new_version + 1
        rolled_back.version = new_version

        # Write the restored content back to the store. The storage adapter
        # performs an INSERT OR REPLACE keyed on memory_id, so the version
        # value on the row is what we pass in.
        store.update(rolled_back)

        # Record the rolled-back state as a version snapshot so the history
        # is preserved. record_version auto-increments — we use INSERT OR
        # REPLACE via the connection so it doesn't double-increment.
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO memory_versions
                        (memory_id, version, content, embedding, importance,
                         metadata, tags, memory_type, confidence, session_id,
                         namespace, source, changed_at, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rolled_back.memory_id,
                        new_version,
                        rolled_back.content,
                        _pack_embedding(rolled_back.embedding),
                        rolled_back.importance,
                        json.dumps(rolled_back.metadata or {}),
                        json.dumps(rolled_back.tags or []),
                        rolled_back.memory_type.value
                        if hasattr(rolled_back.memory_type, "value")
                        else str(rolled_back.memory_type),
                        rolled_back.confidence,
                        rolled_back.session_id,
                        rolled_back.namespace,
                        rolled_back.source.value
                        if hasattr(rolled_back.source, "value")
                        else str(rolled_back.source),
                        datetime.now(timezone.utc).isoformat(),
                        changed_by,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

        # Sync the memory object's version field so callers see the new value
        current.version = new_version
        rolled_back.version = new_version

        logger.info(
            f"Rolled back memory {memory_id} from version {snapshot.version} "
            f"to version {new_version}"
        )

        return RollbackResult(
            memory_id=memory_id,
            from_version=snapshot.version,
            to_version=new_version,
            rolled_back_at=datetime.now(timezone.utc),
        )

    # -------------------------------------------------------------------------
    # Diff
    # -------------------------------------------------------------------------

    def diff(
        self,
        memory_id: str,
        from_version: int,
        to_version: int,
    ) -> DiffResult | None:
        """Show what changed between two versions of a memory.

        Args:
            memory_id: ID of the memory.
            from_version: Starting version number.
            to_version: Ending version number.

        Returns:
            DiffResult listing all field changes, or None if either version not found.
        """
        snap_from = self.get_version(memory_id, from_version)
        snap_to = self.get_version(memory_id, to_version)

        if snap_from is None or snap_to is None:
            return None

        changes: dict[str, tuple[Any, Any]] = {}
        fields = [
            ("content", snap_from.content, snap_to.content),
            ("importance", snap_from.importance, snap_to.importance),
            ("metadata", snap_from.metadata, snap_to.metadata),
            ("tags", snap_from.tags, snap_to.tags),
            ("memory_type", snap_from.memory_type, snap_to.memory_type),
            ("confidence", snap_from.confidence, snap_to.confidence),
            ("session_id", snap_from.session_id, snap_to.session_id),
        ]

        for field_name, old_val, new_val in fields:
            old_normalized = self._normalize_field_value(old_val)
            new_normalized = self._normalize_field_value(new_val)
            if old_normalized != new_normalized:
                changes[field_name] = (old_val, new_val)

        return DiffResult(
            memory_id=memory_id,
            from_version=from_version,
            to_version=to_version,
            field_changes=changes,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _row_to_snapshot(self, row) -> VersionSnapshot:
        import json

        return VersionSnapshot(
            memory_id=row["memory_id"],
            version=row["version"],
            content=row["content"],
            embedding=_unpack_embedding(row["embedding"]),
            importance=row["importance"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            tags=json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"],
            memory_type=row["memory_type"],
            confidence=row["confidence"],
            session_id=row["session_id"],
            namespace=row["namespace"],
            source=row["source"],
            changed_at=datetime.fromisoformat(row["changed_at"]),
            changed_by=row["changed_by"],
        )

    def _normalize_field_value(self, value: Any) -> str:
        """Normalize a field value for comparison."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value)


def diff_memories(
    before: MemoryObject,
    after: MemoryObject,
) -> DiffResult:
    """Diff two memory objects and return field-level changes.

    Convenience function that doesn't need a version store —
    useful for previewing what an update would change.

    Args:
        before: Memory state before the change.
        after: Memory state after the change.

    Returns:
        DiffResult with all changed fields.
    """
    changes: dict[str, tuple[Any, Any]] = {}
    for field in ("content", "importance", "confidence", "tags", "metadata", "memory_type"):
        old_val = getattr(before, field, None)
        new_val = getattr(after, field, None)
        if old_val != new_val:
            changes[field] = (old_val, new_val)

    return DiffResult(
        memory_id=before.memory_id,
        from_version=before.version,
        to_version=after.version,
        field_changes=changes,
    )


# ---------------------------------------------------------------------------
# Attach versioning to Memory core (patch-in hooks)
# ---------------------------------------------------------------------------

def enable_versioning(memory: Any, db_path: str | None = None) -> MemoryVersionStore:
    """Enable memory versioning on a Memory instance.

    Returns a MemoryVersionStore that can be used to record, list,
    and rollback memory versions.

    Args:
        memory: A Memory instance to enable versioning on.
        db_path: Optional path to the SQLite DB (defaults to ~.kemi/memories.db).

    Usage::

        vs = enable_versioning(memory)
        vs.record_version(updated_memory, changed_by="update")
        snapshots = vs.list_versions("mem-123")
        vs.rollback("mem-123", target_version=2, store=memory._store)
    """
    if db_path is None:
        import os

        db_path = os.path.join(os.path.expanduser("~"), ".kemi", "memories.db")

    return MemoryVersionStore(db_path=db_path)

