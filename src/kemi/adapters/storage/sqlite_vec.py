"""SQLite storage adapter with ANN vector index via sqlite-vec.

Same database file as the standard SQLite adapter, but search uses an
HNSW approximate nearest neighbor index for sub-millisecond vector search
instead of brute-force cosine similarity.

Install: pip install kemi[vec]

Falls back to brute-force if sqlite-vec is not installed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.memory.model import LifecycleState, MemoryObject

if TYPE_CHECKING:
    from kemi.infra.encryption import EncryptionConfig

logger = logging.getLogger(__name__)

try:
    import sqlite3

    import sqlite_vec as _sqlite_vec

    # Verify the extension can actually be loaded (not just imported).
    # Some environments have the Python package but lack the native .so.
    _test_conn = sqlite3.connect(":memory:")
    _test_conn.enable_load_extension(True)
    _sqlite_vec.load(_test_conn)
    _test_conn.enable_load_extension(False)
    _test_conn.close()
    _SQLITE_VEC_AVAILABLE = True
except Exception:  # pragma: no cover
    _sqlite_vec = None
    _SQLITE_VEC_AVAILABLE = False


def _embedding_to_json(embedding: list[float]) -> str:
    """Serialize embedding to a JSON string for vec0."""
    return json.dumps(embedding)


class SQLiteVecStorageAdapter(SQLiteStorageAdapter):
    """SQLite storage with HNSW vector index via sqlite-vec.

    Uses the same SQLite database file and ``memories`` table as the
    standard adapter, but adds a ``memories_vec`` vec0 virtual table
    for fast approximate nearest neighbor search.

    On ``search()``, the adapter queries the ANN index instead of loading
    every row and computing cosine similarity in Python.  All other methods
    (get, count, export, etc.) delegate to the parent.

    If ``sqlite-vec`` is not installed, the adapter silently falls back
    to brute-force (same as ``SQLiteStorageAdapter``).

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    embedding_dim : int
        Dimension of embeddings (e.g. 384 for fastembed, 1536 for OpenAI).
        Must match the dimension in use.  Default 384.
    lazy : bool
        If True, defer HNSW index updates until search time.
        Inserts are faster (no vec0 index maintenance), but the first
        search after a batch of inserts will be slightly slower while
        the index catches up.  Default False.
    """

    CURRENT_VERSION = 7

    def __init__(
        self,
        db_path: str = "kemi.db",
        embedding_dim: int = 384,
        lazy: bool = False,
        encryption: EncryptionConfig | None = None,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._lazy = lazy
        self._vec_loaded = False
        self._pending_count: int | None = None
        super().__init__(db_path, encryption=encryption)

    def is_lazy(self) -> bool:
        """Returns True if deferred HNSW insertion is enabled."""
        return self._lazy

    # ── Connection (load vec0 extension once on the shared conn) ─────

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        if _SQLITE_VEC_AVAILABLE:
            # Load the vec0 extension per-connection (connections are
            # thread-local, so each new connection needs its own load).
            if not getattr(self._local, "vec_loaded", False):
                try:
                    conn.enable_load_extension(True)
                    if _sqlite_vec is not None:
                        _sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                    self._local.vec_loaded = True
                except (sqlite3.OperationalError, AttributeError):  # pragma: no cover
                    pass
        return conn

    # ── Schema ──────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user_stated',
                    importance REAL NOT NULL DEFAULT 0.5,
                    lifecycle_state TEXT NOT NULL DEFAULT 'active',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    tags TEXT NOT NULL DEFAULT '',
                    vec_rowid INTEGER,                    confidence REAL NOT NULL DEFAULT 1.0,
                    memory_type TEXT NOT NULL DEFAULT 'episodic',
                    session_id TEXT,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    version INTEGER NOT NULL DEFAULT 1,
                    agent_id TEXT,
                    run_id TEXT,
                    app_id TEXT,
                    expires_at TEXT
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle_state)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_lifecycle "
                "ON memories(user_id, lifecycle_state)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")

            # Pending vec0 inserts for lazy mode
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memories_vec_pending (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL
                )"""
            )

            self._init_vec_table(conn)
            self._run_migrations(conn)
            conn.commit()

    def _init_vec_table(self, conn: sqlite3.Connection) -> None:
        if not _SQLITE_VEC_AVAILABLE:
            return

        dim = self._embedding_dim
        try:
            conn.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                    USING vec0(
                        embedding float[{dim}],
                        memory_id text,
                        user_id text,
                        lifecycle_state text
                    )"""
            )
            self._vec_loaded = True
        except sqlite3.OperationalError:  # pragma: no cover
            pass

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        current = self._get_schema_version(conn)
        if current >= self.CURRENT_VERSION:
            return

        if current < 2:
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (2)")

        if current < 3:
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN vec_rowid INTEGER")
            except sqlite3.OperationalError:
                pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (3)")

        if current < 4:
            try:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS memories_vec_pending (
                        memory_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        lifecycle_state TEXT NOT NULL
                    )"""
                )
            except sqlite3.OperationalError:
                pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (4)")

        if current < 5:
            for col, dtype in [
                ("confidence", "REAL NOT NULL DEFAULT 1.0"),
                ("memory_type", "TEXT NOT NULL DEFAULT 'episodic'"),
                ("session_id", "TEXT"),
                ("namespace", "TEXT NOT NULL DEFAULT 'default'"),
                ("version", "INTEGER NOT NULL DEFAULT 1"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (5)")

        if current < 6:
            for col, dtype in [
                ("agent_id", "TEXT"),
                ("run_id", "TEXT"),
                ("app_id", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (6)")

        if current < 7:
            # TTL: add expires_at column and index
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_expires_at "
                    "ON memories(expires_at)"
                )
            except sqlite3.OperationalError:
                pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (7)")

    # ── Store ───────────────────────────────────────────────────

    def store(self, memory: MemoryObject) -> None:
        with self._get_connection() as conn:
            # Preserve existing vec_rowid from DB if not set in metadata.
            # Without this, INSERT OR REPLACE overwrites vec_rowid to NULL,
            # causing _flush_pending to create duplicate vec0 entries.
            if self._lazy and self._vec_loaded and memory.metadata.get("_vec_rowid") is None:
                existing = conn.execute(
                    "SELECT vec_rowid FROM memories WHERE memory_id = ?",
                    (memory.memory_id,),
                ).fetchone()
                if existing and existing["vec_rowid"] is not None:
                    memory.metadata["_vec_rowid"] = existing["vec_rowid"]

            row = self._memory_to_row(memory)
            conn.execute(
                """INSERT OR REPLACE INTO memories
                (memory_id, user_id, content, embedding, embedding_dim,
                 created_at, last_accessed_at, source, importance,
                 lifecycle_state, metadata, tags, vec_rowid,
                 confidence, memory_type, session_id, namespace, version,
                 agent_id, run_id, app_id, expires_at)
                VALUES (:memory_id, :user_id, :content, :embedding, :embedding_dim,
                        :created_at, :last_accessed_at, :source, :importance,
                        :lifecycle_state, :metadata, :tags, :vec_rowid,
                        :confidence, :memory_type, :session_id, :namespace, :version,
                        :agent_id, :run_id, :app_id, :expires_at)""",
                row,
            )

            if not self._vec_ready() or memory.embedding is None:
                return

            if self._lazy:
                self._store_pending_on_conn(conn, memory)
            else:
                self._store_vec_direct_on_conn(conn, memory)

    def _store_pending_on_conn(self, conn: sqlite3.Connection, memory: MemoryObject) -> None:
        """Store embedding in the pending table (no HNSW index update)."""
        assert memory.embedding is not None
        embedding_json = _embedding_to_json(memory.embedding)
        conn.execute(
            """INSERT OR REPLACE INTO memories_vec_pending
               (memory_id, user_id, embedding, lifecycle_state)
               VALUES (?, ?, ?, ?)""",
            (
                memory.memory_id,
                memory.user_id,
                embedding_json,
                memory.lifecycle_state.value,
            ),
        )
        # Invalidate cached pending count so next _has_pending() is accurate
        self._pending_count = None

    def _store_vec_direct_on_conn(self, conn: sqlite3.Connection, memory: MemoryObject) -> None:
        """Insert/update the vector in the vec0 HNSW index directly.

        Since the HNSW index is built incrementally, this is slower
        than lazy insertion but keeps the index always up-to-date.
        """
        assert memory.embedding is not None
        embedding_json = _embedding_to_json(memory.embedding)
        existing_row = conn.execute(
            "SELECT vec_rowid FROM memories WHERE memory_id = ?",
            (memory.memory_id,),
        ).fetchone()
        vec_rowid: int | None = (
            existing_row[0] if existing_row and existing_row[0] is not None else None
        )

        if vec_rowid is not None:
            conn.execute(
                "UPDATE memories_vec SET embedding=?, user_id=?, lifecycle_state=? WHERE rowid=?",
                (embedding_json, memory.user_id, memory.lifecycle_state.value, vec_rowid),
            )
        else:
            cursor = conn.execute(
                """INSERT INTO memories_vec (embedding, memory_id, user_id, lifecycle_state)
                   VALUES (?, ?, ?, ?)""",
                (embedding_json, memory.memory_id, memory.user_id, memory.lifecycle_state.value),
            )
            new_rowid = cursor.lastrowid
            if new_rowid is not None:
                conn.execute(
                    "UPDATE memories SET vec_rowid = ? WHERE memory_id = ?",
                    (new_rowid, memory.memory_id),
                )
                memory.metadata["_vec_rowid"] = new_rowid

    # ── Flush pending → vec0 ───────────────────────────────────

    def _count_pending(self) -> int:
        """Return how many memories are waiting in the pending table."""
        if self._pending_count is not None:
            return self._pending_count
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM memories_vec_pending").fetchone()
            self._pending_count = row[0] if row else 0
        return self._pending_count

    def _has_pending(self) -> bool:
        return self._count_pending() > 0

    def _flush_pending(self) -> None:
        """Batch-insert all pending vectors into the vec0 HNSW index.

        During lazy mode the HNSW index is not updated per insert, so
        we batch all pending entries here in a single atomic transaction
        for efficiency.

        Handles re-flush gracefully: if a memory already has a vec_rowid
        (e.g. it was flushed before, then updated and re-stored), we
        UPDATE the existing vec0 row instead of creating a duplicate.
        """
        if not self._vec_ready():
            return
        if not self._has_pending():
            return

        count = self._pending_count or 0
        logger.info("Flushing %d pending vectors to ANN index…", count)

        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT p.memory_id, p.user_id, p.embedding, p.lifecycle_state, "
                "       m.vec_rowid "
                "FROM memories_vec_pending p "
                "LEFT JOIN memories m ON m.memory_id = p.memory_id"
            ).fetchall()

            for row in rows:
                mid = row["memory_id"]
                existing_vec_rowid = row["vec_rowid"]

                if existing_vec_rowid is not None:
                    conn.execute(
                        "UPDATE memories_vec SET embedding=?, user_id=?, lifecycle_state=? "
                        "WHERE rowid=?",
                        (
                            row["embedding"],
                            row["user_id"],
                            row["lifecycle_state"],
                            existing_vec_rowid,
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT INTO memories_vec
                               (embedding, memory_id, user_id, lifecycle_state)
                           VALUES (?, ?, ?, ?)""",
                        (row["embedding"], mid, row["user_id"], row["lifecycle_state"]),
                    )
                    result = conn.execute("SELECT last_insert_rowid()").fetchone()
                    new_rowid = result[0] if result else None
                    if new_rowid is not None:
                        conn.execute(
                            "UPDATE memories SET vec_rowid = ? WHERE memory_id = ?",
                            (new_rowid, mid),
                        )

            conn.execute("DELETE FROM memories_vec_pending")

        self._pending_count = 0
        logger.info("Flushed %d vectors to ANN index", count)

    # ── Search ──────────────────────────────────────────────

    def _vec_ready(self) -> bool:
        """True if vec0 is available and the current thread's connection has it loaded."""
        if not _SQLITE_VEC_AVAILABLE or not self._vec_loaded:
            return False
        return getattr(self._local, "vec_loaded", False)

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states_list = [s.value for s in lifecycle_filter]

        if not _SQLITE_VEC_AVAILABLE or not self._vec_loaded or not query_embedding:
            return super().search(
                user_id, query_embedding, top_k, lifecycle_filter, namespace, session_id
            )

        # Ensure the extension is loaded on this thread before using vec0.
        self._get_connection()
        if not getattr(self._local, "vec_loaded", False):
            return super().search(
                user_id, query_embedding, top_k, lifecycle_filter, namespace, session_id
            )
        if self._lazy and self._has_pending():
            self._flush_pending()
        return self._search_vec(user_id, query_embedding, top_k, states_list, namespace)

    def _search_vec(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        states_list: list[str],
        namespace: str = "default",
    ) -> list[MemoryObject]:
        """Search using the vec0 ANN index."""
        embedding_json = _embedding_to_json(query_embedding)
        placeholders = ",".join("?" * len(states_list))

        # Over-fetch from vec0 because namespace is filtered post-hoc.
        # vec0 doesn't index namespace, so we can't push it into the ANN query.
        # Multiplying by 3 is a heuristic; matches core.py's recall() which also
        # fetches top_k * 3 from the storage layer.
        fetch_k = top_k * 3

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""SELECT rowid, distance, memory_id
                    FROM memories_vec
                    WHERE embedding MATCH ?
                      AND user_id = ?
                      AND lifecycle_state IN ({placeholders})
                    ORDER BY distance
                    LIMIT ?""",
                [embedding_json, user_id] + states_list + [fetch_k],
            ).fetchall()

            if not rows:
                return []

            memory_ids = [r["memory_id"] for r in rows]
            distances = {r["memory_id"]: r["distance"] for r in rows}

            id_placeholders = ",".join("?" * len(memory_ids))
            memory_rows = conn.execute(
                f"SELECT * FROM memories WHERE memory_id IN ({id_placeholders})",
                memory_ids,
            ).fetchall()

            mem_map = {r["memory_id"]: r for r in memory_rows}
            results: list[MemoryObject] = []
            for mid in memory_ids:
                if mid not in mem_map:
                    continue
                mem = self._row_to_memory(mem_map[mid])
                # vec0 returns cosine distance in [0, 2]; convert to [0, 1] score
                distance = distances.get(mid, 0.0)
                mem.score = max(0.0, min(1.0, 1.0 - distance / 2.0))
                # Filter by namespace (vec0 doesn't support this in the query)
                if mem.namespace == namespace:
                    results.append(mem)
                    if len(results) >= top_k:
                        break

        return results

    # ── Delete ──────────────────────────────────────────────

    def delete_by_id(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            if self._vec_ready():
                row = conn.execute(
                    "SELECT vec_rowid FROM memories WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                if row and row["vec_rowid"] is not None:
                    conn.execute(
                        "DELETE FROM memories_vec WHERE rowid = ?",
                        (row["vec_rowid"],),
                    )

            conn.execute(
                "DELETE FROM memories_vec_pending WHERE memory_id = ?",
                (memory_id,),
            )
            self._pending_count = None

            cursor = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            return cursor.rowcount > 0

    def delete_by_user(self, user_id: str) -> int:
        with self._get_connection() as conn:
            if self._vec_ready():
                rows = conn.execute(
                    "SELECT vec_rowid FROM memories WHERE user_id = ? AND vec_rowid IS NOT NULL",
                    (user_id,),
                ).fetchall()
                for r in rows:
                    conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (r[0],))

            conn.execute(
                "DELETE FROM memories_vec_pending WHERE user_id = ?",
                (user_id,),
            )
            self._pending_count = None

            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            return cursor.rowcount

    # ── Update ──────────────────────────────────────────────────

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    # ── Row helpers ──────────────────────────────────────────────

    def _memory_to_row(self, memory: MemoryObject) -> dict[str, Any]:
        row = super()._memory_to_row(memory)
        vec_rowid = memory.metadata.get("_vec_rowid") if memory.metadata else None
        row["vec_rowid"] = vec_rowid
        return row

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryObject:
        mem = super()._row_to_memory(row)
        try:
            vec_rowid = row["vec_rowid"]
            if vec_rowid is not None:
                mem.metadata["_vec_rowid"] = vec_rowid
        except (IndexError, KeyError):  # pragma: no cover
            pass
        return mem

    # ── Utility ──────────────────────────────────────────────

    @classmethod
    def is_vec_available(cls) -> bool:
        """Returns True if sqlite-vec is installed and usable."""
        return _SQLITE_VEC_AVAILABLE
