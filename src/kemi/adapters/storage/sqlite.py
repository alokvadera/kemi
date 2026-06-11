from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from kemi.adapters.base import StorageAdapter
from kemi.memory import scoring
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType

if TYPE_CHECKING:
    from kemi.infra.encryption import EncryptionConfig

logger = logging.getLogger(__name__)


class SQLiteStorageAdapter(StorageAdapter):
    """SQLite storage adapter with WAL mode and thread-local connections.

    Embedding stored as BLOB (float32 bytes) for compactness.
    Schema version tracked in schema_version table.

    Thread-safety: Uses thread-local storage for connections, giving each
    thread its own connection. This avoids SQLite's thread-safety issues
    while allowing concurrent access from multiple threads.

    Encryption: When encryption config is provided, uses Fernet field-level
    encryption for content and metadata fields. Pass ``encryption=`` to
    ``__init__`` or set ``KEMI_ENCRYPTION_KEY`` / ``KEMI_ENCRYPTION_ENABLED``
    environment variables.
    """

    CURRENT_VERSION = 8

    @staticmethod
    def _parse_tags(tags_csv: str) -> list[str]:
        """Inverse of the CSV tag serialization in ``_row_to_memory``.

        Tags are joined with ``,`` and ``,`` inside tags is escaped as
        ``\\,``. A trailing empty string (from a trailing comma) is
        dropped.
        """
        if not tags_csv:
            return []
        return [t.replace("\\,", ",") for t in tags_csv.split(",") if t]

    def __init__(
        self,
        db_path: str = "kemi.db",
        encryption: EncryptionConfig | None = None,
    ) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()
        # Lazy import to avoid circular dependency at module level
        from kemi.infra.encryption import EncryptionConfig, FieldEncryptor

        if encryption is None:
            # Try environment-based config
            try:
                env_config = EncryptionConfig.from_env()
                self._field_encryptor = FieldEncryptor(env_config) if env_config.enabled else None
            except Exception:
                self._field_encryptor = None
        else:
            self._field_encryptor = FieldEncryptor(encryption) if encryption.enabled else None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a connection for the current thread.

        Each thread gets its own connection to avoid SQLite thread-safety
        issues. Connections are created on-demand and cached per-thread.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            logger.debug("Created new connection for thread %s", threading.current_thread().name)
        return self._local.conn  # type: ignore[no-any-return]

    @contextmanager
    def _transaction(self) -> Any:
        """Context manager for explicit transaction handling.

        Ensures that multiple operations are atomic. If an exception occurs
        within the context, the transaction is rolled back. Otherwise, it
        commits at context exit.

        Usage:
            with adapter._transaction() as conn:
                conn.execute(...)
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            logger.exception("Transaction failed, rolled back")
            raise

    def close(self) -> None:
        """Close the connection for the current thread.

        Note: This only closes the connection for the calling thread.
        Other threads will keep their own connections until they also
        call close() or are garbage collected.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:  # pragma: no cover
                pass
            self._local.conn = None

    @property
    def _shared_conn(self) -> sqlite3.Connection | None:
        """Backward-compat property for code/tests that expect _shared_conn."""
        return getattr(self._local, "conn", None)

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> SQLiteStorageAdapter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
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
                    confidence REAL NOT NULL DEFAULT 1.0,
                    memory_type TEXT NOT NULL DEFAULT 'episodic',
                    session_id TEXT,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    version INTEGER NOT NULL DEFAULT 1,
                    agent_id TEXT,
                    run_id TEXT,
                    app_id TEXT,
                    expires_at TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle_state)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_lifecycle "
                "ON memories(user_id, lifecycle_state)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
            # idx_memories_expires_at is created by migration v7 below, AFTER
            # the column is added. The CREATE TABLE above includes the column
            # for fresh DBs only; legacy DBs need the ALTER TABLE first.
            # A redundant try/except here keeps init safe if migration order
            # ever changes.
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_expires_at "
                    "ON memories(expires_at)"
                )
            except sqlite3.OperationalError:
                logger.debug(
                    "expires_at column not yet present; index will be created "
                    "by migration v7"
                )

            # Create FTS5 virtual table for full-text search with BM25 ranking
            # Using simple FTS5 without content_rowid linkage since we manage sync manually
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id,
                    user_id,
                    content,
                    namespace,
                    session_id,
                    tokenize='porter unicode61'
                )
            """)

            self._run_migrations(conn)
            conn.commit()

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        try:
            cursor = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                return row[0]  # type: ignore[no-any-return]
            return 0
        except sqlite3.OperationalError:  # pragma: no cover
            return 0

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        current = self._get_schema_version(conn)

        if current >= self.CURRENT_VERSION:
            return

        if current < 2:
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:  # pragma: no cover
                pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (2)")

        if current < 3:
            for col, dtype in [
                ("confidence", "REAL NOT NULL DEFAULT 1.0"),
                ("memory_type", "TEXT NOT NULL DEFAULT 'episodic'"),
                ("session_id", "TEXT"),
                ("namespace", "TEXT NOT NULL DEFAULT 'default'"),
                ("version", "INTEGER NOT NULL DEFAULT 1"),
                ("agent_id", "TEXT"),
                ("run_id", "TEXT"),
                ("app_id", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (3)")

        if current < 4:
            # Ensure FTS5 table exists for BM25 search
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        memory_id,
                        user_id,
                        content,
                        namespace,
                        session_id,
                        tokenize='porter unicode61'
                    )
                """)
            except sqlite3.OperationalError:
                pass  # Table already exists
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (4)")

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
            # TTL: add expires_at column and index for fast sweeper queries
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

        if current < 8:
            # API key authentication table for multi-tenant FastAPI server
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    hashed_key TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used_at TEXT,
                    revoked_at TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_hashed_key ON api_keys(hashed_key)"
            )
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (8)")

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryObject:
        embedding = None
        if row["embedding"] is not None:
            num_floats = len(row["embedding"]) // 4
            embedding = list(struct.unpack(f"{num_floats}f", row["embedding"]))

        # Use dict-style get for columns that may not exist in older schemas
        row_dict = dict(row)

        # Decrypt content and metadata if encryption is enabled.
        # Both are stored as JSON strings (either plain values or encrypted envelopes).
        content_str: str = row["content"]
        metadata_str: str = row["metadata"]

        try:
            content_val = json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            content_val = content_str  # fallback for plain text without JSON wrapping

        try:
            metadata_val = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata_val = {}

        if self._field_encryptor is not None:
            if self._field_encryptor._is_encrypted(content_val):
                content_val = self._field_encryptor.decrypt_field("content", content_val)
            if self._field_encryptor._is_encrypted(metadata_val):
                metadata_val = self._field_encryptor.decrypt_field("metadata", metadata_val)
        else:
            # Without encryption, content_val is a JSON string (the double-quoted string)
            # or a raw string. Normalize it.
            if isinstance(content_val, str):
                pass  # already a string
            elif content_val is None:
                content_val = ""
            else:
                content_val = str(content_val)

        # Decrypt user_id only if user_id encryption is enabled
        user_id_val: str = row["user_id"]
        if self._field_encryptor is not None and self._field_encryptor._encrypt_user_id:
            try:
                uid_parsed = json.loads(user_id_val)
                if self._field_encryptor._is_encrypted(uid_parsed):
                    user_id_val = self._field_encryptor.decrypt_field("user_id", uid_parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        expires_at = None
        if row_dict.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(row_dict["expires_at"])
            except (TypeError, ValueError):
                expires_at = None

        return MemoryObject(
            memory_id=row["memory_id"],
            user_id=user_id_val,
            content=str(content_val),
            embedding=embedding,
            score=0.0,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
            source=MemorySource(row["source"]),
            importance=row["importance"],
            lifecycle_state=LifecycleState(row["lifecycle_state"]),
            metadata=metadata_val if isinstance(metadata_val, dict) else {},
            embedding_dim=row["embedding_dim"],
            tags=[t.replace("\\,", ",") for t in row["tags"].split(",")] if row["tags"] else [],
            confidence=row["confidence"],
            memory_type=MemoryType(row["memory_type"]),
            session_id=row_dict.get("session_id"),
            namespace=row_dict.get("namespace", "default"),
            version=row_dict.get("version", 1),
            agent_id=row_dict.get("agent_id") if row_dict.get("agent_id") is not None else None,
            run_id=row_dict.get("run_id") if row_dict.get("run_id") is not None else None,
            app_id=row_dict.get("app_id") if row_dict.get("app_id") is not None else None,
            expires_at=expires_at,
        )

    def _memory_to_row(self, memory: MemoryObject) -> dict[str, Any]:
        embedding_blob = None
        if memory.embedding is not None:
            embedding_blob = struct.pack(f"{len(memory.embedding)}f", *memory.embedding)

        # Start with content and metadata as their native types so the
        # field encryptor can process them before JSON serialization.
        content_val: Any = memory.content
        metadata_val: Any = memory.metadata
        user_id_val: Any = memory.user_id

        if self._field_encryptor is not None:
            content_val = self._field_encryptor.encrypt_field("content", memory.content)
            metadata_val = self._field_encryptor.encrypt_field("metadata", memory.metadata)
            if self._field_encryptor._encrypt_user_id:
                user_id_val = self._field_encryptor.encrypt_field("user_id", memory.user_id)
                user_id_val = json.dumps(user_id_val)

        content_json = json.dumps(content_val)
        metadata_json = json.dumps(metadata_val)

        return {
            "memory_id": memory.memory_id,
            "user_id": user_id_val,
            "content": content_json,
            "embedding": embedding_blob,
            "embedding_dim": memory.embedding_dim,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata": metadata_json,
            "tags": ",".join(t.replace(",", "\\,") for t in memory.tags) if memory.tags else "",
            "confidence": memory.confidence,
            "memory_type": memory.memory_type.value,
            "session_id": memory.session_id,
            "namespace": memory.namespace,
            "version": memory.version,
            "agent_id": memory.agent_id,
            "run_id": memory.run_id,
            "app_id": memory.app_id,
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        }

    def store(self, memory: MemoryObject) -> None:
        with self._get_connection() as conn:
            row = self._memory_to_row(memory)
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (memory_id, user_id, content, embedding, embedding_dim, created_at,
                 last_accessed_at, source, importance, lifecycle_state, metadata, tags,
                 confidence, memory_type, session_id, namespace, version,
                 agent_id, run_id, app_id, expires_at)
                VALUES (:memory_id, :user_id, :content, :embedding, :embedding_dim,
                        :created_at, :last_accessed_at, :source, :importance,
                        :lifecycle_state, :metadata, :tags,
                        :confidence, :memory_type, :session_id, :namespace, :version,
                        :agent_id, :run_id, :app_id, :expires_at)
            """,
                row,
            )
            # Sync to FTS5 index for BM25 search
            self._sync_fts_single(conn, memory)

    def store_many(self, memories: list[MemoryObject]) -> int:
        """Store multiple memories in a single atomic transaction.

        All stores are wrapped in a transaction for atomicity. If any store
        fails, all changes are rolled back.

        Args:
            memories: List of MemoryObjects to store.

        Returns:
            Number of memories stored.
        """
        if not memories:
            return 0

        with self._transaction() as conn:
            for memory in memories:
                row = self._memory_to_row(memory)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (memory_id, user_id, content, embedding, embedding_dim, created_at,
                     last_accessed_at, source, importance, lifecycle_state, metadata, tags,
                     confidence, memory_type, session_id, namespace, version,
                     agent_id, run_id, app_id)
                    VALUES (:memory_id, :user_id, :content, :embedding, :embedding_dim,
                            :created_at, :last_accessed_at, :source, :importance,
                            :lifecycle_state, :metadata, :tags,
                            :confidence, :memory_type, :session_id, :namespace, :version,
                            :agent_id, :run_id, :app_id)
                """,
                    row,
                )
                # Sync to FTS5 index
                self._sync_fts_single(conn, memory)
        return len(memories)

    def _sync_fts_single(self, conn: sqlite3.Connection, memory: MemoryObject) -> None:
        """Sync a single memory to FTS5 index.

        Deletes any existing FTS row for this memory_id first, then inserts
        the new one. INSERT OR REPLACE alone is not sufficient because the
        memories_fts table is contentless and has no UNIQUE constraint on
        memory_id — it would add a new row on every store(), causing FTS
        duplicates to accumulate.
        """
        try:
            conn.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                (memory.memory_id,),
            )
            conn.execute(
                """
                INSERT INTO memories_fts (memory_id, user_id, content, namespace, session_id)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    memory.memory_id,
                    memory.user_id,
                    memory.content,
                    memory.namespace,
                    memory.session_id,
                ),
            )
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 sync failed: %s", e)

    def rebuild_fts_index(self, user_id: str | None = None) -> int:
        """Rebuild the FTS5 index from memories table.

        Use this if the FTS index gets out of sync with the memories table.

        Args:
            user_id: If provided, only reindex memories for this user. Otherwise
                the full index is rebuilt.

        Returns:
            Number of memories indexed.
        """
        with self._get_connection() as conn:
            if user_id is None:
                # Clear and rebuild the full FTS index
                conn.execute("DELETE FROM memories_fts")
                cursor = conn.execute("""
                    SELECT memory_id, user_id, content, namespace, session_id
                    FROM memories
                """)
            else:
                # Rebuild for a specific user: delete their existing FTS rows,
                # then re-insert them. The rest of the index is untouched.
                conn.execute("DELETE FROM memories_fts WHERE user_id = ?", (user_id,))
                cursor = conn.execute(
                    """
                    SELECT memory_id, user_id, content, namespace, session_id
                    FROM memories
                    WHERE user_id = ?
                """,
                    (user_id,),
                )
            rows = cursor.fetchall()

            # Batch insert via executemany instead of one INSERT per row.
            if rows:
                conn.executemany(
                    """
                    INSERT INTO memories_fts (memory_id, user_id, content, namespace, session_id)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    rows,
                )

            return len(rows)

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

        states = [s.value for s in lifecycle_filter]
        params: list[Any] = [user_id, namespace] + states

        sql = """
            SELECT * FROM memories
            WHERE user_id = ? AND namespace = ? AND lifecycle_state IN ({})
        """.format(",".join("?" * len(states)))

        if session_id is not None:
            sql += " AND (session_id = ? OR session_id IS NULL)"
            params.append(session_id)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        memories = []
        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding is not None:
                similarity = scoring.cosine_similarity(memory.embedding, query_embedding)
                memory.score = (similarity + 1.0) / 2.0
                memories.append(memory)

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:top_k]

    def get(self, memory_id: str) -> MemoryObject | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()

        if row:
            return self._row_to_memory(row)
        return None

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def update_many(self, memories: list[MemoryObject]) -> int:
        """Update multiple memories in a single atomic transaction.

        Uses ``executemany`` to batch UPDATE statements within one
        transaction, reducing round-trip cost from O(N) to O(1).

        Args:
            memories: List of MemoryObjects to update.

        Returns:
            Number of memories updated.
        """
        if not memories:
            return 0

        with self._transaction() as conn:
            for memory in memories:
                row = self._memory_to_row(memory)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (memory_id, user_id, content, embedding, embedding_dim, created_at,
                     last_accessed_at, source, importance, lifecycle_state, metadata, tags,
                     confidence, memory_type, session_id, namespace, version,
                     agent_id, run_id, app_id, expires_at)
                    VALUES (:memory_id, :user_id, :content, :embedding, :embedding_dim,
                            :created_at, :last_accessed_at, :source, :importance,
                            :lifecycle_state, :metadata, :tags,
                            :confidence, :memory_type, :session_id, :namespace, :version,
                            :agent_id, :run_id, :app_id, :expires_at)
                """,
                    row,
                )
                # Sync to FTS5 index
                self._sync_fts_single(conn, memory)
        return len(memories)

    def delete_by_user(self, user_id: str) -> int:
        with self._get_connection() as conn:
            # First delete from FTS5 index
            try:
                conn.execute("DELETE FROM memories_fts WHERE user_id = ?", (user_id,))
            except sqlite3.OperationalError:
                pass  # FTS table might not exist

            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        return cursor.rowcount

    def delete_by_id(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            # Also delete from FTS5 index to keep it in sync
            try:
                conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            except sqlite3.OperationalError:
                pass  # FTS table might not exist in old databases
        return cursor.rowcount > 0

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]
        params: list[Any] = [user_id, namespace] + states

        sql = """
            SELECT * FROM memories
            WHERE user_id = ? AND namespace = ? AND lifecycle_state IN ({})
        """.format(",".join("?" * len(states)))

        if session_id is not None:
            sql += " AND (session_id = ? OR session_id IS NULL)"
            params.append(session_id)

        if offset is not None:
            sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            # Use -1 as "no limit" when limit is None but offset is provided
            params.extend([limit if limit is not None else -1, offset])
        elif limit is not None:
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def count(self, user_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
            return cursor.fetchone()[0]  # type: ignore[no-any-return]

    def count_aggregates(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """SQL-based aggregate pushdown for ``stats()``.

        Runs three GROUP BY queries (lifecycle, source, tag presence) plus
        an AVG(importance) in a single connection block. O(1) per state
        instead of O(N) per memory.
        """
        from kemi.memory.model import LifecycleState as _LifecycleState
        from kemi.memory.model import MemorySource as _MemorySource

        with self._get_connection() as conn:
            where = ["user_id = ?"]
            params: list[Any] = [user_id]
            if lifecycle_filter is not None:
                placeholders = ",".join("?" * len(lifecycle_filter))
                where.append(f"lifecycle_state IN ({placeholders})")
                params.extend(s.value for s in lifecycle_filter)
            if session_id is not None:
                where.append("session_id = ?")
                params.append(session_id)
            where_sql = " AND ".join(where)

            total_row = conn.execute(
                f"SELECT COUNT(*), COALESCE(AVG(importance), 0.0) FROM memories WHERE {where_sql}",
                params,
            ).fetchone()
            total = int(total_row[0])
            avg_imp_numer = float(total_row[1]) * total  # recover the sum

            lifecycle_rows = conn.execute(
                f"SELECT lifecycle_state, COUNT(*) FROM memories WHERE {where_sql} GROUP BY lifecycle_state",  # noqa: E501
                params,
            ).fetchall()
            by_lifecycle = {state.value: 0 for state in _LifecycleState}
            for state_value, count in lifecycle_rows:
                if state_value in by_lifecycle:
                    by_lifecycle[state_value] = int(count)

            source_rows = conn.execute(
                f"SELECT source, COUNT(*) FROM memories WHERE {where_sql} GROUP BY source",
                params,
            ).fetchall()
            by_source = {source.value: 0 for source in _MemorySource}
            for source_value, count in source_rows:
                if source_value in by_source:
                    by_source[source_value] = int(count)

            # tags: stored as CSV with backslash-escaped commas. We have
            # to split in Python because SQLite's string functions don't
            # understand our escape convention. This is still O(N) on the
            # tag column, but cheaper than the full get_all_by_user scan
            # because we only fetch one column instead of all 21 fields.
            tag_counts: dict[str, int] = {}
            total_with_tags = 0
            mem_rows = conn.execute(
                f"SELECT tags FROM memories WHERE {where_sql}", params
            ).fetchall()
            for (tags_csv,) in mem_rows:
                parsed = self._parse_tags(tags_csv) if tags_csv else []
                if parsed:
                    total_with_tags += 1
                    for tag in parsed:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total": total,
            "by_lifecycle": by_lifecycle,
            "by_source": by_source,
            "total_with_tags": int(total_with_tags),
            "tag_counts": tag_counts,
            "avg_importance_numerator": float(avg_imp_numer),
        }

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        sql = "SELECT * FROM memories"
        params: list[Any] = []

        if offset is not None:
            sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            # Use -1 as "no limit" when limit is None but offset is provided
            params.extend([limit if limit is not None else -1, offset])
        elif limit is not None:
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Search for memories using FTS5 full-text search with native BM25 ranking.

        This method uses SQLite's FTS5 for fast full-text search with BM25 scoring.
        This is much faster than Python-based BM25 scoring for large datasets.

        Falls back to Python-based scoring if FTS5 query fails.
        """
        try:
            return self._fts5_search(user_id, query, top_k, lifecycle_filter, namespace, session_id)
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 search failed, falling back to Python BM25: %s", e)
            return self._bm25_python_fallback(
                user_id, query, top_k, lifecycle_filter, namespace, session_id
            )

    def _fts5_search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        lifecycle_filter: list[LifecycleState] | None,
        namespace: str,
        session_id: str | None,
    ) -> list[MemoryObject]:
        """Native FTS5 BM25 search - much faster than Python scoring."""
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]

        # Build lifecycle filter for subquery
        lifecycle_placeholders = ",".join("?" * len(states))

        # Use FTS5 MATCH on content column only with BM25 ranking
        # Join with main memories table to get full memory objects with lifecycle filtering
        # Build full FTS5 query with content: prefix in Python to avoid parameter binding issues
        fts_query = f"content:{self._prepare_fts_query(query)}"

        sql = f"""
            SELECT m.*, bm25(memories_fts) as fts_score
            FROM memories m
            INNER JOIN memories_fts fts ON m.memory_id = fts.memory_id
            WHERE fts.user_id = ?
              AND fts.namespace = ?
              AND m.lifecycle_state IN ({lifecycle_placeholders})
              AND memories_fts MATCH ?
        """

        params: list[Any] = [user_id, namespace] + states

        if session_id is not None:
            sql += " AND (m.session_id = ? OR m.session_id IS NULL)"
            params.append(session_id)

        sql += " ORDER BY fts_score LIMIT ?"
        params.append(top_k)

        # Append FTS query at the end (after lifecycle states and top_k)
        params.insert(-1, fts_query)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        memories = []
        rank = 0
        for row in rows:
            memory = self._row_to_memory(row)
            # Use rank-based scoring: higher rank position = lower score
            # BM25 returns negative scores where lower = better, so invert
            rank += 1
            memory.score = (
                1.0 / rank
            )  # Rank-based normalization (1st result = 1.0, 2nd = 0.5, etc.)
            memories.append(memory)

        return memories

    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query string for FTS5 MATCH on content column.

        Handles escaping special FTS5 characters to prevent query syntax errors.
        """
        if not query or not query.strip():
            return '""'

        # Escape FTS5 special characters: " * ( ) : ~
        # Replace with spaces to preserve word separation
        escaped = query
        for char in '"*():~':
            escaped = escaped.replace(char, " ")

        # Tokenize and create phrase queries for each term (prefix matching)
        terms = escaped.strip().split()
        if not terms:
            return '""'

        if len(terms) == 1:
            # Single term - prefix match
            term = terms[0].strip()
            if term:
                return f'"{term}"*'
            return '""'

        # Multiple terms - OR for matching any term
        phrase_terms = []
        for term in terms:
            term = term.strip()
            if term:
                phrase_terms.append(f'"{term}"*')

        if phrase_terms:
            return " OR ".join(phrase_terms)
        return '""'

    def _bm25_python_fallback(
        self,
        user_id: str,
        query: str,
        top_k: int,
        lifecycle_filter: list[LifecycleState] | None,
        namespace: str,
        session_id: str | None,
    ) -> list[MemoryObject]:
        """Python-based BM25 fallback when FTS5 is unavailable."""
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        # Limit fetch to prevent memory issues with large datasets
        fetch_limit = min(top_k * 20, 200)

        states = [s.value for s in lifecycle_filter]
        params: list[Any] = [user_id, namespace] + states

        sql = """
            SELECT * FROM memories
            WHERE user_id = ? AND namespace = ? AND lifecycle_state IN ({})
        """.format(",".join("?" * len(states)))

        if session_id is not None:
            sql += " AND (session_id = ? OR session_id IS NULL)"
            params.append(session_id)

        sql += f" ORDER BY created_at DESC LIMIT {fetch_limit}"

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        memories = [self._row_to_memory(row) for row in rows]

        if not memories:
            return []

        corpus = [m.content for m in memories]
        for memory in memories:
            memory.score = scoring.bm25_score_corpus(query, memory.content, corpus)

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:top_k]

    def get_all_users(self) -> list[str]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT user_id FROM memories")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def upgrade_schema(
        self, from_version: int | None = None, to_version: int | None = None
    ) -> int:
        target = to_version if to_version is not None else self.CURRENT_VERSION
        with self._get_connection() as conn:
            self._run_migrations(conn)
            conn.commit()
        return target

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]

        with self._get_connection() as conn:
            placeholders = ",".join("?" * len(states))
            # Escape LIKE wildcards in the user-supplied tag so a tag like
            # "a_b" or "a%b" matches literally instead of treating _ / % as
            # wildcards.
            escaped_tag = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            cursor = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE user_id = ? AND namespace = ? AND lifecycle_state IN ({placeholders})
                AND (',' || tags || ',') LIKE ('%,' || ? || ',%') ESCAPE '\\'
            """,
                [user_id, namespace] + states + [escaped_tag],
            )
            rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def get_namespaces(self, user_id: str) -> list[str]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT namespace FROM memories WHERE user_id = ?",
                (user_id,),
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def get_api_key_manager(self) -> Any:
        """Return an APIKeyManager bound to this adapter's connection.

        Lazy import to avoid a hard dependency on the api_keys module.
        Returns a fresh manager instance each call; the manager is cheap to
        create because it shares the underlying connection pool.
        """
        from kemi.infra.api_keys import APIKeyManager

        return APIKeyManager(connection=self._get_connection())
