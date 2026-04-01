import json
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from kemi.adapters.base import StorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource
from kemi import scoring


class SQLiteStorageAdapter(StorageAdapter):
    """SQLite storage adapter with WAL mode.

    Embedding stored as BLOB (float32 bytes) for compactness.
    Schema version tracked in schema_version table.
    """

    CURRENT_VERSION = 1

    def __init__(self, db_path: str = "kemi.db"):
        self._db_path = db_path
        self._shared_conn = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.execute("PRAGMA journal_mode=WAL")
            self._shared_conn.row_factory = sqlite3.Row
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
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
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle_state)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_lifecycle ON memories(user_id, lifecycle_state)"
            )

    def _row_to_memory(self, row) -> MemoryObject:
        embedding = None
        if row["embedding"]:
            num_floats = len(row["embedding"]) // 4
            embedding = list(struct.unpack(f"{num_floats}f", row["embedding"]))

        return MemoryObject(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            content=row["content"],
            embedding=embedding,
            score=0.0,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
            source=MemorySource(row["source"]),
            importance=row["importance"],
            lifecycle_state=LifecycleState(row["lifecycle_state"]),
            metadata=json.loads(row["metadata"]),
            embedding_dim=row["embedding_dim"],
        )

    def _memory_to_row(self, memory: MemoryObject) -> dict:
        embedding_blob = None
        if memory.embedding:
            embedding_blob = struct.pack(f"{len(memory.embedding)}f", *memory.embedding)

        return {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "content": memory.content,
            "embedding": embedding_blob,
            "embedding_dim": memory.embedding_dim,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata": json.dumps(memory.metadata),
        }

    def store(self, memory: MemoryObject) -> None:
        with self._get_connection() as conn:
            row = self._memory_to_row(memory)
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (memory_id, user_id, content, embedding, embedding_dim, created_at, 
                 last_accessed_at, source, importance, lifecycle_state, metadata)
                VALUES (:memory_id, :user_id, :content, :embedding, :embedding_dim,
                        :created_at, :last_accessed_at, :source, :importance, 
                        :lifecycle_state, :metadata)
            """,
                row,
            )

    def search(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM memories 
                WHERE user_id = ? AND lifecycle_state IN ({})
            """.format(",".join("?" * len(states))),
                [user_id] + states,
            )

            rows = cursor.fetchall()

        memories = []
        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                similarity = scoring.cosine_similarity(
                    memory.embedding, query_embedding
                )
                memory.score = (similarity + 1.0) / 2.0
                memories.append(memory)

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:top_k]

    def get(self, memory_id: str) -> Optional[MemoryObject]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            )
            row = cursor.fetchone()

        if row:
            return self._row_to_memory(row)
        return None

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def delete_by_user(self, user_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        return cursor.rowcount

    def delete_by_id(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
        return cursor.rowcount > 0

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM memories 
                WHERE user_id = ? AND lifecycle_state IN ({})
            """.format(",".join("?" * len(states))),
                [user_id] + states,
            )

            rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def count(self, user_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
            )
            return cursor.fetchone()[0]

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_version (version, applied_at)
                VALUES (?, datetime('now'))
            """,
                (to_version,),
            )
