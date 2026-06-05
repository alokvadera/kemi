"""PostgreSQL storage adapter with pgvector ANN search.

Requires: pip install kemi[postgres]

Schema mirrors SQLite adapter:
- memories table with VECTOR(dim) for embeddings, JSONB for metadata, TEXT[] for tags
- schema_version table for migrations
- Indexes: user_id, lifecycle_state, user_lifecycle, namespace, GIN on tags, GIN on FTS, ivfflat on embedding

Uses psycopg_pool.ConnectionPool for connection management.
ANN search via <=> (cosine) operator, FTS via to_tsvector.
Hybrid weighted combination for combined vector + keyword search.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from kemi.adapters.base import StorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource, MemoryType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_psycopg_pool: type | None = None


def _get_psycopg_pool() -> type:
    global _psycopg_pool
    if _psycopg_pool is None:
        import psycopg_pool

        _psycopg_pool = psycopg_pool
    return _psycopg_pool


class PostgresStorageAdapter(StorageAdapter):
    """PostgreSQL storage adapter with pgvector ANN support.

    Uses a connection pool (psycopg_pool.ConnectionPool) for concurrent access.
    DSN is read from the PG_DSN environment variable by default.

    Embedding stored as VECTOR(dim) via pgvector extension.
    Metadata stored as JSONB. Tags stored as TEXT[].
    Full-text search via PostgreSQL to_tsvector/to_tsquery.
    Hybrid search combines vector similarity with BM25 and recency.

    Parameters
    ----------
    dsn : str
        PostgreSQL connection string. Defaults to the PG_DSN env var.
    embedding_dim : int
        Dimension of embeddings. Used for VECTOR type. Default 384.
    min_connections : int
        Minimum connections in the pool. Default 1.
    max_connections : int
        Maximum connections in the pool. Default 10.
    """

    CURRENT_VERSION = 1

    def __init__(
        self,
        dsn: str | None = None,
        embedding_dim: int = 384,
        min_connections: int = 1,
        max_connections: int = 10,
    ) -> None:
        self._dsn = dsn or os.environ.get("PG_DSN", "")
        if not self._dsn:
            raise ValueError(
                "DSN must be provided or PG_DSN environment variable must be set. "
                "Example: postgresql://user:pass@localhost:5432/kemi"
            )
        self._embedding_dim = embedding_dim
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._pool: Any = None
        self._init_schema()

    def _get_pool(self) -> Any:
        pool_type = _get_psycopg_pool()
        if self._pool is None:
            self._pool = pool_type.ConnectionPool(
                self._dsn,
                min_size=self._min_connections,
                max_size=self._max_connections,
            )
        return self._pool

    @contextmanager
    def _transaction(self) -> Any:
        """Context manager for explicit transaction handling."""
        conn = self._get_pool().getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Transaction failed, rolled back")
            raise
        finally:
            self._get_pool().putconn(conn)

    @contextmanager
    def _connection(self) -> Any:
        """Get a connection from the pool as a context manager.

        Sets dict row factory so rows support row['colname'] and row.get('colname')
        for compatibility with SQLite-style dict-row access patterns.
        """
        conn = self._get_pool().getconn()
        try:
            # Use MappingRow so cursor results behave like dicts (row['col'], row.get())
            from psycopg.rows import MappingRow

            conn.row_factory = MappingRow
            yield conn
        finally:
            self._get_pool().putconn(conn)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __enter__(self) -> "PostgresStorageAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Schema ──────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connection() as conn:
            self._ensure_pgvector(conn)
            self._create_tables(conn)
            self._run_migrations(conn)

    def _ensure_pgvector(self, conn: Any) -> None:
        """Ensure pgvector extension exists."""
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        except Exception as e:
            logger.warning("Could not create pgvector extension: %s", e)

    def _create_tables(self, conn: Any) -> None:
        """Create memories and schema_version tables."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({self._embedding_dim}),
                embedding_dim INTEGER,
                created_at TIMESTAMPTZ NOT NULL,
                last_accessed_at TIMESTAMPTZ NOT NULL,
                source TEXT NOT NULL DEFAULT 'user_stated',
                importance REAL NOT NULL DEFAULT 0.5,
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                metadata JSONB NOT NULL DEFAULT '{{}}',
                tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                confidence REAL NOT NULL DEFAULT 1.0,
                memory_type TEXT NOT NULL DEFAULT 'episodic',
                session_id TEXT,
                namespace TEXT NOT NULL DEFAULT 'default',
                version INTEGER NOT NULL DEFAULT 1,
                agent_id TEXT,
                run_id TEXT,
                app_id TEXT,
                expires_at TIMESTAMPTZ
            )
            """
        )
        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle_state)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_lifecycle ON memories(user_id, lifecycle_state)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at)")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100)"
        )
        # FTS expression index using GIN on to_tsvector(content)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON memories USING GIN(to_tsvector('english', content))"
        )
        conn.commit()

    def _get_schema_version(self, conn: Any) -> int:
        try:
            cursor = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row["version"] if row else 0
        except Exception:
            return 0

    def _run_migrations(self, conn: Any) -> None:
        current = self._get_schema_version(conn)
        if current >= self.CURRENT_VERSION:
            return

        # Migrations applied here as we add new versions
        # Example migration from version 0 to 1:
        # if current < 1:
        #     conn.execute("ALTER TABLE memories ADD COLUMN new_col TEXT")
        #     ...

        conn.execute(
            "INSERT INTO schema_version (version) VALUES (%s)",
            (self.CURRENT_VERSION,),
        )
        conn.commit()

    # ── Row helpers ──────────────────────────────────────────────

    def _row_to_memory(self, row: Any) -> MemoryObject:
        """Convert a database row (dict-like) to a MemoryObject."""
        embedding: list[float] | None = None
        if row.get("embedding") is not None:
            emb = row["embedding"]
            if hasattr(emb, "tolist"):
                embedding = emb.tolist()
            elif isinstance(emb, (list, tuple)):
                embedding = list(emb)
            else:
                embedding = list(emb)

        # Parse metadata JSONB
        metadata_val: dict[str, Any] = {}
        raw_metadata = row.get("metadata")
        if raw_metadata:
            if isinstance(raw_metadata, dict):
                metadata_val = raw_metadata
            else:
                try:
                    metadata_val = json.loads(raw_metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata_val = {}

        # Parse content
        content_str = row.get("content", "")
        try:
            content_val: Any = json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            content_val = content_str
        if isinstance(content_val, str):
            pass
        elif content_val is None:
            content_val = ""
        else:
            content_val = str(content_val)

        expires_at: datetime | None = None
        raw_expires = row.get("expires_at")
        if raw_expires:
            try:
                if isinstance(raw_expires, str):
                    expires_at = datetime.fromisoformat(raw_expires.replace("Z", "+00:00"))
                else:
                    expires_at = raw_expires
            except (TypeError, ValueError):
                expires_at = None

        # Parse tags (TEXT[] -> list[str])
        raw_tags = row.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = raw_tags.strip("{}").split(",") if raw_tags != "{}" else []
        tags_list: list[str] = []
        for t in raw_tags:
            if t:
                tags_list.append(t.replace("\\,", ",").replace("\\\\", "\\"))
        tags_list = [t for t in tags_list if t]

        # Parse source
        source_val = MemorySource.USER_STATED
        raw_source = row.get("source")
        if raw_source:
            try:
                source_val = MemorySource(raw_source)
            except ValueError:
                source_val = MemorySource.USER_STATED

        # Parse memory_type
        memory_type_val = MemoryType.EPISODIC
        raw_mtype = row.get("memory_type")
        if raw_mtype:
            try:
                memory_type_val = MemoryType(raw_mtype)
            except ValueError:
                memory_type_val = MemoryType.EPISODIC

        # Parse lifecycle
        lifecycle_val = LifecycleState.ACTIVE
        raw_lifecycle = row.get("lifecycle_state")
        if raw_lifecycle:
            try:
                lifecycle_val = LifecycleState(raw_lifecycle)
            except ValueError:
                lifecycle_val = LifecycleState.ACTIVE

        # Parse timestamps
        created_at = row.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        last_accessed = row.get("last_accessed_at")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))

        return MemoryObject(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            content=str(content_val),
            embedding=embedding,
            score=0.0,
            created_at=created_at or datetime.now(timezone.utc),
            last_accessed_at=last_accessed or datetime.now(timezone.utc),
            source=source_val,
            importance=row.get("importance", 0.5),
            lifecycle_state=lifecycle_val,
            metadata=metadata_val,
            embedding_dim=row.get("embedding_dim"),
            tags=tags_list,
            confidence=row.get("confidence", 1.0),
            memory_type=memory_type_val,
            session_id=row.get("session_id"),
            namespace=row.get("namespace", "default"),
            version=row.get("version", 1),
            agent_id=row.get("agent_id"),
            run_id=row.get("run_id"),
            app_id=row.get("app_id"),
            expires_at=expires_at,
        )

    def _memory_to_row(self, memory: MemoryObject) -> dict[str, Any]:
        """Convert a MemoryObject to a dict for parameterized query."""
        return {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "content": json.dumps(memory.content),
            "embedding": memory.embedding,
            "embedding_dim": memory.embedding_dim,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata": json.dumps(memory.metadata or {}),
            "tags": memory.tags or [],
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

    # ── Store ───────────────────────────────────────────────────

    def store(self, memory: MemoryObject) -> None:
        with self._connection() as conn:
            row = self._memory_to_row(memory)
            conn.execute(
                """
                INSERT INTO memories
                (memory_id, user_id, content, embedding, embedding_dim, created_at,
                 last_accessed_at, source, importance, lifecycle_state, metadata, tags,
                 confidence, memory_type, session_id, namespace, version,
                 agent_id, run_id, app_id, expires_at)
                VALUES (
                    %(memory_id)s, %(user_id)s, %(content)s, %(embedding)s,
                    %(embedding_dim)s, %(created_at)s, %(last_accessed_at)s,
                    %(source)s, %(importance)s, %(lifecycle_state)s, %(metadata)s,
                    %(tags)s, %(confidence)s, %(memory_type)s, %(session_id)s,
                    %(namespace)s, %(version)s, %(agent_id)s, %(run_id)s, %(app_id)s,
                    %(expires_at)s
                )
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    embedding_dim = EXCLUDED.embedding_dim,
                    last_accessed_at = EXCLUDED.last_accessed_at,
                    source = EXCLUDED.source,
                    importance = EXCLUDED.importance,
                    lifecycle_state = EXCLUDED.lifecycle_state,
                    metadata = EXCLUDED.metadata,
                    tags = EXCLUDED.tags,
                    confidence = EXCLUDED.confidence,
                    memory_type = EXCLUDED.memory_type,
                    session_id = EXCLUDED.session_id,
                    namespace = EXCLUDED.namespace,
                    version = EXCLUDED.version,
                    agent_id = EXCLUDED.agent_id,
                    run_id = EXCLUDED.run_id,
                    app_id = EXCLUDED.app_id,
                    expires_at = EXCLUDED.expires_at
                """,
                row,
            )
            conn.commit()

    def store_many(self, memories: list[MemoryObject]) -> int:
        """Store multiple memories in a single atomic transaction."""
        if not memories:
            return 0

        with self._transaction() as conn:
            from psycopg.rows import MappingRow

            conn.row_factory = MappingRow
            for memory in memories:
                row = self._memory_to_row(memory)
                conn.execute(
                    """
                    INSERT INTO memories
                    (memory_id, user_id, content, embedding, embedding_dim, created_at,
                     last_accessed_at, source, importance, lifecycle_state, metadata, tags,
                     confidence, memory_type, session_id, namespace, version,
                     agent_id, run_id, app_id, expires_at)
                    VALUES (
                        %(memory_id)s, %(user_id)s, %(content)s, %(embedding)s,
                        %(embedding_dim)s, %(created_at)s, %(last_accessed_at)s,
                        %(source)s, %(importance)s, %(lifecycle_state)s, %(metadata)s,
                        %(tags)s, %(confidence)s, %(memory_type)s, %(session_id)s,
                        %(namespace)s, %(version)s, %(agent_id)s, %(run_id)s, %(app_id)s,
                        %(expires_at)s
                    )
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        embedding_dim = EXCLUDED.embedding_dim,
                        last_accessed_at = EXCLUDED.last_accessed_at,
                        source = EXCLUDED.source,
                        importance = EXCLUDED.importance,
                        lifecycle_state = EXCLUDED.lifecycle_state,
                        metadata = EXCLUDED.metadata,
                        tags = EXCLUDED.tags,
                        confidence = EXCLUDED.confidence,
                        memory_type = EXCLUDED.memory_type,
                        session_id = EXCLUDED.session_id,
                        namespace = EXCLUDED.namespace,
                        version = EXCLUDED.version,
                        agent_id = EXCLUDED.agent_id,
                        run_id = EXCLUDED.run_id,
                        app_id = EXCLUDED.app_id,
                        expires_at = EXCLUDED.expires_at
                    """,
                    row,
                )
            # _transaction context manager commits on exit — no explicit commit needed here
        return len(memories)

    # ── Search ──────────────────────────────────────────────

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Search using pgvector cosine distance (<=>) for ANN search."""
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]
        states_placeholder = ",".join(["%s"] * len(states))

        sql = f"""
            SELECT m.*,
                   1 - (m.embedding <=> %s) AS vector_similarity
            FROM memories m
            WHERE m.user_id = %s
              AND m.namespace = %s
              AND m.lifecycle_state IN ({states_placeholder})
        """
        params: list[Any] = [query_embedding, user_id, namespace] + states

        if session_id is not None:
            sql += " AND (m.session_id = %s OR m.session_id IS NULL)"
            params.append(session_id)

        sql += " ORDER BY m.embedding <=> %s LIMIT %s"
        params.append(query_embedding)
        params.append(top_k)

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        memories = []
        for row in rows:
            mem = self._row_to_memory(row)
            vs = row.get("vector_similarity", 0.0)
            mem.score = max(0.0, min(1.0, 1.0 - vs / 2.0))
            memories.append(mem)

        return memories

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Search using PostgreSQL full-text search with to_tsvector/to_tsquery."""
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]
        states_placeholder = ",".join(["%s"] * len(states))

        fts_query = "websearch_to_tsquery('english', %s)"

        sql = f"""
            SELECT m.*,
                   ts_rank(to_tsvector('english', m.content), {fts_query}) AS fts_rank
            FROM memories m
            WHERE m.user_id = %s
              AND m.namespace = %s
              AND m.lifecycle_state IN ({states_placeholder})
              AND to_tsvector('english', m.content) @@ {fts_query}
        """
        params: list[Any] = [user_id, namespace] + states + [query, query]

        if session_id is not None:
            sql += " AND (m.session_id = %s OR m.session_id IS NULL)"
            params.append(session_id)

        sql += f" ORDER BY fts_rank DESC LIMIT %s"
        params.append(top_k)

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        memories = []
        rank = 0
        for row in rows:
            mem = self._row_to_memory(row)
            rank += 1
            mem.score = 1.0 / rank
            memories.append(mem)

        return memories

    def search_hybrid(
        self,
        user_id: str,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        weight_vector: float = 0.6,
        weight_bm25: float = 0.3,
        weight_recency: float = 0.1,
    ) -> list[MemoryObject]:
        """Hybrid search combining ANN vector + FTS text + recency scoring."""
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = [s.value for s in lifecycle_filter]
        states_placeholder = ",".join(["%s"] * len(states))

        sql = f"""
            SELECT m.*,
                   1 - (m.embedding <=> %s) AS vector_sim,
                   ts_rank(to_tsvector('english', m.content),
                           websearch_to_tsquery('english', %s)) AS fts_rank,
                   EXTRACT(EPOCH FROM (NOW() - m.last_accessed_at)) AS age_seconds
            FROM memories m
            WHERE m.user_id = %s
              AND m.namespace = %s
              AND m.lifecycle_state IN ({states_placeholder})
              AND m.embedding IS NOT NULL
        """
        params: list[Any] = [query_embedding, query_text, user_id, namespace] + states

        if session_id is not None:
            sql += " AND (m.session_id = %s OR m.session_id IS NULL)"
            params.append(session_id)

        sql += f" ORDER BY vector_sim DESC LIMIT %s"
        params.append(top_k * 3)

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        if not rows:
            return []

        memories: list[MemoryObject] = []
        max_rank = max(r.get("fts_rank", 0.0) for r in rows) if rows else 1.0

        for row in rows:
            mem = self._row_to_memory(row)

            vec_sim = row.get("vector_sim", 0.0)
            vec_score = max(0.0, min(1.0, 1.0 - vec_sim / 2.0))

            fts_rank = row.get("fts_rank", 0.0)
            bm25_score = fts_rank / max_rank if max_rank > 0 else 0.0

            age_seconds = row.get("age_seconds", 0.0) or 0.0
            if age_seconds <= 0:
                recency_score = 1.0
            else:
                recency_score = 2.0 ** (-age_seconds / 604800.0)

            final_score = (
                vec_score * weight_vector
                + bm25_score * weight_bm25
                + recency_score * weight_recency
            )
            mem.score = final_score
            memories.append(mem)

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:top_k]

    # ── Get / Update / Delete ─────────────────────────────────

    def get(self, memory_id: str) -> MemoryObject | None:
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE memory_id = %s",
                (memory_id,),
            )
            row = cursor.fetchone()
        if row:
            return self._row_to_memory(row)
        return None

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def delete_by_id(self, memory_id: str) -> bool:
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE memory_id = %s",
                (memory_id,),
            )
            deleted = cursor.rowcount > 0 if hasattr(cursor, "rowcount") else False
            conn.commit()
        return deleted

    def delete_by_user(self, user_id: str) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = %s",
                (user_id,),
            )
            count = cursor.rowcount if hasattr(cursor, "rowcount") else 0
            conn.commit()
        return count

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
        states_placeholder = ",".join(["%s"] * len(states))

        sql = f"""
            SELECT * FROM memories
            WHERE user_id = %s AND namespace = %s AND lifecycle_state IN ({states_placeholder})
        """
        params: list[Any] = [user_id, namespace] + states

        if session_id is not None:
            sql += " AND (session_id = %s OR session_id IS NULL)"
            params.append(session_id)

        if offset is not None:
            sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.append(limit if limit is not None else -1)
            params.append(offset)
        elif limit is not None:
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def count(self, user_id: str) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
        return row["count"] if row else 0

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
        *,
        namespace: str | None = None,
    ) -> list[MemoryObject]:
        sql = "SELECT * FROM memories"
        params: list[Any] = []

        where_parts: list[str] = []
        if namespace is not None:
            where_parts.append("namespace = %s")
            params.append(namespace)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        if offset is not None:
            sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.append(limit if limit is not None else -1)
            params.append(offset)
        elif limit is not None:
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

        with self._connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def get_all_users(self) -> list[str]:
        with self._connection() as conn:
            cursor = conn.execute("SELECT DISTINCT user_id FROM memories")
            rows = cursor.fetchall()
        return [row["user_id"] for row in rows]

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
        states_placeholder = ",".join(["%s"] * len(states))

        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE user_id = %s AND namespace = %s
                  AND lifecycle_state IN ({states_placeholder})
                  AND %s = ANY(tags)
                ORDER BY created_at DESC
                """,
                [user_id, namespace] + states + [tag],
            )
            rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        """Run schema migrations from from_version to to_version.

        Applies any needed ALTER TABLE statements or data migrations.
        """
        if from_version >= to_version:
            return

        with self._connection() as conn:
            current = self._get_schema_version(conn)
            if current >= to_version:
                return  # Already at or beyond target version

            # Apply incremental migrations
            # Example:
            # if from_version < 2 <= to_version:
            #     conn.execute("ALTER TABLE memories ADD COLUMN new_field TEXT")
            #     ...

            # Mark as applied
            if current < to_version:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (%s)",
                    (to_version,),
                )
                conn.commit()

    def rebuild_fts_index(self, user_id: str | None = None) -> int:
        """Rebuild the GIN index on to_tsvector(content).

        Uses REINDEX to rebuild the FTS index. Returns the number of indexed rows.

        Parameters
        ----------
        user_id : str | None
            Optional user ID to limit scope. If None, rebuilds for all memories.
        """
        count = 0
        with self._connection() as conn:
            if user_id is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id = %s",
                    (user_id,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
            row = cursor.fetchone()
            count = row["count"] if row else 0

        try:
            with self._connection() as conn:
                conn.execute("REINDEX INDEX idx_memories_content_fts")
                conn.commit()
        except Exception as e:
            logger.warning("Could not reindex FTS index: %s", e)

        return count

    def rebuild_embedding_index(self, user_id: str | None = None) -> int:
        """Rebuild the ivfflat index on embeddings.

        Uses REINDEX to rebuild the index. Returns the number of indexed rows.

        Parameters
        ----------
        user_id : str | None
            Optional user ID to limit reindex scope. If None, rebuilds the entire index.
        """
        count = 0
        with self._connection() as conn:
            if user_id is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id = %s AND embedding IS NOT NULL",
                    (user_id,),
                )
                count = cursor.fetchone()["count"] if cursor.fetchone() else 0
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
                )
                count = cursor.fetchone()["count"] if cursor.fetchone() else 0

        try:
            with self._connection() as conn:
                conn.execute("REINDEX INDEX idx_memories_embedding")
                conn.commit()
        except Exception as e:
            logger.warning("Could not reindex embedding index: %s", e)

        return count