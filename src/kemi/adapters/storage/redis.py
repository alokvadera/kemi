"""Redis storage adapter for kemi.

Mirrors the SQLiteStorageAdapter API.  Uses Redis hashes for storage,
sets for user/tag indexing, and Python-side cosine similarity for search.

Install: pip install kemi[redis]

Usage::

    from kemi import Memory
    from kemi.adapters.storage.redis import RedisStorageAdapter

    adapter = RedisStorageAdapter()  # uses REDIS_URL env var
    memory = Memory(store=adapter)
"""

import json
import logging
import os
from datetime import datetime

from kemi.adapters.base import StorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource, MemoryType

logger = logging.getLogger(__name__)

try:
    import redis

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REDIS_AVAILABLE = False

_REDIS_ERR = "redis>=5.0 is required for RedisStorageAdapter. Install with: pip install kemi[redis]"


class RedisStorageAdapter(StorageAdapter):
    """Redis storage adapter.

    Parameters
    ----------
    url : str, optional
        Redis connection URL.  Defaults to ``REDIS_URL`` env var,
        then ``redis://localhost:6379/0``.
    prefix : str
        Key prefix for all stored data (default ``kemi``).
    """

    CURRENT_VERSION = 1

    def __init__(self, url: str | None = None, prefix: str = "kemi") -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(_REDIS_ERR)
        self._prefix = prefix
        self._redis = redis.Redis.from_url(
            url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )

    def _k(self, *parts: str) -> str:
        return ":".join((self._prefix, *parts))

    def close(self) -> None:
        self._redis.close()

    def __enter__(self) -> "RedisStorageAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass

    # ── Conversion ───────────────────────────────────────────────────────

    def _memory_to_hash(self, m: MemoryObject) -> dict[str, str]:
        return {
            "memory_id": m.memory_id,
            "user_id": m.user_id,
            "content": m.content,
            "embedding": json.dumps(m.embedding) if m.embedding is not None else "",
            "embedding_dim": str(m.embedding_dim) if m.embedding_dim is not None else "",
            "created_at": m.created_at.isoformat(),
            "last_accessed_at": m.last_accessed_at.isoformat(),
            "source": m.source.value,
            "importance": str(m.importance),
            "lifecycle_state": m.lifecycle_state.value,
            "metadata": json.dumps(m.metadata),
            "tags": json.dumps(m.tags),
            "confidence": str(m.confidence),
            "memory_type": m.memory_type.value,
            "session_id": m.session_id or "",
            "namespace": m.namespace,
            "version": str(m.version),
            "agent_id": m.agent_id or "",
            "run_id": m.run_id or "",
            "app_id": m.app_id or "",
            "expires_at": m.expires_at.isoformat() if m.expires_at else "",
        }

    def _hash_to_memory(self, d: dict[str, str]) -> MemoryObject:
        emb = json.loads(d["embedding"]) if d.get("embedding") else None
        dim_str = d.get("embedding_dim", "")
        expires_raw = d.get("expires_at", "")
        expires_at = (
            datetime.fromisoformat(expires_raw) if expires_raw else None
        )
        return MemoryObject(
            memory_id=d["memory_id"],
            user_id=d["user_id"],
            content=d["content"],
            embedding=emb,
            score=0.0,
            created_at=datetime.fromisoformat(d["created_at"]),
            last_accessed_at=datetime.fromisoformat(d["last_accessed_at"]),
            source=MemorySource(d["source"]),
            importance=float(d["importance"]),
            lifecycle_state=LifecycleState(d["lifecycle_state"]),
            metadata=json.loads(d["metadata"]) if d.get("metadata") else {},
            embedding_dim=int(dim_str) if dim_str else None,
            tags=json.loads(d["tags"]) if d.get("tags") else [],
            confidence=float(d.get("confidence", "1.0")),
            memory_type=MemoryType(d.get("memory_type", "episodic")),
            session_id=d.get("session_id") or None,
            namespace=d.get("namespace", "default"),
            version=int(d.get("version", "1")),
            agent_id=d.get("agent_id") or None,
            run_id=d.get("run_id") or None,
            app_id=d.get("app_id") or None,
            expires_at=expires_at,
        )

    def _tag_index_key(self, user_id: str, tag: str) -> str:
        return self._k("user", user_id, "tag", tag)

    def _user_memories_key(self, user_id: str) -> str:
        return self._k("user", user_id, "mem")

    def _mem_key(self, memory_id: str) -> str:
        return self._k("mem", memory_id)

    def _mem_ids_for_user(self, user_id: str) -> set[str]:
        return self._redis.smembers(self._user_memories_key(user_id))  # type: ignore[no-any-return]

    def _fetch_memories(self, ids: set[str]) -> list[dict[str, str]]:
        if not ids:
            return []
        keys = [self._mem_key(mid) for mid in ids]
        pipe = self._redis.pipeline()
        for k in keys:
            pipe.hgetall(k)
        results = pipe.execute()
        return [r for r in results if r]

    # ── Store ────────────────────────────────────────────────────────────

    def store(self, memory: MemoryObject) -> None:
        pipe = self._redis.pipeline()
        h = self._memory_to_hash(memory)
        mem_key = self._mem_key(memory.memory_id)
        pipe.hset(mem_key, mapping=h)
        pipe.sadd(self._user_memories_key(memory.user_id), memory.memory_id)
        pipe.sadd(self._k("users"), memory.user_id)
        for tag in memory.tags:
            pipe.sadd(self._tag_index_key(memory.user_id, tag), memory.memory_id)
        pipe.execute()

    def store_many(self, memories: list[MemoryObject]) -> int:
        if not memories:
            return 0
        pipe = self._redis.pipeline()
        for memory in memories:
            h = self._memory_to_hash(memory)
            mem_key = self._mem_key(memory.memory_id)
            pipe.hset(mem_key, mapping=h)
            pipe.sadd(self._user_memories_key(memory.user_id), memory.memory_id)
            pipe.sadd(self._k("users"), memory.user_id)
            for tag in memory.tags:
                pipe.sadd(self._tag_index_key(memory.user_id, tag), memory.memory_id)
        pipe.execute()
        return len(memories)

    def get(self, memory_id: str) -> MemoryObject | None:
        d = self._redis.hgetall(self._mem_key(memory_id))
        if not d:
            return None
        return self._hash_to_memory(d)

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    # ── Delete ───────────────────────────────────────────────────────────

    def delete_by_id(self, memory_id: str) -> bool:
        d = self._redis.hgetall(self._mem_key(memory_id))
        if not d:
            return False
        pipe = self._redis.pipeline()
        pipe.delete(self._mem_key(memory_id))
        if d.get("user_id") and d.get("tags"):
            user_id = d["user_id"]
            for tag in json.loads(d["tags"]):
                pipe.srem(self._tag_index_key(user_id, tag), memory_id)
        if d.get("user_id"):
            pipe.srem(self._user_memories_key(d["user_id"]), memory_id)
        pipe.execute()
        return True

    def delete_by_user(self, user_id: str) -> int:
        ids = self._mem_ids_for_user(user_id)
        if not ids:
            return 0
        pipe = self._redis.pipeline()
        for mid in ids:
            pipe.delete(self._mem_key(mid))
        pipe.delete(self._user_memories_key(user_id))
        results = pipe.execute()
        # Sum the per-key delete results (skip the trailing user-set delete).
        return sum(int(r) for r in results[:-1])

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        from kemi.scoring import cosine_similarity

        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]
        states = {s.value for s in lifecycle_filter}

        ids = self._mem_ids_for_user(user_id)
        mems = self._fetch_memories(ids)

        results: list[MemoryObject] = []
        for d in mems:
            if d.get("lifecycle_state") not in states:
                continue
            if d.get("namespace", "default") != namespace:
                continue
            if session_id is not None and d.get("session_id") not in (session_id, ""):
                continue
            mem = self._hash_to_memory(d)
            if mem.embedding is not None:
                sim = cosine_similarity(mem.embedding, query_embedding)
                mem.score = (sim + 1.0) / 2.0
                results.append(mem)

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]
        states = {s.value for s in lifecycle_filter}
        q = query.lower()

        ids = self._mem_ids_for_user(user_id)
        mems = self._fetch_memories(ids)

        candidates: list[MemoryObject] = []
        for d in mems:
            if d.get("lifecycle_state") not in states:
                continue
            if d.get("namespace", "default") != namespace:
                continue
            if session_id is not None and d.get("session_id") not in (session_id, ""):
                continue
            if q in d.get("content", "").lower():
                mem = self._hash_to_memory(d)
                mem.score = len(query) / max(len(mem.content), 1)
                candidates.append(mem)

        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]

    # ── Bulk retrieval ───────────────────────────────────────────────────

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
        states = {s.value for s in lifecycle_filter}

        ids = self._mem_ids_for_user(user_id)
        mems = self._fetch_memories(ids)

        results = [
            self._hash_to_memory(d)
            for d in mems
            if d.get("lifecycle_state") in states
            and d.get("namespace", "default") == namespace
            and (session_id is None or d.get("session_id") in (session_id, ""))
        ]

        if offset is not None:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        users = self.get_all_users()
        all_mems: list[MemoryObject] = []
        for uid in users:
            ids = self._mem_ids_for_user(uid)
            mems = self._fetch_memories(ids)
            all_mems.extend(self._hash_to_memory(d) for d in mems)
        if offset is not None:
            all_mems = all_mems[offset:]
        if limit is not None:
            all_mems = all_mems[:limit]
        return all_mems

    def count(self, user_id: str) -> int:
        return self._redis.scard(self._user_memories_key(user_id))  # type: ignore[no-any-return]

    def get_all_users(self) -> list[str]:
        return sorted(self._redis.smembers(self._k("users")))

    # ── Tags ─────────────────────────────────────────────────────────────

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]
        states = {s.value for s in lifecycle_filter}

        ids = self._redis.smembers(self._tag_index_key(user_id, tag))
        mems = self._fetch_memories(ids)

        return [
            self._hash_to_memory(d)
            for d in mems
            if d.get("lifecycle_state") in states and d.get("namespace", "default") == namespace
        ]

    # ── Schema ───────────────────────────────────────────────────────────

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        self._redis.set(self._k("schema_version"), str(to_version))
