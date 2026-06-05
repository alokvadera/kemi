"""LRU cache for `Memory.recall()` results.

Kept as a small private class because the cache is tightly coupled to
`MemoryObject` semantics (returns shallow copies to prevent cache corruption
when callers mutate the returned list).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from kemi.models import LifecycleState, MemoryObject


class _QueryCache:
    """Simple LRU cache for `recall()` query results.

    Caches lists of `MemoryObject`s keyed by query parameters.
    Returns a *shallow copy* of the cached list so callers can
    safely mutate the returned result without corrupting the cache.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, list[MemoryObject]] = OrderedDict()

    def _make_key(
        self,
        user_id: str,
        query: str,
        top_k: int,
        max_tokens: int | None,
        lifecycle_filter: list[LifecycleState] | None,
        hybrid_search: bool | None,
        namespace: str,
        session_id: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> str:
        """Build a stable string key from query parameters."""
        lf = tuple(sorted(s.value for s in lifecycle_filter)) if lifecycle_filter else ()
        mf = tuple(sorted((k, v) for k, v in (metadata_filter or {}).items()))
        return "|".join(
            [
                user_id,
                query,
                str(top_k),
                str(max_tokens),
                str(lf),
                str(hybrid_search),
                namespace,
                str(session_id),
                str(mf),
            ]
        )

    def _copy_memories(self, memories: list[MemoryObject]) -> list[MemoryObject]:
        """Return a list of MemoryObject copies with mutable fields duplicated."""
        return [
            MemoryObject(
                memory_id=m.memory_id,
                user_id=m.user_id,
                content=m.content,
                embedding=m.embedding,
                score=m.score,
                created_at=m.created_at,
                last_accessed_at=m.last_accessed_at,
                source=m.source,
                importance=m.importance,
                lifecycle_state=m.lifecycle_state,
                metadata=m.metadata.copy(),
                embedding_dim=m.embedding_dim,
                tags=list(m.tags),
                confidence=m.confidence,
                memory_type=m.memory_type,
                session_id=m.session_id,
                namespace=m.namespace,
                version=m.version,
                agent_id=m.agent_id,
                run_id=m.run_id,
                app_id=m.app_id,
            )
            for m in memories
        ]

    def get(self, key: str) -> list[MemoryObject] | None:
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            # Return copies so callers cannot mutate the cached objects.
            return self._copy_memories(self._cache[key])
        return None

    def put(self, key: str, value: list[MemoryObject]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        # Store copies so internal mutations (e.g., lifecycle updates on
        # the result list returned by recall) don't corrupt the cache.
        self._cache[key] = self._copy_memories(value)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
