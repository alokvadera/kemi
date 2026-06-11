from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from kemi.memory.model import LifecycleState, MemoryObject


class EmbeddingAdapter(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats.
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """Embed a single text into a vector.

        Args:
            text: String to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings this adapter produces.

        Used for:
        - Detecting dimension mismatches during migration
        - Validating compatibility at query time
        - Storing embedding_dim on MemoryObject

        Returns:
            Integer dimension (e.g., 384 for bge-small, 1536 for OpenAI).
        """
        pass


class StorageAdapter(ABC):
    @abstractmethod
    def store(self, memory: MemoryObject) -> None:
        """Persist a memory object.

        The adapter should:
        - Store all fields except score (which is query-time only)
        - Serialize embedding as bytes for compactness (or JSON, adapter's choice)
        - Serialize metadata as JSON string
        - Use ISO 8601 strings for datetime fields

        Args:
            memory: The MemoryObject to persist.
        """
        pass

    @abstractmethod
    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Search for memories similar to the query embedding.

        The adapter is responsible for:
        - Filtering by user_id (mandatory scoping)
        - Filtering by lifecycle_state (if lifecycle_filter provided)
        - Computing similarity and returning top_k results
        - Returning MemoryObjects with embedding populated

        Note: The adapter returns results sorted by similarity.
        The scoring engine in core.py will re-rank with temporal decay + importance.

        Args:
            user_id: Scope search to this user's memories.
            query_embedding: The vector to search against.
            top_k: Maximum number of results.
            lifecycle_filter: Only include memories in these states.
                              If None, default to [ACTIVE, DECAYING].

        Returns:
            List of MemoryObjects, sorted by vector similarity (descending).
        """
        pass

    @abstractmethod
    def get(self, memory_id: str) -> MemoryObject | None:
        """Retrieve a single memory by ID.

        Args:
            memory_id: The UUID of the memory.

        Returns:
            MemoryObject if found, None otherwise.
        """
        pass

    @abstractmethod
    def update(self, memory: MemoryObject) -> None:
        """Update an existing memory.

        Used for:
        - Refreshing last_accessed_at after recall
        - Updating lifecycle_state
        - Updating embedding after migration

        Args:
            memory: The MemoryObject with updated fields.
                    Must have memory_id set to identify which row to update.
        """
        pass

    def update_many(self, memories: list[MemoryObject]) -> int:
        """Update multiple memories in a single batch operation.

        Default implementation falls back to individual ``update`` calls.
        Storage adapters backed by SQL SHOULD override this with an
        ``executemany`` or equivalent for O(1) round-trip cost.

        Args:
            memories: List of MemoryObjects to update.

        Returns:
            Number of memories updated.
        """
        for mem in memories:
            self.update(mem)
        return len(memories)

    @abstractmethod
    def delete_by_user(self, user_id: str) -> int:
        """Delete ALL memories for a user. GDPR compliance.

        Args:
            user_id: The user whose memories to delete.

        Returns:
            Number of memories deleted.
        """
        pass

    @abstractmethod
    def delete_by_id(self, memory_id: str) -> bool:
        """Delete a single memory by ID.

        Args:
            memory_id: The UUID of the memory to delete.

        Returns:
            True if found and deleted, False if not found.
        """
        pass

    @abstractmethod
    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        """Get all memories for a user.

        Used for:
        - Migration (re-embedding all memories)
        - Consolidation (v2)
        - Export/backup

        Args:
            user_id: The user whose memories to retrieve.
            lifecycle_filter: Only include memories in these states.
            namespace: Filter by namespace (default: 'default').
            session_id: Optional session ID filter.
            limit: Maximum number of memories to return (None for no limit).
            offset: Number of memories to skip (for pagination).

        Returns:
            List of all matching MemoryObjects.
        """
        pass

    @abstractmethod
    def count(self, user_id: str) -> int:
        """Count memories for a user.

        Args:
            user_id: The user whose memories to count.

        Returns:
            Number of memories stored for this user.
        """
        pass

    def count_aggregates(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute ``stats()`` aggregates in a single round-trip.

        Default implementation iterates over ``get_all_by_user`` in Python.
        Storage adapters backed by a SQL database SHOULD override this with
        a ``GROUP BY``-based implementation for O(states) cost instead of
        O(memories) cost.

        Args:
            user_id: User to aggregate over.
            lifecycle_filter: Restrict to these lifecycle states (same
                semantics as ``get_all_by_user``).
            session_id: Restrict to this session (same semantics).

        Returns:
            Dict with keys:
              - ``total``: int
              - ``by_lifecycle``: dict[str, int]
              - ``by_source``: dict[str, int]
              - ``total_with_tags``: int
              - ``tag_counts``: dict[str, int]
              - ``avg_importance_numerator``: float (sum of importance)
            The caller divides ``avg_importance_numerator`` by ``total``
            to get the average. Splitting numerator/denominator avoids
            a second pass.
        """
        all_memories = self.get_all_by_user(
            user_id, lifecycle_filter=lifecycle_filter, session_id=session_id
        )
        from kemi.memory.model import LifecycleState as _LifecycleState
        from kemi.memory.model import MemorySource as _MemorySource

        by_lifecycle = {state.value: 0 for state in _LifecycleState}
        by_source = {source.value: 0 for source in _MemorySource}
        tag_counts: dict[str, int] = {}
        total_with_tags = 0
        importance_sum = 0.0
        for mem in all_memories:
            by_lifecycle[mem.lifecycle_state.value] += 1
            by_source[mem.source.value] += 1
            importance_sum += mem.importance
            if mem.tags:
                total_with_tags += 1
                for tag in mem.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return {
            "total": len(all_memories),
            "by_lifecycle": by_lifecycle,
            "by_source": by_source,
            "total_with_tags": total_with_tags,
            "tag_counts": tag_counts,
            "avg_importance_numerator": importance_sum,
        }

    @abstractmethod
    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        """Get ALL memories from the store.

        Used for export/backup.

        Args:
            limit: Maximum number of memories to return (None for no limit).
            offset: Number of memories to skip (for pagination).

        Returns:
            List of all MemoryObjects in the store.
        """
        pass

    @abstractmethod
    def get_all_users(self) -> list[str]:
        """Get all unique user IDs that have memories.

        Used for listing users.

        Returns:
            List of unique user IDs.
        """
        pass

    @abstractmethod
    def upgrade_schema(
        self, from_version: int | None = None, to_version: int | None = None
    ) -> int:
        """Migrate the storage schema to ``to_version``.

        Called by :meth:`MemoryService.upgrade`.

        Args:
            from_version: Source schema version. If ``None``, the adapter
                should detect the current version from the database.
            to_version: Target schema version. If ``None``, the adapter
                uses its own ``CURRENT_VERSION`` class attribute.

        Returns:
            The schema version after the upgrade (i.e. ``to_version``).
        """
        pass

    @abstractmethod
    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        """Get all memories with a specific tag for a user.

        Args:
            user_id: The user whose memories to search.
            tag: The tag to filter by.
            lifecycle_filter: Only include memories in these states.
            namespace: Filter by namespace (default: 'default').

        Returns:
            List of MemoryObjects with the specified tag.
        """

    def get_namespaces(self, user_id: str) -> list[str]:
        """Return the distinct namespaces that contain memories for a user.

        Default implementation scans ``get_all`` in Python.
        Storage adapters backed by SQL SHOULD override this with
        ``SELECT DISTINCT namespace FROM ...`` for O(namespaces) cost
        instead of O(memories) cost.

        Args:
            user_id: The user whose namespaces to retrieve.

        Returns:
            List of namespace strings.
        """
        namespaces: set[str] = set()
        for mem in self.get_all():
            if mem.user_id == user_id:
                namespaces.add(mem.namespace)
        return sorted(namespaces)

    @abstractmethod
    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Search for memories using keyword matching (no embeddings required).

        This is a fallback search method for when embeddings are not available
        or when keyword-only search is preferred. Uses BM25-style scoring.

        Args:
            user_id: The user whose memories to search.
            query: The search query string.
            top_k: Maximum number of results to return.
            lifecycle_filter: Only include memories in these states.
            namespace: Filter by namespace (default: 'default').
            session_id: Optional session ID filter.

        Returns:
            List of MemoryObjects matching the query, sorted by relevance.
        """
        pass

    def delete_expired(
        self,
        before: datetime,
        lifecycle_filter: list[LifecycleState] | None = None,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Delete memories whose ``expires_at`` is before *before*.

        Default implementation loops over ``get_all_by_user`` and deletes
        individually. Storage adapters backed by SQL SHOULD override this
        with a single ``DELETE`` statement for O(1) round-trip cost.

        Args:
            before: Cutoff datetime; memories with ``expires_at <= before``
                are deleted.
            lifecycle_filter: Only consider memories in these lifecycle states.
                Defaults to ``[ACTIVE, DECAYING]``.
            user_id: If provided, only sweep this user's memories.
            namespace: If provided, only sweep this namespace.

        Returns:
            Number of memories deleted.
        """
        deleted = 0
        target_states = lifecycle_filter or [LifecycleState.ACTIVE, LifecycleState.DECAYING]
        users = [user_id] if user_id is not None else self.get_all_users()
        for uid in users:
            namespaces = [namespace] if namespace is not None else self.get_namespaces(uid)
            for ns in namespaces:
                for mem in self.get_all_by_user(
                    uid, lifecycle_filter=list(target_states), namespace=ns
                ):
                    if mem.expires_at is not None and mem.expires_at <= before:
                        if self.delete_by_id(mem.memory_id):
                            deleted += 1
        return deleted
