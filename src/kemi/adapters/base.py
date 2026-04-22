from abc import ABC, abstractmethod

from kemi.models import LifecycleState, MemoryObject


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
    ) -> list[MemoryObject]:
        """Get all memories for a user.

        Used for:
        - Migration (re-embedding all memories)
        - Consolidation (v2)
        - Export/backup

        Args:
            user_id: The user whose memories to retrieve.
            lifecycle_filter: Only include memories in these states.

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

    @abstractmethod
    def get_all(self) -> list[MemoryObject]:
        """Get ALL memories from the store.

        Used for export/backup.

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
    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        """Migrate the storage schema between versions.

        Called by Memory.upgrade().

        Args:
            from_version: Current schema version.
            to_version: Target schema version.
        """
        pass

    @abstractmethod
    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Get all memories with a specific tag for a user.

        Args:
            user_id: The user whose memories to search.
            tag: The tag to filter by.
            lifecycle_filter: Only include memories in these states.

        Returns:
            List of MemoryObjects with the specified tag.
        """
        pass
