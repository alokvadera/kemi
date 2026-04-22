import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from kemi import dedup, lifecycle, sanitize, scoring
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.models import (
    LifecycleState,
    MemoryConfig,
    MemoryObject,
    MemorySource,
)

logger = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        embed: EmbeddingAdapter | None = None,
        store: StorageAdapter | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        if embed is None:
            try:
                from kemi.adapters.embedding.fastembed import FastEmbedAdapter

                self._embed: EmbeddingAdapter = FastEmbedAdapter()
            except ImportError as e:
                raise ImportError(
                    "No embedding adapter provided and fastembed is not installed. "
                    "Install with: pip install kemi[local] or provide your own: "
                    "Memory(embed=YourAdapter())"
                ) from e
        else:
            self._embed = embed

        if store is None:
            from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

            default_db_path = os.path.join(os.path.expanduser("~"), ".kemi", "memories.db")
            os.makedirs(os.path.dirname(default_db_path), exist_ok=True)
            self._store: StorageAdapter = SQLiteStorageAdapter(db_path=default_db_path)
        else:
            self._store = store

        if config is None:
            self._config: MemoryConfig = MemoryConfig()
        else:
            self._config = config

    def remember(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        metadata: dict[str, Any] | None = None,
        sanitize_input: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty — there is nothing to remember")
        if not isinstance(importance, (int, float)):
            raise TypeError(
                f"importance must be a number between 0.0 and 1.0, got {type(importance).__name__}"
            )

        if sanitize_input:
            content = sanitize.sanitize(content, strict=self._config.sanitize)

        embedding = self._embed.embed_single(content)
        embedding_dim = len(embedding)

        clamped_importance = max(0.0, min(1.0, importance))

        new_memory = MemoryObject(
            memory_id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            embedding=embedding,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=source,
            importance=clamped_importance,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata=metadata or {},
            embedding_dim=embedding_dim,
            tags=tags or [],
        )

        existing = self._store.get_all_by_user(
            user_id,
            lifecycle_filter=[
                LifecycleState.ACTIVE,
                LifecycleState.DECAYING,
                LifecycleState.ARCHIVED,
            ],
        )

        duplicates = dedup.find_duplicates(new_memory, existing, self._config.dedup_threshold)

        if duplicates:
            resolved = dedup.resolve_duplicate(new_memory, duplicates[0])
            self._store.update(resolved)
            logger.info(f"Resolved duplicate for user {user_id}: {resolved.memory_id}")
            return resolved.memory_id

        conflicts = dedup.find_conflicts(
            new_memory,
            existing,
            self._config.conflict_threshold,
            self._config.dedup_threshold,
        )

        if conflicts:
            new_memory.metadata["conflict_flagged"] = True
            logger.warning(
                f"Potential conflict detected for user {user_id}: "
                f"new memory '{content[:50]}...' conflicts with existing memory "
                f"'{conflicts[0].content[:50]}...'"
            )

        self._store.store(new_memory)
        return new_memory.memory_id

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
    ) -> list[MemoryObject]:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValueError("query cannot be empty — what should kemi search for?")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        if hybrid_search is None:
            hybrid_search = self._config.hybrid_search

        query_embedding = self._embed.embed_single(query)

        if lifecycle_filter is None:
            lifecycle_filter = lifecycle.get_recall_filter()

        search_results = self._store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            lifecycle_filter=lifecycle_filter,
        )

        current_dim = self._embed.dimension()
        if search_results:
            stored_dim = search_results[0].embedding_dim
            if stored_dim is not None and stored_dim != current_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: stored memories use {stored_dim} dimensions "
                    f"but current adapter produces {current_dim} dimensions. "
                    f"Run memory.migrate(user_id, new_adapter) to re-embed your memories."
                )

        ranked = scoring.rank_memories(search_results, query_embedding, query, hybrid_search)

        if len(ranked) > top_k and query_embedding is not None and top_k > 1:
            ranked = scoring.mmr_rerank(ranked, query_embedding, top_k, lambda_param=0.7)

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._config.max_tokens_default
        )

        if effective_max_tokens is not None:
            truncated = scoring.truncate_by_tokens(ranked, effective_max_tokens)
        else:
            truncated = ranked

        final_results = truncated[:top_k]

        for mem in final_results:
            mem.last_accessed_at = datetime.now(timezone.utc)

            new_state = lifecycle.evaluate_lifecycle(mem, self._config.decay_threshold_hours)

            if new_state != mem.lifecycle_state:
                updated = lifecycle.transition(mem, new_state)
                self._store.update(updated)

        return final_results

    def forget(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> int:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        if memory_id is not None:
            deleted = self._store.delete_by_id(memory_id)
            return 1 if deleted else 0
        else:
            count = self._store.delete_by_user(user_id)
            return count

    def context_block(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int = 1500,
        prefix: str = "Relevant context from memory:",
    ) -> str:
        memories = self.recall(
            user_id=user_id,
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        if not memories:
            return ""

        lines = [prefix]
        for mem in memories:
            lines.append(f"- {mem.content}")

        return "\n".join(lines)

    async def aremember(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        metadata: dict[str, Any] | None = None,
        sanitize_input: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        import asyncio
        import warnings

        return await asyncio.to_thread(
            self.remember, user_id, content, importance, source, metadata, sanitize_input, tags
        )

    async def arecall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
    ) -> list[MemoryObject]:
        import asyncio

        return await asyncio.to_thread(
            self.recall, user_id, query, top_k, max_tokens, lifecycle_filter, hybrid_search
        )

    async def aforget(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> int:
        import asyncio

        return await asyncio.to_thread(self.forget, user_id, memory_id)

    async def acontext_block(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int = 1500,
        prefix: str = "Relevant context from memory:",
    ) -> str:
        import asyncio

        return await asyncio.to_thread(
            self.context_block, user_id, query, top_k, max_tokens, prefix
        )

    def migrate(
        self,
        user_id: str,
        new_embed_fn: EmbeddingAdapter,
        batch_size: int = 100,
    ) -> int:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        memories = self._store.get_all_by_user(
            user_id,
            lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
        )

        if not memories:
            return 0

        count = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            contents = [m.content for m in batch]
            new_embeddings = new_embed_fn.embed(contents)
            new_dim = new_embed_fn.dimension()

            for j, mem in enumerate(batch):
                mem.embedding = new_embeddings[j]
                mem.embedding_dim = new_dim
                self._store.update(mem)
                count += 1

        logger.info(f"Migrated {count} memories for user {user_id}")
        return count

    def export(self, file_path: str) -> int:
        """Export all memories to a JSON file."""
        import json

        all_memories = self._store.get_all()
        memories_data = []
        for mem in all_memories:
            memories_data.append(
                {
                    "memory_id": mem.memory_id,
                    "user_id": mem.user_id,
                    "content": mem.content,
                    "embedding": mem.embedding,
                    "score": mem.score,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None,
                    "last_accessed_at": mem.last_accessed_at.isoformat()
                    if mem.last_accessed_at
                    else None,
                    "source": mem.source.value if mem.source else None,
                    "importance": mem.importance,
                    "lifecycle_state": mem.lifecycle_state.value if mem.lifecycle_state else None,
                    "metadata": mem.metadata,
                    "embedding_dim": mem.embedding_dim,
                    "tags": mem.tags,
                }
            )

        with open(file_path, "w") as f:
            json.dump(memories_data, f, indent=2)

        logger.info(f"Exported {len(memories_data)} memories to {file_path}")
        return len(memories_data)

    def import_from(self, file_path: str) -> int:
        """Import memories from a JSON file."""
        import json

        with open(file_path) as f:
            memories_data = json.load(f)

        imported_count = 0
        for mem_data in memories_data:
            existing = self._store.get(mem_data["memory_id"])
            if existing is not None:
                continue

            from datetime import datetime

            from kemi.models import LifecycleState, MemorySource

            created_at = (
                datetime.fromisoformat(mem_data["created_at"])
                if mem_data.get("created_at")
                else datetime.now(timezone.utc)
            )
            last_accessed_at = (
                datetime.fromisoformat(mem_data["last_accessed_at"])
                if mem_data.get("last_accessed_at")
                else datetime.now(timezone.utc)
            )

            memory = MemoryObject(
                memory_id=mem_data["memory_id"],
                user_id=mem_data["user_id"],
                content=mem_data["content"],
                embedding=mem_data.get("embedding"),
                score=mem_data.get("score", 0.0),
                created_at=created_at,
                last_accessed_at=last_accessed_at,
                source=MemorySource(mem_data["source"])
                if mem_data.get("source")
                else MemorySource.USER_STATED,
                importance=mem_data.get("importance", 0.5),
                lifecycle_state=LifecycleState(mem_data["lifecycle_state"])
                if mem_data.get("lifecycle_state")
                else LifecycleState.ACTIVE,
                metadata=mem_data.get("metadata", {}),
                embedding_dim=mem_data.get("embedding_dim"),
                tags=mem_data.get("tags", []),
            )

            self._store.store(memory)
            imported_count += 1

        logger.info(f"Imported {imported_count} memories from {file_path}")
        return imported_count

    async def aexport(self, file_path: str) -> int:
        import asyncio

        return await asyncio.to_thread(self.export, file_path)

    async def aimport_from(self, file_path: str) -> int:
        import asyncio

        return await asyncio.to_thread(self.import_from, file_path)

    def upgrade(self) -> None:
        self._store.upgrade_schema(from_version=1, to_version=1)
        logger.info("Schema upgraded to version 1")

    def remember_many(
        self,
        user_id: str,
        contents: list[str],
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Store multiple memories at once.

        Args:
            user_id: User ID.
            contents: List of content strings to remember.
            importance: Importance value (0.0-1.0) for all.
            source: Memory source.
            tags: Optional list of tags to apply to all memories.

        Returns:
            List of memory IDs.
        """
        if not contents:
            return []

        memory_ids = []
        for content in contents:
            memory_id = self.remember(user_id, content, importance, source, tags=tags)
            memory_ids.append(memory_id)
        return memory_ids

    def list_users(self) -> list[str]:
        """Get all unique user IDs that have memories.

        Returns:
            List of user IDs.
        """
        return self._store.get_all_users()

    def stats(self, user_id: str) -> dict:
        """Return health statistics for a user's memory store.

        Returns a dict with these keys:
          total: int - total number of memories
          by_lifecycle: dict - count per lifecycle state
            e.g. {"active": 10, "decaying": 3, "archived": 1, "deleted": 0}
          by_source: dict - count per memory source
            e.g. {"user_stated": 8, "agent_inferred": 5}
          avg_importance: float - average importance score (0.0 if no memories)
          tag_counts: dict - how many memories each tag appears in
            e.g. {"food": 3, "work": 7}
          total_with_tags: int - number of memories that have at least one tag
          total_without_tags: int - number of memories with no tags
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        all_memories = self._store.get_all_by_user(user_id, lifecycle_filter=None)

        by_lifecycle = {state.value: 0 for state in LifecycleState}
        by_source = {source.value: 0 for source in MemorySource}
        tag_counts: dict[str, int] = {}
        total_with_tags = 0
        total_importance = 0.0

        for mem in all_memories:
            by_lifecycle[mem.lifecycle_state.value] += 1
            by_source[mem.source.value] += 1
            total_importance += mem.importance

            if mem.tags:
                total_with_tags += 1
                for tag in mem.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        total = len(all_memories)
        avg_importance = total_importance / total if total > 0 else 0.0
        total_without_tags = total - total_with_tags

        return {
            "total": total,
            "by_lifecycle": by_lifecycle,
            "by_source": by_source,
            "avg_importance": avg_importance,
            "tag_counts": tag_counts,
            "total_with_tags": total_with_tags,
            "total_without_tags": total_without_tags,
        }

    async def astats(self, user_id: str) -> dict:
        """Async version of stats()."""
        import asyncio

        return await asyncio.to_thread(self.stats, user_id)

    def recall_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Recall memories by tag.

        Args:
            user_id: User ID to search for.
            tag: Tag to filter by.
            lifecycle_filter: Filter by lifecycle state.

        Returns:
            List of MemoryObjects with the specified tag.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not tag or not tag.strip():
            raise ValueError("tag cannot be empty")

        return self._store.get_by_tag(user_id, tag, lifecycle_filter)

    async def arecall_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Async version of recall_by_tag()."""
        import asyncio

        return await asyncio.to_thread(self.recall_by_tag, user_id, tag, lifecycle_filter)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
    ) -> str:
        """Update an existing memory.

        Args:
            memory_id: ID of memory to update.
            content: New content (if provided, will re-embed).
            importance: New importance value (0.0-1.0).

        Returns:
            The memory_id of updated memory.

        Raises:
            ValueError: If memory_id not found.
        """
        if content is None and importance is None:
            return memory_id

        memory = self._store.get(memory_id)
        if memory is None:
            raise ValueError(f"Memory not found: {memory_id}")

        if content is not None:
            memory.content = content
            memory.embedding = self._embed.embed_single(content)
            memory.embedding_dim = len(memory.embedding)

        if importance is not None:
            memory.importance = max(0.0, min(1.0, importance))

        self._store.update(memory)
        logger.info(f"Updated memory: {memory_id}")
        return memory_id

    def recall_since(
        self,
        user_id: str,
        query: str,
        hours: float = 24.0,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Recall memories created in the last N hours.

        Args:
            user_id: User ID to search for.
            query: Search query.
            hours: Only return memories created in last N hours.
            top_k: Maximum memories to return.
            max_tokens: Token budget for context_block.
            lifecycle_filter: Filter by lifecycle state.

        Returns:
            List of MemoryObjects.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        all_results = self.recall(
            user_id=user_id,
            query=query,
            top_k=top_k * 3,
            max_tokens=max_tokens,
            lifecycle_filter=lifecycle_filter,
        )

        filtered = [m for m in all_results if m.created_at and m.created_at >= cutoff]
        return filtered[:top_k]

    async def alist_users(self) -> list[str]:
        """Async version of list_users()."""
        import asyncio

        return await asyncio.to_thread(self.list_users)

    async def aupdate(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
    ) -> str:
        """Async version of update()."""
        import asyncio

        return await asyncio.to_thread(self.update, memory_id, content, importance)

    async def arecall_since(
        self,
        user_id: str,
        query: str,
        hours: float = 24.0,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Async version of recall_since()."""
        import asyncio

        return await asyncio.to_thread(
            self.recall_since, user_id, query, hours, top_k, max_tokens, lifecycle_filter
        )

    async def aremember_many(
        self,
        user_id: str,
        contents: list[str],
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Async version of remember_many()."""
        import asyncio

        return await asyncio.to_thread(
            self.remember_many, user_id, contents, importance, source, tags
        )
