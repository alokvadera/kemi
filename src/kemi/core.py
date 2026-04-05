import logging
import uuid
from datetime import datetime
from typing import Any, Optional

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
        embed: Optional[EmbeddingAdapter] = None,
        store: Optional[StorageAdapter] = None,
        config: Optional[MemoryConfig] = None,
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

            self._store: StorageAdapter = SQLiteStorageAdapter(db_path="kemi.db")
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
        metadata: Optional[dict[str, Any]] = None,
        sanitize_input: bool = False,
    ) -> str:
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
            created_at=datetime.utcnow(),
            last_accessed_at=datetime.utcnow(),
            source=source,
            importance=clamped_importance,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata=metadata or {},
            embedding_dim=embedding_dim,
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
        max_tokens: Optional[int] = None,
        lifecycle_filter: Optional[list[LifecycleState]] = None,
    ) -> list[MemoryObject]:
        query_embedding = self._embed.embed_single(query)

        if lifecycle_filter is None:
            lifecycle_filter = lifecycle.get_recall_filter()

        search_results = self._store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            lifecycle_filter=lifecycle_filter,
        )

        ranked = scoring.rank_memories(search_results, query_embedding)

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._config.max_tokens_default
        )

        if effective_max_tokens is not None:
            truncated = scoring.truncate_by_tokens(ranked, effective_max_tokens)
        else:
            truncated = ranked

        final_results = truncated[:top_k]

        for mem in final_results:
            mem.last_accessed_at = datetime.utcnow()

            new_state = lifecycle.evaluate_lifecycle(mem, self._config.decay_threshold_hours)

            if new_state != mem.lifecycle_state:
                updated = lifecycle.transition(mem, new_state)
                self._store.update(updated)

        return final_results

    def forget(
        self,
        user_id: str,
        memory_id: Optional[str] = None,
    ) -> int:
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

    def migrate(
        self,
        user_id: str,
        new_embed_fn: EmbeddingAdapter,
        batch_size: int = 100,
    ) -> int:
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

    def upgrade(self) -> None:
        self._store.upgrade_schema(from_version=1, to_version=1)
        logger.info("Schema upgraded to version 1")
