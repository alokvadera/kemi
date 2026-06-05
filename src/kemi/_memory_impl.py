from __future__ import annotations

import logging
import os
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kemi.encryption import EncryptionConfig

from kemi.versions import (
    DiffResult,
    MemoryVersionStore,
    RollbackResult,
    VersionSnapshot,
)
from kemi.webhooks import WebhookDispatcher, WebhookEventType, WebhookStore, build_payload


def _memory_to_dict(memory: "MemoryObject") -> dict[str, Any]:
    """Convert a MemoryObject to a JSON-serialisable dict for webhook payloads."""
    if not isinstance(memory, MemoryObject):
        return {}
    return {
        "memory_id": memory.memory_id,
        "content": memory.content,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "lifecycle_state": memory.lifecycle_state.value if memory.lifecycle_state else None,
        "memory_type": memory.memory_type.value if memory.memory_type else None,
        "source": memory.source.value if memory.source else None,
        "tags": memory.tags,
        "namespace": memory.namespace,
        "session_id": memory.session_id,
        "version": memory.version,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
        "metadata": memory.metadata,
        "agent_id": memory.agent_id,
        "run_id": memory.run_id,
        "app_id": memory.app_id,
    }

from kemi import dedup, lifecycle, sanitize, scoring
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.entities import EntityLinker, NoopEntityLinker, RegexEntityLinker
from kemi.models import (
    LifecycleState,
    MemoryConfig,
    MemoryObject,
    MemorySource,
    MemoryType,
)

logger = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        embed: EmbeddingAdapter | None = None,
        store: StorageAdapter | None = None,
        config: MemoryConfig | None = None,
        encryption: "EncryptionConfig | None" = None,
        entity_linker: "EntityLinker | None" = None,
    ) -> None:
        # Lazy import to avoid circular dependencies
        from kemi.encryption import EncryptionConfig

        if encryption is None:
            try:
                encryption = EncryptionConfig.from_env()
            except Exception:
                # Broad catch intentional: EncryptionConfig.from_env() reads
                # env vars and can fail for many reasons (missing key, bad
                # Fernet format, base64 decode errors). Encryption is opt-in;
                # fall back to disabled rather than blocking Memory init.
                encryption = None
        if embed is None:
            try:  # pragma: no cover
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
            # Honor an explicit KEMI_DB_PATH so background-task workers and
            # ad-hoc Memory() instantiations point at the same database as
            # the main app, rather than silently defaulting to ~/.kemi.
            env_path = os.environ.get("KEMI_DB_PATH")
            if env_path:
                default_db_path = os.path.expanduser(env_path)
            else:
                default_db_path = os.path.join(os.path.expanduser("~"), ".kemi", "memories.db")
            os.makedirs(os.path.dirname(default_db_path), exist_ok=True)

            try:
                from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter

                if SQLiteVecStorageAdapter.is_vec_available():
                    embedding_dim = self._embed.dimension()
                    self._store: StorageAdapter = SQLiteVecStorageAdapter(
                        db_path=default_db_path,
                        embedding_dim=embedding_dim,
                        encryption=encryption if isinstance(encryption, EncryptionConfig) and encryption.enabled else None,
                    )
                    logger.info(
                        "Using SQLiteVecStorageAdapter with ANN vector search "
                        "(sqlite-vec installed)"
                    )
                else:
                    from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

                    self._store = SQLiteStorageAdapter(
                        db_path=default_db_path,
                        encryption=encryption if isinstance(encryption, EncryptionConfig) and encryption.enabled else None,
                    )
            except ImportError:  # pragma: no cover
                from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

                self._store = SQLiteStorageAdapter(
                    db_path=default_db_path,
                    encryption=encryption if isinstance(encryption, EncryptionConfig) and encryption.enabled else None,
                )  # pragma: no cover
        else:
            self._store = store

        if config is None:
            self._config: MemoryConfig = MemoryConfig()
        else:
            self._config = config

        # Optional observability
        self._metrics: Any | None = None
        try:
            from kemi.observability import get_metrics_collector

            self._metrics = get_metrics_collector()
        except ImportError:
            pass

        self._audit_trail: Any | None = None
        self._adaptive_retriever: Any | None = None
        self._event_hooks: dict[str, list[Callable[..., Any]]] = {"pre": [], "post": []}
        self._query_cache: _QueryCache | None = None
        self._version_store: MemoryVersionStore | None = None
        self._max_versions_per_memory: int = 50
        self._auto_prune_versions: bool = True
        self._webhook_dispatcher: WebhookDispatcher | None = None

        if entity_linker is not None:
            self._entity_linker: EntityLinker = entity_linker
        elif self._config.enable_entity_boost:
            self._entity_linker = RegexEntityLinker()
        else:
            self._entity_linker = NoopEntityLinker()

    def _latency_tracker(self, operation: str) -> Any:
        """Return a context manager that tracks operation latency if metrics are enabled."""
        from kemi.operations import _ops_metrics
        return _ops_metrics.latency_tracker(self, operation)

    def remember(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        metadata: dict[str, Any] | None = None,
        sanitize_input: bool = False,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty — there is nothing to remember")
        if not isinstance(importance, (int, float)):
            raise TypeError(
                f"importance must be a number between 0.0 and 1.0, got {type(importance).__name__}"
            )
        if ttl_seconds is not None and (
            not isinstance(ttl_seconds, int) or ttl_seconds <= 0
        ):
            raise ValueError(f"ttl_seconds must be a positive integer, got {ttl_seconds}")

        with self._latency_tracker("remember"):
            if sanitize_input:
                content = sanitize.sanitize(content, strict=self._config.sanitize)

            try:
                embedding = self._embed.embed_single(content)
            except Exception:
                # Broad catch intentional: embedding adapters wrap many
                # different backends (fastembed, OpenAI, custom) each with
                # their own exception types. We just need to count the
                # error in metrics, then re-raise the original.
                self._record_embed_error()
                raise
            embedding_dim = len(embedding)

            clamped_importance = max(0.0, min(1.0, importance))

            self._run_hooks(
                "pre",
                "remember",
                user_id=user_id,
                content=content,
                namespace=namespace,
            )

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
                confidence=max(0.0, min(1.0, confidence)),
                memory_type=memory_type,
                session_id=session_id,
                namespace=namespace,
                version=1,
                agent_id=agent_id,
                run_id=run_id,
                app_id=app_id,
                expires_at=(
                    datetime.now(timezone.utc).replace(microsecond=0)
                    + timedelta(seconds=ttl_seconds)
                    if ttl_seconds is not None
                    else None
                ),
            )

            existing = self._store.get_all_by_user(
                user_id,
                lifecycle_filter=[
                    LifecycleState.ACTIVE,
                    LifecycleState.DECAYING,
                    LifecycleState.ARCHIVED,
                ],
                namespace=namespace,
            )

            duplicates = dedup.find_duplicates(new_memory, existing, self._config.dedup_threshold)

            if duplicates:
                resolved = dedup.resolve_duplicate(new_memory, duplicates[0])
                # Content changed during merge — invalidate stale cached entities
                resolved.metadata.pop("extracted_entities", None)
                # Record version BEFORE overwriting
                try:
                    vs = self._get_version_store()
                    vs.record_version(duplicates[0], changed_by="merge")
                    self._auto_prune_versions_for_memory(duplicates[0].memory_id)
                except (RuntimeError, Exception):
                    pass
                self._store.update(resolved)
                # Remove the other near-duplicates so they don't re-trigger on
                # the next insert. duplicates[0] is the canonical we merged into.
                for extra in duplicates[1:]:
                    if extra.memory_id != resolved.memory_id:
                        self._store.delete_by_id(extra.memory_id)
                # Dispatch updated webhook for merged duplicate
                snapshot = _memory_to_dict(resolved)
                self._dispatch_webhook_event(
                    WebhookEventType.UPDATED,
                    memory_id=resolved.memory_id,
                    user_id=user_id,
                    snapshot=snapshot,
                )
                logger.info(f"Resolved duplicate for user {user_id}: {resolved.memory_id}")
                if self._metrics is not None:
                    self._metrics.duplicates_detected.inc(1)
                self._track_operation(
                    "remember",
                    user_id,
                    {"memory_id": resolved.memory_id, "duplicate": True},
                    resolved.memory_id,
                    namespace,
                )
                return resolved.memory_id

            conflicts = dedup.find_conflicts(
                new_memory,
                existing,
                self._config.conflict_threshold,
                self._config.dedup_threshold,
            )

            conflict_detected = False
            if conflicts:
                new_memory.metadata["conflict_flagged"] = True
                conflict_detected = True
                logger.warning(
                    f"Potential conflict detected for user {user_id}: "
                    f"new memory '{content[:50]}...' conflicts with existing memory "
                    f"'{conflicts[0].content[:50]}...'"
                )
                if self._metrics is not None:
                    self._metrics.conflicts_detected.inc(1)

            if self._config.enable_entity_boost:
                new_memory.metadata["extracted_entities"] = list(
                    self._entity_linker.extract(content)
                )

            try:
                self._store.store(new_memory)
            except Exception:
                # Broad catch intentional: storage adapters can raise from
                # many layers (SQLite, JSON, Postgres, encryption). Record
                # the error in metrics and re-raise the original.
                self._record_store_error()
                raise
            if self._metrics is not None:
                self._metrics.embed_total.inc(1)
                self._metrics.embed_bytes_total.inc(len(content))
                self._metrics.total_memories.set(self._store.count(user_id))

            # Dispatch webhooks
            snapshot = _memory_to_dict(new_memory)
            self._dispatch_webhook_event(
                WebhookEventType.REMEMBERED,
                memory_id=new_memory.memory_id,
                user_id=user_id,
                snapshot=snapshot,
            )
            if conflict_detected:
                self._dispatch_webhook_event(
                    WebhookEventType.CONFLICT,
                    memory_id=new_memory.memory_id,
                    user_id=user_id,
                    snapshot=snapshot,
                    conflict_with=conflicts[0].memory_id,
                )

            remember_details: dict[str, Any] = {
                "memory_id": new_memory.memory_id,
                "content_length": len(content),
            }
            if conflict_detected:
                remember_details["conflict"] = True
                remember_details["conflict_with"] = conflicts[0].memory_id
            self._run_hooks(
                "post",
                "remember",
                user_id=user_id,
                memory_id=new_memory.memory_id,
                namespace=namespace,
            )
            self._track_operation(
                "remember",
                user_id,
                remember_details,
                new_memory.memory_id,
                namespace,
            )
            return new_memory.memory_id

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryObject]:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValueError("query cannot be empty — what should kemi search for?")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        with self._latency_tracker("recall"):
            if hybrid_search is None:
                hybrid_search = self._config.hybrid_search

            query_embedding = self._embed.embed_single(query)

            if lifecycle_filter is None:
                lifecycle_filter = lifecycle.get_recall_filter()

            # Check query cache
            if self._query_cache is not None:
                cache_key = self._query_cache._make_key(
                    user_id,
                    query,
                    top_k,
                    max_tokens,
                    lifecycle_filter,
                    hybrid_search,
                    namespace,
                    session_id,
                    metadata_filter,
                )
                cached = self._query_cache.get(cache_key)
                if cached is not None:
                    self._track_operation(
                        "recall",
                        user_id,
                        {"query": query, "results_count": len(cached), "cache_hit": True},
                        namespace=namespace,
                    )
                    return cached

            self._run_hooks("pre", "recall", user_id=user_id, query=query, namespace=namespace)

            # When metadata_filter is active we may need more than top_k
            # results from storage because filtering is applied post-hoc.
            # Use a larger multiplier to increase the chance of returning
            # top_k results after filtering.
            fetch_multiplier = 10 if metadata_filter is not None else 3
            search_results = self._store.search(
                user_id=user_id,
                query_embedding=query_embedding,
                top_k=top_k * fetch_multiplier,
                lifecycle_filter=lifecycle_filter,
                namespace=namespace,
                session_id=session_id,
            )

            if metadata_filter is not None:
                search_results = [
                    m
                    for m in search_results
                    if all(m.metadata.get(k) == v for k, v in metadata_filter.items())
                ]

            current_dim = self._embed.dimension()
            if search_results:
                stored_dim = search_results[0].embedding_dim
                if stored_dim is not None and stored_dim != current_dim:
                    raise ValueError(
                        "Embedding dimension mismatch: stored memories use "
                        f"{stored_dim} dimensions but current adapter produces "
                        f"{current_dim} dimensions. Run memory.migrate(user_id, "
                        "new_adapter) to re-embed your memories."
                    )

            query_entities: set[str] | None = None
            memory_entities_map: dict[str, set[str]] | None = None
            if self._config.enable_entity_boost:
                query_entities = self._entity_linker.extract(query)
                memory_entities_map = {}
                for m in search_results:
                    cached = m.metadata.get("extracted_entities")
                    if cached is not None:
                        memory_entities_map[m.memory_id] = set(cached)
                    else:
                        memory_entities_map[m.memory_id] = self._entity_linker.extract(m.content)

            ranked = scoring.rank_memories(
                search_results,
                query_embedding,
                query,
                hybrid_search,
                weight_semantic=self._config.weight_semantic,
                weight_recency=self._config.weight_recency,
                weight_bm25=self._config.weight_bm25,
                weight_semantic_no_embed=self._config.weight_semantic_no_embed,
                weight_recency_no_embed=self._config.weight_recency_no_embed,
                weight_importance=self._config.weight_importance,
                query_entities=query_entities,
                memory_entities_map=memory_entities_map,
                weight_entity=self._config.entity_boost_weight,
            )

            if len(ranked) > top_k and top_k > 1:
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
                    if self._metrics is not None:
                        self._metrics.lifecycle_transitions.inc(1)

            if self._metrics is not None:
                self._metrics.total_memories.set(self._store.count(user_id))
            if self._query_cache is not None:
                cache_key = self._query_cache._make_key(
                    user_id,
                    query,
                    top_k,
                    max_tokens,
                    lifecycle_filter,
                    hybrid_search,
                    namespace,
                    session_id,
                    metadata_filter,
                )
                self._query_cache.put(cache_key, final_results)

            self._run_hooks(
                "post",
                "recall",
                user_id=user_id,
                query=query,
                results=final_results,
                namespace=namespace,
            )
            self._track_operation(
                "recall",
                user_id,
                {"query": query, "results_count": len(final_results), "cache_hit": False},
                namespace=namespace,
            )
            if self._adaptive_retriever is not None:
                try:
                    profile = self._adaptive_retriever.analyze_query(query)
                    self._adaptive_retriever.record_feedback(user_id, query, profile)
                except Exception:
                    logger.debug("Adaptive retrieval analysis failed", exc_info=True)
            return final_results

    def recall_many(
        self,
        user_ids: list[str],
        queries: list[str],
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, list[MemoryObject]]:
        """Recall memories for multiple users and queries at once.

        Args:
            user_ids: List of user IDs to recall for.
            queries: List of query strings (same length as user_ids).
            top_k: Max memories per result.
            max_tokens: Token budget.
            lifecycle_filter: Optional lifecycle state filter.
            hybrid_search: Override hybrid search.
            namespace: Memory namespace.
            session_id: Optional session ID filter.
            metadata_filter: Optional metadata key-value filter dict.

        Returns:
            Dict mapping user_id -> list of MemoryObjects.
        """
        if len(user_ids) != len(queries):
            raise ValueError("user_ids and queries must have the same length")
        results: dict[str, list[MemoryObject]] = {}
        for uid, q in zip(user_ids, queries, strict=True):
            results[uid] = self.recall(
                user_id=uid,
                query=q,
                top_k=top_k,
                max_tokens=max_tokens,
                lifecycle_filter=lifecycle_filter,
                hybrid_search=hybrid_search,
                namespace=namespace,
                session_id=session_id,
                metadata_filter=metadata_filter,
            )
        return results

    def update_many(
        self,
        memory_ids: list[str],
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Update multiple memories at once.

        Args:
            memory_ids: List of memory IDs to update.
            content: New content for all (if provided, will re-embed).
            importance: New importance for all.
            confidence: New confidence for all.
            memory_type: New memory type for all.

        Returns:
            List of updated memory IDs.
        """
        updated: list[str] = []
        for mid in memory_ids:
            self.update(
                mid,
                content=content,
                importance=importance,
                confidence=confidence,
                memory_type=memory_type,
                metadata=metadata,
            )
            updated.append(mid)
        return updated

    def forget_many(
        self,
        memory_ids: list[str],
    ) -> int:
        """Delete multiple memories by ID at once.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            Number of memories deleted.

        Note:
            Unlike :meth:`forget`, this method does not fire pre/post
            event hooks for each individual delete. Hooks are intentionally
            skipped for batch performance.
        """
        count = 0
        for mid in memory_ids:
            if self._store.delete_by_id(mid):
                count += 1
        return count

    def forget(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> int:
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        with self._latency_tracker("forget"):
            self._run_hooks(
                "pre", "forget", user_id=user_id, memory_id=memory_id, namespace="default"
            )
            if memory_id is not None:
                deleted = self._store.delete_by_id(memory_id)
                if deleted:
                    self._dispatch_webhook_event(
                        WebhookEventType.DELETED,
                        memory_id=memory_id,
                        user_id=user_id,
                    )
                self._run_hooks(
                    "post",
                    "forget",
                    user_id=user_id,
                    memory_id=memory_id,
                    deleted=deleted,
                    namespace="default",
                )
                self._track_operation(
                    "forget",
                    user_id,
                    {"memory_id": memory_id, "deleted": deleted},
                    memory_id,
                    namespace="default",
                )
                return 1 if deleted else 0
            else:
                count = self._store.delete_by_user(user_id)
                if count:
                    self._dispatch_webhook_event(
                        WebhookEventType.DELETED,
                        memory_id="batch",
                        user_id=user_id,
                        deleted_count=count,
                    )
                self._run_hooks(
                    "post", "forget", user_id=user_id, deleted_count=count, namespace="default"
                )
                self._track_operation(
                    "forget", user_id, {"deleted_count": count}, namespace="default"
                )
                return count

    def context_block(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int = 1500,
        prefix: str = "Relevant context from memory:",
        namespace: str = "default",
        session_id: str | None = None,
    ) -> str:
        memories = self.recall(
            user_id=user_id,
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
            namespace=namespace,
            session_id=session_id,
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
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
    ) -> str:
        import asyncio

        return await asyncio.to_thread(
            self.remember,
            user_id,
            content,
            importance,
            source,
            metadata,
            sanitize_input,
            tags,
            namespace,
            session_id,
            memory_type,
            confidence,
            agent_id,
            run_id,
            app_id,
        )

    async def recall_stream(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ):
        """Stream recall results as an async generator.

        Scores all candidate memories, then yields each one as MMR reranking
        selects it, providing progressive delivery instead of waiting for
        full ranking.

        Args:
            Same as :meth:`recall`.

        Yields:
            MemoryObject instances in ranked order.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValueError("query cannot be empty — what should kemi search for?")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        import asyncio

        if hybrid_search is None:
            hybrid_search = self._config.hybrid_search

        query_embedding = await asyncio.to_thread(self._embed.embed_single, query)

        if lifecycle_filter is None:
            lifecycle_filter = lifecycle.get_recall_filter()

        # Run the synchronous store search in a thread
        search_results = await asyncio.to_thread(
            self._store.search,
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        if metadata_filter is not None:
            search_results = [
                m
                for m in search_results
                if all(m.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        current_dim = self._embed.dimension()
        if search_results:
            stored_dim = search_results[0].embedding_dim
            if stored_dim is not None and stored_dim != current_dim:
                raise ValueError(
                    "Embedding dimension mismatch: stored memories use "
                    f"{stored_dim} dimensions but current adapter produces "
                    f"{current_dim} dimensions. Run memory.migrate(user_id, "
                    "new_adapter) to re-embed your memories."
                )

        # Entity extraction for boost
        query_entities_stream: set[str] | None = None
        memory_entities_map_stream: dict[str, set[str]] | None = None
        if self._config.enable_entity_boost:
            query_entities_stream = self._entity_linker.extract(query)
            memory_entities_map_stream = {}
            for m in search_results:
                cached = m.metadata.get("extracted_entities")
                if cached is not None:
                    memory_entities_map_stream[m.memory_id] = set(cached)
                else:
                    memory_entities_map_stream[m.memory_id] = self._entity_linker.extract(m.content)

        # Score all candidates (same as rank_memories)
        corpus = [m.content for m in search_results] if len(search_results) > 1 else None
        for memory in search_results:
            mem_entities = None
            if memory_entities_map_stream is not None:
                mem_entities = memory_entities_map_stream.get(memory.memory_id)
            memory.score = scoring.score_memory(
                memory,
                query_embedding,
                query,
                hybrid_search,
                corpus,
                self._config.weight_semantic,
                self._config.weight_recency,
                self._config.weight_bm25,
                self._config.weight_semantic_no_embed,
                self._config.weight_recency_no_embed,
                self._config.weight_importance,
                query_entities_stream,
                mem_entities,
                self._config.entity_boost_weight,
            )

        # Sort by score descending first so mmr_rerank_stream gets pre-sorted input
        search_results.sort(key=lambda m: m.score, reverse=True)

        # Truncate to token budget before MMR
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._config.max_tokens_default
        )

        if effective_max_tokens is not None:
            search_results = scoring.truncate_by_tokens(search_results, effective_max_tokens)

        # Apply MMR and yield progressively
        yielded_memories: list[MemoryObject] = []
        for memory in scoring.mmr_rerank_stream(
            search_results, query_embedding, top_k, lambda_param=0.7
        ):
            # Update lifecycle and access time
            memory.last_accessed_at = datetime.now(timezone.utc)
            new_state = lifecycle.evaluate_lifecycle(memory, self._config.decay_threshold_hours)
            if new_state != memory.lifecycle_state:
                updated = lifecycle.transition(memory, new_state)
                self._store.update(updated)
                if self._metrics is not None:
                    self._metrics.lifecycle_transitions.inc(1)
            yielded_memories.append(memory)
            yield memory

        # Update metrics after all yielded
        if self._metrics is not None:
            self._metrics.total_memories.set(self._store.count(user_id))
        self._run_hooks(
            "post",
            "recall",
            user_id=user_id,
            query=query,
            results=yielded_memories,
            namespace=namespace,
        )
        self._track_operation(
            "recall",
            user_id,
            {"query": query, "results_count": len(yielded_memories), "cache_hit": False, "stream": True},
            namespace=namespace,
        )
        if self._adaptive_retriever is not None:
            try:
                profile = self._adaptive_retriever.analyze_query(query)
                self._adaptive_retriever.record_feedback(user_id, query, profile)
            except Exception:
                logger.debug("Adaptive retrieval analysis failed", exc_info=True)

    async def arecall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        stream: bool = False,
    ):
        import asyncio

        if stream:
            return self.recall_stream(
                user_id,
                query,
                top_k,
                max_tokens,
                lifecycle_filter,
                hybrid_search,
                namespace,
                session_id,
                metadata_filter,
            )

        return await asyncio.to_thread(
            self.recall,
            user_id,
            query,
            top_k,
            max_tokens,
            lifecycle_filter,
            hybrid_search,
            namespace,
            session_id,
            metadata_filter,
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
        namespace: str = "default",
        session_id: str | None = None,
    ) -> str:
        import asyncio

        return await asyncio.to_thread(
            self.context_block, user_id, query, top_k, max_tokens, prefix, namespace, session_id
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

        with self._latency_tracker("migrate"):
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
            self._track_operation("migrate", user_id, {"count": count})
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
                    "confidence": mem.confidence,
                    "memory_type": mem.memory_type.value,
                    "session_id": mem.session_id,
                    "namespace": mem.namespace,
                    "version": mem.version,
                    "agent_id": mem.agent_id,
                    "run_id": mem.run_id,
                    "app_id": mem.app_id,
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
                confidence=mem_data.get("confidence", 1.0),
                memory_type=MemoryType(mem_data["memory_type"])
                if mem_data.get("memory_type")
                else MemoryType.EPISODIC,
                session_id=mem_data.get("session_id"),
                namespace=mem_data.get("namespace", "default"),
                version=mem_data.get("version", 1),
                agent_id=mem_data.get("agent_id"),
                run_id=mem_data.get("run_id"),
                app_id=mem_data.get("app_id"),
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
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
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

        Note:
            Fires pre/post ``remember`` event hooks for each item so
            batch behavior is consistent with :meth:`recall_many` and
            :meth:`update_many`.
        """
        if not contents:
            return []

        with self._latency_tracker("remember_many"):
            # Batch embed all contents at once for performance
            embeddings = self._embed.embed(contents)

            memory_ids: list[str] = []
            audit_batch: list[dict[str, Any]] | None = [] if self._audit_trail is not None else None
            for i, content in enumerate(contents):
                if not content or not content.strip():
                    raise ValueError("content cannot be empty — there is nothing to remember")
                self._run_hooks(
                    "pre", "remember", user_id=user_id, content=content, namespace=namespace
                )
                memory_id = self._remember_with_embedding(
                    user_id=user_id,
                    content=content,
                    embedding=embeddings[i],
                    importance=importance,
                    source=source,
                    tags=tags,
                    namespace=namespace,
                    session_id=session_id,
                    memory_type=memory_type,
                    confidence=confidence,
                    audit_batch=audit_batch,
                    agent_id=agent_id,
                    run_id=run_id,
                    app_id=app_id,
                    ttl_seconds=ttl_seconds,
                )
                self._run_hooks(
                    "post", "remember", user_id=user_id, memory_id=memory_id, namespace=namespace
                )
                memory_ids.append(memory_id)

            if self._metrics is not None:
                self._metrics.remember_many_total.inc(1)
                self._metrics.total_memories.set(self._store.count(user_id))

            # Append batch-level audit entry and flush all entries in one transaction
            if audit_batch is not None:
                self._track_operation(
                    "remember_many",
                    user_id,
                    {"count": len(memory_ids)},
                    namespace=namespace,
                    audit_batch=audit_batch,
                )
                if self._audit_trail is not None:
                    try:
                        self._audit_trail.log_operation_batch(audit_batch)
                    except Exception:
                        logger.warning("Audit log batch failed for remember_many", exc_info=True)
            return memory_ids

    def _remember_with_embedding(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        importance: float,
        source: MemorySource,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        audit_batch: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """Internal: store a memory with a pre-computed embedding."""
        clamped_importance = max(0.0, min(1.0, importance))
        embedding_dim = len(embedding)
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
            confidence=max(0.0, min(1.0, confidence)),
            memory_type=memory_type,
            session_id=session_id,
            namespace=namespace,
            version=1,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            expires_at=(
                datetime.now(timezone.utc).replace(microsecond=0)
                + timedelta(seconds=ttl_seconds)
                if ttl_seconds is not None
                else None
            ),
        )

        existing = self._store.get_all_by_user(
            user_id,
            lifecycle_filter=[
                LifecycleState.ACTIVE,
                LifecycleState.DECAYING,
                LifecycleState.ARCHIVED,
            ],
            namespace=namespace,
        )

        duplicates = dedup.find_duplicates(new_memory, existing, self._config.dedup_threshold)

        if duplicates:
            resolved = dedup.resolve_duplicate(new_memory, duplicates[0])
            # Content changed during merge — invalidate stale cached entities
            resolved.metadata.pop("extracted_entities", None)
            # Record version BEFORE overwriting
            try:
                vs = self._get_version_store()
                vs.record_version(duplicates[0], changed_by="merge")
                self._auto_prune_versions_for_memory(duplicates[0].memory_id)
            except (RuntimeError, Exception):
                pass
            self._store.update(resolved)
            # Dispatch updated webhook for merged duplicate
            snapshot = _memory_to_dict(resolved)
            self._dispatch_webhook_event(
                WebhookEventType.UPDATED,
                memory_id=resolved.memory_id,
                user_id=user_id,
                snapshot=snapshot,
            )
            logger.info(f"Resolved duplicate for user {user_id}: {resolved.memory_id}")
            if self._metrics is not None:
                self._metrics.duplicates_detected.inc(1)
            self._track_operation(
                "remember",
                user_id,
                {"memory_id": resolved.memory_id, "duplicate": True},
                resolved.memory_id,
                namespace,
                audit_batch=audit_batch,
            )
            return resolved.memory_id

        conflicts = dedup.find_conflicts(
            new_memory,
            existing,
            self._config.conflict_threshold,
            self._config.dedup_threshold,
        )

        conflict_detected = False
        if conflicts:
            new_memory.metadata["conflict_flagged"] = True
            conflict_detected = True
            logger.warning(
                f"Potential conflict detected for user {user_id}: "
                f"new memory '{content[:50]}...' conflicts with existing memory "
                f"'{conflicts[0].content[:50]}...'"
            )
            if self._metrics is not None:
                self._metrics.conflicts_detected.inc(1)

        if self._config.enable_entity_boost:
            new_memory.metadata["extracted_entities"] = list(
                self._entity_linker.extract(content)
            )

        self._store.store(new_memory)
        if self._metrics is not None:
            self._metrics.embed_total.inc(1)
            self._metrics.embed_bytes_total.inc(len(content))

        # Dispatch webhooks
        snapshot = _memory_to_dict(new_memory)
        self._dispatch_webhook_event(
            WebhookEventType.REMEMBERED,
            memory_id=new_memory.memory_id,
            user_id=user_id,
            snapshot=snapshot,
        )
        if conflict_detected:
            self._dispatch_webhook_event(
                WebhookEventType.CONFLICT,
                memory_id=new_memory.memory_id,
                user_id=user_id,
                snapshot=snapshot,
                conflict_with=conflicts[0].memory_id,
            )

        remember_details: dict[str, Any] = {
            "memory_id": new_memory.memory_id,
            "content_length": len(content),
        }
        if conflict_detected:
            remember_details["conflict"] = True
            remember_details["conflict_with"] = conflicts[0].memory_id
        self._track_operation(
            "remember",
            user_id,
            remember_details,
            new_memory.memory_id,
            namespace,
            audit_batch=audit_batch,
        )
        return new_memory.memory_id

    def list_users(self) -> list[str]:
        """Get all unique user IDs that have memories.

        Returns:
            List of user IDs.
        """
        return self._store.get_all_users()

    def prune(
        self,
        user_id: str,
        max_age_days: float | None = None,
        min_importance: float | None = None,
        lifecycle_states: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> int:
        """Auto-prune old or low-importance memories.

        Args:
            user_id: User ID to prune.
            max_age_days: Delete memories older than this many days.
            min_importance: Delete memories with importance below this threshold.
            lifecycle_states: Only prune memories in these states.
            namespace: Memory namespace to prune.

        Returns:
            Number of memories deleted.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        with self._latency_tracker("prune"):
            all_memories = self._store.get_all_by_user(
                user_id,
                lifecycle_filter=lifecycle_states or [LifecycleState.DECAYING],
                namespace=namespace,
            )

            to_delete: list[str] = []
            now = datetime.now(timezone.utc)

            for mem in all_memories:
                if max_age_days is not None:
                    age_days = (now - mem.created_at).total_seconds() / 86400.0
                    if age_days > max_age_days:
                        to_delete.append(mem.memory_id)
                        continue

                if min_importance is not None:
                    if mem.importance < min_importance:
                        to_delete.append(mem.memory_id)
                        continue

            for mid in to_delete:
                self._store.delete_by_id(mid)

            logger.info(f"Pruned {len(to_delete)} memories for user {user_id}")
            self._track_operation(
                "prune", user_id, {"deleted_count": len(to_delete)}, namespace=namespace
            )
            return len(to_delete)

    def prune_expired(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Delete memories whose ``expires_at`` has passed.

        Sweeps ACTIVE and DECAYING memories with a non-null ``expires_at``
        in the past, transitions them to DELETED, and removes them from
        the store.  Called automatically by :meth:`run_maintenance`.

        Args:
            user_id: If provided, only sweep this user.  If None, sweep all
                users.
            namespace: If provided, only sweep this namespace.  If None,
                sweep memories across all namespaces.

        Returns:
            Number of memories deleted.
        """
        with self._latency_tracker("prune_expired"):
            now = datetime.now(timezone.utc)
            deleted = 0

            if user_id is not None:
                users = [user_id]
            else:
                users = self._store.get_all_users()

            for uid in users:
                if namespace is not None:
                    memories = self._store.get_all_by_user(
                        uid,
                        lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
                        namespace=namespace,
                    )
                else:
                    # Sweep across all namespaces.  The base adapter's
                    # ``get_all_by_user`` defaults ``namespace="default"``,
                    # so we fetch from each known namespace explicitly.
                    namespaces = self._known_namespaces(uid)
                    memories = []
                    for ns in namespaces:
                        memories.extend(
                            self._store.get_all_by_user(
                                uid,
                                lifecycle_filter=[
                                    LifecycleState.ACTIVE,
                                    LifecycleState.DECAYING,
                                ],
                                namespace=ns,
                            )
                        )
                for mem in memories:
                    if mem.expires_at is not None and mem.expires_at <= now:
                        if self._store.delete_by_id(mem.memory_id):
                            deleted += 1

            if deleted > 0:
                logger.info(
                    f"Pruned {deleted} expired memories"
                    + (f" for user {user_id}" if user_id else "")
                )
            self._track_operation(
                "prune_expired",
                user_id or "all",
                {"deleted_count": deleted},
                namespace=namespace or "all",
            )
            return deleted

    def _known_namespaces(self, user_id: str) -> set[str]:
        """Return the set of distinct namespaces holding memories for a user."""
        namespaces: set[str] = set()
        # Default namespace is always present even if empty.
        namespaces.add("default")
        try:
            # Use get_all_by_user with no namespace filter is not possible
            # (the base API defaults to "default"), so we sample a small
            # batch from get_all() and collect namespaces.
            for mem in self._store.get_all(limit=1000):
                if mem.user_id == user_id:
                    namespaces.add(mem.namespace)
        except Exception:
            pass
        return namespaces

    def recall_between(
        self,
        user_id: str,
        query: str,
        start: datetime,
        end: datetime,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Recall memories created within a specific date range.

        Args:
            user_id: User ID to search for.
            query: Search query.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            top_k: Maximum memories to return.
            max_tokens: Token budget.
            lifecycle_filter: Filter by lifecycle state.
            namespace: Memory namespace.
            session_id: Filter by session ID.

        Returns:
            List of MemoryObjects created within the date range.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        all_results = self.recall(
            user_id=user_id,
            query=query,
            top_k=top_k * 3,
            max_tokens=max_tokens,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        filtered = [m for m in all_results if m.created_at and start <= m.created_at <= end]
        return filtered[:top_k]

    def recall_user_profile(
        self,
        user_id: str,
        *,
        top_k: int = 20,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        """Recall a user's long-lived profile — semantic facts and preferences.

        Filters for SEMANTIC memories that are ACTIVE, DECAYING, or ARCHIVED,
        then ranks by importance (highest first).  This is the ergonomic
        equivalent of:

        .. code-block:: python

            memories = memory.recall(
                user_id="alice",
                query="profile preferences facts",
                top_k=20,
                lifecycle_filter=[ACTIVE, DECAYING, ARCHIVED],
            )

        Args:
            user_id: User whose profile to retrieve.
            top_k: Maximum number of profile facts to return.
            namespace: Memory namespace.

        Returns:
            List of MemoryObjects sorted by importance descending.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        with self._latency_tracker("recall_user_profile"):
            self._run_hooks("pre", "recall_user_profile", user_id=user_id, namespace=namespace)

            all_memories = self._store.get_all_by_user(
                user_id,
                lifecycle_filter=[
                    LifecycleState.ACTIVE,
                    LifecycleState.DECAYING,
                    LifecycleState.ARCHIVED,
                ],
                namespace=namespace,
            )

            profile_memories = [
                m for m in all_memories if m.memory_type == MemoryType.SEMANTIC
            ]
            profile_memories.sort(key=lambda m: m.importance, reverse=True)

            for mem in profile_memories[:top_k]:
                mem.last_accessed_at = datetime.now(timezone.utc)
                new_state = lifecycle.evaluate_lifecycle(mem, self._config.decay_threshold_hours)
                if new_state != mem.lifecycle_state:
                    updated = lifecycle.transition(mem, new_state)
                    self._store.update(updated)

            self._run_hooks(
                "post",
                "recall_user_profile",
                user_id=user_id,
                results=profile_memories[:top_k],
                namespace=namespace,
            )
            self._track_operation(
                "recall_user_profile",
                user_id,
                {"results_count": len(profile_memories[:top_k])},
                namespace=namespace,
            )
            return profile_memories[:top_k]

    def recall_session_context(
        self,
        user_id: str,
        session_id: str,
        *,
        top_k: int = 20,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        """Recall recent episodic memories scoped to a specific session.

        Filters for EPISODIC memories that belong to the given *session_id*,
        then ranks by recency (most recent first).  This is the ergonomic
        equivalent of:

        .. code-block:: python

            memories = memory.recall(
                user_id="alice",
                query="session context",
                top_k=20,
                lifecycle_filter=[ACTIVE, DECAYING],
                session_id="sess_123",
            )

        Args:
            user_id: User whose session to retrieve.
            session_id: Session identifier.
            top_k: Maximum number of session memories to return.
            namespace: Memory namespace.

        Returns:
            List of MemoryObjects sorted by created_at descending.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        with self._latency_tracker("recall_session_context"):
            self._run_hooks(
                "pre", "recall_session_context", user_id=user_id, session_id=session_id, namespace=namespace
            )

            all_memories = self._store.get_all_by_user(
                user_id,
                lifecycle_filter=[
                    LifecycleState.ACTIVE,
                    LifecycleState.DECAYING,
                    LifecycleState.ARCHIVED,
                ],
                namespace=namespace,
                session_id=session_id,
            )

            session_memories = [
                m for m in all_memories if m.memory_type == MemoryType.EPISODIC
            ]
            session_memories.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

            for mem in session_memories[:top_k]:
                mem.last_accessed_at = datetime.now(timezone.utc)
                new_state = lifecycle.evaluate_lifecycle(mem, self._config.decay_threshold_hours)
                if new_state != mem.lifecycle_state:
                    updated = lifecycle.transition(mem, new_state)
                    self._store.update(updated)

            self._run_hooks(
                "post",
                "recall_session_context",
                user_id=user_id,
                session_id=session_id,
                results=session_memories[:top_k],
                namespace=namespace,
            )
            self._track_operation(
                "recall_session_context",
                user_id,
                {"results_count": len(session_memories[:top_k]), "session_id": session_id},
                namespace=namespace,
            )
            return session_memories[:top_k]

    def recall_agent_knowledge(
        self,
        agent_id: str,
        *,
        namespace: str = "default",
        top_k: int = 50,
    ) -> list[MemoryObject]:
        """Recall memories that belong to a specific agent.

        Scans across all users (via :meth:`list_users`) and returns the
        agent's most important memories in the given namespace.

        Args:
            agent_id: Agent identifier to filter by.
            namespace: Memory namespace.
            top_k: Maximum number of memories to return.

        Returns:
            List of MemoryObjects sorted by importance descending.
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        with self._latency_tracker("recall_agent_knowledge"):
            self._run_hooks(
                "pre", "recall_agent_knowledge", agent_id=agent_id, namespace=namespace
            )

            all_users = self._store.get_all_users()
            agent_memories: list[MemoryObject] = []
            for uid in all_users:
                user_memories = self._store.get_all_by_user(
                    uid,
                    lifecycle_filter=[
                        LifecycleState.ACTIVE,
                        LifecycleState.DECAYING,
                        LifecycleState.ARCHIVED,
                    ],
                    namespace=namespace,
                )
                for mem in user_memories:
                    if mem.agent_id == agent_id:
                        agent_memories.append(mem)

            agent_memories.sort(key=lambda m: m.importance, reverse=True)

            for mem in agent_memories[:top_k]:
                mem.last_accessed_at = datetime.now(timezone.utc)
                new_state = lifecycle.evaluate_lifecycle(mem, self._config.decay_threshold_hours)
                if new_state != mem.lifecycle_state:
                    updated = lifecycle.transition(mem, new_state)
                    self._store.update(updated)

            self._run_hooks(
                "post",
                "recall_agent_knowledge",
                agent_id=agent_id,
                results=agent_memories[:top_k],
                namespace=namespace,
            )
            self._track_operation(
                "recall_agent_knowledge",
                "all",
                {"results_count": len(agent_memories[:top_k]), "agent_id": agent_id},
                namespace=namespace,
            )
            return agent_memories[:top_k]

    def recall_explain(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories with detailed score breakdowns.

        Returns each memory with an 'explanation' dict showing:
        - semantic_score: cosine similarity contribution
        - recency_score: temporal recency contribution
        - bm25_score: keyword match contribution (if hybrid)
        - importance_score: importance contribution
        - final_score: the combined score

        Args:
            user_id: User ID.
            query: Search query.
            top_k: Max results.
            namespace: Memory namespace.
            session_id: Filter by session.

        Returns:
            List of dicts with 'memory' (MemoryObject) and 'explanation'.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        query_embedding = self._embed.embed_single(query)

        search_results = self._store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            lifecycle_filter=lifecycle.get_recall_filter(),
            namespace=namespace,
            session_id=session_id,
        )

        corpus = [m.content for m in search_results] if len(search_results) > 1 else None

        query_entities_explain: set[str] | None = None
        memory_entities_map_explain: dict[str, set[str]] | None = None
        if self._config.enable_entity_boost:
            query_entities_explain = self._entity_linker.extract(query)
            memory_entities_map_explain = {}
            for m in search_results:
                cached = m.metadata.get("extracted_entities")
                if cached is not None:
                    memory_entities_map_explain[m.memory_id] = set(cached)
                else:
                    memory_entities_map_explain[m.memory_id] = self._entity_linker.extract(m.content)

        explained: list[dict[str, Any]] = []
        for memory in search_results:
            semantic = scoring.cosine_similarity(memory.embedding, query_embedding)
            semantic_norm = (semantic + 1.0) / 2.0
            recency = scoring.temporal_recency(memory.last_accessed_at)

            entity_score = 0.0
            if (
                self._config.enable_entity_boost
                and query_entities_explain is not None
                and memory_entities_map_explain is not None
            ):
                mem_entities = memory_entities_map_explain.get(memory.memory_id, set())
                entity_score = scoring.jaccard_similarity(query_entities_explain, mem_entities)

            if self._config.hybrid_search:
                if corpus and len(corpus) > 1:
                    bm25 = scoring.bm25_score_corpus(query, memory.content, corpus)
                else:
                    bm25 = scoring.bm25_score(query, memory.content)
                final = (
                    semantic_norm * self._config.weight_semantic
                    + recency * self._config.weight_recency
                    + bm25 * self._config.weight_bm25
                    + entity_score * self._config.entity_boost_weight
                )
                explanation = {
                    "semantic_score": round(semantic_norm, 4),
                    "recency_score": round(recency, 4),
                    "bm25_score": round(bm25, 4),
                    "importance_score": None,
                    "entity_score": round(entity_score, 4),
                    "final_score": round(final, 4),
                    "weights": {
                        "semantic": self._config.weight_semantic,
                        "recency": self._config.weight_recency,
                        "bm25": self._config.weight_bm25,
                        "entity": self._config.entity_boost_weight,
                    },
                }
            else:
                importance = max(0.0, min(1.0, memory.importance))
                final = (
                    semantic_norm * self._config.weight_semantic_no_embed
                    + recency * self._config.weight_recency_no_embed
                    + importance * self._config.weight_importance
                    + entity_score * self._config.entity_boost_weight
                )
                explanation = {
                    "semantic_score": round(semantic_norm, 4),
                    "recency_score": round(recency, 4),
                    "bm25_score": None,
                    "importance_score": round(importance, 4),
                    "entity_score": round(entity_score, 4),
                    "final_score": round(final, 4),
                    "weights": {
                        "semantic": self._config.weight_semantic_no_embed,
                        "recency": self._config.weight_recency_no_embed,
                        "importance": self._config.weight_importance,
                        "entity": self._config.entity_boost_weight,
                    },
                }

            memory.score = final
            explained.append({"memory": memory, "explanation": explanation})

        explained.sort(key=lambda x: x["explanation"]["final_score"], reverse=True)
        return explained[:top_k]

    def consolidate(
        self,
        user_id: str,
        namespace: str = "default",
        min_memories: int = 5,
        max_age_days: float = 30.0,
        with_llm_summary: bool = False,
    ) -> str | None:
        """Consolidate old episodic memories into a semantic summary.

        Uses local extractive summarization (no LLM required) by default.
        When ``with_llm_summary=True``, uses LLM-powered abstractive
        summarization via the configured provider (see ``MemoryConfig``).

        Finds clusters of related old memories, generates a summary for
        each, stores it as a SEMANTIC memory, and archives the old ones.

        Args:
            user_id: User to consolidate.
            namespace: Memory namespace.
            min_memories: Minimum memories needed to form a cluster.
            max_age_days: Only consider memories older than this.
            with_llm_summary: If True, use LLM-powered abstractive summary.

        Returns:
            Memory ID of the consolidated summary, or None if no consolidation occurred.
        """
        try:
            from kemi import consolidation
        except ImportError:  # pragma: no cover
            logger.warning("consolidation module not available")
            return None

        mid = consolidation.consolidate(
            store=self._store,
            embed=self._embed,
            user_id=user_id,
            namespace=namespace,
            min_memories=min_memories,
            max_age_days=max_age_days,
            with_llm_summary=with_llm_summary,
            summarizer_llm_provider=self._config.summarizer_llm_provider,
            summarizer_llm_model=self._config.summarizer_llm_model,
            summarizer_prompt_template=self._config.summarizer_prompt_template,
        )
        if mid is not None:
            self._dispatch_webhook_event(
                WebhookEventType.CONSOLIDATED,
                memory_id=mid,
                user_id=user_id,
            )
        return mid

    def cluster_topics(
        self,
        user_id: str,
        n_clusters: int = 3,
        namespace: str = "default",
    ) -> dict[str, list[MemoryObject]]:
        """Cluster memories into topic groups using embeddings.

        Requires scikit-learn to be installed.

        Args:
            user_id: User ID.
            n_clusters: Number of topic clusters.
            namespace: Memory namespace.

        Returns:
            Dict mapping topic labels to lists of memories.
        """
        try:
            from kemi import topics
        except ImportError:  # pragma: no cover
            logger.warning("topics module not available")
            return {}

        return topics.cluster_memories(
            store=self._store,
            user_id=user_id,
            n_clusters=n_clusters,
            namespace=namespace,
        )

    def extract_entities(self, memory_id: str) -> list[dict[str, Any]]:
        """Extract named entities from a memory's content.

        Uses regex/heuristic-based extraction (no external NER model required).

        Args:
            memory_id: Memory ID.

        Returns:
            List of entity dicts with 'text', 'label', 'start', 'end'.
        """
        try:
            from kemi import graph
        except ImportError:  # pragma: no cover
            logger.warning("graph module not available")
            return []

        memory = self._store.get(memory_id)
        if memory is None:
            raise ValueError(f"Memory not found: {memory_id}")

        return graph.extract_entities(memory.content)

    def get_memory_graph(
        self,
        user_id: str,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Build a memory graph of entities and relations.

        Args:
            user_id: User ID.
            namespace: Memory namespace.

        Returns:
            Dict with 'entities' (list) and 'relations' (list of dicts).
        """
        try:
            from kemi import graph
        except ImportError:  # pragma: no cover
            logger.warning("graph module not available")
            return {"entities": [], "relations": []}

        return graph.build_memory_graph(
            store=self._store,
            user_id=user_id,
            namespace=namespace,
        )

    def stats(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return health statistics for a user's memory store.

        Args:
            user_id: The user whose memories to analyze.
            lifecycle_filter: Optional list of lifecycle states to filter by.
            session_id: Optional session ID to filter by.

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

        all_memories = self._store.get_all_by_user(
            user_id, lifecycle_filter=lifecycle_filter, session_id=session_id
        )

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

    async def astats(self, user_id: str) -> dict[str, Any]:
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
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Update an existing memory.

        Args:
            memory_id: ID of memory to update.
            content: New content (if provided, will re-embed).
            importance: New importance value (0.0-1.0).
            confidence: New confidence value (0.0-1.0).
            memory_type: New memory type.
            metadata: Metadata dict to merge into existing metadata.
            tags: New tags to replace existing tags.

        Returns:
            The memory_id of updated memory.

        Raises:
            ValueError: If memory_id not found.
        """
        if (
            content is None
            and importance is None
            and confidence is None
            and memory_type is None
            and metadata is None
            and tags is None
        ):
            return memory_id

        with self._latency_tracker("update"):
            self._run_hooks("pre", "update", memory_id=memory_id)
            memory = self._store.get(memory_id)
            if memory is None:
                raise ValueError(f"Memory not found: {memory_id}")

            # Capture pre-update state BEFORE mutating the memory object.
            # record_before_update expects two separate objects (before, after).
            memory_before = MemoryObject(
                memory_id=memory.memory_id,
                user_id=memory.user_id,
                content=memory.content,
                embedding=memory.embedding,
                score=memory.score,
                created_at=memory.created_at,
                last_accessed_at=memory.last_accessed_at,
                source=memory.source,
                importance=memory.importance,
                lifecycle_state=memory.lifecycle_state,
                metadata=memory.metadata.copy() if memory.metadata else {},
                embedding_dim=memory.embedding_dim,
                tags=list(memory.tags) if memory.tags else [],
                confidence=memory.confidence,
                memory_type=memory.memory_type,
                session_id=memory.session_id,
                namespace=memory.namespace,
                version=memory.version,
                agent_id=memory.agent_id,
                run_id=memory.run_id,
                app_id=memory.app_id,
            )

            # Now apply all mutations
            if content is not None:
                memory.content = content
                memory.embedding = self._embed.embed_single(content)
                memory.embedding_dim = len(memory.embedding)
                memory.last_accessed_at = datetime.now(timezone.utc)
                if self._config.enable_entity_boost:
                    memory.metadata["extracted_entities"] = list(
                        self._entity_linker.extract(content)
                    )

            if importance is not None:
                memory.importance = max(0.0, min(1.0, importance))

            if confidence is not None:
                memory.confidence = max(0.0, min(1.0, confidence))

            if memory_type is not None:
                memory.memory_type = memory_type

            if metadata is not None:
                memory.metadata.update(metadata)

            if tags is not None:
                memory.tags = tags

            # Record version BEFORE and AFTER the update (pre + post snapshot)
            try:
                vs = self._get_version_store()
                vs.record_before_update(memory_before, memory, changed_by="update")
                self._auto_prune_versions_for_memory(memory_id)
            except (RuntimeError, Exception):
                pass  # Versioning is optional

            previous_state = _memory_to_dict(memory)
            memory.version += 1

            snapshot = _memory_to_dict(memory)
            self._dispatch_webhook_event(
                WebhookEventType.UPDATED,
                memory_id=memory_id,
                user_id=memory.user_id,
                snapshot=snapshot,
                previous_state=previous_state,
            )

            self._store.update(memory)
            self._run_hooks("post", "update", memory_id=memory_id, version=memory.version)
            logger.info(f"Updated memory: {memory_id} (version {memory.version})")
            self._track_operation(
                "update",
                memory.user_id,
                {"memory_id": memory_id, "version": memory.version},
                memory_id,
                memory.namespace,
            )
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
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Async version of update()."""
        import asyncio

        return await asyncio.to_thread(
            self.update,
            memory_id,
            content,
            importance,
            confidence,
            memory_type,
            metadata,
        )

    async def aupdate_many(
        self,
        memory_ids: list[str],
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Async version of update_many() — runs updates concurrently.

        Small batches (≤10) use individual concurrent aupdate calls via
        asyncio.gather. Larger batches fall back to a single threaded
        update_many to avoid excessive thread overhead.
        """
        import asyncio

        if not memory_ids:
            return []

        if len(memory_ids) <= 10:
            tasks = [
                self.aupdate(
                    mid,
                    content=content,
                    importance=importance,
                    confidence=confidence,
                    memory_type=memory_type,
                    metadata=metadata,
                )
                for mid in memory_ids
            ]
            return await asyncio.gather(*tasks)

        return await asyncio.to_thread(
            self.update_many,
            memory_ids,
            content,
            importance,
            confidence,
            memory_type,
            metadata,
        )

    async def aforget_many(
        self,
        memory_ids: list[str],
    ) -> int:
        """Async version of forget_many() — runs deletions concurrently."""
        import asyncio

        if not memory_ids:
            return 0

        tasks = [asyncio.to_thread(self._store.delete_by_id, mid) for mid in memory_ids]
        results = await asyncio.gather(*tasks)
        return sum(1 for r in results if r)

    async def arecall_many(
        self,
        user_ids: list[str],
        queries: list[str],
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        hybrid_search: bool | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, list[MemoryObject]]:
        """Async version of recall_many() — runs individual recalls concurrently."""
        import asyncio

        if len(user_ids) != len(queries):
            raise ValueError("user_ids and queries must have the same length")

        tasks = [
            self.arecall(
                uid,
                q,
                top_k=top_k,
                max_tokens=max_tokens,
                lifecycle_filter=lifecycle_filter,
                hybrid_search=hybrid_search,
                namespace=namespace,
                session_id=session_id,
                metadata_filter=metadata_filter,
            )
            for uid, q in zip(user_ids, queries, strict=True)
        ]
        results = await asyncio.gather(*tasks)
        return {uid: res for uid, res in zip(user_ids, results, strict=True)}

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
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
    ) -> list[str]:
        """Async version of remember_many()."""
        import asyncio

        return await asyncio.to_thread(
            self.remember_many,
            user_id,
            contents,
            importance,
            source,
            tags,
            namespace,
            session_id,
            memory_type,
            confidence,
            agent_id,
            run_id,
            app_id,
        )

    def feedback(
        self,
        user_id: str,
        memory_id: str,
        helpful: bool = True,
        namespace: str = "default",
    ) -> None:
        """Record user feedback on a recalled memory.

        Stores feedback in metadata and adjusts importance:
        - helpful=True: boosts importance slightly (up to 1.0)
        - helpful=False: reduces importance slightly (down to 0.0)

        Args:
            user_id: User ID.
            memory_id: Memory ID that was recalled.
            helpful: Whether the memory was helpful.
            namespace: Memory namespace.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id cannot be empty")

        memory = self._store.get(memory_id)
        if memory is None:
            raise ValueError(f"Memory not found: {memory_id}")
        if memory.user_id != user_id:
            raise ValueError("Memory does not belong to this user")
        if memory.namespace != namespace:
            raise ValueError("Memory does not belong to this namespace")

        # Record feedback history
        feedback_entry = {
            "helpful": helpful,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if "feedback" not in memory.metadata:
            memory.metadata["feedback"] = []
        memory.metadata["feedback"].append(feedback_entry)

        # Adjust importance based on feedback
        adjustment = 0.05 if helpful else -0.05
        memory.importance = max(0.0, min(1.0, memory.importance + adjustment))

        self._store.update(memory)
        logger.info(
            f"Feedback recorded for memory {memory_id}: helpful={helpful}, "
            f"new_importance={memory.importance:.2f}"
        )
        self._track_operation(
            "feedback", user_id, {"memory_id": memory_id, "helpful": helpful}, memory_id, namespace
        )

    def backfill_entities(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Backfill ``extracted_entities`` for memories that don't have them yet.

        Iterates over memories (optionally filtered by *user_id* and
        *namespace*), extracts entities from their content using the
        configured :attr:`_entity_linker`, and persists the result in
        ``memory.metadata["extracted_entities"]``.

        This is useful after enabling entity boost on an existing store
        so that subsequent recall calls can read cached entities instead
        of falling back to on-the-fly extraction.

        Args:
            user_id: If provided, only backfill this user's memories.
                If ``None``, backfill across all users.
            namespace: If provided, only backfill memories in this
                namespace. If ``None``, backfill across all namespaces.

        Returns:
            Number of memories that were backfilled.
        """
        if not self._config.enable_entity_boost:
            logger.info("Entity boost is disabled; skipping entity backfill")
            return 0

        with self._latency_tracker("backfill_entities"):
            if user_id is not None:
                users = [user_id]
            else:
                users = self._store.get_all_users()

            backfilled = 0
            for uid in users:
                if namespace is not None:
                    namespaces = [namespace]
                else:
                    namespaces = self._known_namespaces(uid)

                for ns in namespaces:
                    memories = self._store.get_all_by_user(
                        uid,
                        lifecycle_filter=[
                            LifecycleState.ACTIVE,
                            LifecycleState.DECAYING,
                            LifecycleState.ARCHIVED,
                        ],
                        namespace=ns,
                    )
                    for mem in memories:
                        if "extracted_entities" in mem.metadata:
                            continue
                        entities = self._entity_linker.extract(mem.content)
                        mem.metadata["extracted_entities"] = list(entities)
                        self._store.update(mem)
                        backfilled += 1

            if backfilled > 0:
                logger.info(
                    f"Backfilled extracted_entities for {backfilled} memories"
                    + (f" (user={user_id})" if user_id else "")
                )
            self._track_operation(
                "backfill_entities",
                user_id or "all",
                {"backfilled": backfilled},
                namespace=namespace or "all",
            )
            return backfilled

    async def abackfill_entities(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Async version of :meth:`backfill_entities`."""
        import asyncio

        return await asyncio.to_thread(
            self.backfill_entities, user_id, namespace
        )

    def run_maintenance(
        self,
        user_id: str,
        namespace: str = "default",
        auto_prune: bool = True,
        auto_consolidate: bool = True,
        auto_backfill_entities: bool = False,
        prune_max_age_days: float = 90.0,
        prune_min_importance: float = 0.1,
        consolidate_min_memories: int = 5,
        consolidate_max_age_days: float = 30.0,
        auto_prune_expired: bool = True,
        consolidate_with_llm_summary: bool = False,
    ) -> dict[str, Any]:
        """Run automatic maintenance tasks for a user's memories.

        This is a one-shot maintenance run. For periodic maintenance,
        call this method from a scheduler (e.g., cron, APScheduler).

        Args:
            user_id: User ID to maintain.
            namespace: Memory namespace.
            auto_prune: Whether to prune old memories.
            auto_consolidate: Whether to consolidate old episodic memories.
            auto_backfill_entities: Whether to backfill missing
                ``extracted_entities`` metadata.
            prune_max_age_days: Delete DECAYING memories older than this.
            prune_min_importance: Delete memories below this importance.
            consolidate_min_memories: Minimum memories to consolidate.
            consolidate_max_age_days: Only consolidate memories older than this.
            auto_prune_expired: Whether to delete TTL-expired memories.
            consolidate_with_llm_summary: Use LLM-powered summarization.

        Returns:
            Dict with 'pruned' (int), 'expired' (int),
            'consolidated' (str | None), and 'backfilled' (int) keys.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        with self._latency_tracker("run_maintenance"):
            result: dict[str, Any] = {
                "pruned": 0,
                "expired": 0,
                "consolidated": None,
                "backfilled": 0,
            }

            if auto_backfill_entities:
                backfilled = self.backfill_entities(
                    user_id=user_id, namespace=namespace
                )
                result["backfilled"] = backfilled

            if auto_prune:
                pruned = self.prune(
                    user_id=user_id,
                    max_age_days=prune_max_age_days,
                    min_importance=prune_min_importance,
                    namespace=namespace,
                )
                result["pruned"] = pruned

            if auto_prune_expired:
                expired = self.prune_expired(
                    user_id=user_id, namespace=namespace
                )
                result["expired"] = expired

            if auto_consolidate:
                consolidated_id = self.consolidate(
                    user_id=user_id,
                    namespace=namespace,
                    min_memories=consolidate_min_memories,
                    max_age_days=consolidate_max_age_days,
                    with_llm_summary=consolidate_with_llm_summary,
                )
                result["consolidated"] = consolidated_id

            logger.info(f"Maintenance complete for {user_id}: {result}")
            self._track_operation("run_maintenance", user_id, result, namespace=namespace)
            return result

    def get_metrics(self) -> dict[str, Any] | None:
        """Return current metrics as a dictionary."""
        if self._metrics is None:
            return None
        return self._metrics.to_dict()  # type: ignore[no-any-return]

    def get_metrics_prometheus(self) -> str | None:
        """Return current metrics in Prometheus text format."""
        if self._metrics is None:
            return None
        return self._metrics.to_prometheus()  # type: ignore[no-any-return]

    def enable_adaptive_retrieval(self, enable: bool = True) -> None:
        """Enable or disable adaptive retrieval."""
        if enable:
            try:
                from kemi.adaptive import AdaptiveRetriever

                self._adaptive_retriever = AdaptiveRetriever()
            except ImportError:
                logger.warning("Adaptive retrieval module not available")
        else:
            self._adaptive_retriever = None

    def _track_operation(
        self,
        operation: str,
        user_id: str,
        details: dict[str, Any] | None = None,
        memory_id: str | None = None,
        namespace: str = "default",
        status: str = "success",
        audit_batch: list[dict[str, Any]] | None = None,
    ) -> None:
        """Track an operation in metrics and audit trail."""
        from kemi.operations import _ops_metrics
        _ops_metrics.track_operation_full(
            self, operation, user_id, details, memory_id, namespace, status, audit_batch
        )

    def _record_embed_error(self) -> None:
        """Record an embedding error in metrics if available."""
        from kemi.operations import _ops_metrics
        _ops_metrics.record_embed_error(self)

    def _record_store_error(self) -> None:
        """Record a storage error in metrics if available."""
        from kemi.operations import _ops_metrics
        _ops_metrics.record_store_error(self)

    def add_event_hook(self, phase: str, callback: Callable[..., Any]) -> None:
        """Register an event hook callback.

        Args:
            phase: "pre" or "post" — called before or after the operation.
            callback: Callable that receives (operation, **kwargs).
        """
        from kemi.operations import _ops_hooks
        _ops_hooks.add(self, phase, callback)

    def remove_event_hook(self, phase: str, callback: Callable[..., Any]) -> bool:
        """Remove a previously registered event hook callback.

        Returns True if removed, False if not found.
        """
        from kemi.operations import _ops_hooks
        return _ops_hooks.remove(self, phase, callback)

    def _run_hooks(
        self,
        phase: str,
        operation: str,
        *,
        raise_on_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Run all hooks registered for a phase/operation.

        Args:
            phase: "pre" or "post".
            operation: Name of the operation triggering the hook.
            raise_on_error: If True, exceptions from hooks are re-raised so
                a failing pre-hook can abort the operation. If None (default),
                the value is taken from ``self._config.hooks_raise_on_error``.
            **kwargs: Passed through to each callback.
        """
        from kemi.operations import _ops_hooks
        _ops_hooks.run(self, phase, operation, raise_on_error=raise_on_error, **kwargs)

    def enable_query_cache(self, max_size: int = 128) -> None:
        """Enable an LRU cache for recall() results.

        Args:
            max_size: Maximum number of cached query results.
        """
        from kemi.operations import _ops_metrics
        _ops_metrics.enable_query_cache(self, max_size)

    def disable_query_cache(self) -> None:
        """Disable the query cache."""
        from kemi.operations import _ops_metrics
        _ops_metrics.disable_query_cache(self)

    def configure_versioning(
        self,
        db_path: str | None = None,
        max_versions_per_memory: int = 50,
        auto_prune_versions: bool = True,
    ) -> None:
        """Enable memory version history tracking.

        Args:
            db_path: Path to the SQLite database. Defaults to the store's db_path.
            max_versions_per_memory: Maximum versions to keep per memory before pruning.
            auto_prune_versions: If True, prune old versions when limits are exceeded.
        """
        from kemi.operations import _ops_versioning
        _ops_versioning.configure(
            self, db_path, max_versions_per_memory, auto_prune_versions
        )

    def _get_version_store(self) -> MemoryVersionStore:
        """Get the version store, initialising it lazily from the storage adapter's db.

        Falls back to an in-memory SQLite database when the storage adapter
        does not expose a ``_db_path`` (e.g. mock adapters used in tests,
        in-memory backends).
        """
        from kemi.operations import _ops_versioning
        return _ops_versioning.get_store(self)

    def get_history(
        self,
        memory_id: str,
        limit: int = 100,
    ) -> list["VersionSnapshot"]:
        """Return version history for a memory, newest first."""
        from kemi.operations import _ops_versioning
        return _ops_versioning.get_history(self, memory_id, limit)

    def diff_versions(
        self,
        memory_id: str,
        from_version: int,
        to_version: int,
    ) -> "DiffResult | None":
        """Show field-level differences between two versions of a memory."""
        from kemi.operations import _ops_versioning
        return _ops_versioning.diff(self, memory_id, from_version, to_version)

    def rollback_memory(
        self,
        memory_id: str,
        target_version: int,
    ) -> "RollbackResult | None":
        """Roll a memory back to a previous version."""
        from kemi.operations import _ops_versioning
        return _ops_versioning.rollback(self, memory_id, target_version)

    def _auto_prune_versions_for_memory(self, memory_id: str) -> None:
        """Prune old versions for a memory, keeping only the most recent ones."""
        from kemi.operations import _ops_versioning
        _ops_versioning.auto_prune(self, memory_id)

    def configure_webhooks(self, db_path: str | None = None) -> None:
        """Enable webhook dispatch for memory lifecycle events.

        Args:
            db_path: Path to SQLite database for webhook config storage.
                Defaults to the same path used by the storage adapter.
        """
        from kemi.operations import _ops_webhooks
        _ops_webhooks.configure(self, db_path)

    def _dispatch_webhook_event(
        self,
        event: WebhookEventType,
        memory_id: str,
        user_id: str,
        snapshot: dict[str, Any] | None = None,
        previous_state: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Dispatch a webhook event if a dispatcher is configured."""
        from kemi.operations import _ops_webhooks
        _ops_webhooks.dispatch(
            self, event, memory_id, user_id, snapshot, previous_state, **extra
        )

    def enable_audit_trail(
        self,
        retention_days: int = 365,
        auto_purge: bool = True,
    ) -> None:
        """Enable the audit trail for compliance logging."""
        from kemi.operations import _ops_metrics
        _ops_metrics.enable_audit_trail(self, retention_days, auto_purge)

    def get_metrics(self) -> dict[str, Any] | None:
        """Return current metrics snapshot as a dict, or None if disabled."""
        from kemi.operations import _ops_metrics
        return _ops_metrics.get_metrics(self)

    def get_metrics_prometheus(self) -> str | None:
        """Return metrics in Prometheus text format, or None if disabled."""
        from kemi.operations import _ops_metrics
        return _ops_metrics.get_metrics_prometheus(self)

    def enable_adaptive_retrieval(self, enable: bool = True) -> None:
        """Enable or disable adaptive retrieval (re-weights hybrid scores per user)."""
        from kemi.operations import _ops_metrics
        _ops_metrics.enable_adaptive_retrieval(self, enable)


class _QueryCache:
    """DEPRECATED shim — moved to :mod:`kemi.operations._query_cache`.

    Kept as a re-export so existing imports of ``kemi.core._QueryCache``
    keep working. The canonical location is ``kemi.operations._query_cache._QueryCache``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        from kemi.operations._query_cache import _QueryCache as _Impl

        return _Impl(*args, **kwargs)


_QueryCache.__doc__ = _QueryCache.__doc__
