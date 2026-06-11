"""Read-side facade: recall + stats + read-only data queries.

Methods on this facade are pure reads (or reads with side-effecting
lifecycle re-evaluations, like ``recall_stream`` and ``recall``).
Cross-facade composition is handled by the public ``MemoryService``
shim, not here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from kemi.exceptions import ValidationError
from kemi.memory import lifecycle, scoring
from kemi.memory.model import LifecycleState, MemoryObject

if TYPE_CHECKING:
    from kemi.memory.core import _MemoryCore

logger = logging.getLogger(__name__)


class MemoryReadService:
    """Read-path methods: recall, stats, graph queries."""

    def __init__(self, core: _MemoryCore) -> None:
        self._core = core

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
        """Recall memories matching ``query``."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValidationError("query cannot be empty — what should kemi search for?")
        if top_k < 1:
            raise ValidationError(f"top_k must be at least 1, got {top_k}")

        with self._core._latency_tracker("recall"):
            return self._core._recall_via_pipeline(
                user_id=user_id,
                query=query,
                top_k=top_k,
                max_tokens=max_tokens,
                lifecycle_filter=lifecycle_filter,
                hybrid_search=hybrid_search,
                namespace=namespace,
                session_id=session_id,
                metadata_filter=metadata_filter,
            )

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
        """Recall memories for multiple users and queries at once."""
        if len(user_ids) != len(queries):
            raise ValidationError("user_ids and queries must have the same length")
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
    ) -> AsyncGenerator[MemoryObject, None]:
        """Stream recall results as an async generator."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty")
        if not query or not query.strip():
            raise ValidationError("query cannot be empty — what should kemi search for?")
        if top_k < 1:
            raise ValidationError(f"top_k must be at least 1, got {top_k}")

        if hybrid_search is None:
            hybrid_search = self._core._config.hybrid_search

        query_embedding = await asyncio.to_thread(self._core._embed.embed_single, query)

        if lifecycle_filter is None:
            lifecycle_filter = lifecycle.get_recall_filter()

        search_results = await asyncio.to_thread(
            self._core._store.search,
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

        current_dim = self._core._embed.dimension()
        if search_results:
            stored_dim = search_results[0].embedding_dim
            if stored_dim is not None and stored_dim != current_dim:
                raise ValidationError(
                    "Embedding dimension mismatch: stored memories use "
                    f"{stored_dim} dimensions but current adapter produces "
                    f"{current_dim} dimensions. Run memory.migrate(user_id, "
                    "new_adapter) to re-embed your memories."
                )

        query_entities_stream: set[str] | None = None
        memory_entities_map_stream: dict[str, set[str]] | None = None
        if self._core._config.enable_entity_boost:
            query_entities_stream = self._core._entity_linker.extract(query)
            memory_entities_map_stream = {}
            for m in search_results:
                cached = m.metadata.get("extracted_entities")
                if cached is not None:
                    memory_entities_map_stream[m.memory_id] = set(cached)
                else:
                    memory_entities_map_stream[m.memory_id] = self._core._entity_linker.extract(m.content)  # noqa: E501

        corpus = [m.content for m in search_results] if len(search_results) > 1 else None
        score_cfg = scoring.ScoreConfig.from_memory_config(
            self._core._config, hybrid_search=hybrid_search
        )
        for memory in search_results:
            mem_entities = None
            if memory_entities_map_stream is not None:
                mem_entities = memory_entities_map_stream.get(memory.memory_id)
            memory.score = scoring.score_memory(
                memory,
                query_embedding,
                query=query,
                corpus=corpus,
                query_entities=query_entities_stream,
                memory_entities=mem_entities,
                config=score_cfg,
            )

        search_results.sort(key=lambda m: float(m.score), reverse=True)

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._core._config.max_tokens_default
        )
        if effective_max_tokens is not None:
            search_results = scoring.truncate_by_tokens(search_results, effective_max_tokens)

        yielded_memories: list[MemoryObject] = []
        for memory in scoring.mmr_rerank_stream(
            search_results, query_embedding, top_k, lambda_param=0.7
        ):
            tz = getattr(memory.last_accessed_at, "tzinfo", None) or timezone.utc
            memory.last_accessed_at = datetime.now(tz=tz)
            new_state = lifecycle.evaluate_lifecycle(memory, self._core._config.decay_threshold_hours)  # noqa: E501
            if new_state != memory.lifecycle_state:
                updated = lifecycle.transition(memory, new_state)
                self._core._store.update(updated)
                if self._core._metrics is not None:
                    self._core._metrics.lifecycle_transitions.inc(1)
            yielded_memories.append(memory)
            yield memory

        if self._core._metrics is not None:
            self._core._metrics.total_memories.set(self._core._store.count(user_id))
        self._core._run_hooks(
            "post",
            "recall",
            user_id=user_id,
            query=query,
            results=yielded_memories,
            namespace=namespace,
        )
        self._core._track_operation(
            "recall",
            user_id,
            {"query": query, "results_count": len(yielded_memories), "cache_hit": False, "stream": True},  # noqa: E501
            namespace=namespace,
        )
        if self._core._adaptive_retriever is not None:
            try:
                profile = self._core._adaptive_retriever.analyze_query(query)
                self._core._adaptive_retriever.record_feedback(user_id, query, profile)
            except Exception:
                logger.debug("Adaptive retrieval analysis failed", exc_info=True)

    def list_users(self) -> list[str]:
        """Get all unique user IDs that have memories."""
        from kemi.operations import _io

        return _io.list_users(self._core.build_io_runtime())

    def stats(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return health statistics for a user's memory store."""
        from kemi.operations import _io

        return _io.stats(self._core.build_io_runtime(), user_id, lifecycle_filter, session_id)

    async def alist_users(self) -> list[str]:
        """Async version of :meth:`list_users`."""
        from kemi.operations import _io

        return await _io.alist_users(self._core.build_io_runtime())

    async def astats(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`stats`."""
        from kemi.operations import _io

        return await _io.astats(self._core.build_io_runtime(), user_id, lifecycle_filter, session_id)  # noqa: E501

    def get_memory_graph(
        self,
        user_id: str,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Build a memory graph of entities and relations."""
        from kemi.operations import _io

        return _io.get_memory_graph(self._core.build_io_runtime(), user_id, namespace)

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
        """Recall memories created within a specific date range."""
        from kemi.operations import _io

        return _io.recall_between(
            self._core.build_io_runtime(),
            user_id,
            query,
            start,
            end,
            top_k,
            max_tokens,
            lifecycle_filter,
            namespace,
            session_id,
        )

    def recall_user_profile(
        self,
        user_id: str,
        *,
        top_k: int = 20,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        """Recall a user's long-lived profile — semantic facts and preferences."""
        from kemi.operations import _io

        return _io.recall_user_profile(
            self._core.build_io_runtime(), user_id, top_k=top_k, namespace=namespace
        )

    def recall_session_context(
        self,
        user_id: str,
        session_id: str,
        *,
        top_k: int = 20,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        """Recall recent episodic memories scoped to a specific session."""
        from kemi.operations import _io

        return _io.recall_session_context(
            self._core.build_io_runtime(),
            user_id,
            session_id,
            top_k=top_k,
            namespace=namespace,
        )

    def recall_agent_knowledge(
        self,
        agent_id: str,
        *,
        namespace: str = "default",
        top_k: int = 50,
    ) -> list[MemoryObject]:
        """Recall memories that belong to a specific agent."""
        from kemi.operations import _io

        return _io.recall_agent_knowledge(
            self._core.build_io_runtime(), agent_id, namespace=namespace, top_k=top_k
        )

    def recall_explain(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories with detailed score breakdowns."""
        from kemi.operations import _io

        return _io.recall_explain(
            self._core.build_io_runtime(),
            user_id,
            query,
            top_k,
            namespace,
            session_id,
        )

    def recall_since(
        self,
        user_id: str,
        query: str,
        hours: float = 24.0,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Recall memories created in the last N hours."""
        from kemi.operations import _io

        return _io.recall_since(
            self._core.build_io_runtime(),
            user_id,
            query,
            hours,
            top_k,
            max_tokens,
            lifecycle_filter,
        )

    def recall_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        """Recall memories by tag."""
        from kemi.operations import _io

        return _io.recall_by_tag(
            self._core.build_io_runtime(), user_id, tag, lifecycle_filter
        )

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
        """Recall memories and format as a context block."""
        from kemi.operations import _io

        return _io.context_block(
            self._core.build_io_runtime(),
            user_id,
            query,
            top_k,
            max_tokens,
            prefix,
            namespace,
            session_id,
        )

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
    ) -> list[MemoryObject] | AsyncGenerator[MemoryObject, None]:
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
        if len(user_ids) != len(queries):
            raise ValidationError("user_ids and queries must have the same length")

        async def _gather() -> dict[str, list[MemoryObject]]:
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

        return await _gather()

    async def arecall_since(
        self,
        user_id: str,
        query: str,
        hours: float = 24.0,
        top_k: int = 5,
        max_tokens: int | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        from kemi.operations import _io

        return await _io.arecall_since(
            self._core.build_io_runtime(),
            user_id,
            query,
            hours,
            top_k,
            max_tokens,
            lifecycle_filter,
        )

    async def arecall_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        from kemi.operations import _io

        return await _io.arecall_by_tag(
            self._core.build_io_runtime(), user_id, tag, lifecycle_filter
        )

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
        from kemi.operations import _io

        return await _io.acontext_block(
            self._core.build_io_runtime(),
            user_id,
            query,
            top_k,
            max_tokens,
            prefix,
            namespace,
            session_id,
        )
