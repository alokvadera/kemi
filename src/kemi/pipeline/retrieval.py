"""Retrieval pipeline: turn a ``(user_id, query, ...)`` request into ranked results.

Extracted from ``kemi._memory_impl``. The pipeline is a stateful
object (``RetrievalPipeline``) that holds a :class:`RetrievalContext`
with all its dependencies. It does not reference the ``Memory``
class — everything it needs comes through the context, which keeps
the pipeline independently testable.

The pipeline owns the full recall flow: default resolution, query
embedding, cache check, hook firing, storage search, metadata
filtering, embedding-dimension check, entity extraction, scoring,
MMR reranking, token truncation, lifecycle updates, metric
increments, cache write, and adaptive retrieval feedback. The
caller (:meth:`kemi.memory.facade.Memory.recall`) is responsible for
input validation and latency tracking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kemi.memory import lifecycle, scoring
from kemi.memory.model import LifecycleState, MemoryConfig, MemoryObject
from kemi.pipeline import _steps

if TYPE_CHECKING:
    from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
    from kemi.memory.entities import EntityLinker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Dependencies required to run a single retrieval.

    The context is intentionally explicit — no global state, no
    hidden coupling to the ``Memory`` class. Side-effect callbacks
    (``run_hooks``, ``track_operation``) are passed as callables
    that close over the orchestrator's state, which keeps the
    pipeline testable in isolation.
    """

    store: StorageAdapter
    embed: EmbeddingAdapter
    config: MemoryConfig
    entity_linker: EntityLinker
    query_cache: Any | None
    metrics: Any | None
    adaptive_retriever: Any | None

    # Side-effect callbacks. The ``Memory`` orchestrator wires these
    # to the implementations in ``kemi.operations._ops_*``.
    run_hooks: Callable[..., None] = lambda *args, **kwargs: None
    track_operation: Callable[..., None] = lambda *args, **kwargs: None


class RetrievalPipeline:
    """Encapsulates the recall flow.

    Public entry point is :meth:`retrieve` — the rest are private
    helpers that split the flow into testable steps.
    """

    # Magic numbers, kept as class constants for clarity.
    _METADATA_FETCH_MULTIPLIER = 10
    _DEFAULT_FETCH_MULTIPLIER = 3
    _MMR_LAMBDA = 0.7

    def __init__(self, ctx: RetrievalContext) -> None:
        self._ctx = ctx

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
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
        """Run the recall flow and return the top-k ranked results."""
        if hybrid_search is None:
            hybrid_search = self._ctx.config.hybrid_search

        query_embedding = self._embed_query(query)

        if lifecycle_filter is None:
            lifecycle_filter = lifecycle.get_recall_filter()

        cached = self._check_cache(
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
        if cached is not None:
            return cached

        self._ctx.run_hooks(
            "pre", "recall", user_id=user_id, query=query, namespace=namespace
        )

        # When metadata_filter is active we may need more than top_k
        # results from storage because filtering is applied post-hoc.
        # Use a larger multiplier to increase the chance of returning
        # top_k results after filtering.
        fetch_multiplier = (
            self._METADATA_FETCH_MULTIPLIER
            if metadata_filter is not None
            else self._DEFAULT_FETCH_MULTIPLIER
        )
        search_results = self._search_storage(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
            fetch_multiplier=fetch_multiplier,
            metadata_filter=metadata_filter,
        )

        self._validate_embedding_dim(search_results)

        query_entities, memory_entities_map = self._build_entity_maps(query, search_results)

        ranked = self._rank(
            search_results=search_results,
            query_embedding=query_embedding,
            query=query,
            hybrid_search=hybrid_search,
            query_entities=query_entities,
            memory_entities_map=memory_entities_map,
        )

        ranked = self._mmr_rerank(ranked, query_embedding, top_k)

        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._ctx.config.max_tokens_default
        )
        ranked = self._truncate(ranked, effective_max_tokens)

        final_results = ranked[:top_k]

        self._update_lifecycle(final_results)

        if self._ctx.metrics is not None:
            self._ctx.metrics.total_memories.set(self._ctx.store.count(user_id))

        self._cache_results(
            user_id=user_id,
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
            lifecycle_filter=lifecycle_filter,
            hybrid_search=hybrid_search,
            namespace=namespace,
            session_id=session_id,
            metadata_filter=metadata_filter,
            results=final_results,
        )

        self._ctx.run_hooks(
            "post",
            "recall",
            user_id=user_id,
            query=query,
            results=final_results,
            namespace=namespace,
        )
        self._ctx.track_operation(
            "recall",
            user_id,
            {"query": query, "results_count": len(final_results), "cache_hit": False},
            namespace=namespace,
        )
        self._adaptive_feedback(user_id, query)

        return final_results

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    # Extraction policy: only methods that add meaningful pipeline logic
    # (filtering, scoring, lifecycle transitions, etc.) are extracted to
    # _steps.py.  The remaining one-liners below are pure adapter
    # delegations — the pipeline adds no logic, so extracting them would
    # just create pass-through functions with no incremental testability.
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """Pure delegation to the embedding adapter — no pipeline logic."""
        return self._ctx.embed.embed_single(query)

    def _check_cache(
        self,
        user_id: str,
        query: str,
        top_k: int,
        max_tokens: int | None,
        lifecycle_filter: list[LifecycleState],
        hybrid_search: bool,
        namespace: str,
        session_id: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[MemoryObject] | None:
        """Return cached results on hit, else None. Records the cache hit."""
        if self._ctx.query_cache is None:
            return None
        cache_key = self._ctx.query_cache._make_key(
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
        cached = self._ctx.query_cache.get(cache_key)
        if cached is None:
            return None
        self._ctx.track_operation(
            "recall",
            user_id,
            {"query": query, "results_count": len(cached), "cache_hit": True},
            namespace=namespace,
        )
        return cached

    def _search_storage(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        lifecycle_filter: list[LifecycleState],
        namespace: str,
        session_id: str | None,
        fetch_multiplier: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[MemoryObject]:
        """Run the storage search and apply the metadata filter post-hoc."""
        return _steps.search_and_filter_storage(
            store=self._ctx.store,
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
            fetch_multiplier=fetch_multiplier,
            metadata_filter=metadata_filter,
        )

    def _validate_embedding_dim(self, search_results: list[MemoryObject]) -> None:
        """Raise if stored memories have a different embedding dimension than the current adapter."""  # noqa: E501
        _steps.validate_embedding_dimension(search_results, self._ctx.embed.dimension())

    def _build_entity_maps(
        self,
        query: str,
        search_results: list[MemoryObject],
    ) -> tuple[set[str] | None, dict[str, set[str]] | None]:
        """Extract query entities and a per-memory entity map (if entity boost is enabled)."""
        return _steps.build_entity_boost_maps(
            query,
            search_results,
            self._ctx.config.enable_entity_boost,
            self._ctx.entity_linker,
        )

    def _rank(
        self,
        search_results: list[MemoryObject],
        query_embedding: list[float],
        query: str,
        hybrid_search: bool,
        query_entities: set[str] | None,
        memory_entities_map: dict[str, set[str]] | None,
    ) -> list[MemoryObject]:
        return scoring.rank_memories(
            search_results,
            query_embedding,
            query=query,
            query_entities=query_entities,
            memory_entities_map=memory_entities_map,
            config=scoring.ScoreConfig.from_memory_config(
                self._ctx.config, hybrid_search=hybrid_search
            ),
        )

    def _mmr_rerank(
        self,
        ranked: list[MemoryObject],
        query_embedding: list[float],
        top_k: int,
    ) -> list[MemoryObject]:
        if len(ranked) <= top_k or top_k <= 1:
            return ranked
        return scoring.mmr_rerank(ranked, query_embedding, top_k, lambda_param=self._MMR_LAMBDA)

    def _truncate(
        self,
        ranked: list[MemoryObject],
        max_tokens: int | None,
    ) -> list[MemoryObject]:
        if max_tokens is None:
            return ranked
        return scoring.truncate_by_tokens(ranked, max_tokens)

    def _update_lifecycle(self, results: list[MemoryObject]) -> None:
        """Bump ``last_accessed_at`` and apply lifecycle transitions."""
        _steps.update_lifecycle_inplace(
            results,
            self._ctx.config.decay_threshold_hours,
            self._ctx.store,
            self._ctx.metrics,
        )

    def _cache_results(
        self,
        user_id: str,
        query: str,
        top_k: int,
        max_tokens: int | None,
        lifecycle_filter: list[LifecycleState],
        hybrid_search: bool,
        namespace: str,
        session_id: str | None,
        metadata_filter: dict[str, Any] | None,
        results: list[MemoryObject],
    ) -> None:
        if self._ctx.query_cache is None:
            return
        cache_key = self._ctx.query_cache._make_key(
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
        self._ctx.query_cache.put(cache_key, results)

    def _adaptive_feedback(self, user_id: str, query: str) -> None:
        _steps.adaptive_feedback(self._ctx.adaptive_retriever, user_id, query)


__all__ = ["RetrievalContext", "RetrievalPipeline"]
