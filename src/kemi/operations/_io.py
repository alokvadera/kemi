"""Direct memory I/O operations: ``update``, ``forget``, ``context_block``, ``stats``,
``list_users``, and their async / batch variants.

These are orchestrator-level CRUD operations that don't fit into the
:class:`~kemi.pipeline.ingestion.IngestionPipeline` or
:class:`~kemi.pipeline.retrieval.RetrievalPipeline` subsystems. Each
function takes a :class:`MemoryIORuntime` containing the dependencies
it needs (storage, embedding, entity linker, config, side-effect
callables) so the operations are decoupled from the
:class:`~kemi.memory.facade.Memory` class.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from kemi.exceptions import NotFoundError, ValidationError
from kemi.infra.webhooks import WebhookEventType
from kemi.memory import lifecycle, scoring
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType
from kemi.memory.versions import DiffResult, RollbackResult, VersionSnapshot

if TYPE_CHECKING:
    from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
    from kemi.memory.entities import EntityLinker
    from kemi.memory.versions import MemoryVersionStore

logger = logging.getLogger(__name__)


def _memory_to_dict(memory: MemoryObject) -> dict[str, Any]:
    """Convert a MemoryObject to a JSON-serialisable dict for webhook payloads.

    Delegates to :func:`kemi.memory.model.memory_to_dict` — the canonical
    implementation lives on the model module so both this module and
    :mod:`kemi.pipeline.ingestion` share a single source.
    """
    from kemi.memory.model import memory_to_dict as _to_dict

    return _to_dict(memory)


@dataclass
class MemoryIORuntime:
    """Dependencies required by the I/O operations in this module.

    The runtime is intentionally explicit — no global state, no hidden
    coupling to ``Memory``. Side-effect callbacks are passed as
    callables that close over the orchestrator's state.
    """

    store: StorageAdapter
    embed: EmbeddingAdapter
    entity_linker: EntityLinker
    config: Any
    metrics: Any | None

    # Side-effect callbacks.
    run_hooks: Callable[..., None] = lambda *args, **kwargs: None
    track_operation: Callable[..., None] = lambda *args, **kwargs: None
    log_audit: Callable[..., None] = lambda *args, **kwargs: None
    dispatch_webhook: Callable[..., None] = lambda *args, **kwargs: None
    latency_tracker: Callable[[str], Any] = lambda operation: nullcontext()
    # ``recall_fn`` is used by ``context_block`` to fetch memories.
    # It is bound to ``Memory.recall`` (which delegates to the
    # retrieval pipeline) by the orchestrator.
    recall_fn: Callable[..., list[MemoryObject]] = lambda *args, **kwargs: []
    get_version_store: Callable[[], MemoryVersionStore | None] = lambda: None
    auto_prune_versions: Callable[[str], None] = lambda memory_id: None
    auto_prune_versions_enabled: bool = False
    max_versions_per_memory: int = 0


# ----------------------------------------------------------------------
# update
# ----------------------------------------------------------------------


def update(
    ctx: MemoryIORuntime,
    memory_id: str,
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    memory_type: MemoryType | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update an existing memory. Returns the memory_id.

    Raises ``ValueError`` if the memory does not exist.
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

    with ctx.latency_tracker("update"):
        ctx.run_hooks("pre", "update", memory_id=memory_id)
        memory = ctx.store.get(memory_id)
        if memory is None:
            raise NotFoundError(f"Memory not found: {memory_id}")

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

        previous_state = _memory_to_dict(memory_before)

        # Now apply all mutations.
        if content is not None:
            memory.content = content
            memory.embedding = ctx.embed.embed_single(content)
            memory.embedding_dim = len(memory.embedding)
            memory.last_accessed_at = datetime.now(timezone.utc)
            if ctx.config.enable_entity_boost:
                memory.metadata["extracted_entities"] = list(
                    ctx.entity_linker.extract(content)
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

        vs = ctx.get_version_store()
        if vs is not None:
            try:
                keep = (
                    ctx.max_versions_per_memory
                    if getattr(ctx, "auto_prune_versions_enabled", False)
                    else None
                )
                vs.record_and_update(
                    memory_before,
                    memory,
                    ctx.store,
                    changed_by="update",
                    keep_count=keep,
                )
            except Exception as exc:
                try:
                    vs.record_before_update(memory_before, memory, changed_by="update")
                    ctx.auto_prune_versions(memory_id)
                except Exception as fallback_exc:
                    logger.error(
                        "Version recording failed for update %s: %s",
                        memory_id,
                        fallback_exc,
                    )
                    raise fallback_exc from exc
        memory.version = (memory_before.version or 0) + 1
        ctx.store.update(memory)

        snapshot = _memory_to_dict(memory)
        ctx.dispatch_webhook(
            WebhookEventType.UPDATED,
            memory_id=memory_id,
            user_id=memory.user_id,
            snapshot=snapshot,
            previous_state=previous_state,
        )

        ctx.run_hooks("post", "update", memory_id=memory_id, version=memory.version)
        logger.info(f"Updated memory: {memory_id} (version {memory.version})")
        ctx.track_operation(
            "update",
            memory.user_id,
            {"memory_id": memory_id, "version": memory.version},
            memory_id,
            memory.namespace,
        )
        return memory_id


def update_many(
    ctx: MemoryIORuntime,
    memory_ids: list[str],
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    memory_type: MemoryType | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Update multiple memories in sequence."""
    updated: list[str] = []
    for mid in memory_ids:
        update(
            ctx,
            mid,
            content=content,
            importance=importance,
            confidence=confidence,
            memory_type=memory_type,
            metadata=metadata,
        )
        updated.append(mid)
    return updated


async def aupdate(
    ctx: MemoryIORuntime,
    memory_id: str,
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    memory_type: MemoryType | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Async version of :func:`update`."""
    return await asyncio.to_thread(
        update,
        ctx,
        memory_id,
        content,
        importance,
        confidence,
        memory_type,
        metadata,
    )


async def aupdate_many(
    ctx: MemoryIORuntime,
    memory_ids: list[str],
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    memory_type: MemoryType | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Async version of :func:`update_many` — runs updates concurrently.

    Small batches (≤10) use individual concurrent ``aupdate`` calls via
    ``asyncio.gather``. Larger batches fall back to a single threaded
    ``update_many`` to avoid excessive thread overhead.
    """
    if not memory_ids:
        return []
    if len(memory_ids) <= 10:
        tasks = [
            aupdate(
                ctx,
                mid,
                content=content,
                importance=importance,
                confidence=confidence,
                memory_type=memory_type,
                metadata=metadata,
            )
            for mid in memory_ids
        ]
        return list(await asyncio.gather(*tasks))
    return await asyncio.to_thread(
        update_many,
        ctx,
        memory_ids,
        content,
        importance,
        confidence,
        memory_type,
        metadata,
    )


# ----------------------------------------------------------------------
# forget
# ----------------------------------------------------------------------


def forget(ctx: MemoryIORuntime, user_id: str, memory_id: str | None = None) -> int:
    """Delete a single memory by ID, or all memories for a user.

    Returns 1 if a single memory was deleted, 0 if it was not found, or
    the number of memories deleted when ``memory_id`` is None.
    """
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")

    with ctx.latency_tracker("forget"):
        ctx.run_hooks(
            "pre", "forget", user_id=user_id, memory_id=memory_id, namespace="default"
        )
        if memory_id is not None:
            deleted = ctx.store.delete_by_id(memory_id)
            if deleted:
                ctx.dispatch_webhook(
                    WebhookEventType.DELETED,
                    memory_id=memory_id,
                    user_id=user_id,
                )
            ctx.run_hooks(
                "post",
                "forget",
                user_id=user_id,
                memory_id=memory_id,
                deleted=deleted,
                namespace="default",
            )
            ctx.track_operation(
                "forget",
                user_id,
                {"memory_id": memory_id, "deleted": deleted},
                memory_id,
                namespace="default",
            )
            ctx.log_audit(
                "forget",
                user_id,
                memory_id=memory_id,
                namespace="default",
                details={"deleted": deleted},
            )
            return 1 if deleted else 0
        else:
            count = ctx.store.delete_by_user(user_id)
            if count:
                ctx.dispatch_webhook(
                    WebhookEventType.DELETED,
                    memory_id="batch",
                    user_id=user_id,
                    deleted_count=count,
                )
            ctx.run_hooks(
                "post", "forget", user_id=user_id, deleted_count=count, namespace="default"
            )
            ctx.track_operation(
                "forget", user_id, {"deleted_count": count}, namespace="default"
            )
            ctx.log_audit(
                "forget",
                user_id,
                memory_id="batch",
                namespace="default",
                details={"deleted_count": count},
            )
            return count


def forget_many(ctx: MemoryIORuntime, memory_ids: list[str]) -> int:
    """Delete multiple memories by ID in sequence.

    Fires pre/post event hooks, webhooks, metrics, and audit for each
    individual delete.
    """
    count = 0
    for mid in memory_ids:
        memory = ctx.store.get(mid)
        if memory is None:
            continue
        ctx.run_hooks(
            "pre", "forget", user_id=memory.user_id, memory_id=mid, namespace=memory.namespace
        )
        if ctx.store.delete_by_id(mid):
            count += 1
            ctx.dispatch_webhook(
                WebhookEventType.DELETED,
                memory_id=mid,
                user_id=memory.user_id,
            )
            ctx.run_hooks(
                "post",
                "forget",
                user_id=memory.user_id,
                memory_id=mid,
                deleted=True,
                namespace=memory.namespace,
            )
            ctx.track_operation(
                "forget",
                memory.user_id,
                {"memory_id": mid, "deleted": True},
                mid,
                memory.namespace,
            )
            ctx.log_audit(
                "forget",
                memory.user_id,
                memory_id=mid,
                namespace=memory.namespace,
                details={"deleted": True},
            )
    return count


async def aforget(
    ctx: MemoryIORuntime, user_id: str, memory_id: str | None = None
) -> int:
    """Async version of :func:`forget`."""
    return await asyncio.to_thread(forget, ctx, user_id, memory_id)


async def aforget_many(ctx: MemoryIORuntime, memory_ids: list[str]) -> int:
    """Async version of :func:`forget_many`."""
    return await asyncio.to_thread(forget_many, ctx, memory_ids)


# ----------------------------------------------------------------------
# context_block
# ----------------------------------------------------------------------


def context_block(
    ctx: MemoryIORuntime,
    user_id: str,
    query: str,
    top_k: int = 5,
    max_tokens: int = 1500,
    prefix: str = "Relevant context from memory:",
    namespace: str = "default",
    session_id: str | None = None,
) -> str:
    """Recall memories for ``query`` and format them as a context block."""
    memories = ctx.recall_fn(
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


async def acontext_block(
    ctx: MemoryIORuntime,
    user_id: str,
    query: str,
    top_k: int = 5,
    max_tokens: int = 1500,
    prefix: str = "Relevant context from memory:",
    namespace: str = "default",
    session_id: str | None = None,
) -> str:
    """Async version of :func:`context_block`."""
    return await asyncio.to_thread(
        context_block, ctx, user_id, query, top_k, max_tokens, prefix, namespace, session_id
    )


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------


def stats(
    ctx: MemoryIORuntime,
    user_id: str,
    lifecycle_filter: list[LifecycleState] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return health statistics for a user's memory store.

    Returns a dict with these keys:
      total: int — total number of memories
      by_lifecycle: dict — count per lifecycle state
      by_source: dict — count per memory source
      avg_importance: float — average importance score (0.0 if no memories)
      tag_counts: dict — how many memories each tag appears in
      total_with_tags: int — number of memories that have at least one tag
      total_without_tags: int — number of memories with no tags

    Backed by ``StorageAdapter.count_aggregates`` which uses SQL GROUP BY
    for SQL adapters (O(1) per state) and falls back to a Python scan for
    non-SQL adapters.
    """
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")

    agg = ctx.store.count_aggregates(
        user_id, lifecycle_filter=lifecycle_filter, session_id=session_id
    )
    total = agg["total"]
    avg_importance = (
        agg["avg_importance_numerator"] / total if total > 0 else 0.0
    )
    return {
        "total": total,
        "by_lifecycle": agg["by_lifecycle"],
        "by_source": agg["by_source"],
        "avg_importance": avg_importance,
        "tag_counts": agg["tag_counts"],
        "total_with_tags": agg["total_with_tags"],
        "total_without_tags": total - agg["total_with_tags"],
    }


async def astats(
    ctx: MemoryIORuntime,
    user_id: str,
    lifecycle_filter: list[LifecycleState] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Async version of :func:`stats`."""
    return await asyncio.to_thread(
        stats, ctx, user_id, lifecycle_filter, session_id
    )


# ----------------------------------------------------------------------
# list_users
# ----------------------------------------------------------------------


def list_users(ctx: MemoryIORuntime) -> list[str]:
    """Get all unique user IDs that have memories."""
    return ctx.store.get_all_users()


async def alist_users(ctx: MemoryIORuntime) -> list[str]:
    """Async version of :func:`list_users`."""
    return await asyncio.to_thread(list_users, ctx)


# ----------------------------------------------------------------------
# Maintenance: prune, prune_expired
# ----------------------------------------------------------------------


def _known_namespaces(store: StorageAdapter, user_id: str) -> set[str]:
    """Return the set of distinct namespaces holding memories for a user.

    Delegates to ``store.get_namespaces(user_id)`` so adapters with
    SQL backing can use ``SELECT DISTINCT namespace`` instead of
    sampling a limited batch in Python.
    """
    namespaces: set[str] = set()
    # Default namespace is always present even if empty.
    namespaces.add("default")
    try:
        store_namespaces = store.get_namespaces(user_id)
        namespaces.update(store_namespaces)
    except Exception as exc:
        logger.warning(
            "get_namespaces failed for user %s: %s", user_id, exc
        )
    return namespaces


def prune(
    ctx: MemoryIORuntime,
    user_id: str,
    max_age_days: float | None = None,
    min_importance: float | None = None,
    lifecycle_states: list[LifecycleState] | None = None,
    namespace: str = "default",
) -> int:
    """Auto-prune old or low-importance memories. Returns the count deleted."""
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")

    with ctx.latency_tracker("prune"):
        all_memories = ctx.store.get_all_by_user(
            user_id,
            lifecycle_filter=lifecycle_states or [LifecycleState.DECAYING],
            namespace=namespace,
        )

        to_delete: list[str] = []
        now = datetime.now(timezone.utc)

        for mem in all_memories:
            # Both filters are ANDed when both are present: a memory must
            # exceed max_age_days AND fall below min_importance. When only
            # one filter is supplied it alone determines deletion.
            if max_age_days is not None or min_importance is not None:
                age_days = (now - mem.created_at).total_seconds() / 86400.0
                should_delete_age = (
                    max_age_days is None or age_days > max_age_days
                )
                should_delete_importance = (
                    min_importance is None or mem.importance < min_importance
                )
                if should_delete_age and should_delete_importance:
                    to_delete.append(mem.memory_id)

        for mid in to_delete:
            ctx.store.delete_by_id(mid)

        logger.info(f"Pruned {len(to_delete)} memories for user {user_id}")
        ctx.track_operation(
            "prune", user_id, {"deleted_count": len(to_delete)}, namespace=namespace
        )
        return len(to_delete)


def prune_expired(
    ctx: MemoryIORuntime,
    user_id: str | None = None,
    namespace: str | None = None,
) -> int:
    """Delete memories whose ``expires_at`` has passed.

    Sweeps ACTIVE and DECAYING memories with a non-null ``expires_at``
    in the past, then removes them from the store. Called automatically
    by :func:`run_maintenance`.
    """
    with ctx.latency_tracker("prune_expired"):
        now = datetime.now(timezone.utc)
        deleted = 0

        if user_id is not None:
            users = [user_id]
        else:
            users = ctx.store.get_all_users()

        for uid in users:
            if namespace is not None:
                memories = ctx.store.get_all_by_user(
                    uid,
                    lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
                    namespace=namespace,
                )
            else:
                # Sweep across all namespaces — the base adapter's
                # ``get_all_by_user`` defaults ``namespace="default"``,
                # so we fetch from each known namespace explicitly.
                namespaces = _known_namespaces(ctx.store, uid)
                memories = []
                for ns in namespaces:
                    memories.extend(
                        ctx.store.get_all_by_user(
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
                    if ctx.store.delete_by_id(mem.memory_id):
                        deleted += 1

        if deleted > 0:
            logger.info(
                f"Pruned {deleted} expired memories"
                + (f" for user {user_id}" if user_id else "")
            )
        ctx.track_operation(
            "prune_expired",
            user_id or "all",
            {"deleted_count": deleted},
            namespace=namespace or "all",
        )
        return deleted


# ----------------------------------------------------------------------
# Analysis: consolidate, cluster_topics, extract_entities, get_memory_graph
# ----------------------------------------------------------------------


def consolidate(
    ctx: MemoryIORuntime,
    user_id: str,
    namespace: str = "default",
    min_memories: int = 5,
    max_age_days: float = 30.0,
    with_llm_summary: bool = False,
) -> str | None:
    """Consolidate old episodic memories into a semantic summary.

    Returns the memory ID of the consolidated summary, or ``None`` if
    the consolidation module is not available.
    """
    try:
        from kemi.memory import consolidation
    except ImportError:  # pragma: no cover
        logger.warning("consolidation module not available")
        return None

    mid = consolidation.consolidate(
        store=ctx.store,
        embed=ctx.embed,
        user_id=user_id,
        namespace=namespace,
        min_memories=min_memories,
        max_age_days=max_age_days,
        with_llm_summary=with_llm_summary,
        summarizer_llm_provider=ctx.config.summarizer_llm_provider,
        summarizer_llm_model=ctx.config.summarizer_llm_model,
        summarizer_prompt_template=ctx.config.summarizer_prompt_template,
    )
    if mid is not None:
        ctx.dispatch_webhook(
            WebhookEventType.CONSOLIDATED,
            memory_id=mid,
            user_id=user_id,
        )
    return mid


def cluster_topics(
    ctx: MemoryIORuntime,
    user_id: str,
    n_clusters: int = 3,
    namespace: str = "default",
) -> dict[str, list[MemoryObject]]:
    """Cluster memories into topic groups using embeddings.

    Requires the optional topics module: ``from kemi.nlp.topics import topics``
    and a topic model to be installed.
    """
    try:
        from kemi.nlp import topics
    except ImportError:  # pragma: no cover
        logger.warning("topics module not available")
        return {}

    return topics.cluster_memories(
        store=ctx.store,
        user_id=user_id,
        n_clusters=n_clusters,
        namespace=namespace,
    )


def extract_entities(
    ctx: MemoryIORuntime,
    memory_id: str,
) -> list[dict[str, Any]]:
    """Extract named entities from a memory's content.

    Uses regex-based and graph-based extraction depending on the
    configured :class:`EntityLinker`.
    """
    try:
        from kemi.nlp import graph
    except ImportError:  # pragma: no cover
        logger.warning("graph module not available")
        return []

    memory = ctx.store.get(memory_id)
    if memory is None:
        raise NotFoundError(f"Memory not found: {memory_id}")

    return graph.extract_entities(memory.content)


def get_memory_graph(
    ctx: MemoryIORuntime,
    user_id: str,
    namespace: str = "default",
) -> dict[str, Any]:
    """Build a memory graph of entities and relations.

    Requires the optional graph module: ``from kemi.nlp import graph``.
    """
    try:
        from kemi.nlp import graph
    except ImportError:  # pragma: no cover
        logger.warning("graph module not available")
        return {"entities": [], "relations": []}

    return graph.build_memory_graph(
        store=ctx.store,
        user_id=user_id,
        namespace=namespace,
    )


# ----------------------------------------------------------------------
# Feedback
# ----------------------------------------------------------------------


def feedback(
    ctx: MemoryIORuntime,
    user_id: str,
    memory_id: str,
    helpful: bool = True,
    namespace: str = "default",
) -> None:
    """Record user feedback on a recalled memory.

    Stores feedback in ``metadata["feedback"]`` and adjusts importance:
    ``helpful=True`` boosts importance (up to 1.0), ``helpful=False`` reduces
    it (down to 0.0).
    """
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if not memory_id or not memory_id.strip():
        raise ValidationError("memory_id cannot be empty")

    memory = ctx.store.get(memory_id)
    if memory is None:
        raise NotFoundError(f"Memory not found: {memory_id}")
    if memory.user_id != user_id:
        raise ValidationError("Memory does not belong to this user")
    if memory.namespace != namespace:
        raise ValidationError("Memory does not belong to this namespace")

    feedback_entry = {
        "helpful": helpful,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if "feedback" not in memory.metadata:
        memory.metadata["feedback"] = []
    memory.metadata["feedback"].append(feedback_entry)

    adjustment = 0.05 if helpful else -0.05
    memory.importance = max(0.0, min(1.0, memory.importance + adjustment))

    ctx.store.update(memory)
    logger.info(
        f"Feedback recorded for memory {memory_id}: helpful={helpful}, "
        f"new_importance={memory.importance:.2f}"
    )
    ctx.track_operation(
        "feedback", user_id, {"memory_id": memory_id, "helpful": helpful}, memory_id, namespace
    )


# ----------------------------------------------------------------------
# Entity backfill
# ----------------------------------------------------------------------


def backfill_entities(
    ctx: MemoryIORuntime,
    user_id: str | None = None,
    namespace: str | None = None,
) -> int:
    """Backfill ``extracted_entities`` for memories that don't have them yet.

    Returns the number of memories that were backfilled. Skips entirely
    when entity boost is disabled.
    """
    if not ctx.config.enable_entity_boost:
        logger.info("Entity boost is disabled; skipping entity backfill")
        return 0

    with ctx.latency_tracker("backfill_entities"):
        if user_id is not None:
            users = [user_id]
        else:
            users = ctx.store.get_all_users()

        backfilled = 0
        for uid in users:
            if namespace is not None:
                namespaces = [namespace]
            else:
                namespaces = _known_namespaces(ctx.store, uid)

            for ns in namespaces:
                memories = ctx.store.get_all_by_user(
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
                    entities = ctx.entity_linker.extract(mem.content)
                    mem.metadata["extracted_entities"] = list(entities)
                    ctx.store.update(mem)
                    backfilled += 1

        if backfilled > 0:
            logger.info(
                f"Backfilled extracted_entities for {backfilled} memories"
                + (f" (user={user_id})" if user_id else "")
            )
        ctx.track_operation(
            "backfill_entities",
            user_id or "all",
            {"backfilled": backfilled},
            namespace=namespace or "all",
        )
        return backfilled


async def abackfill_entities(
    ctx: MemoryIORuntime,
    user_id: str | None = None,
    namespace: str | None = None,
) -> int:
    """Async version of :func:`backfill_entities`."""
    return await asyncio.to_thread(backfill_entities, ctx, user_id, namespace)


# ----------------------------------------------------------------------
# run_maintenance
# ----------------------------------------------------------------------


def run_maintenance(
    ctx: MemoryIORuntime,
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

    Returns a dict with ``pruned`` (int), ``expired`` (int),
    ``consolidated`` (str | None), and ``backfilled`` (int) keys.
    """
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")

    with ctx.latency_tracker("run_maintenance"):
        result: dict[str, Any] = {
            "pruned": 0,
            "expired": 0,
            "consolidated": None,
            "backfilled": 0,
        }

        if auto_backfill_entities:
            result["backfilled"] = backfill_entities(
                ctx, user_id=user_id, namespace=namespace
            )

        if auto_prune:
            result["pruned"] = prune(
                ctx,
                user_id=user_id,
                max_age_days=prune_max_age_days,
                min_importance=prune_min_importance,
                namespace=namespace,
            )

        if auto_prune_expired:
            result["expired"] = prune_expired(ctx, user_id=user_id, namespace=namespace)

        if auto_consolidate:
            result["consolidated"] = consolidate(
                ctx,
                user_id=user_id,
                namespace=namespace,
                min_memories=consolidate_min_memories,
                max_age_days=consolidate_max_age_days,
                with_llm_summary=consolidate_with_llm_summary,
            )

        logger.info(f"Maintenance complete for {user_id}: {result}")
        ctx.track_operation("run_maintenance", user_id, result, namespace=namespace)
        return result


def recall_between(
    ctx: MemoryIORuntime,
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

    Delegates to :func:`ctx.recall_fn` and filters results by ``created_at``.
    """
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if not query or not query.strip():
        raise ValidationError("query cannot be empty")
    if top_k < 1:
        raise ValidationError(f"top_k must be at least 1, got {top_k}")

    all_results = ctx.recall_fn(
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


def _touch_lifecycle(
    ctx: MemoryIORuntime,
    memories: list[MemoryObject],
) -> None:
    """Bump last_accessed_at, evaluate lifecycle, persist any transitions."""
    for mem in memories:
        mem.last_accessed_at = datetime.now(timezone.utc)
        new_state = lifecycle.evaluate_lifecycle(mem, ctx.config.decay_threshold_hours)
        if new_state != mem.lifecycle_state:
            updated = lifecycle.transition(mem, new_state)
            ctx.store.update(updated)


def recall_user_profile(
    ctx: MemoryIORuntime,
    user_id: str,
    *,
    top_k: int = 20,
    namespace: str = "default",
) -> list[MemoryObject]:
    """Recall a user's long-lived profile — semantic facts and preferences.

    Filters for SEMANTIC memories that are ACTIVE/DECAYING/ARCHIVED, then ranks
    by importance (highest first).
    """
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if top_k < 1:
        raise ValidationError(f"top_k must be at least 1, got {top_k}")

    with ctx.latency_tracker("recall_user_profile"):
        ctx.run_hooks("pre", "recall_user_profile", user_id=user_id, namespace=namespace)

        all_memories = ctx.store.get_all_by_user(
            user_id,
            lifecycle_filter=[
                LifecycleState.ACTIVE,
                LifecycleState.DECAYING,
                LifecycleState.ARCHIVED,
            ],
            namespace=namespace,
        )

        profile_memories = [m for m in all_memories if m.memory_type == MemoryType.SEMANTIC]
        profile_memories.sort(key=lambda m: m.importance, reverse=True)

        _touch_lifecycle(ctx, profile_memories[:top_k])

        ctx.run_hooks(
            "post",
            "recall_user_profile",
            user_id=user_id,
            results=profile_memories[:top_k],
            namespace=namespace,
        )
        ctx.track_operation(
            "recall_user_profile",
            user_id,
            {"results_count": len(profile_memories[:top_k])},
            namespace=namespace,
        )
        return profile_memories[:top_k]


def recall_session_context(
    ctx: MemoryIORuntime,
    user_id: str,
    session_id: str,
    *,
    top_k: int = 20,
    namespace: str = "default",
) -> list[MemoryObject]:
    """Recall recent episodic memories scoped to a specific session."""
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if not session_id or not session_id.strip():
        raise ValidationError("session_id cannot be empty")
    if top_k < 1:
        raise ValidationError(f"top_k must be at least 1, got {top_k}")

    with ctx.latency_tracker("recall_session_context"):
        ctx.run_hooks(
            "pre",
            "recall_session_context",
            user_id=user_id,
            session_id=session_id,
            namespace=namespace,
        )

        all_memories = ctx.store.get_all_by_user(
            user_id,
            lifecycle_filter=[
                LifecycleState.ACTIVE,
                LifecycleState.DECAYING,
                LifecycleState.ARCHIVED,
            ],
            namespace=namespace,
            session_id=session_id,
        )

        session_memories = [m for m in all_memories if m.memory_type == MemoryType.EPISODIC]
        session_memories.sort(
            key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        _touch_lifecycle(ctx, session_memories[:top_k])

        ctx.run_hooks(
            "post",
            "recall_session_context",
            user_id=user_id,
            session_id=session_id,
            results=session_memories[:top_k],
            namespace=namespace,
        )
        ctx.track_operation(
            "recall_session_context",
            user_id,
            {"results_count": len(session_memories[:top_k]), "session_id": session_id},
            namespace=namespace,
        )
        return session_memories[:top_k]


def recall_agent_knowledge(
    ctx: MemoryIORuntime,
    agent_id: str,
    *,
    namespace: str = "default",
    top_k: int = 50,
) -> list[MemoryObject]:
    """Recall memories that belong to a specific agent (scans all users)."""
    if not agent_id or not agent_id.strip():
        raise ValidationError("agent_id cannot be empty")
    if top_k < 1:
        raise ValidationError(f"top_k must be at least 1, got {top_k}")

    with ctx.latency_tracker("recall_agent_knowledge"):
        ctx.run_hooks("pre", "recall_agent_knowledge", agent_id=agent_id, namespace=namespace)

        all_users = ctx.store.get_all_users()
        agent_memories: list[MemoryObject] = []
        for uid in all_users:
            user_memories = ctx.store.get_all_by_user(
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
        _touch_lifecycle(ctx, agent_memories[:top_k])

        ctx.run_hooks(
            "post",
            "recall_agent_knowledge",
            agent_id=agent_id,
            results=agent_memories[:top_k],
            namespace=namespace,
        )
        ctx.track_operation(
            "recall_agent_knowledge",
            "all",
            {"results_count": len(agent_memories[:top_k]), "agent_id": agent_id},
            namespace=namespace,
        )
        return agent_memories[:top_k]


def recall_explain(
    ctx: MemoryIORuntime,
    user_id: str,
    query: str,
    top_k: int = 5,
    namespace: str = "default",
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Recall memories with detailed score breakdowns (semantic/recency/bm25/importance)."""
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if not query or not query.strip():
        raise ValidationError("query cannot be empty")
    if top_k < 1:
        raise ValidationError(f"top_k must be at least 1, got {top_k}")

    query_embedding = ctx.embed.embed_single(query)

    search_results = ctx.store.search(
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
    if ctx.config.enable_entity_boost:
        query_entities_explain = ctx.entity_linker.extract(query)
        memory_entities_map_explain = {}
        for m in search_results:
            cached = m.metadata.get("extracted_entities")
            if cached is not None:
                memory_entities_map_explain[m.memory_id] = set(cached)
            else:
                memory_entities_map_explain[m.memory_id] = ctx.entity_linker.extract(m.content)

    explained: list[dict[str, Any]] = []
    for memory in search_results:
        semantic = scoring.cosine_similarity(memory.embedding, query_embedding)
        semantic_norm = (semantic + 1.0) / 2.0
        recency = scoring.temporal_recency(memory.last_accessed_at)

        entity_score = 0.0
        if (
            ctx.config.enable_entity_boost
            and query_entities_explain is not None
            and memory_entities_map_explain is not None
        ):
            mem_entities = memory_entities_map_explain.get(memory.memory_id, set())
            entity_score = scoring.jaccard_similarity(query_entities_explain, mem_entities)

        if ctx.config.hybrid_search:
            if corpus and len(corpus) > 1:
                bm25 = scoring.bm25_score_corpus(query, memory.content, corpus)
            else:
                bm25 = scoring.bm25_score(query, memory.content)
            final = (
                semantic_norm * ctx.config.weight_semantic
                + recency * ctx.config.weight_recency
                + bm25 * ctx.config.weight_bm25
                + entity_score * ctx.config.entity_boost_weight
            )
            explanation = {
                "semantic_score": round(semantic_norm, 4),
                "recency_score": round(recency, 4),
                "bm25_score": round(bm25, 4),
                "importance_score": None,
                "entity_score": round(entity_score, 4),
                "final_score": round(final, 4),
                "weights": {
                    "semantic": ctx.config.weight_semantic,
                    "recency": ctx.config.weight_recency,
                    "bm25": ctx.config.weight_bm25,
                    "entity": ctx.config.entity_boost_weight,
                },
            }
        else:
            importance = max(0.0, min(1.0, memory.importance))
            final = (
                semantic_norm * ctx.config.weight_semantic_no_embed
                + recency * ctx.config.weight_recency_no_embed
                + importance * ctx.config.weight_importance
                + entity_score * ctx.config.entity_boost_weight
            )
            explanation = {
                "semantic_score": round(semantic_norm, 4),
                "recency_score": round(recency, 4),
                "bm25_score": None,
                "importance_score": round(importance, 4),
                "entity_score": round(entity_score, 4),
                "final_score": round(final, 4),
                "weights": {
                    "semantic": ctx.config.weight_semantic_no_embed,
                    "recency": ctx.config.weight_recency_no_embed,
                    "importance": ctx.config.weight_importance,
                    "entity": ctx.config.entity_boost_weight,
                },
            }

        memory.score = final
        explained.append({"memory": memory, "explanation": explanation})

    explained.sort(key=lambda x: x["explanation"]["final_score"], reverse=True)
    return explained[:top_k]


def recall_by_tag(
    ctx: MemoryIORuntime,
    user_id: str,
    tag: str,
    lifecycle_filter: list[LifecycleState] | None = None,
) -> list[MemoryObject]:
    """Recall memories by tag (direct store call)."""
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if not tag or not tag.strip():
        raise ValidationError("tag cannot be empty")

    return ctx.store.get_by_tag(user_id, tag, lifecycle_filter)


async def arecall_by_tag(
    ctx: MemoryIORuntime,
    user_id: str,
    tag: str,
    lifecycle_filter: list[LifecycleState] | None = None,
) -> list[MemoryObject]:
    """Async version of :func:`recall_by_tag`."""
    return await asyncio.to_thread(recall_by_tag, ctx, user_id, tag, lifecycle_filter)


def recall_since(
    ctx: MemoryIORuntime,
    user_id: str,
    query: str,
    hours: float = 24.0,
    top_k: int = 5,
    max_tokens: int | None = None,
    lifecycle_filter: list[LifecycleState] | None = None,
) -> list[MemoryObject]:
    """Recall memories created in the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    all_results = ctx.recall_fn(
        user_id=user_id,
        query=query,
        top_k=top_k * 3,
        max_tokens=max_tokens,
        lifecycle_filter=lifecycle_filter,
    )

    filtered = [m for m in all_results if m.created_at and m.created_at >= cutoff]
    return filtered[:top_k]


async def arecall_since(
    ctx: MemoryIORuntime,
    user_id: str,
    query: str,
    hours: float = 24.0,
    top_k: int = 5,
    max_tokens: int | None = None,
    lifecycle_filter: list[LifecycleState] | None = None,
) -> list[MemoryObject]:
    """Async version of :func:`recall_since`."""
    return await asyncio.to_thread(
        recall_since, ctx, user_id, query, hours, top_k, max_tokens, lifecycle_filter
    )


def migrate(
    ctx: MemoryIORuntime,
    user_id: str,
    new_embed_fn: EmbeddingAdapter,
    batch_size: int = 100,
) -> int:
    """Re-embed all ACTIVE/DECAYING memories for a user with a new embedder."""
    if not user_id or not user_id.strip():
        raise ValidationError("user_id cannot be empty")
    if batch_size < 1:
        raise ValidationError(f"batch_size must be at least 1, got {batch_size}")

    with ctx.latency_tracker("migrate"):
        memories = ctx.store.get_all_by_user(
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
                ctx.store.update(mem)
                count += 1

        logger.info(f"Migrated {count} memories for user {user_id}")
        ctx.track_operation("migrate", user_id, {"count": count})
        return count


def _memory_to_export_dict(mem: MemoryObject) -> dict[str, Any]:
    """Serialise a MemoryObject to the full export JSON shape.

    Includes the embedding vector itself (unlike the webhook payload helper),
    so the file can be re-imported losslessly via :func:`import_from`.
    """
    return {
        "memory_id": mem.memory_id,
        "user_id": mem.user_id,
        "content": mem.content,
        "embedding": mem.embedding,
        "score": mem.score,
        "created_at": mem.created_at.isoformat() if mem.created_at else None,
        "last_accessed_at": mem.last_accessed_at.isoformat() if mem.last_accessed_at else None,
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


def _memory_from_dict(mem_data: dict[str, Any]) -> MemoryObject:
    """Reverse of :func:`_memory_to_export_dict` — rebuild a MemoryObject.

    All fields are optional with sensible defaults to support partial exports
    and forward compatibility with newer fields.
    """
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

    return MemoryObject(
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


def export(ctx: MemoryIORuntime, file_path: str) -> int:
    """Export all memories to a JSON file. Returns the count written."""
    all_memories = ctx.store.get_all()
    memories_data = [_memory_to_export_dict(m) for m in all_memories]

    with open(file_path, "w") as f:
        json.dump(memories_data, f, indent=2)

    logger.info(f"Exported {len(memories_data)} memories to {file_path}")
    return len(memories_data)


def import_from(ctx: MemoryIORuntime, file_path: str) -> int:
    """Import memories from a JSON file, skipping any that already exist.

    Validates the file schema before persisting anything so that a
    malformed entry does not leave a half-imported dataset.
    """
    with open(file_path) as f:
        memories_data = json.load(f)

    if not isinstance(memories_data, list):
        raise ValidationError("Import file must contain a JSON array of memory objects")

    required_fields = ("memory_id", "user_id", "content")
    for i, mem_data in enumerate(memories_data):
        if not isinstance(mem_data, dict):
            raise ValidationError(f"Import item at index {i} is not a dict")
        for field in required_fields:
            if field not in mem_data:
                raise ValidationError(
                    f"Import item at index {i} missing required field '{field}'"
                )

    imported_count = 0
    for mem_data in memories_data:
        existing = ctx.store.get(mem_data["memory_id"])
        if existing is not None:
            continue

        memory = _memory_from_dict(mem_data)
        ctx.store.store(memory)
        imported_count += 1

    logger.info(f"Imported {imported_count} memories from {file_path}")
    return imported_count


async def aexport(ctx: MemoryIORuntime, file_path: str) -> int:
    """Async version of :func:`export`."""
    return await asyncio.to_thread(export, ctx, file_path)


async def aimport_from(ctx: MemoryIORuntime, file_path: str) -> int:
    """Async version of :func:`import_from`."""
    return await asyncio.to_thread(import_from, ctx, file_path)


def get_history(
    ctx: MemoryIORuntime,
    memory_id: str,
    limit: int = 100,
) -> list[VersionSnapshot]:
    """Return version history for a memory, newest first.

    Returns an empty list when versioning is not configured. Raises the
    underlying exception when the version store is misconfigured or
    unreachable so callers can distinguish "no versions" from "store
    broken".
    """
    vs = ctx.get_version_store()
    if vs is None:
        return []
    snapshots = vs.list_versions(memory_id)
    return snapshots[:limit]


def diff_versions(
    ctx: MemoryIORuntime,
    memory_id: str,
    from_version: int,
    to_version: int,
) -> DiffResult | None:
    """Show field-level differences between two versions of a memory.

    Returns ``None`` when versioning is not configured. Raises the
    underlying exception when the version store is unreachable.
    """
    vs = ctx.get_version_store()
    if vs is None:
        return None
    return vs.diff(memory_id, from_version, to_version)


def rollback_memory(
    ctx: MemoryIORuntime,
    memory_id: str,
    target_version: int,
) -> RollbackResult | None:
    """Roll a memory back to a previous version (writes the rolled-back state to ctx.store).

    Returns ``None`` when versioning is not configured. Raises the
    underlying exception when the version store is unreachable or the
    target version does not exist.
    """
    vs = ctx.get_version_store()
    if vs is None:
        return None
    return vs.rollback(
        memory_id=memory_id,
        target_version=target_version,
        store=ctx.store,
    )


__all__ = [
    "MemoryIORuntime",
    "update",
    "update_many",
    "aupdate",
    "aupdate_many",
    "forget",
    "forget_many",
    "aforget",
    "aforget_many",
    "context_block",
    "acontext_block",
    "stats",
    "astats",
    "list_users",
    "alist_users",
    "prune",
    "prune_expired",
    "consolidate",
    "cluster_topics",
    "extract_entities",
    "get_memory_graph",
    "feedback",
    "backfill_entities",
    "abackfill_entities",
    "run_maintenance",
    "recall_between",
    "recall_user_profile",
    "recall_session_context",
    "recall_agent_knowledge",
    "recall_explain",
    "recall_by_tag",
    "arecall_by_tag",
    "recall_since",
    "arecall_since",
    "migrate",
    "export",
    "import_from",
    "aexport",
    "aimport_from",
    "get_history",
    "diff_versions",
    "rollback_memory",
]
