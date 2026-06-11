"""Pure-function pipeline steps extracted from
:mod:`kemi.pipeline.retrieval` and :mod:`kemi.pipeline.ingestion`.

These functions are intentionally free-standing — they take data in,
return data (or perform the documented side effect with explicit
dependencies). The :class:`RetrievalPipeline` and
:class:`IngestionPipeline` wrap them with their contexts, which lets:

- The steps be unit-tested in isolation (no ``RetrievalContext`` /
  ``IngestionContext`` fixture required).
- The pipelines shrink — methods that just unpacked context fields
  are now single-line delegations to these functions.
- Future work to inline the pipelines (e.g. into async generators) be
  a textual change rather than an architectural one.

Extraction criteria
-------------------
A method is extracted to ``_steps.py`` when it **adds meaningful pipeline
logic** that is worth testing in isolation (e.g. scoring heuristics,
metadata filtering, lifecycle transitions, conflict detection, or
duplicate resolution).  One-liner adapter delegations — such as
``_embed_query``, ``_check_cache``, or ``_cache_results`` — stay
inline in the pipeline class because the pipeline adds no logic; moving
them to ``_steps.py`` would only create pass-through functions with no
incremental testability.

Side-effect-bearing functions take their dependencies as parameters
rather than reading from a context object.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from kemi.exceptions import ValidationError
from kemi.infra.webhooks import WebhookEventType
from kemi.memory import dedup, lifecycle
from kemi.memory.model import LifecycleState, MemoryObject

logger = logging.getLogger(__name__)


def validate_embedding_dimension(
    search_results: list[MemoryObject],
    current_dim: int,
) -> None:
    """Raise ``ValueError`` if any stored memory has a mismatched embedding dimension.

    Args:
        search_results: Candidates returned from the storage adapter.
        current_dim: The dimensionality produced by the active embedder.

    Raises:
        ValueError: When ``search_results[0].embedding_dim`` differs from
            ``current_dim``. The message tells the user to run
            ``memory.migrate`` to re-embed their memories.
    """
    if not search_results:
        return
    stored_dim = search_results[0].embedding_dim
    if stored_dim is not None and stored_dim != current_dim:
        raise ValidationError(
            "Embedding dimension mismatch: stored memories use "
            f"{stored_dim} dimensions but current adapter produces "
            f"{current_dim} dimensions. Run memory.migrate(user_id, "
            "new_adapter) to re-embed your memories."
        )


def build_entity_boost_maps(
    query: str,
    search_results: list[MemoryObject],
    enable_entity_boost: bool,
    entity_linker: Any,
) -> tuple[set[str] | None, dict[str, set[str]] | None]:
    """Extract query + per-memory entity sets for hybrid scoring.

    Returns ``(None, None)`` when entity boost is disabled so callers
    can short-circuit on the result without an extra check.

    Args:
        query: The recall query string.
        search_results: Candidates returned from the storage adapter.
        enable_entity_boost: Feature flag from :class:`MemoryConfig`.
        entity_linker: An :class:`EntityLinker` (or ``NoopEntityLinker``)
            that implements ``extract(text) -> set[str]``.

    Returns:
        A 2-tuple ``(query_entities, memory_entities_map)``. Either may
        be ``None`` when entity boost is off.
    """
    if not enable_entity_boost:
        return None, None
    query_entities = entity_linker.extract(query)
    memory_entities_map: dict[str, set[str]] = {}

    # Separate memories with cached entities from those that need extraction.
    need_extraction: list[MemoryObject] = []
    for m in search_results:
        cached = m.metadata.get("extracted_entities")
        if cached is not None:
            memory_entities_map[m.memory_id] = set(cached)
        else:
            need_extraction.append(m)

    if need_extraction:
        contents = [m.content for m in need_extraction]
        batch_entities = entity_linker.extract_batch(contents)
        for m, entities in zip(need_extraction, batch_entities, strict=True):
            memory_entities_map[m.memory_id] = entities

    return query_entities, memory_entities_map


def update_lifecycle_inplace(
    results: list[MemoryObject],
    decay_threshold_hours: float,
    store: Any,
    metrics: Any | None,
) -> None:
    """Bump ``last_accessed_at`` and apply any lifecycle transitions.

    Mutates ``results`` in place. Batches ``store.update_many`` for all
    memories whose lifecycle state changed in a single round-trip.
    Increments the ``lifecycle_transitions`` counter on *metrics* if
    available.

    Args:
        results: The memories to update.
        decay_threshold_hours: Hours of inactivity before decay.
        store: A :class:`StorageAdapter` (for the ``update_many`` call).
        metrics: A metrics collector or ``None``.
    """
    now = datetime.now(timezone.utc)
    changed: list[MemoryObject] = []
    for mem in results:
        mem.last_accessed_at = now
        new_state = lifecycle.evaluate_lifecycle(mem, decay_threshold_hours)
        if new_state != mem.lifecycle_state:
            updated = lifecycle.transition(mem, new_state)
            changed.append(updated)
    if changed:
        store.update_many(changed)
        if metrics is not None:
            metrics.lifecycle_transitions.inc(len(changed))


def adaptive_feedback(
    adaptive_retriever: Any | None,
    user_id: str,
    query: str,
) -> None:
    """Best-effort adaptive retriever feedback (never raises).

    Adaptive retrieval is opt-in. Failures must not break the recall
    response that was already returned to the caller, but they are
    logged at ERROR so operators can diagnose misconfiguration without
    enabling DEBUG.
    """
    if adaptive_retriever is None:
        return
    try:
        profile = adaptive_retriever.analyze_query(query)
        adaptive_retriever.record_feedback(user_id, query, profile)
    except (ValueError, TypeError, AttributeError, RuntimeError):
        logger.error(
            "Adaptive retrieval feedback failed for user %s", user_id, exc_info=True
        )


def search_and_filter_storage(
    store: Any,
    user_id: str,
    query_embedding: list[float],
    top_k: int,
    lifecycle_filter: list[LifecycleState],
    namespace: str,
    session_id: str | None,
    fetch_multiplier: int,
    metadata_filter: dict[str, Any] | None,
) -> list[MemoryObject]:
    """Run storage search and apply metadata filter post-hoc.

    Args:
        store: A :class:`StorageAdapter`.
        user_id: The user to search within.
        query_embedding: The embedded query vector.
        top_k: Desired result count (multiplied by *fetch_multiplier* before
            searching so filtering has enough candidates).
        lifecycle_filter: Which lifecycle states to include.
        namespace: Namespace filter.
        session_id: Optional session filter.
        fetch_multiplier: Multiplier for ``top_k`` to fetch extra candidates.
        metadata_filter: Optional dict of ``{key: value}`` pairs to match
            against ``memory.metadata``.

    Returns:
        Filtered list of candidate memories.
    """
    results = store.search(
        user_id=user_id,
        query_embedding=query_embedding,
        top_k=top_k * fetch_multiplier,
        lifecycle_filter=lifecycle_filter,
        namespace=namespace,
        session_id=session_id,
    )
    if metadata_filter is not None:
        results = [
            m
            for m in results
            if all(m.metadata.get(k) == v for k, v in metadata_filter.items())
        ]
    return results


def annotate_memory_for_ingestion(
    memory: MemoryObject,
    existing_memories: list[MemoryObject],
    conflict_threshold: float,
    dedup_threshold: float,
    enable_entity_boost: bool,
    entity_linker: Any,
    metrics: Any | None,
) -> tuple[bool, list[MemoryObject]]:
    """Detect conflicts and extract entities, mutating *memory* in place.

    Sets ``memory.metadata["conflict_flagged"]`` when a conflict is found
    and ``memory.metadata["extracted_entities"]`` when entity boost is
    enabled.

    Args:
        memory: The candidate memory to annotate.
        existing_memories: Prior memories for the same user/namespace.
        conflict_threshold: Similarity floor for conflict detection.
        dedup_threshold: Similarity ceiling for conflict detection.
        enable_entity_boost: Whether to extract and cache entities.
        entity_linker: An :class:`EntityLinker` instance.
        metrics: A metrics collector or ``None``.

    Returns:
        A 2-tuple ``(conflict_detected, conflicts)``.
    """
    conflicts = dedup.find_conflicts(
        memory, existing_memories, conflict_threshold, dedup_threshold
    )
    conflict_detected = False
    if conflicts:
        memory.metadata["conflict_flagged"] = True
        conflict_detected = True
        logger.warning(
            f"Potential conflict detected for user {memory.user_id}: "
            f"new memory '{memory.content[:50]}...' conflicts with existing memory "
            f"'{conflicts[0].content[:50]}...'"
        )
        if metrics is not None:
            metrics.conflicts_detected.inc(1)

    if enable_entity_boost:
        memory.metadata["extracted_entities"] = list(
            entity_linker.extract(memory.content)
        )

    return conflict_detected, conflicts


def handle_duplicate_resolution(
    memory: MemoryObject,
    duplicates: list[MemoryObject],
    store: Any,
    get_version_store: Callable[[], Any],
    auto_prune_versions: Callable[[str], None],
    dispatch_webhook: Callable[..., None],
    track_operation: Callable[..., None],
    metrics: Any | None,
    audit_batch: list[dict[str, Any]] | None,
) -> MemoryObject:
    """Merge *memory* into the canonical duplicate and persist the result.

    Performs version recording, stale-entity cleanup, removal of
    extra near-duplicates, webhook dispatch, metrics, and audit tracking.

    Args:
        memory: The new memory that matched as a duplicate.
        duplicates: List of existing duplicates (first element is the
            canonical one to merge into).
        store: A :class:`StorageAdapter`.
        get_version_store: Callable that returns a
            :class:`MemoryVersionStore` or ``None``.
        auto_prune_versions: Callable that prunes versions for a memory ID.
        dispatch_webhook: Callable for webhook dispatch.
        track_operation: Callable for audit/operation tracking.
        metrics: A metrics collector or ``None``.
        audit_batch: Optional audit batch list.

    Returns:
        The resolved (canonical) memory after merge.
    """
    from kemi.memory.model import memory_to_dict

    resolved = dedup.resolve_duplicate(memory, duplicates[0])
    # Content changed during merge — invalidate stale cached entities.
    resolved.metadata.pop("extracted_entities", None)
    # Record version BEFORE overwriting.
    try:
        version_store = get_version_store()
        if version_store is not None:
            version_store.record_version(duplicates[0], changed_by="merge")
            auto_prune_versions(duplicates[0].memory_id)
    except Exception:
        pass
    store.update(resolved)
    # Remove the other near-duplicates so they don't re-trigger on
    # the next insert. duplicates[0] is the canonical we merged into.
    for extra in duplicates[1:]:
        if extra.memory_id != resolved.memory_id:
            store.delete_by_id(extra.memory_id)
    snapshot = memory_to_dict(resolved)
    dispatch_webhook(
        WebhookEventType.UPDATED,
        memory_id=resolved.memory_id,
        user_id=resolved.user_id,
        snapshot=snapshot,
    )
    logger.info(
        f"Resolved duplicate for user {resolved.user_id}: {resolved.memory_id}"
    )
    if metrics is not None:
        metrics.duplicates_detected.inc(1)
    track_operation(
        "remember",
        resolved.user_id,
        {"memory_id": resolved.memory_id, "duplicate": True},
        resolved.memory_id,
        resolved.namespace,
        audit_batch=audit_batch,
    )
    return resolved


__all__ = [
    "validate_embedding_dimension",
    "build_entity_boost_maps",
    "update_lifecycle_inplace",
    "adaptive_feedback",
    "search_and_filter_storage",
    "annotate_memory_for_ingestion",
    "handle_duplicate_resolution",
]
