"""Ingestion pipeline: take a fully-built ``MemoryObject`` and store it.

The pipeline is the "ingest" half of the remember flow. It is purely
about "what to do with a candidate memory": dedup, conflict detection,
entity extraction, storage, webhook dispatch, and audit tracking.

Composition (validation, sanitization, embedding, ``MemoryObject``
construction, pre/post hooks) is the caller's responsibility — see
:class:`kemi.memory.facade.Memory.remember`.

The pipeline does not depend on the ``Memory`` class. It takes an
:class:`IngestionContext` with the storage adapter, configuration,
and the side-effect callables that the orchestrator wires up. This
keeps the pipeline independently testable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kemi.infra.webhooks import WebhookEventType
from kemi.memory import dedup
from kemi.memory.model import LifecycleState, MemoryConfig, MemoryObject
from kemi.pipeline import _steps

if TYPE_CHECKING:
    from kemi.adapters.base import StorageAdapter
    from kemi.memory.entities import EntityLinker
    from kemi.memory.versions import MemoryVersionStore

logger = logging.getLogger(__name__)


def _memory_to_dict(memory: MemoryObject) -> dict[str, Any]:
    """Convert a MemoryObject to a JSON-serialisable dict for webhook payloads.

    Delegates to :func:`kemi.memory.model.memory_to_dict` (canonical impl).
    """
    from kemi.memory.model import memory_to_dict as _to_dict

    return _to_dict(memory)


@dataclass
class IngestionContext:
    """Dependencies required to ingest a single ``MemoryObject``.

    The pipeline is "what to do with a candidate memory" — it needs
    the storage adapter, configuration, and the side-effect callables
    that the orchestrator wires up to ``kemi.operations._ops_*``.
    No global state, no hidden coupling to ``Memory``.
    """

    store: StorageAdapter
    config: MemoryConfig
    entity_linker: EntityLinker
    metrics: Any | None

    # Side-effect callbacks. The ``Memory`` orchestrator wires these
    # to the implementations in ``kemi.operations._ops_*``.
    record_store_error: Callable[[], None] = lambda: None
    dispatch_webhook: Callable[..., None] = lambda *args, **kwargs: None
    track_operation: Callable[..., None] = lambda *args, **kwargs: None
    get_version_store: Callable[[], MemoryVersionStore] = lambda: None  # type: ignore[return-value]
    auto_prune_versions: Callable[[str], None] = lambda memory_id: None


class IngestionPipeline:
    """Ingest a fully-built ``MemoryObject`` and return the stored result.

    The pipeline mutates ``memory.metadata`` to attach
    ``conflict_flagged`` and ``extracted_entities`` annotations, then
    stores it. In the dedup case the input is merged into the canonical
    existing memory and that canonical is returned instead.

    The pipeline does NOT fire user hooks. The caller
    (:meth:`kemi.memory.facade.Memory.remember`) fires the pre/post
    hooks around the pipeline call to preserve the historical
    contract: ``_remember_with_embedding`` (used by ``remember_many``)
    fires no hooks, while the public ``remember`` fires both.
    """

    def __init__(self, ctx: IngestionContext) -> None:
        self._ctx = ctx

    def ingest(
        self,
        memory: MemoryObject,
        *,
        audit_batch: list[dict[str, Any]] | None = None,
    ) -> MemoryObject:
        """Ingest ``memory`` and return the stored ``MemoryObject``."""
        existing = self._ctx.store.get_all_by_user(
            memory.user_id,
            lifecycle_filter=[
                LifecycleState.ACTIVE,
                LifecycleState.DECAYING,
                LifecycleState.ARCHIVED,
            ],
            namespace=memory.namespace,
        )

        duplicates = dedup.find_duplicates(
            memory, existing, self._ctx.config.dedup_threshold
        )
        if duplicates:
            return _steps.handle_duplicate_resolution(
                memory=memory,
                duplicates=duplicates,
                store=self._ctx.store,
                get_version_store=self._ctx.get_version_store,
                auto_prune_versions=self._ctx.auto_prune_versions,
                dispatch_webhook=self._ctx.dispatch_webhook,
                track_operation=self._ctx.track_operation,
                metrics=self._ctx.metrics,
                audit_batch=audit_batch,
            )

        conflict_detected, conflicts = _steps.annotate_memory_for_ingestion(
            memory=memory,
            existing_memories=existing,
            conflict_threshold=self._ctx.config.conflict_threshold,
            dedup_threshold=self._ctx.config.dedup_threshold,
            enable_entity_boost=self._ctx.config.enable_entity_boost,
            entity_linker=self._ctx.entity_linker,
            metrics=self._ctx.metrics,
        )

        try:
            self._ctx.store.store(memory)
        except Exception:
            # Broad catch intentional: storage adapters can raise from
            # many layers (SQLite, JSON, Postgres, encryption). Record
            # the error in metrics and re-raise the original.
            self._ctx.record_store_error()
            raise

        if self._ctx.metrics is not None:
            self._ctx.metrics.embed_total.inc(1)
            self._ctx.metrics.embed_bytes_total.inc(len(memory.content))
            self._ctx.metrics.total_memories.set(
                self._ctx.store.count(memory.user_id)
            )

        snapshot = _memory_to_dict(memory)
        self._ctx.dispatch_webhook(
            WebhookEventType.REMEMBERED,
            memory_id=memory.memory_id,
            user_id=memory.user_id,
            snapshot=snapshot,
        )
        if conflict_detected:
            self._ctx.dispatch_webhook(
                WebhookEventType.CONFLICT,
                memory_id=memory.memory_id,
                user_id=memory.user_id,
                snapshot=snapshot,
                conflict_with=conflicts[0].memory_id,
            )

        details: dict[str, Any] = {
            "memory_id": memory.memory_id,
            "content_length": len(memory.content),
        }
        if conflict_detected:
            details["conflict"] = True
            details["conflict_with"] = conflicts[0].memory_id
        self._ctx.track_operation(
            "remember",
            memory.user_id,
            details,
            memory.memory_id,
            memory.namespace,
            audit_batch=audit_batch,
        )
        return memory




__all__ = ["IngestionContext", "IngestionPipeline"]
