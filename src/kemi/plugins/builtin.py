"""Built-in plugin implementations.

Each class here is a thin adapter that wraps an existing module
(``kemi.webhooks``, ``kemi.audit``, ``kemi.operations._query_cache``) and
exposes only the surface required by the protocols in
:mod:`kemi.plugins.protocols`. User code can construct these directly and
pass them to ``MemoryService.add_*_sink()`` / ``set_query_cache()``, or it
can implement the protocols on its own classes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kemi.exceptions import ValidationError

if TYPE_CHECKING:
    from kemi.infra.audit import AuditTrail
    from kemi.infra.webhooks import WebhookDispatcher, WebhookEventType
    from kemi.memory.model import MemoryObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebhookSink
# ---------------------------------------------------------------------------


class WebhookDispatcherSink:
    """Adapter that exposes
    :class:`kemi.infra.webhooks.WebhookDispatcher` as a :class:`WebhookSink`.

    The dispatcher manages its own list of endpoints, so this adapter is
    typically a single-element sink in the registry. Use
    :func:`kemi.infra.webhooks.WebhookStore` directly to register endpoints.
    """

    def __init__(self, dispatcher: WebhookDispatcher) -> None:
        self._dispatcher = dispatcher

    def send(
        self,
        event: WebhookEventType,
        payload: dict[str, Any],
    ) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._dispatcher.dispatch_sync(payload, event)
            except Exception:
                logger.warning(
                    "Sync webhook dispatch failed for %s", event.value, exc_info=True
                )
            return

        try:
            asyncio.ensure_future(self._dispatcher.dispatch_async(payload, event))
        except Exception:
            logger.warning(
                "Async webhook dispatch failed for %s", event.value, exc_info=True
            )


__all__ = [
    "WebhookDispatcherSink",
    "AuditTrailSink",
    "LruQueryCache",
    "CallbackHookSink",
]


# ---------------------------------------------------------------------------
# AuditSink
# ---------------------------------------------------------------------------


class AuditTrailSink:
    """Adapter that exposes :class:`kemi.infra.audit.AuditTrail` as an :class:`AuditSink`."""

    def __init__(self, audit: AuditTrail) -> None:
        self._audit = audit

    def log(
        self,
        user_id: str,
        operation: str,
        details: dict[str, Any] | None = None,
        memory_id: str | None = None,
        namespace: str = "default",
        status: str = "success",
    ) -> None:
        try:
            self._audit.log_operation(
                user_id=user_id,
                operation=operation,
                details=details or {},
                memory_id=memory_id,
                namespace=namespace,
                status=status,
            )
        except Exception:
            logger.warning(
                f"Audit log failed for {operation}", exc_info=True
            )

    def log_batch(self, entries: list[dict[str, Any]]) -> None:
        try:
            self._audit.log_operation_batch(entries)
        except Exception:
            logger.warning("Audit batch log failed", exc_info=True)


# ---------------------------------------------------------------------------
# QueryCacheProvider
# ---------------------------------------------------------------------------


class LruQueryCache:
    """LRU cache for ``recall()`` results, satisfying :class:`QueryCacheProvider`.

    Stores *shallow copies* on :meth:`put` and returns copies on :meth:`get`
    so caller mutations cannot corrupt the cache. Replaces the older
    :class:`kemi.operations._query_cache._QueryCache` as the canonical
    built-in; that class is kept as a re-export shim.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, list[MemoryObject]] = OrderedDict()

    def _make_key(
        self,
        user_id: str,
        query: str,
        top_k: int,
        max_tokens: int | None,
        lifecycle_filter: list[Any] | None,
        hybrid_search: bool | None,
        namespace: str,
        session_id: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> str:
        """Build a stable string key from query parameters.

        Kept as a method on the cache so the retrieval pipeline can build
        a key without depending on the cache's internal storage.
        """
        lf = tuple(sorted(s.value for s in lifecycle_filter)) if lifecycle_filter else ()
        mf = tuple(sorted((k, v) for k, v in (metadata_filter or {}).items()))
        return "|".join(
            [
                user_id,
                query,
                str(top_k),
                str(max_tokens),
                str(lf),
                str(hybrid_search),
                namespace,
                str(session_id),
                str(mf),
            ]
        )

    def get(self, key: str) -> list[MemoryObject] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._copy_memories(self._cache[key])
        return None

    def put(self, key: str, value: list[MemoryObject]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = self._copy_memories(value)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _copy_memories(memories: list[MemoryObject]) -> list[MemoryObject]:
        from kemi.memory.model import MemoryObject

        return [
            MemoryObject(
                memory_id=m.memory_id,
                user_id=m.user_id,
                content=m.content,
                embedding=m.embedding,
                score=m.score,
                created_at=m.created_at,
                last_accessed_at=m.last_accessed_at,
                source=m.source,
                importance=m.importance,
                lifecycle_state=m.lifecycle_state,
                metadata=m.metadata.copy(),
                embedding_dim=m.embedding_dim,
                tags=list(m.tags),
                confidence=m.confidence,
                memory_type=m.memory_type,
                session_id=m.session_id,
                namespace=m.namespace,
                version=m.version,
                agent_id=m.agent_id,
                run_id=m.run_id,
                app_id=m.app_id,
            )
            for m in memories
        ]


# ---------------------------------------------------------------------------
# HookSink
# ---------------------------------------------------------------------------


class CallbackHookSink:
    """In-memory registry of pre/post hooks, satisfying :class:`HookSink`.

    Replaces the implicit ``_event_hooks`` dict previously stored on
    ``Memory`` instances. Backed by a ``dict[phase, list[callback]]`` keyed
    on ``"pre"`` / ``"post"``.
    """

    _VALID_PHASES = ("pre", "post")

    def __init__(self, hooks: dict[str, list[Any]] | None = None) -> None:
        self._hooks: dict[str, list[Any]] = (
            hooks if hooks is not None else {"pre": [], "post": []}
        )

    def add(self, phase: str, callback: Callable[..., Any]) -> None:
        if phase not in self._VALID_PHASES:
            raise ValidationError(f"phase must be one of {self._VALID_PHASES!r}, got {phase!r}")
        self._hooks[phase].append(callback)

    def remove(self, phase: str, callback: Callable[..., Any]) -> bool:
        bucket = self._hooks.get(phase)
        if not bucket or callback not in bucket:
            return False
        bucket.remove(callback)
        return True

    def run(
        self,
        phase: str,
        operation: str,
        *,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> None:
        for hook in list(self._hooks.get(phase, [])):
            try:
                hook(operation, **kwargs)
            except Exception:
                if raise_on_error:
                    raise
                logger.warning(
                    f"Event hook failed for {phase}:{operation}", exc_info=True
                )

    def clear(self) -> None:
        """Remove all registered hooks (used by ``Memory.clear_event_hooks``)."""
        for phase in self._VALID_PHASES:
            self._hooks[phase].clear()

    def count(self, phase: str | None = None) -> int:
        """Return the number of registered hooks (all phases if *phase* is ``None``)."""
        if phase is None:
            return sum(len(v) for v in self._hooks.values())
        return len(self._hooks.get(phase, []))
