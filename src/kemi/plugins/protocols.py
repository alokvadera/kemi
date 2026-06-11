"""Plugin Protocols — the contracts every plugin implementation must satisfy.

Each protocol is the minimal surface that the ``MemoryService`` core calls into.
Built-in implementations live in :mod:`kemi.plugins.builtin`; user code may
substitute any object whose methods match the protocol (structural typing
via :mod:`typing.Protocol`).

The four plugin slots are:

* :class:`WebhookSink` — receives a memory lifecycle event with a JSON payload.
* :class:`AuditSink` — receives a structured operation record for compliance logging.
* :class:`QueryCacheProvider` — caches ``recall()`` results keyed by a string.
* :class:`HookSink` — runs user-supplied callbacks before/after operations.

Plugins MUST NOT raise from any of these methods — exceptions break the calling
operation. Implementations should swallow internal errors and log them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kemi.infra.webhooks import WebhookEventType
    from kemi.memory.model import MemoryObject


@runtime_checkable
class WebhookSink(Protocol):
    """Receives memory lifecycle webhook events.

    Implementations should be fire-and-forget: never raise, never block the
    caller for more than a few milliseconds (or offload to a background
    thread/queue internally).
    """

    def send(
        self,
        event: WebhookEventType,
        payload: dict[str, Any],
    ) -> None:
        """Deliver *payload* to the underlying transport for *event*.

        Args:
            event: The lifecycle event type that fired.
            payload: JSON-serialisable dict (already built via
                :func:`kemi.infra.webhooks.build_payload`).
        """
        ...


@runtime_checkable
class AuditSink(Protocol):
    """Receives structured operation records for compliance / observability.

    Implementations are called on every operation that the core tracks
    (``remember``, ``recall``, ``forget``, ``update``, etc.). They MUST be
    fast (sub-millisecond) and MUST NOT raise.
    """

    def log(
        self,
        user_id: str,
        operation: str,
        details: dict[str, Any] | None = None,
        memory_id: str | None = None,
        namespace: str = "default",
        status: str = "success",
    ) -> None:
        """Record a single operation.

        Args:
            user_id: Owner of the affected memory.
            operation: Operation name (e.g. ``"remember"``, ``"recall"``).
            details: Free-form operation details.
            memory_id: Affected memory ID (if any).
            namespace: Memory namespace.
            status: ``"success"`` / ``"error"`` / ``"denied"``.
        """
        ...

    def log_batch(self, entries: list[dict[str, Any]]) -> None:
        """Record a batch of operations atomically (default: loop and call :meth:`log`)."""
        ...


@runtime_checkable
class QueryCacheProvider(Protocol):
    """Caches ``recall()`` results keyed by a stable string.

    Implementations should be safe to call from multiple threads. The core
    stores *shallow copies* on :meth:`put` and returns copies on :meth:`get`
    to prevent caller mutations from corrupting the cache.
    """

    def get(self, key: str) -> list[MemoryObject] | None:
        """Return the cached list for *key* (or ``None`` on miss)."""
        ...

    def put(self, key: str, value: list[MemoryObject]) -> None:
        """Store *value* under *key*, evicting if over capacity."""
        ...


@runtime_checkable
class HookSink(Protocol):
    """Runs user callbacks before/after memory operations.

    The core calls :meth:`run` with a *phase* (``"pre"`` or ``"post"``) and
    an *operation* name (e.g. ``"remember"``). Implementations should swallow
    callback exceptions unless ``raise_on_error`` is set.
    """

    def add(self, phase: str, callback: Callable[..., Any]) -> None:
        """Register *callback* to fire on *phase* (``"pre"`` or ``"post"``)."""
        ...

    def remove(self, phase: str, callback: Callable[..., Any]) -> bool:
        """Unregister *callback*; return True if it was registered."""
        ...

    def run(
        self,
        phase: str,
        operation: str,
        *,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> None:
        """Invoke every callback registered for (*phase*, *operation*).

        Args:
            phase: ``"pre"`` or ``"post"``.
            operation: Operation name.
            raise_on_error: If True, re-raise the first callback exception.
            **kwargs: Forwarded to each callback as keyword arguments.
        """
        ...
