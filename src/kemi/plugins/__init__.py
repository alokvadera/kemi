"""Plugin system for ``kemi``.

The plugin system lets users extend or replace the built-in subsystems
that the ``MemoryService`` core calls into:

* **Webhooks** (:class:`WebhookSink`) — fan-out delivery of lifecycle events.
* **Audit** (:class:`AuditSink`) — structured operation logging.
* **Query cache** (:class:`QueryCacheProvider`) — caching layer for ``recall()``.
* **Hooks** (:class:`HookSink`) — pre/post operation callbacks.

Each subsystem has a :mod:`~kemi.plugins.protocols` Protocol and one or
more :mod:`~kemi.plugins.builtin` adapters. The active plugins are
stored on a :class:`~kemi.plugins.registry.PluginRegistry` attached to
each :class:`~kemi.memory.service.MemoryService` instance.

Example: a Slack webhook sink + a stdout audit sink alongside the built-ins.

.. code-block:: python

    from kemi import Memory
    from kemi.plugins import WebhookSink, AuditSink

    class SlackSink:
        def __init__(self, webhook_url: str) -> None:
            self._url = webhook_url
        def send(self, event, payload) -> None:
            import httpx
            httpx.post(self._url, json=payload, timeout=2.0)

    class StdoutAuditSink:
        def log(self, user_id, operation, **kw) -> None:
            print(f"[audit] {operation} user={user_id}")
        def log_batch(self, entries) -> None:
            for e in entries: self.log(**e)

    mem = Memory()
    mem.configure_webhooks(db_path="webhooks.db")   # built-in SQLite dispatcher
    mem.add_webhook_sink(SlackSink("https://hooks.slack.com/..."))
    mem.add_audit_sink(StdoutAuditSink())
"""

from kemi.plugins._version import KEMI_PROTOCOL_VERSION, parse_version
from kemi.plugins.builtin import (
    AuditTrailSink,
    CallbackHookSink,
    LruQueryCache,
    WebhookDispatcherSink,
)
from kemi.plugins.protocols import (
    AuditSink,
    HookSink,
    QueryCacheProvider,
    WebhookSink,
)
from kemi.plugins.registry import PluginRegistry

__all__ = [
    # Protocols
    "WebhookSink",
    "AuditSink",
    "QueryCacheProvider",
    "HookSink",
    # Registry
    "PluginRegistry",
    # Built-in adapters
    "WebhookDispatcherSink",
    "AuditTrailSink",
    "LruQueryCache",
    "CallbackHookSink",
    # Protocol version
    "KEMI_PROTOCOL_VERSION",
    "parse_version",
]
