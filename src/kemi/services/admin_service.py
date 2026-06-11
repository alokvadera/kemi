"""Admin-side facade: configuration, maintenance, plugins, versioning.

Methods on this facade configure the service, run scheduled maintenance,
or expose plugin/version/webhook/audit machinery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kemi.exceptions import ValidationError
from kemi.memory.model import LifecycleState

if TYPE_CHECKING:
    from kemi.memory.core import _MemoryCore

logger = logging.getLogger(__name__)


class MemoryAdminService:
    """Admin-path methods: configure, maintain, version, plugins."""

    def __init__(self, core: _MemoryCore) -> None:
        self._core = core

    def configure_webhooks(self, db_path: str | None = None) -> None:
        """Enable webhook dispatch for memory lifecycle events."""
        from kemi.operations import _ops_webhooks

        _ops_webhooks.configure(self._core, db_path)

    def configure_versioning(
        self,
        db_path: str | None = None,
        max_versions_per_memory: int = 50,
        auto_prune_versions: bool = True,
    ) -> None:
        """Enable memory version history tracking."""
        from kemi.operations import _ops_versioning

        _ops_versioning.configure(
            self._core, db_path, max_versions_per_memory, auto_prune_versions
        )

    def enable_audit_trail(
        self,
        retention_days: int = 365,
        auto_purge: bool = True,
    ) -> None:
        """Enable the audit trail for compliance logging."""
        from kemi.operations import _ops_metrics

        _ops_metrics.enable_audit_trail(self._core, retention_days, auto_purge)

    def enable_query_cache(self, max_size: int = 128) -> None:
        """Enable an LRU cache for recall() results."""
        from kemi.operations import _ops_metrics

        _ops_metrics.enable_query_cache(self._core, max_size)

    def disable_query_cache(self) -> None:
        """Disable the query cache."""
        from kemi.operations import _ops_metrics

        _ops_metrics.disable_query_cache(self._core)

    def enable_adaptive_retrieval(self, enable: bool = True) -> None:
        """Enable or disable adaptive retrieval."""
        from kemi.operations import _ops_metrics

        _ops_metrics.enable_adaptive_retrieval(self._core, enable)

    def get_metrics(self) -> dict[str, Any] | None:
        """Return current metrics snapshot as a dict, or None if disabled."""
        from kemi.operations import _ops_metrics

        return _ops_metrics.get_metrics(self._core)

    def get_metrics_prometheus(self) -> str | None:
        """Return metrics in Prometheus text format, or None if disabled."""
        from kemi.operations import _ops_metrics

        return _ops_metrics.get_metrics_prometheus(self._core)

    def add_event_hook(self, phase: str, callback: Any) -> None:
        """Register an event hook callback."""
        from kemi.operations import _ops_hooks

        _ops_hooks.add(self._core, phase, callback)

    def remove_event_hook(self, phase: str, callback: Any) -> bool:
        """Remove a previously registered event hook callback."""
        from kemi.operations import _ops_hooks

        return _ops_hooks.remove(self._core, phase, callback)

    def get_history(
        self,
        memory_id: str,
        limit: int = 100,
    ) -> list[Any]:
        """Return version history for a memory, newest first."""
        from kemi.operations import _io

        return _io.get_history(self._core.build_io_runtime(), memory_id, limit)

    def diff_versions(
        self,
        memory_id: str,
        from_version: int,
        to_version: int,
    ) -> Any:
        """Show field-level differences between two versions of a memory."""
        from kemi.operations import _io

        return _io.diff_versions(
            self._core.build_io_runtime(), memory_id, from_version, to_version
        )

    def rollback_memory(
        self,
        memory_id: str,
        target_version: int,
    ) -> Any:
        """Roll a memory back to a previous version."""
        from kemi.operations import _io

        return _io.rollback_memory(
            self._core.build_io_runtime(), memory_id, target_version
        )

    def upgrade(self) -> int:
        """Migrate the storage schema to the adapter's current version."""
        target = getattr(self._core._store, "CURRENT_VERSION", 1)
        new_version = self._core._store.upgrade_schema(to_version=target)
        if new_version > 1:
            logger.info("Schema upgraded to version %d", new_version)
        return new_version

    def export(self, file_path: str) -> int:
        """Export all memories to a JSON file."""
        from kemi.operations import _io

        return _io.export(self._core.build_io_runtime(), file_path)

    def import_from(self, file_path: str) -> int:
        """Import memories from a JSON file."""
        from kemi.operations import _io

        return _io.import_from(self._core.build_io_runtime(), file_path)

    async def aexport(self, file_path: str) -> int:
        """Async version of :meth:`export`."""
        from kemi.operations import _io

        return await _io.aexport(self._core.build_io_runtime(), file_path)

    async def aimport_from(self, file_path: str) -> int:
        """Async version of :meth:`import_from`."""
        from kemi.operations import _io

        return await _io.aimport_from(self._core.build_io_runtime(), file_path)

    def prune(
        self,
        user_id: str,
        max_age_days: float | None = None,
        min_importance: float | None = None,
        lifecycle_states: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> int:
        """Auto-prune old or low-importance memories."""
        from kemi.operations import _io

        return _io.prune(
            self._core.build_io_runtime(),
            user_id,
            max_age_days,
            min_importance,
            lifecycle_states,
            namespace,
        )

    def prune_expired(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Delete memories whose ``expires_at`` has passed."""
        from kemi.operations import _io

        return _io.prune_expired(self._core.build_io_runtime(), user_id, namespace)

    def consolidate(
        self,
        user_id: str,
        namespace: str = "default",
        min_memories: int = 5,
        max_age_days: float = 30.0,
        with_llm_summary: bool = False,
    ) -> str | None:
        """Consolidate old episodic memories into a semantic summary."""
        from kemi.operations import _io

        return _io.consolidate(
            self._core.build_io_runtime(),
            user_id,
            namespace,
            min_memories,
            max_age_days,
            with_llm_summary,
        )

    def cluster_topics(
        self,
        user_id: str,
        n_clusters: int = 3,
        namespace: str = "default",
    ) -> dict[str, list[Any]]:
        """Cluster memories into topic groups using embeddings."""
        from kemi.operations import _io

        return _io.cluster_topics(
            self._core.build_io_runtime(), user_id, n_clusters, namespace
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
        """Run automatic maintenance tasks for a user's memories."""
        from kemi.operations import _io

        return _io.run_maintenance(
            self._core.build_io_runtime(),
            user_id,
            namespace,
            auto_prune,
            auto_consolidate,
            auto_backfill_entities,
            prune_max_age_days,
            prune_min_importance,
            consolidate_min_memories,
            consolidate_max_age_days,
            auto_prune_expired,
            consolidate_with_llm_summary,
        )

    def get_plugins(self) -> Any:
        """Return the :class:`PluginRegistry` holding this instance's plugins."""
        return self._core._plugins

    def add_webhook_sink(self, sink: Any) -> None:
        """Register an additional :class:`~kemi.plugins.WebhookSink`."""
        from kemi.plugins import WebhookSink as _WS

        if not isinstance(sink, _WS):
            raise ValidationError(
                f"sink must satisfy the WebhookSink protocol, got {type(sink).__name__}"
            )
        self._core._plugins.webhook_sinks.append(sink)

    def add_audit_sink(self, sink: Any) -> None:
        """Register an additional :class:`~kemi.plugins.AuditSink`."""
        from kemi.plugins import AuditSink as _AS

        if not isinstance(sink, _AS):
            raise ValidationError(
                f"sink must satisfy the AuditSink protocol, got {type(sink).__name__}"
            )
        self._core._plugins.audit_sinks.append(sink)

    def add_hook_sink(self, sink: Any) -> None:
        """Register an additional :class:`~kemi.plugins.HookSink`."""
        from kemi.plugins import HookSink as _HS

        if not isinstance(sink, _HS):
            raise ValidationError(
                f"sink must satisfy the HookSink protocol, got {type(sink).__name__}"
            )
        self._core._plugins.hook_sinks.append(sink)

    def set_query_cache(self, cache: Any) -> None:
        """Set (or clear) the active query cache."""
        from kemi.plugins import QueryCacheProvider as _QC

        if cache is not None and not isinstance(cache, _QC):
            raise ValidationError(
                f"cache must satisfy the QueryCacheProvider protocol, got {type(cache).__name__}"
            )
        self._core._query_cache = cache
        self._core._plugins.query_cache = cache

    def clear_webhook_sinks(self) -> None:
        """Remove all webhook sinks (including the built-in dispatcher)."""
        self._core._plugins.clear_webhook_sinks()
        self._core._webhook_dispatcher = None

    def clear_audit_sinks(self) -> None:
        """Remove all audit sinks (including the built-in AuditTrail)."""
        self._core._plugins.clear_audit_sinks()
        self._core._audit_trail = None

    def clear_hook_sinks(self) -> None:
        """Remove all custom hook sinks (keeps the default CallbackHookSink)."""
        from kemi.plugins import CallbackHookSink

        self._core._plugins.hook_sinks = [
            s for s in self._core._plugins.hook_sinks if isinstance(s, CallbackHookSink)
        ]

    def clear_event_hooks(self) -> None:
        """Drop every callback registered on the default :class:`HookSink`."""
        from kemi.plugins import CallbackHookSink

        for sink in self._core._plugins.hook_sinks:
            if isinstance(sink, CallbackHookSink):
                sink.clear()
