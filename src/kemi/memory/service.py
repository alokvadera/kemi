"""Public ``MemoryService`` — composition of three facades.

This module is a delegation shell. All logic lives in:

- :mod:`kemi.services.read_service`  (recall, stats, graph)
- :mod:`kemi.services.write_service` (remember, update, forget)
- :mod:`kemi.services.admin_service` (configure, maintain, plugins)

The shared mutable state lives in :class:`_MemoryCore`. This file
preserves the historical public surface of :class:`MemoryService` by
forwarding every method to the appropriate facade.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.exceptions import ConfigurationError
from kemi.memory.core import _MemoryCore, build_default_store, default_entity_linker
from kemi.memory.model import (
    MemoryConfig,
    MemoryObject,
)
from kemi.services import MemoryAdminService, MemoryReadService, MemoryWriteService

if TYPE_CHECKING:
    from kemi.infra.encryption import EncryptionConfig
    from kemi.memory.entities import EntityLinker
    from kemi.plugins import (
        PluginRegistry,
    )

logger = logging.getLogger(__name__)


class MemoryService:
    """Public façade for the ``kemi`` memory library.

    Internally composed of three service objects (read / write / admin)
    that all share a single :class:`_MemoryCore` instance. Every public
    method here simply forwards to the appropriate facade.
    """

    def __init__(
        self,
        embed: EmbeddingAdapter | None = None,
        store: StorageAdapter | None = None,
        config: MemoryConfig | None = None,
        encryption: EncryptionConfig | None = None,
        entity_linker: EntityLinker | None = None,
    ) -> None:
        from kemi.infra.encryption import EncryptionConfig

        if encryption is None:
            try:
                encryption = EncryptionConfig.from_env()
            except Exception as exc:
                # Broad catch intentional: EncryptionConfig.from_env() reads
                # env vars and can fail for many reasons (missing key, bad
                # Fernet format, base64 decode errors). Encryption is opt-in;
                # fall back to disabled rather than blocking Memory init.
                logger.warning(
                    "EncryptionConfig.from_env() failed (%s). "
                    "Encryption will be disabled. To enable it, set a valid "
                    "KEMI_ENCRYPTION_KEY environment variable.",
                    exc,
                )
                encryption = None

        if embed is None:
            try:  # pragma: no cover
                from kemi.adapters.embedding.fastembed import FastEmbedAdapter

                embed = FastEmbedAdapter()
            except ImportError as e:
                raise ConfigurationError(
                    "No embedding adapter provided and fastembed is not installed. "
                    "Install with: pip install kemi[local] or provide your own: "
                    "Memory(embed=YourAdapter())"
                ) from e

        if store is None:
            store = build_default_store(embed, encryption)

        if config is None:
            config = MemoryConfig()

        if entity_linker is None:
            entity_linker = default_entity_linker(config)

        # Per-instance observability (isolates metrics per MemoryService so
        # multi-tenant deployments and test fixtures don't cross-talk).
        self._core = _MemoryCore(
            embed=embed,
            store=store,
            config=config,
            entity_linker=entity_linker,
            encryption=encryption,
        )
        try:
            from kemi.infra.observability import MetricsCollector

            self._core._metrics = MetricsCollector()
        except ImportError:
            pass

        # Compose the three facades against the shared core.
        self._read = MemoryReadService(self._core)
        self._write = MemoryWriteService(self._core)
        self._admin = MemoryAdminService(self._core)

    # ------------------------------------------------------------------
    # Internal helpers (kept for legacy test fixtures that may touch them)
    # ------------------------------------------------------------------

    def _latency_tracker(self, operation: str) -> Any:
        return self._core._latency_tracker(operation)

    def _validate_remember_inputs(
        self,
        user_id: str,
        content: str,
        importance: Any,
        ttl_seconds: int | None,
    ) -> None:
        _MemoryCore.validate_remember_inputs(user_id, content, importance, ttl_seconds)

    def _build_memory_object(self, **kwargs: Any) -> MemoryObject:
        return _MemoryCore.build_memory_object(**kwargs)

    def _build_io_runtime(self) -> Any:
        return self._core.build_io_runtime()

    def _build_ingestion_context(self) -> Any:
        return self._core.build_ingestion_context()

    def _build_retrieval_context(self) -> Any:
        return self._core.build_retrieval_context()

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
        self._core._track_operation(
            operation, user_id, details, memory_id, namespace, status, audit_batch
        )

    def _record_embed_error(self) -> None:
        self._core._record_embed_error()

    def _record_store_error(self) -> None:
        self._core._record_store_error()

    def _run_hooks(
        self,
        phase: str,
        operation: str,
        *,
        raise_on_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self._core._run_hooks(phase, operation, raise_on_error=raise_on_error, **kwargs)

    def _dispatch_webhook_event(
        self,
        event: Any,
        memory_id: str,
        user_id: str,
        snapshot: dict[str, Any] | None = None,
        previous_state: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self._core._dispatch_webhook_event(
            event, memory_id, user_id, snapshot, previous_state, **extra
        )

    def _get_version_store(self) -> Any:
        return self._core._get_version_store()

    def _auto_prune_versions_for_memory(self, memory_id: str) -> None:
        self._core._auto_prune_versions_for_memory(memory_id)

    def _remember_with_embedding(self, *args: Any, **kwargs: Any) -> str:
        return self._write._remember_with_embedding(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy ``_xxx`` legacy attributes to :class:`_MemoryCore`.

        Lots of legacy code paths (api_server, background_tasks, tests)
        reach into ``mem._store``, ``mem._audit_trail``, ``mem._plugins``,
        etc. They all work transparently because :class:`_MemoryCore`
        stores the same attributes with the same names. Public methods
        (non-``_`` prefix) are defined on this class and are not proxied.
        """
        if name.startswith("_") and not name.startswith("__"):
            try:
                core = object.__getattribute__(self, "_core")
            except AttributeError as exc:
                raise AttributeError(name) from exc
            return getattr(core, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy ``_xxx`` legacy writes to :class:`_MemoryCore`.

        Symmetric with :meth:`__getattr__` so legacy code can do
        ``mem._audit_trail = AuditTrail(...)`` and the assignment lands
        on the shared :class:`_MemoryCore` instance.
        """
        if name.startswith("_") and not name.startswith("__") and name not in {
            "_core",
            "_read",
            "_write",
            "_admin",
        }:
            try:
                core = object.__getattribute__(self, "_core")
                setattr(core, name, value)
                return
            except AttributeError:
                pass
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Write facade
    # ------------------------------------------------------------------

    def remember(self, *args: Any, **kwargs: Any) -> str:
        return self._write.remember(*args, **kwargs)

    def remember_many(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._write.remember_many(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> str:
        return self._write.update(*args, **kwargs)

    def update_many(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._write.update_many(*args, **kwargs)

    def forget(self, *args: Any, **kwargs: Any) -> int:
        return self._write.forget(*args, **kwargs)

    def forget_many(self, *args: Any, **kwargs: Any) -> int:
        return self._write.forget_many(*args, **kwargs)

    def feedback(self, *args: Any, **kwargs: Any) -> None:
        self._write.feedback(*args, **kwargs)

    def backfill_entities(self, *args: Any, **kwargs: Any) -> int:
        return self._write.backfill_entities(*args, **kwargs)

    def extract_entities(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._write.extract_entities(*args, **kwargs)

    def migrate(self, *args: Any, **kwargs: Any) -> int:
        return self._write.migrate(*args, **kwargs)

    async def aremember(self, *args: Any, **kwargs: Any) -> str:
        return await self._write.aremember(*args, **kwargs)

    async def aremember_many(self, *args: Any, **kwargs: Any) -> list[str]:
        return await self._write.aremember_many(*args, **kwargs)

    async def aupdate(self, *args: Any, **kwargs: Any) -> str:
        return await self._write.aupdate(*args, **kwargs)

    async def aupdate_many(self, *args: Any, **kwargs: Any) -> list[str]:
        return await self._write.aupdate_many(*args, **kwargs)

    async def aforget(self, *args: Any, **kwargs: Any) -> int:
        return await self._write.aforget(*args, **kwargs)

    async def aforget_many(self, *args: Any, **kwargs: Any) -> int:
        return await self._write.aforget_many(*args, **kwargs)

    async def abackfill_entities(self, *args: Any, **kwargs: Any) -> int:
        return await self._write.abackfill_entities(*args, **kwargs)

    # ------------------------------------------------------------------
    # Read facade
    # ------------------------------------------------------------------

    def recall(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall(*args, **kwargs)

    def recall_many(self, *args: Any, **kwargs: Any) -> dict[str, list[MemoryObject]]:
        return self._read.recall_many(*args, **kwargs)

    def recall_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[MemoryObject, None]:
        return self._read.recall_stream(*args, **kwargs)

    def recall_between(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall_between(*args, **kwargs)

    def recall_user_profile(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall_user_profile(*args, **kwargs)

    def recall_session_context(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall_session_context(*args, **kwargs)

    def recall_agent_knowledge(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall_agent_knowledge(*args, **kwargs)

    def recall_explain(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._read.recall_explain(*args, **kwargs)

    def recall_since(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall_since(*args, **kwargs)

    def recall_by_tag(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return self._read.recall_by_tag(*args, **kwargs)

    def context_block(self, *args: Any, **kwargs: Any) -> str:
        return self._read.context_block(*args, **kwargs)

    def list_users(self) -> list[str]:
        return self._read.list_users()

    def stats(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._read.stats(*args, **kwargs)

    def get_memory_graph(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._read.get_memory_graph(*args, **kwargs)

    async def arecall(
        self, *args: Any, **kwargs: Any
    ) -> list[MemoryObject] | AsyncGenerator[MemoryObject, None]:
        return await self._read.arecall(*args, **kwargs)

    async def arecall_many(self, *args: Any, **kwargs: Any) -> dict[str, list[MemoryObject]]:
        return await self._read.arecall_many(*args, **kwargs)

    async def arecall_since(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return await self._read.arecall_since(*args, **kwargs)

    async def arecall_by_tag(self, *args: Any, **kwargs: Any) -> list[MemoryObject]:
        return await self._read.arecall_by_tag(*args, **kwargs)

    async def acontext_block(self, *args: Any, **kwargs: Any) -> str:
        return await self._read.acontext_block(*args, **kwargs)

    async def alist_users(self) -> list[str]:
        return await self._read.alist_users()

    async def astats(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._read.astats(*args, **kwargs)

    # ------------------------------------------------------------------
    # Admin facade
    # ------------------------------------------------------------------

    def configure_webhooks(self, *args: Any, **kwargs: Any) -> None:
        self._admin.configure_webhooks(*args, **kwargs)

    def configure_versioning(self, *args: Any, **kwargs: Any) -> None:
        self._admin.configure_versioning(*args, **kwargs)

    def enable_audit_trail(self, *args: Any, **kwargs: Any) -> None:
        self._admin.enable_audit_trail(*args, **kwargs)

    def enable_query_cache(self, *args: Any, **kwargs: Any) -> None:
        self._admin.enable_query_cache(*args, **kwargs)

    def disable_query_cache(self) -> None:
        self._admin.disable_query_cache()

    def enable_adaptive_retrieval(self, *args: Any, **kwargs: Any) -> None:
        self._admin.enable_adaptive_retrieval(*args, **kwargs)

    def get_metrics(self) -> dict[str, Any] | None:
        return self._admin.get_metrics()

    def get_metrics_prometheus(self) -> str | None:
        return self._admin.get_metrics_prometheus()

    def add_event_hook(self, *args: Any, **kwargs: Any) -> None:
        self._admin.add_event_hook(*args, **kwargs)

    def remove_event_hook(self, *args: Any, **kwargs: Any) -> bool:
        return self._admin.remove_event_hook(*args, **kwargs)

    def get_history(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self._admin.get_history(*args, **kwargs)

    def diff_versions(self, *args: Any, **kwargs: Any) -> Any:
        return self._admin.diff_versions(*args, **kwargs)

    def rollback_memory(self, *args: Any, **kwargs: Any) -> Any:
        return self._admin.rollback_memory(*args, **kwargs)

    def upgrade(self) -> int:
        return self._admin.upgrade()

    def export(self, file_path: str) -> int:
        return self._admin.export(file_path)

    def import_from(self, file_path: str) -> int:
        return self._admin.import_from(file_path)

    async def aexport(self, file_path: str) -> int:
        return await self._admin.aexport(file_path)

    async def aimport_from(self, file_path: str) -> int:
        return await self._admin.aimport_from(file_path)

    def prune(self, *args: Any, **kwargs: Any) -> int:
        return self._admin.prune(*args, **kwargs)

    def prune_expired(self, *args: Any, **kwargs: Any) -> int:
        return self._admin.prune_expired(*args, **kwargs)

    def consolidate(self, *args: Any, **kwargs: Any) -> str | None:
        return self._admin.consolidate(*args, **kwargs)

    def cluster_topics(self, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
        return self._admin.cluster_topics(*args, **kwargs)

    def run_maintenance(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._admin.run_maintenance(*args, **kwargs)

    def get_plugins(self) -> PluginRegistry:
        return self._admin.get_plugins()

    def close(self) -> None:
        """Close the underlying storage adapter and release resources.

        Safe to call multiple times.  After closing, the MemoryService
        instance should not be used for further operations.
        """
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
        # Drop references to heavy resources (embedding models, webhook
        # dispatchers) so ephemeral Memory instances created in background
        # tasks can be fully garbage collected.
        try:
            core = object.__getattribute__(self, "_core")
            core._embed = None
            core._webhook_dispatcher = None
            core._metrics = None
            core._query_cache = None
            core._version_store = None
            core._audit_trail = None
            core._adaptive_retriever = None
        except AttributeError:
            pass

    def add_webhook_sink(self, *args: Any, **kwargs: Any) -> None:
        self._admin.add_webhook_sink(*args, **kwargs)

    def add_audit_sink(self, *args: Any, **kwargs: Any) -> None:
        self._admin.add_audit_sink(*args, **kwargs)

    def add_hook_sink(self, *args: Any, **kwargs: Any) -> None:
        self._admin.add_hook_sink(*args, **kwargs)

    def set_query_cache(self, *args: Any, **kwargs: Any) -> None:
        self._admin.set_query_cache(*args, **kwargs)

    def clear_webhook_sinks(self) -> None:
        self._admin.clear_webhook_sinks()

    def clear_audit_sinks(self) -> None:
        self._admin.clear_audit_sinks()

    def clear_hook_sinks(self) -> None:
        self._admin.clear_hook_sinks()

    def clear_event_hooks(self) -> None:
        self._admin.clear_event_hooks()


class _QueryCache:
    """DEPRECATED shim — moved to :mod:`kemi.operations._query_cache`.

    Kept as a re-export so existing imports of ``kemi.core._QueryCache``
    keep working. The canonical location is ``kemi.operations._query_cache._QueryCache``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        from kemi.operations._query_cache import _QueryCache as _Impl

        return _Impl(*args, **kwargs)


_QueryCache.__doc__ = _QueryCache.__doc__
