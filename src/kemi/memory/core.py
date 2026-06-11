from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kemi.infra.encryption import EncryptionConfig
    from kemi.infra.webhooks import WebhookDispatcher
    from kemi.memory.entities import EntityLinker
    from kemi.plugins import PluginRegistry

from kemi.exceptions import ValidationError
from kemi.memory.entities import NoopEntityLinker, RegexEntityLinker
from kemi.memory.model import LifecycleState, MemoryConfig, MemoryObject, MemorySource, MemoryType
from kemi.memory.versions import MemoryVersionStore
from kemi.plugins import CallbackHookSink, PluginRegistry

logger = logging.getLogger(__name__)


class _MemoryCore:
    """Shared mutable state owned by every :class:`MemoryService` facade.

    All attributes use the same ``_``-prefixed names that the legacy
    monolithic :class:`MemoryService` exposed, so the helper modules under
    :mod:`kemi.operations` can keep reading ``memory._metrics``,
    ``memory._plugins``, etc. without any changes.

    The core is intentionally NOT a dataclass: lifecycle is non-uniform
    (``_metrics`` and ``_webhook_dispatcher`` start as ``None`` and are
    lazily initialised), and we want free-form ``_xxx`` attributes
    matching the historical surface for legacy test fixtures.
    """

    def __init__(
        self,
        embed: Any,
        store: Any,
        config: MemoryConfig,
        entity_linker: EntityLinker,
        encryption: EncryptionConfig | None = None,
    ) -> None:
        self._embed = embed
        self._store = store
        self._config = config
        self._entity_linker = entity_linker
        self._encryption = encryption

        self._metrics: Any | None = None
        self._audit_trail: Any | None = None
        self._adaptive_retriever: Any | None = None
        self._event_hooks: dict[str, list[Callable[..., Any]]] = {"pre": [], "post": []}
        self._query_cache: Any | None = None
        self._version_store: MemoryVersionStore | None = None
        self._max_versions_per_memory: int = 50
        self._auto_prune_versions: bool = True
        self._webhook_dispatcher: WebhookDispatcher | None = None

        self._plugins: PluginRegistry = PluginRegistry()
        self._plugins.hook_sinks.append(CallbackHookSink(hooks=self._event_hooks))

    def build_io_runtime(self) -> Any:
        from kemi.operations._io import MemoryIORuntime

        return MemoryIORuntime(
            store=self._store,
            embed=self._embed,
            entity_linker=self._entity_linker,
            config=self._config,
            metrics=self._metrics,
            run_hooks=self._run_hooks,
            track_operation=self._track_operation,
            log_audit=self._log_audit,
            dispatch_webhook=self._dispatch_webhook_event,
            latency_tracker=self._latency_tracker,
            recall_fn=self._recall_via_pipeline,
            get_version_store=self._get_version_store,
            auto_prune_versions=self._auto_prune_versions_for_memory,
            auto_prune_versions_enabled=bool(self._auto_prune_versions),
            max_versions_per_memory=int(self._max_versions_per_memory or 0),
        )

    def build_ingestion_context(self) -> Any:
        from kemi.pipeline.ingestion import IngestionContext

        return IngestionContext(
            store=self._store,
            config=self._config,
            entity_linker=self._entity_linker,
            metrics=self._metrics,
            record_store_error=self._record_store_error,
            dispatch_webhook=self._dispatch_webhook_event,
            track_operation=self._track_operation,
            get_version_store=self._get_version_store,
            auto_prune_versions=self._auto_prune_versions_for_memory,
        )

    def build_retrieval_context(self) -> Any:
        from kemi.pipeline.retrieval import RetrievalContext

        return RetrievalContext(
            store=self._store,
            embed=self._embed,
            config=self._config,
            entity_linker=self._entity_linker,
            query_cache=self._query_cache,
            metrics=self._metrics,
            adaptive_retriever=self._adaptive_retriever,
            run_hooks=self._run_hooks,
            track_operation=self._track_operation,
        )

    def _latency_tracker(self, operation: str) -> Any:
        from kemi.operations import _ops_metrics

        return _ops_metrics.latency_tracker(self, operation)

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
        from kemi.operations import _ops_metrics

        _ops_metrics.track_operation_full(
            self, operation, user_id, details, memory_id, namespace, status, audit_batch
        )

    def _log_audit(
        self,
        operation: str,
        user_id: str,
        memory_id: str | None = None,
        namespace: str = "default",
        details: dict[str, Any] | None = None,
    ) -> None:
        if self._audit_trail is None:
            return
        try:
            self._audit_trail.log_operation(
                user_id=user_id,
                operation=operation,
                details=details or {},
                memory_id=memory_id,
                namespace=namespace,
                status="success",
            )
        except Exception:
            pass

    def _record_embed_error(self) -> None:
        from kemi.operations import _ops_metrics

        _ops_metrics.record_embed_error(self)

    def _record_store_error(self) -> None:
        from kemi.operations import _ops_metrics

        _ops_metrics.record_store_error(self)

    def _run_hooks(
        self,
        phase: str,
        operation: str,
        *,
        raise_on_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        from kemi.operations import _ops_hooks

        _ops_hooks.run(self, phase, operation, raise_on_error=raise_on_error, **kwargs)

    def _dispatch_webhook_event(
        self,
        event: Any,
        memory_id: str,
        user_id: str,
        snapshot: dict[str, Any] | None = None,
        previous_state: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        from kemi.operations import _ops_webhooks

        _ops_webhooks.dispatch(
            self, event, memory_id, user_id, snapshot, previous_state, **extra
        )

    def _get_version_store(self) -> MemoryVersionStore | None:
        from kemi.operations import _ops_versioning

        return _ops_versioning.get_store(self)

    def _auto_prune_versions_for_memory(self, memory_id: str) -> None:
        from kemi.operations import _ops_versioning

        _ops_versioning.auto_prune(self, memory_id)

    def _recall_via_pipeline(self, **kwargs: Any) -> Any:
        from kemi.pipeline.retrieval import RetrievalPipeline

        return RetrievalPipeline(self.build_retrieval_context()).retrieve(**kwargs)

    @staticmethod
    def validate_remember_inputs(
        user_id: str,
        content: str,
        importance: Any,
        ttl_seconds: int | None,
    ) -> None:
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty")
        if not content or not content.strip():
            raise ValidationError("content cannot be empty — there is nothing to remember")
        if not isinstance(importance, (int, float)):
            raise ValidationError(
                f"importance must be a number between 0.0 and 1.0, got {type(importance).__name__}"
            )
        if ttl_seconds is not None and (
            not isinstance(ttl_seconds, int) or ttl_seconds <= 0
        ):
            raise ValidationError(f"ttl_seconds must be a positive integer, got {ttl_seconds}")

    @staticmethod
    def build_memory_object(
        user_id: str,
        content: str,
        embedding: list[float],
        importance: float,
        source: MemorySource,
        metadata: dict[str, Any] | None,
        tags: list[str] | None,
        namespace: str,
        session_id: str | None,
        memory_type: MemoryType,
        confidence: float,
        agent_id: str | None,
        run_id: str | None,
        app_id: str | None,
        ttl_seconds: int | None,
    ) -> MemoryObject:
        """Construct a fresh ACTIVE ``MemoryObject`` from raw inputs."""
        clamped_importance = max(0.0, min(1.0, importance))
        return MemoryObject(
            memory_id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            embedding=embedding,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=source,
            importance=clamped_importance,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata=metadata or {},
            embedding_dim=len(embedding),
            tags=tags or [],
            confidence=max(0.0, min(1.0, confidence)),
            memory_type=memory_type,
            session_id=session_id,
            namespace=namespace,
            version=1,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            expires_at=(
                datetime.now(timezone.utc).replace(microsecond=0)
                + timedelta(seconds=ttl_seconds)
                if ttl_seconds is not None
                else None
            ),
        )


def build_default_store(
    embed: Any, encryption: EncryptionConfig | None
) -> Any:
    """Pick the right storage adapter for the runtime environment."""
    from kemi.infra.encryption import EncryptionConfig

    env_path = os.environ.get("KEMI_DB_PATH")
    if env_path:
        default_db_path = os.path.expanduser(env_path)
    else:
        default_db_path = os.path.join(os.path.expanduser("~"), ".kemi", "memories.db")
    os.makedirs(os.path.dirname(default_db_path), exist_ok=True)

    enc_or_none = (
        encryption
        if isinstance(encryption, EncryptionConfig) and encryption.enabled
        else None
    )

    try:
        from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter

        if SQLiteVecStorageAdapter.is_vec_available():
            embedding_dim = embed.dimension()
            return SQLiteVecStorageAdapter(
                db_path=default_db_path,
                embedding_dim=embedding_dim,
                encryption=enc_or_none,
            )
    except ImportError:  # pragma: no cover
        pass

    from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

    return SQLiteStorageAdapter(db_path=default_db_path, encryption=enc_or_none)


def default_entity_linker(config: MemoryConfig) -> EntityLinker:
    if config.enable_entity_boost:
        return RegexEntityLinker()
    return NoopEntityLinker()
