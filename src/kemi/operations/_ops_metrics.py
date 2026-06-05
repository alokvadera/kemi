"""Metrics, audit, and query-cache enable/disable operations."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kemi._memory_impl import Memory

logger = logging.getLogger(__name__)


def latency_tracker(memory: "Memory", operation: str) -> Any:
    """Return a context manager that tracks operation latency if metrics are enabled."""
    if memory._metrics is not None:
        return memory._metrics.track(operation)
    return nullcontext()


def track_operation(memory: "Memory", operation: str, **details: Any) -> None:
    """Record a single operation in the metrics collector (no-op if disabled)."""
    if memory._metrics is None:
        return
    try:
        if hasattr(memory._metrics, "track_operation"):
            memory._metrics.track_operation(operation, **details)
        elif hasattr(memory._metrics, "inc"):
            memory._metrics.inc(f"{operation}_total")
    except (AttributeError, TypeError):
        logger.debug("metrics.track_operation failed for %s", operation, exc_info=True)


def track_operation_full(
    memory: "Memory",
    operation: str,
    user_id: str,
    details: dict[str, Any] | None,
    memory_id: str | None,
    namespace: str,
    status: str,
    audit_batch: list[dict[str, Any]] | None,
) -> None:
    """Track an operation in metrics and audit trail.

    If *audit_batch* is provided, the audit entry is appended to that list
    instead of being written immediately. The caller is responsible for
    passing the list to ``AuditTrail.log_operation_batch``.
    """
    if memory._metrics is not None:
        counter_name = f"{operation}_total"
        counter = getattr(memory._metrics, counter_name, None)
        if counter is not None:
            try:
                counter.inc(1)
            except (AttributeError, TypeError):
                pass
    if audit_batch is not None:
        audit_batch.append(
            {
                "user_id": user_id,
                "operation": operation,
                "details": details or {},
                "memory_id": memory_id,
                "namespace": namespace,
                "status": status,
            }
        )
        return
    if memory._audit_trail is not None:
        try:
            memory._audit_trail.log_operation(
                user_id=user_id,
                operation=operation,
                details=details or {},
                memory_id=memory_id,
                namespace=namespace,
                status=status,
            )
        except (AttributeError, TypeError, ValueError):
            # Broad catch: audit log can fail for many reasons (DB locked,
            # schema mismatch, IO error). Audit must never break the caller.
            logger.warning(f"Audit log failed for {operation}", exc_info=True)


def record_embed_error(memory: "Memory") -> None:
    """Increment the embed error counter (no-op if metrics disabled)."""
    if memory._metrics is not None and hasattr(memory._metrics, "embed_errors_total"):
        try:
            memory._metrics.embed_errors_total.inc(1)
        except (AttributeError, TypeError):
            pass


def record_store_error(memory: "Memory") -> None:
    """Increment the store error counter (no-op if metrics disabled)."""
    if memory._metrics is not None and hasattr(memory._metrics, "store_errors_total"):
        try:
            memory._metrics.store_errors_total.inc(1)
        except (AttributeError, TypeError):
            pass


def get_metrics(memory: "Memory") -> dict[str, Any] | None:
    """Return current metrics snapshot as a dict, or None if disabled."""
    if memory._metrics is None:
        return None
    try:
        if hasattr(memory._metrics, "to_dict"):
            return memory._metrics.to_dict()
        if hasattr(memory._metrics, "snapshot"):
            return memory._metrics.snapshot()
    except (AttributeError, TypeError):
        logger.debug("get_metrics failed", exc_info=True)
    return None


def get_metrics_prometheus(memory: "Memory") -> str | None:
    """Return metrics in Prometheus text format, or None if disabled."""
    if memory._metrics is None:
        return None
    try:
        if hasattr(memory._metrics, "to_prometheus"):
            return memory._metrics.to_prometheus()
    except (AttributeError, TypeError):
        logger.debug("get_metrics_prometheus failed", exc_info=True)
    return None


def enable_adaptive_retrieval(memory: "Memory", enable: bool = True) -> None:
    """Enable or disable adaptive retrieval (re-weights hybrid scores per user)."""
    if not enable:
        memory._adaptive_retriever = None
        return
    try:
        from kemi.adaptive import AdaptiveRetriever

        memory._adaptive_retriever = AdaptiveRetriever()
        logger.info("Adaptive retrieval enabled")
    except ImportError as e:
        logger.warning(f"Adaptive retrieval module not available: {e}")


def enable_audit_trail(
    memory: "Memory",
    retention_days: int = 365,
    auto_purge: bool = True,
) -> None:
    """Enable the audit trail for compliance logging."""
    try:
        from kemi.audit import AuditTrail

        conn = memory._store._get_connection()  # type: ignore[attr-defined]
        memory._audit_trail = AuditTrail(
            db_connection=conn,
            retention_days=retention_days,
            auto_purge=auto_purge,
        )
    except (ImportError, AttributeError) as e:
        logger.warning(f"Audit trail not available: {e}")


def enable_query_cache(memory: "Memory", max_size: int = 128) -> None:
    """Enable an LRU cache for ``recall()`` results."""
    from kemi.operations._query_cache import _QueryCache

    memory._query_cache = _QueryCache(max_size=max_size)


def disable_query_cache(memory: "Memory") -> None:
    """Disable the query cache."""
    memory._query_cache = None
