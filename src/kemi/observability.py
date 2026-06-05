"""Observability and metrics for kemi operations.

Provides Prometheus-compatible metrics for monitoring and debugging:
- Operation counters (remember, recall, forget, etc.)
- Latency histograms
- Embedding generation tracking
- Storage operation tracking
- Error counters

All metrics are collected in-memory with zero external dependencies.
Export formats: Prometheus text format, JSON, and Python dict.

Usage:
    from kemi.observability import MetricsCollector

    metrics = MetricsCollector()
    with metrics.track("remember"):
        memory.remember("user123", "content")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A single metric value with metadata."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str, help_text: str = "", namespace: str = "kemi") -> None:
        self._name = f"{namespace}_{name}"
        self._help = help_text
        self._value: int = 0
        self._lock = threading.Lock()
        self._created = time.time()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def value(self) -> int:
        with self._lock:
            return self._value

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self._name} {self._help}", f"# TYPE {self._name} counter"]
        lines.append(f"{self._name} {self._value}")
        return "\n".join(lines)


class Histogram:
    """Histogram for tracking distributions (e.g., latency)."""

    _DEFAULT_BUCKETS = (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    )

    def __init__(
        self,
        name: str,
        help_text: str = "",
        namespace: str = "kemi",
        buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
    ) -> None:
        self._name = f"{namespace}_{name}"
        self._help = help_text
        self._buckets = buckets
        self._count: int = 0
        self._sum: float = 0.0
        self._bucket_counts: dict[float, int] = {b: 0 for b in buckets}
        self._lock = threading.Lock()
        self._created = time.time()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            for bound in self._buckets:
                if value <= bound:
                    self._bucket_counts[bound] += 1

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self._name} {self._help}", f"# TYPE {self._name} histogram"]
        for bound in self._buckets:
            count = self._bucket_counts[bound]
            lines.append(f'{self._name}_bucket{{le="{bound}"}} {count}')
        lines.append(f'{self._name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self._name}_count {self._count}")
        lines.append(f"{self._name}_sum {self._sum:.6f}")
        return "\n".join(lines)


class Gauge:
    """Gauge that can go up and down."""

    def __init__(self, name: str, help_text: str = "", namespace: str = "kemi") -> None:
        self._name = f"{namespace}_{name}"
        self._help = help_text
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def value(self) -> float:
        with self._lock:
            return self._value

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self._name} {self._help}", f"# TYPE {self._name} gauge"]
        lines.append(f"{self._name} {self._value}")
        return "\n".join(lines)


class MetricsCollector:
    """Central metrics collector for kemi operations.

    Tracks:
    - Operation counts and latencies
    - Embedding generation stats
    - Storage operation stats
    - Error counts
    - Memory usage stats (total memories, users, etc.)
    """

    def __init__(self, namespace: str = "kemi") -> None:
        self._ns = namespace

        # Operation counters
        self.remember_total = Counter(
            "remember_total",
            "Total number of remember operations",
            namespace,
        )
        self.recall_total = Counter(
            "recall_total",
            "Total number of recall operations",
            namespace,
        )
        self.forget_total = Counter(
            "forget_total",
            "Total number of forget operations",
            namespace,
        )
        self.remember_many_total = Counter(
            "remember_many_total",
            "Total number of batch remember operations",
            namespace,
        )
        self.update_total = Counter(
            "update_total",
            "Total number of update operations",
            namespace,
        )
        self.prune_total = Counter(
            "prune_total",
            "Total number of prune operations",
            namespace,
        )
        self.migrate_total = Counter(
            "migrate_total",
            "Total number of migrate operations",
            namespace,
        )
        self.consolidate_total = Counter(
            "consolidate_total",
            "Total number of consolidate operations",
            namespace,
        )
        self.export_total = Counter(
            "export_total",
            "Total number of export operations",
            namespace,
        )
        self.import_total = Counter(
            "import_total",
            "Total number of import operations",
            namespace,
        )
        self.feedback_total = Counter(
            "feedback_total",
            "Total number of feedback operations",
            namespace,
        )

        # Latency histograms
        self.remember_latency = Histogram(
            "remember_latency_seconds",
            "Latency of remember operations",
            namespace,
        )
        self.recall_latency = Histogram(
            "recall_latency_seconds",
            "Latency of recall operations",
            namespace,
        )
        self.forget_latency = Histogram(
            "forget_latency_seconds",
            "Latency of forget operations",
            namespace,
        )
        self.remember_many_latency = Histogram(
            "remember_many_latency_seconds",
            "Latency of batch remember operations",
            namespace,
        )
        self.embed_latency = Histogram(
            "embed_latency_seconds",
            "Latency of embedding generation",
            namespace,
        )
        self.store_latency = Histogram(
            "store_latency_seconds",
            "Latency of storage operations",
            namespace,
        )
        self.search_latency = Histogram(
            "search_latency_seconds",
            "Latency of vector search operations",
            namespace,
        )

        # Embedding metrics
        self.embed_total = Counter(
            "embed_total",
            "Total embeddings generated",
            namespace,
        )
        self.embed_errors_total = Counter(
            "embed_errors_total",
            "Total embedding generation errors",
            namespace,
        )
        self.embed_bytes_total = Counter(
            "embed_bytes_total",
            "Approximate total bytes embedded",
            namespace,
        )

        # Storage metrics
        self.store_errors_total = Counter(
            "store_errors_total",
            "Total storage operation errors",
            namespace,
        )

        # Duplicate and conflict metrics
        self.duplicates_detected = Counter(
            "duplicates_detected",
            "Total duplicates detected",
            namespace,
        )
        self.conflicts_detected = Counter(
            "conflicts_detected",
            "Total conflicts detected",
            namespace,
        )

        # Lifecycle metrics
        self.lifecycle_transitions = Counter(
            "lifecycle_transitions",
            "Total lifecycle state transitions",
            namespace,
        )

        # Memory usage gauges
        self.total_memories = Gauge(
            "total_memories",
            "Current total number of memories",
            namespace,
        )
        self.total_users = Gauge(
            "total_users",
            "Current total number of users",
            namespace,
        )

    def _start_timer(self) -> float:
        return time.monotonic()

    def _stop_timer(self, start: float, histogram: Histogram) -> float:
        duration = time.monotonic() - start
        histogram.observe(duration)
        return duration

    def track(self, operation: str) -> "_OperationTracker":
        """Start tracking an operation. Use as context manager.

        Example:
            with metrics.track("remember"):
                memory.remember("user123", "content")
        """
        return _OperationTracker(self, operation)

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed operation with timing."""
        pass  # Go through individual counters

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as a Python dict."""
        return {
            "operations": {
                "remember": self.remember_total.value(),
                "recall": self.recall_total.value(),
                "forget": self.forget_total.value(),
                "remember_many": self.remember_many_total.value(),
                "update": self.update_total.value(),
                "prune": self.prune_total.value(),
                "migrate": self.migrate_total.value(),
                "consolidate": self.consolidate_total.value(),
                "export": self.export_total.value(),
                "import": self.import_total.value(),
                "feedback": self.feedback_total.value(),
            },
            "embeddings": {
                "total": self.embed_total.value(),
                "errors": self.embed_errors_total.value(),
                "bytes_approx": self.embed_bytes_total.value(),
            },
            "storage": {
                "errors": self.store_errors_total.value(),
            },
            "quality": {
                "duplicates_detected": self.duplicates_detected.value(),
                "conflicts_detected": self.conflicts_detected.value(),
                "lifecycle_transitions": self.lifecycle_transitions.value(),
            },
            "memory_usage": {
                "total_memories": self.total_memories.value(),
                "total_users": self.total_users.value(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        metrics = [
            self.remember_total,
            self.recall_total,
            self.forget_total,
            self.remember_many_total,
            self.update_total,
            self.prune_total,
            self.migrate_total,
            self.consolidate_total,
            self.export_total,
            self.import_total,
            self.feedback_total,
            self.embed_total,
            self.embed_errors_total,
            self.embed_bytes_total,
            self.store_errors_total,
            self.duplicates_detected,
            self.conflicts_detected,
            self.lifecycle_transitions,
        ]
        histograms = [
            self.remember_latency,
            self.recall_latency,
            self.forget_latency,
            self.remember_many_latency,
            self.embed_latency,
            self.store_latency,
            self.search_latency,
        ]
        gauges = [
            self.total_memories,
            self.total_users,
        ]

        parts = []
        for m in metrics:
            parts.append(m.to_prometheus())
        for h in histograms:
            parts.append(h.to_prometheus())
        for g in gauges:
            parts.append(g.to_prometheus())

        return "\n\n".join(parts) + "\n"

    def reset(self) -> None:
        """Reset all metrics to zero. Useful for testing."""
        counters = [
            self.remember_total,
            self.recall_total,
            self.forget_total,
            self.remember_many_total,
            self.update_total,
            self.prune_total,
            self.migrate_total,
            self.consolidate_total,
            self.export_total,
            self.import_total,
            self.feedback_total,
            self.embed_total,
            self.embed_errors_total,
            self.embed_bytes_total,
            self.store_errors_total,
            self.duplicates_detected,
            self.conflicts_detected,
            self.lifecycle_transitions,
        ]
        for c in counters:
            with c._lock:
                c._value = 0

        histograms = [
            self.remember_latency,
            self.recall_latency,
            self.forget_latency,
            self.remember_many_latency,
            self.embed_latency,
            self.store_latency,
            self.search_latency,
        ]
        for h in histograms:
            with h._lock:
                h._count = 0
                h._sum = 0.0
                h._bucket_counts = {b: 0 for b in h._buckets}

        gauges = [self.total_memories, self.total_users]
        for g in gauges:
            with g._lock:
                g._value = 0.0


_HISTOGRAM_MAP: dict[str, str] = {
    "remember": "remember_latency",
    "recall": "recall_latency",
    "forget": "forget_latency",
    "remember_many": "remember_many_latency",
    "embed": "embed_latency",
    "store": "store_latency",
    "search": "search_latency",
}


class _OperationTracker:
    """Context manager for tracking operation duration."""

    def __init__(self, collector: MetricsCollector, operation: str) -> None:
        self._collector = collector
        self._operation = operation
        self._start: float = 0.0

    def __enter__(self) -> "_OperationTracker":
        self._start = self._collector._start_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        hist_name = _HISTOGRAM_MAP.get(self._operation, "embed_latency")
        histogram = getattr(self._collector, hist_name, self._collector.embed_latency)
        self._collector._stop_timer(self._start, histogram)

    @property
    def duration(self) -> float:
        if self._start:
            return time.monotonic() - self._start
        return 0.0


# Global singleton for convenience
_global_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector singleton."""
    global _global_collector
    if _global_collector is not None:
        return _global_collector

    with _collector_lock:
        if _global_collector is not None:
            return _global_collector
        _global_collector = MetricsCollector()
        return _global_collector


def reset_metrics() -> None:
    """Reset the global metrics collector. For testing."""
    global _global_collector
    with _collector_lock:
        _global_collector = None
