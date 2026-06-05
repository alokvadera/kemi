"""Tests for kemi observability module."""

import time

from kemi.observability import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    get_metrics_collector,
    reset_metrics,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_starts_at_zero(self) -> None:
        c = Counter("test_counter", "Test counter")
        assert c.value() == 0

    def test_counter_increments(self) -> None:
        c = Counter("test_counter", "Test counter")
        c.inc()
        assert c.value() == 1
        c.inc(5)
        assert c.value() == 6

    def test_counter_prometheus_format(self) -> None:
        c = Counter("test_counter", "Test counter help")
        c.inc(3)
        output = c.to_prometheus()
        assert "kemi_test_counter" in output
        assert "Test counter help" in output
        assert "counter" in output
        assert "3" in output


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_starts_empty(self) -> None:
        h = Histogram("test_hist", "Test histogram")
        output = h.to_prometheus()
        assert "kemi_test_hist_count 0" in output

    def test_histogram_observes(self) -> None:
        h = Histogram("test_hist", "Test histogram")
        h.observe(0.05)
        h.observe(1.5)
        h.observe(15.0)
        output = h.to_prometheus()
        assert "kemi_test_hist_count 3" in output

    def test_histogram_buckets(self) -> None:
        h = Histogram("test_hist", "Test histogram", buckets=(0.1, 1.0, 5.0))
        h.observe(0.05)
        output = h.to_prometheus()
        assert 'le="0.1"' in output


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set_and_get(self) -> None:
        g = Gauge("test_gauge", "Test gauge")
        assert g.value() == 0.0
        g.set(5.0)
        assert g.value() == 5.0

    def test_gauge_inc_dec(self) -> None:
        g = Gauge("test_gauge", "Test gauge")
        g.inc(3.0)
        assert g.value() == 3.0
        g.dec(1.0)
        assert g.value() == 2.0

    def test_gauge_prometheus_format(self) -> None:
        g = Gauge("test_gauge", "Test gauge help")
        g.set(42.0)
        output = g.to_prometheus()
        assert "gauge" in output
        assert "42" in output


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_initial_state(self) -> None:
        collector = MetricsCollector()
        data = collector.to_dict()
        assert data["operations"]["remember"] == 0
        assert data["operations"]["recall"] == 0
        assert data["embeddings"]["total"] == 0
        assert data["quality"]["duplicates_detected"] == 0

    def test_collector_track_operations(self) -> None:
        collector = MetricsCollector()
        collector.remember_total.inc()
        collector.remember_total.inc()
        collector.recall_total.inc()
        data = collector.to_dict()
        assert data["operations"]["remember"] == 2
        assert data["operations"]["recall"] == 1

    def test_collector_embedding_metrics(self) -> None:
        collector = MetricsCollector()
        collector.embed_total.inc(10)
        collector.embed_errors_total.inc(1)
        data = collector.to_dict()
        assert data["embeddings"]["total"] == 10
        assert data["embeddings"]["errors"] == 1

    def test_collector_quality_metrics(self) -> None:
        collector = MetricsCollector()
        collector.duplicates_detected.inc()
        collector.conflicts_detected.inc(2)
        collector.lifecycle_transitions.inc(3)
        data = collector.to_dict()
        assert data["quality"]["duplicates_detected"] == 1
        assert data["quality"]["conflicts_detected"] == 2
        assert data["quality"]["lifecycle_transitions"] == 3

    def test_collector_gauge_metrics(self) -> None:
        collector = MetricsCollector()
        collector.total_memories.set(100)
        collector.total_users.set(5)
        data = collector.to_dict()
        assert data["memory_usage"]["total_memories"] == 100
        assert data["memory_usage"]["total_users"] == 5

    def test_collector_prometheus_export(self) -> None:
        collector = MetricsCollector()
        collector.remember_total.inc()
        output = collector.to_prometheus()
        assert "kemi_remember_total" in output
        assert "counter" in output
        assert "histogram" in output

    def test_collector_latency_tracking(self) -> None:
        collector = MetricsCollector()
        collector.remember_latency.observe(0.05)
        collector.embed_latency.observe(0.10)
        output = collector.to_prometheus()
        assert "kemi_remember_latency_seconds" in output
        assert "kemi_embed_latency_seconds" in output

    def test_collector_track_context_manager(self) -> None:
        collector = MetricsCollector()
        with collector.track("embed") as tracker:
            time.sleep(0.01)
            assert tracker.duration > 0

    def test_collector_reset(self) -> None:
        collector = MetricsCollector()
        collector.remember_total.inc(5)
        collector.recall_total.inc(3)
        collector.embed_total.inc(10)
        collector.remember_latency.observe(0.05)
        collector.embed_latency.observe(0.10)
        collector.total_memories.set(100)
        collector.total_users.set(5)
        collector.reset()
        assert collector.remember_total.value() == 0
        assert collector.recall_total.value() == 0
        assert collector.embed_total.value() == 0
        assert collector.remember_latency._count == 0
        assert collector.embed_latency._count == 0
        assert collector.total_memories.value() == 0.0
        assert collector.total_users.value() == 0.0

    def test_collector_timestamp_present(self) -> None:
        collector = MetricsCollector()
        data = collector.to_dict()
        assert "timestamp" in data

    def test_global_collector_singleton(self) -> None:
        reset_metrics()
        c1 = get_metrics_collector()
        c2 = get_metrics_collector()
        assert c1 is c2
        reset_metrics()
