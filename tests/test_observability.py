"""Tests for kemi observability module."""

import threading
import time

from kemi.infra.observability import (
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

    def test_record_operation_increments_counter(self) -> None:
        collector = MetricsCollector()
        collector.record_operation("remember", success=True, metadata={"key": "val"})
        data = collector.to_dict()
        assert data["operations"]["remember"] == 1

    def test_record_operation_records_duration(self) -> None:
        collector = MetricsCollector()
        collector.record_operation("remember", duration=0.05, success=True)
        data = collector.to_dict()
        assert data["operations"]["remember"] == 1
        assert collector.remember_latency._count == 1

    def test_operation_tracker_duration_before_enter(self) -> None:
        collector = MetricsCollector()
        tracker = collector.track("remember")
        assert tracker.duration == 0.0

    def test_operation_tracker_duration_after_enter(self) -> None:
        collector = MetricsCollector()
        tracker = collector.track("remember")
        with tracker:
            time.sleep(0.01)
            assert tracker.duration > 0
        assert tracker.duration > 0

    def test_operation_tracker_unknown_operation_falls_back_to_other(self) -> None:
        collector = MetricsCollector()
        with collector.track("unknown_op"):
            pass
        output = collector.to_prometheus()
        assert "kemi_other_latency_seconds" in output
        assert collector.other_latency._count == 1

    def test_per_instance_collector_isolation(self) -> None:
        c1 = MetricsCollector()
        c2 = MetricsCollector()
        c1.remember_total.inc(5)
        assert c1.remember_total.value() == 5
        assert c2.remember_total.value() == 0

    def test_operation_tracker_exit_on_exception(self) -> None:
        collector = MetricsCollector()
        try:
            with collector.track("remember"):
                raise ValueError("boom")
        except ValueError:
            pass
        # Latency should still be recorded despite exception
        output = collector.to_prometheus()
        assert "kemi_remember_latency_seconds" in output
        assert collector.remember_latency._count == 1

    def test_histogram_custom_buckets(self) -> None:
        h = Histogram("custom_hist", "Custom buckets", buckets=(0.01, 0.1, 1.0))
        h.observe(0.05)
        output = h.to_prometheus()
        assert 'le="0.01"' in output
        assert 'le="0.1"' in output
        assert 'le="1.0"' in output
        assert 'le="+Inf"' in output

    def test_histogram_high_values(self) -> None:
        h = Histogram("high_hist", "High values", buckets=(1.0, 10.0))
        h.observe(100.0)
        output = h.to_prometheus()
        assert 'le="1.0"' in output
        assert 'le="10.0"' in output
        assert 'le="+Inf"' in output

    def test_counter_negative_start(self) -> None:
        c = Counter("neg_counter", "Counter")
        c.inc(-1)
        assert c.value() == -1

    def test_gauge_negative_value(self) -> None:
        g = Gauge("neg_gauge", "Gauge")
        g.set(-5.0)
        assert g.value() == -5.0
        g.dec(3.0)
        assert g.value() == -8.0

    def test_record_operation_concurrent(self) -> None:
        collector = MetricsCollector()
        errors: list[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)

        def worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(500):
                    collector.record_operation("remember", duration=0.05, success=True)
                    collector.record_operation("unknown_op", duration=0.1, success=True)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Exceptions during concurrent access: {errors}"
        assert collector.remember_total.value() == 5000
        assert collector.remember_latency._count == 5000
        assert collector.other_latency._count == 5000

    def test_track_concurrent_with_exceptions(self) -> None:
        """track() records latency even when the body raises under concurrent load."""
        collector = MetricsCollector()
        errors: list[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)
        success_count: int = 0
        exception_count: int = 0

        def worker(worker_id: int) -> None:
            nonlocal success_count, exception_count
            try:
                barrier.wait(timeout=5.0)
                for i in range(250):
                    if (worker_id + i) % 3 == 0:
                        try:
                            with collector.track("remember"):
                                raise ValueError("intentional")
                        except ValueError:
                            with lock:
                                exception_count += 1
                    else:
                        with collector.track("remember"):
                            pass
                        with lock:
                            success_count += 1
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(wid,)) for wid in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Exceptions during concurrent track: {errors}"
        total_blocks = success_count + exception_count
        assert total_blocks == 2500
        # Every block—success or exception—should have recorded latency.
        assert collector.remember_latency._count == 2500
        assert collector.remember_total.value() == 0  # track() does not increment counters

    def test_record_operation_concurrent_with_mixed_ops(self) -> None:
        """record_operation() is safe with a mix of known/unknown ops and optional duration."""
        collector = MetricsCollector()
        errors: list[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)

        def worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for i in range(500):
                    # known op with duration
                    collector.record_operation("remember", duration=0.01, success=True)
                    # known op without duration
                    collector.record_operation("recall", success=True)
                    # unknown op with duration → falls back to other_latency
                    collector.record_operation(f"unknown_{i}", duration=0.02, success=True)
                    # unknown op without duration → silently ignored
                    collector.record_operation(f"noop_{i}", success=True)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"Exceptions during mixed concurrent record_operation: {errors}"
        assert collector.remember_total.value() == 5000
        assert collector.remember_latency._count == 5000
        assert collector.recall_total.value() == 5000
        # recall has no duration, so its histogram should be untouched
        assert collector.recall_latency._count == 0
        assert collector.other_latency._count == 5000

    def test_reset_metrics_global(self) -> None:
        reset_metrics()
        c1 = get_metrics_collector()
        c1.remember_total.inc(5)
        reset_metrics()
        c2 = get_metrics_collector()
        assert c1 is not c2
        assert c2.remember_total.value() == 0


class TestConcurrentReset:
    """Stress-test reset() under concurrent mutations.

    reset() acquires each metric's individual lock, so it must never
    observe torn state or deadlock when other threads are actively
    incrementing, observing, or exporting.
    """

    def _run_workers(self, worker_fn, num_threads=10, timeout=30.0) -> list[Exception]:
        errors: list[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def wrapper() -> None:
            try:
                barrier.wait(timeout=5.0)
                worker_fn()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=wrapper) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

        return errors

    def test_reset_during_record_operations(self) -> None:
        """reset() races with record_operation() across all metric types."""
        collector = MetricsCollector()

        def worker() -> None:
            for _ in range(500):
                collector.record_operation("remember", duration=0.01, success=True)
                collector.record_operation("recall", duration=0.02, success=True)
                collector.embed_total.inc()
                collector.total_memories.set(42.0)
                collector.total_users.set(7.0)
                if _ % 50 == 0:
                    collector.reset()

        errors = self._run_workers(worker)
        assert not errors, f"Exceptions during concurrent reset/mutation: {errors}"
        # After a final reset the state must be exactly zero.
        collector.reset()
        assert collector.remember_total.value() == 0
        assert collector.recall_total.value() == 0
        assert collector.embed_total.value() == 0
        assert collector.remember_latency._count == 0
        assert collector.recall_latency._count == 0
        assert collector.total_memories.value() == 0.0
        assert collector.total_users.value() == 0.0

    def test_reset_during_prometheus_export(self) -> None:
        """to_prometheus() and reset() run concurrently without crashing."""
        collector = MetricsCollector()

        def worker() -> None:
            for i in range(500):
                collector.remember_total.inc()
                collector.remember_latency.observe(0.01)
                collector.total_memories.set(float(i))
                if i % 25 == 0:
                    collector.reset()
                _ = collector.to_prometheus()

        errors = self._run_workers(worker)
        assert not errors, f"Exceptions during concurrent reset/prometheus: {errors}"

    def test_reset_during_dict_export(self) -> None:
        """to_dict() and reset() run concurrently without crashing."""
        collector = MetricsCollector()

        def worker() -> None:
            for i in range(500):
                collector.recall_total.inc()
                collector.recall_latency.observe(0.02)
                collector.total_users.set(float(i))
                if i % 25 == 0:
                    collector.reset()
                _ = collector.to_dict()

        errors = self._run_workers(worker)
        assert not errors, f"Exceptions during concurrent reset/dict: {errors}"

    def test_reset_then_mutations_resume_cleanly(self) -> None:
        """After reset(), subsequent mutations produce correct values."""
        collector = MetricsCollector()
        collector.remember_total.inc(100)
        collector.remember_latency.observe(1.0)
        collector.total_memories.set(500.0)

        collector.reset()

        assert collector.remember_total.value() == 0
        assert collector.remember_latency._count == 0
        assert collector.total_memories.value() == 0.0

        collector.remember_total.inc(5)
        collector.remember_latency.observe(0.1)
        collector.total_memories.set(10.0)

        assert collector.remember_total.value() == 5
        assert collector.remember_latency._count == 1
        assert collector.total_memories.value() == 10.0

    def test_repeated_reset_is_idempotent(self) -> None:
        """Multiple consecutive reset() calls leave metrics at zero."""
        collector = MetricsCollector()
        collector.remember_total.inc(50)
        collector.embed_errors_total.inc(3)
        collector.remember_latency.observe(0.5)
        collector.total_users.set(25.0)

        for _ in range(10):
            collector.reset()

        assert collector.remember_total.value() == 0
        assert collector.embed_errors_total.value() == 0
        assert collector.remember_latency._count == 0
        assert collector.total_users.value() == 0.0


class TestConcurrentPrometheusExport:
    """Stress-test to_prometheus() under concurrent mutations.

    The _lock is acquired while reading values for export, so
    to_prometheus() must never observe torn or corrupted state.
    """

    def _run_workers(
                        self,
                        worker_fn,
                        num_threads=10,
                        iterations=500,
                        timeout=30.0,
                    ) -> tuple[list[Exception], list[str]]:
        """Spawn threads that execute worker_fn and collect errors/outputs safely."""
        errors: list[Exception] = []
        outputs: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def wrapper() -> None:
            try:
                barrier.wait(timeout=5.0)
                worker_fn(lock, outputs)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=wrapper) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

        return errors, outputs

    def test_counter_concurrent_inc_and_prometheus(self) -> None:
        c = Counter("concurrent_counter", "Concurrent counter")

        def worker(lock, outputs):
            for _ in range(500):
                c.inc()
                with lock:
                    outputs.append(c.to_prometheus())

        errors, outputs = self._run_workers(worker)
        assert not errors, f"Exceptions during concurrent access: {errors}"
        assert c.value() == 5000
        for out in outputs:
            assert "kemi_concurrent_counter" in out
            assert "counter" in out

    def test_histogram_concurrent_observe_and_prometheus(self) -> None:
        h = Histogram("concurrent_hist", "Concurrent histogram")

        def worker(lock, outputs):
            for i in range(500):
                h.observe(float(i % 10) / 1000.0)
                with lock:
                    outputs.append(h.to_prometheus())

        errors, outputs = self._run_workers(worker)
        assert not errors, f"Exceptions during concurrent access: {errors}"
        assert h._count == 5000
        for out in outputs:
            assert "kemi_concurrent_hist" in out
            assert "histogram" in out
            assert "_count" in out
            assert "_sum" in out

    def test_gauge_concurrent_set_and_prometheus(self) -> None:
        g = Gauge("concurrent_gauge", "Concurrent gauge")

        def worker(lock, outputs):
            for i in range(500):
                g.set(float(i))
                with lock:
                    outputs.append(g.to_prometheus())

        errors, outputs = self._run_workers(worker)
        assert not errors, f"Exceptions during concurrent access: {errors}"
        assert 0.0 <= g.value() < 500.0
        for out in outputs:
            assert "kemi_concurrent_gauge" in out
            assert "gauge" in out

    def test_metrics_collector_concurrent_record_and_prometheus(self) -> None:
        collector = MetricsCollector()

        def worker(lock, outputs):
            for i in range(200):
                collector.remember_total.inc()
                collector.remember_latency.observe(0.01 + (i % 10) / 1000.0)
                collector.total_memories.set(float(i))
                with lock:
                    outputs.append(collector.to_prometheus())

        errors, outputs = self._run_workers(worker, iterations=200)
        assert not errors, f"Exceptions during concurrent access: {errors}"
        assert collector.remember_total.value() == 2000
        assert collector.remember_latency._count == 2000
        for out in outputs:
            assert "kemi_remember_total" in out
            assert "kemi_remember_latency_seconds" in out
            assert "kemi_total_memories" in out


