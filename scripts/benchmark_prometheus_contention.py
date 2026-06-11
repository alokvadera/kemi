#!/usr/bin/env python3
"""Benchmark MetricsCollector.to_prometheus() throughput under contention.

Measures how many Prometheus text-format exports per second the
MetricsCollector can sustain while N writer threads continuously
mutate counters, histograms, and gauges.

Scenarios:
  1. Baseline: export with no writers (single-threaded).
  2. Light contention: 4 writer threads + 1 exporter thread.
  3. Heavy contention: 16 writer threads + 1 exporter thread.
  4. Extreme contention: 64 writer threads + 1 exporter thread.

Metrics reported:
  - Exports per second (eps)
  - Median export latency (ms)
  - p99 export latency (ms)
  - Writer mutations per second (total across all writers)

Usage:
    uv run python scripts/benchmark_prometheus_contention.py
"""

import gc
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi.infra.observability import MetricsCollector

# ── Configuration ───────────────────────────────────────────────────
EXPORT_DURATION_S = 5.0        # How long each benchmark phase runs
WARMUP_EXPORTS = 100           # Warm-up exports before measuring
SCALES = [0, 4, 16, 64]        # Number of concurrent writer threads

RESULTS_FILE = Path(__file__).resolve().parent / "benchmark_prometheus_contention_results.json"

# ── Helpers ──────────────────────────────────────────────────────────


def _writer_loop(collector: MetricsCollector, stop_event: threading.Event) -> int:
    """Continuously mutate metrics until stop_event is set.

    Returns the total number of mutations performed.
    """
    mutations = 0
    ops = [
        ("remember", lambda: collector.record_operation("remember", duration=0.05)),
        ("recall", lambda: collector.record_operation("recall", duration=0.03)),
        ("embed", lambda: collector.record_operation("embed", duration=0.01)),
        ("gauge", lambda: collector.total_memories.inc(1.0)),
        ("gauge_dec", lambda: collector.total_users.dec(1.0)),
        ("histogram", lambda: collector.embed_latency.observe(0.015)),
        ("counter", lambda: collector.embed_errors_total.inc(1)),
    ]
    idx = 0
    while not stop_event.is_set():
        _name, fn = ops[idx % len(ops)]
        fn()
        mutations += 1
        idx += 1
    return mutations


def _benchmark_phase(
    collector: MetricsCollector,
    num_writers: int,
    duration_s: float,
) -> dict[str, Any]:
    """Run one benchmark phase and return collected stats."""
    stop_event = threading.Event()
    writer_mutations = [0]
    writer_lock = threading.Lock()

    def _wrapped_writer() -> None:
        count = _writer_loop(collector, stop_event)
        with writer_lock:
            writer_mutations[0] += count

    # Start writers
    writers: list[threading.Thread] = []
    for _ in range(num_writers):
        t = threading.Thread(target=_wrapped_writer, daemon=True)
        t.start()
        writers.append(t)

    # Warm-up (do exports while writers are already running)
    for _ in range(WARMUP_EXPORTS):
        collector.to_prometheus()

    # Measurement phase
    latencies_ms: list[float] = []
    export_count = 0
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        collector.to_prometheus()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        export_count += 1

    # Stop writers
    stop_event.set()
    for t in writers:
        t.join(timeout=2.0)

    # Compute stats
    latencies_ms.sort()
    n = len(latencies_ms)
    median_ms = statistics.median(latencies_ms) if n else 0.0
    p99_ms = latencies_ms[int((n - 1) * 0.99)] if n else 0.0
    eps = export_count / duration_s if duration_s > 0 else 0.0
    writer_eps = writer_mutations[0] / duration_s if duration_s > 0 else 0.0

    return {
        "writers": num_writers,
        "exports": export_count,
        "duration_s": duration_s,
        "eps": round(eps, 2),
        "median_ms": round(median_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "writer_mutations": writer_mutations[0],
        "writer_eps": round(writer_eps, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 74)
    print("  MetricsCollector.to_prometheus() Contention Benchmark")
    print("=" * 74)
    print(f"  Export duration: {EXPORT_DURATION_S}s per phase")
    print(f"  Warm-up exports: {WARMUP_EXPORTS}")
    print(f"  Writer scales:  {SCALES}")
    print()

    results: list[dict[str, Any]] = []

    for num_writers in SCALES:
        label = f"{num_writers} writer{'s' if num_writers != 1 else ''}"
        print(f"  ─── Phase: {label} ───")

        collector = MetricsCollector()
        gc.collect()

        stats = _benchmark_phase(collector, num_writers, EXPORT_DURATION_S)
        results.append(stats)

        print(f"    Exports/sec:        {stats['eps']:>10.2f}")
        print(f"    Median latency:     {stats['median_ms']:>10.4f} ms")
        print(f"    p99 latency:        {stats['p99_ms']:>10.4f} ms")
        print(f"    Writer mutations:   {stats['writer_mutations']:>10,}")
        print(f"    Writer mutations/s: {stats['writer_eps']:>10.2f}")
        print()

    # Summary table
    print("  " + "=" * 74)
    h = (
        f"  {'Writers':>8} | {'Exports':>10} | {'EPS':>10} | "
        f"{'Median ms':>10} | {'p99 ms':>10} | {'Writer mut/s':>14}"
    )
    print(h)
    print("  " + "-" * 74)
    for r in results:
        print(
            f"  {r['writers']:>8} | {r['exports']:>10} | "
            f"{r['eps']:>10.2f} | {r['median_ms']:>10.4f} | "
            f"{r['p99_ms']:>10.4f} | {r['writer_eps']:>14.2f}"
        )
    print("  " + "-" * 74)

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(
            {
                "config": {
                    "export_duration_s": EXPORT_DURATION_S,
                    "warmup_exports": WARMUP_EXPORTS,
                    "scales": SCALES,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
