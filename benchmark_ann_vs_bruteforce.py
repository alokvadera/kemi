"""Benchmark: sqlite-vec ANN search vs brute-force cosine similarity.

Generates a CSV + terminal table + optional ASCII plot at each scale.
"""

import contextlib
import io
import os
import random
import sys
import tempfile
import time

import numpy as np

# Disable coverage clutter in output
os.environ["COV_CORE_SOURCE"] = ""
os.environ["COVERAGE_FILE"] = "/dev/null"

from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter, _SQLITE_VEC_AVAILABLE
from kemi.models import LifecycleState, MemoryObject, MemorySource
from datetime import datetime, timezone

DIM = 384
SCALES = [100, 1_000, 10_000, 50_000, 100_000]
TRIALS = 5  # queries per scale

random.seed(42)
np.random.seed(42)


def make_memory(i: int, user_id: str = "benchmark_user") -> MemoryObject:
    emb = [random.random() for _ in range(DIM)]
    return MemoryObject(
        memory_id=f"mem-{user_id}-{i}",
        user_id=user_id,
        content=f"Benchmark memory #{i} with some filler text to make realistic content length.",
        embedding=emb,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=random.random(),
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=DIM,
        tags=[],
    )


def populate(adapter, n: int, user_id: str = "benchmark_user"):
    for i in range(n):
        adapter.store(make_memory(i, user_id))
    adapter.close()


@contextlib.contextmanager
def bench_db(adapter_cls, n: int, user_id: str = "benchmark_user"):
    tmp = tempfile.mktemp(suffix=".db")
    kwargs: dict = {"db_path": tmp}
    if adapter_cls is SQLiteVecStorageAdapter:
        kwargs["embedding_dim"] = DIM
    adapter = adapter_cls(**kwargs)
    try:
        populate(adapter, n, user_id)
        if adapter_cls is SQLiteVecStorageAdapter:
            adapter = adapter_cls(**kwargs)
        else:
            adapter = adapter_cls(**kwargs)
        yield adapter
    finally:
        adapter.close()
        if os.path.exists(tmp):
            os.unlink(tmp)


def measure(adapter, user_id: str, trials: int = TRIALS) -> dict:
    latencies = []
    for _ in range(trials):
        query = [random.random() for _ in range(DIM)]
        start = time.perf_counter()
        results = adapter.search(user_id, query, top_k=10)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

    if not latencies:
        return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    latencies.sort()
    return {
        "mean_ms": round(np.mean(latencies), 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
        "p50_ms": round(np.median(latencies), 3),
        "p95_ms": round(np.percentile(latencies, 95), 3),
    }


def print_table(results: list[dict]):
    header = f"{'Scale':>10} | {'Method':>10} | {'Mean(ms)':>10} | {'Min(ms)':>8} | {'Max(ms)':>8} | {'P50(ms)':>8} | {'P95(ms)':>8} | {'Speedup':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        speedup = r.get("speedup", "-")
        print(
            f"{r['scale']:>10} | {r['method']:>10} | {r['mean_ms']:>10.3f} | {r['min_ms']:>8.3f} | {r['max_ms']:>8.3f} | {r['p50_ms']:>8.3f} | {r['p95_ms']:>8.3f} | {str(speedup):>8}"
        )
    print(sep)


def ascii_bar_chart(results: list[dict], max_width: int = 50):
    """Simple ASCII bar chart of mean latencies."""
    brute_entries = [r for r in results if r["method"] == "BruteForce"]
    ann_entries = [r for r in results if r["method"] == "ANN"]

    scales = [r["scale"] for r in brute_entries]
    brute_vals = [r["mean_ms"] for r in brute_entries]
    ann_vals = [r["mean_ms"] for r in ann_entries]

    max_val = max(max(brute_vals), max(ann_vals))
    if max_val == 0:
        return

    print("\n--- ASCII Bar Chart (Mean Latency) ---")
    print(f"{'Scale':>10} | {'BruteForce':<{max_width+10}} | {'ANN':<{max_width+10}}")
    print("-" * (max_width * 2 + 30))
    for i, scale in enumerate(scales):
        b_val = brute_vals[i]
        a_val = ann_vals[i]
        b_bar_len = int((b_val / max_val) * max_width) if max_val > 0 else 0
        a_bar_len = int((a_val / max_val) * max_width) if max_val > 0 else 0
        b_bar = "█" * max(1, b_bar_len) if b_val > 0 else ""
        a_bar = "█" * max(1, a_bar_len) if a_val > 0 else ""
        print(
            f"{scale:>10} | {b_val:>8.3f}ms {b_bar:<{max_width}} | {a_val:>8.3f}ms {a_bar:<{max_width}}"
        )


def main():
    print(f"sqlite-vec available: {_SQLITE_VEC_AVAILABLE}")
    print(f"Embedding dim: {DIM}")
    print(f"Trials per scale: {TRIALS}")
    print()

    all_results = []

    for scale in SCALES:
        user_id = f"user_{scale}"

        print(f"\n--- Scale: {scale:,} memories ---")

        # --- Brute Force ---
        with bench_db(SQLiteStorageAdapter, scale, user_id) as adapter:
            bf = measure(adapter, user_id)

        # --- ANN ---
        with bench_db(SQLiteVecStorageAdapter, scale, user_id) as adapter:
            ann = measure(adapter, user_id)

        speedup = round(bf["mean_ms"] / ann["mean_ms"], 1) if ann["mean_ms"] > 0 else float("inf")

        bf["scale"] = scale
        bf["method"] = "BruteForce"
        ann["scale"] = scale
        ann["method"] = "ANN"
        ann["speedup"] = speedup

        all_results.append(bf)
        all_results.append(ann)

        print(f"  BruteForce: {bf['mean_ms']:.3f}ms (p50={bf['p50_ms']:.3f}ms, p95={bf['p95_ms']:.3f}ms)")
        print(f"  ANN:        {ann['mean_ms']:.3f}ms (p50={ann['p50_ms']:.3f}ms, p95={ann['p95_ms']:.3f}ms)")
        print(f"  Speedup:    {speedup}x")

    print("\n\n### ALL RESULTS ###\n")
    print_table(all_results)

    ascii_bar_chart(all_results)

    # Also print as CSV
    print("\n### CSV ###\n")
    print("scale,method,mean_ms,min_ms,max_ms,p50_ms,p95_ms,speedup")
    for r in all_results:
        speedup = r.get("speedup", "")
        print(f"{r['scale']},{r['method']},{r['mean_ms']},{r['min_ms']},{r['max_ms']},{r['p50_ms']},{r['p95_ms']},{speedup}")


if __name__ == "__main__":
    main()
