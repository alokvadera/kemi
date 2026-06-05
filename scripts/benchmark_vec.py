#!/usr/bin/env python3
"""Benchmark ANN (sqlite-vec) vs brute-force vector search across scales.

Runs at multiple vector counts (384-dim) and measures:
  - Insert time (bulk)
  - Query latency (median of 10 runs)
  - Recall@10 (ANN accuracy vs brute-force ground truth)

Uses :memory: databases to avoid file descriptor limits from
per-store-connection overhead.

Saves graph to scripts/benchmark_results.png
"""

import gc
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter, _SQLITE_VEC_AVAILABLE
from kemi.models import LifecycleState, MemoryObject, MemorySource

DIM = 384
SCALES = [1000, 5000, 10000, 25000, 50000]
QUERIES_PER_SCALE = 10
TOP_K = 10
RNG = random.Random(42)


def make_memory(memory_id, user_id, embedding):
    return MemoryObject(
        memory_id=memory_id,
        user_id=user_id,
        content=f"Memory {memory_id}",
        embedding=embedding,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=DIM,
        tags=[],
    )


def random_embedding():
    return [RNG.random() for _ in range(DIM)]


def benchmark_adapter(adapter_cls, label, scale, embeddings, query_embs):
    """Benchmark insert + query for one adapter at one scale using :memory: DB."""
    adapter_args = {"db_path": ":memory:"}
    if issubclass(adapter_cls, SQLiteVecStorageAdapter):
        adapter_args["embedding_dim"] = DIM

    adapter = adapter_cls(**adapter_args)
    user_id = "bench"

    # Warmup
    for i in range(min(3, len(embeddings))):
        adapter.store(make_memory(f"warmup-{i}", user_id, embeddings[i]))
    adapter.delete_by_user(user_id)

    # Bulk insert
    mems = [make_memory(f"mem-{label}-{scale}-{i}", user_id, emb)
            for i, emb in enumerate(embeddings)]

    gc.collect()
    t0 = time.perf_counter()
    for m in mems:
        adapter.store(m)
    t1 = time.perf_counter()
    insert_time = t1 - t0

    # Query
    query_times = []
    for emb in query_embs:
        t0 = time.perf_counter()
        adapter.search(user_id, emb, top_k=TOP_K)
        t1 = time.perf_counter()
        query_times.append((t1 - t0) * 1000)

    adapter.close()
    return insert_time, query_times


def benchmark_bruteforce_python(embeddings, query_embs):
    """Pure Python cosine similarity scan — measures just the math."""
    query_times = []
    for qemb in query_embs:
        t0 = time.perf_counter()
        scores = []
        for emb in embeddings:
            dot = sum(a * b for a, b in zip(emb, qemb))
            n1 = math.sqrt(sum(a * a for a in emb))
            n2 = math.sqrt(sum(b * b for b in qemb))
            sim = dot / (n1 * n2) if n1 * n2 > 0 else 0
            scores.append(sim)
        scores.sort(reverse=True)
        _ = scores[:TOP_K]
        t1 = time.perf_counter()
        query_times.append((t1 - t0) * 1000)
    return query_times


def compute_recall(adapter_cls, label, scale, embeddings, query_embs):
    """Recall@10: what fraction of true top-10 are in ANN top-10."""
    adapter_args = {"db_path": ":memory:"}
    if issubclass(adapter_cls, SQLiteVecStorageAdapter):
        adapter_args["embedding_dim"] = DIM

    adapter = adapter_cls(**adapter_args)
    user_id = "bench"
    for i, emb in enumerate(embeddings):
        adapter.store(make_memory(f"mem-{i}", user_id, emb))

    recalls = []
    for qemb in query_embs:
        ann_results = adapter.search(user_id, qemb, top_k=TOP_K)
        ann_ids = set(r.memory_id for r in ann_results)

        scores = []
        for i, emb in enumerate(embeddings):
            dot = sum(a * b for a, b in zip(emb, qemb))
            n1 = math.sqrt(sum(a * a for a in emb))
            n2 = math.sqrt(sum(b * b for b in qemb))
            sim = dot / (n1 * n2) if n1 * n2 > 0 else 0
            scores.append((sim, i))
        scores.sort(key=lambda x: -x[0])
        truth_ids = set(f"mem-{idx}" for _, idx in scores[:TOP_K])

        overlap = len(ann_ids & truth_ids)
        recalls.append(overlap / TOP_K)

    adapter.close()
    return sum(recalls) / len(recalls)


def plot_results(results_brute, results_vec, recalls, python_times):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping graph")
        return

    scales_str = [str(s) for s in SCALES]
    x = np.arange(len(scales_str))
    width = 0.25

    brute_medians = [r["query_median_ms"] for r in results_brute]
    vec_medians = [r["query_median_ms"] for r in results_vec]
    python_medians = list(python_times)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Plot 1: Query latency ──
    ax = axes[0]
    bars1 = ax.bar(x - width, brute_medians, width,
                   label="Brute-force (SQLite load+scan)", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x, vec_medians, width,
                   label="ANN (sqlite-vec HNSW)", color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x + width, python_medians, width,
                   label="Pure Python (no SQLite)", color="#f39c12", alpha=0.85)
    ax.set_ylabel("Query Latency (ms, lower is better)")
    ax.set_xlabel("Number of Vectors")
    ax.set_title("Search Latency: ANN vs Brute-Force")
    ax.set_xticks(x)
    ax.set_xticklabels(scales_str)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, brute_medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)
    for bar, val in zip(bars2, vec_medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)
    for bar, val in zip(bars3, python_medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

    # ── Plot 2: Recall@10 ──
    ax = axes[1]
    recall_vals = [r * 100 for r in recalls]
    bars = ax.bar(scales_str, recall_vals, color="#3498db", alpha=0.85)
    ax.set_ylabel("Recall@10 (%)")
    ax.set_xlabel("Number of Vectors")
    ax.set_title("ANN Accuracy vs Brute-Force Ground Truth")
    ax.set_ylim(90, 100.5)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, recall_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # ── Plot 3: Insert time ──
    ax = axes[2]
    brute_insert = [r["insert_time_s"] for r in results_brute]
    vec_insert = [r["insert_time_s"] for r in results_vec]
    bars1 = ax.bar(x - width / 2, brute_insert, width,
                   label="Brute-force", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, vec_insert, width,
                   label="ANN", color="#2ecc71", alpha=0.85)
    ax.set_ylabel("Insert Time (seconds)")
    ax.set_xlabel("Number of Vectors")
    ax.set_title("Bulk Insert Time")
    ax.set_xticks(x)
    ax.set_xticklabels(scales_str)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, brute_insert):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7, rotation=45)
    for bar, val in zip(bars2, vec_insert):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7, rotation=45)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved to {out_path}")
    plt.close()


def main():
    if not _SQLITE_VEC_AVAILABLE:
        print("ERROR: sqlite-vec must be installed to run benchmarks.")
        print("Run: pip install sqlite-vec")
        sys.exit(1)

    print("=" * 70)
    print("  Vector Search Benchmark: ANN (sqlite-vec) vs Brute-Force")
    print(f"  Dimension: {DIM}  |  Top-K: {TOP_K}  |  Queries per scale: {QUERIES_PER_SCALE}")
    print(f"  Hardware: in-memory DB (no disk I/O overhead)")
    print("=" * 70)

    results_brute = []
    results_vec = []
    all_recalls = []
    all_python_times = []

    for scale in SCALES:
        print(f"\n--- Scale: {scale} vectors ---")

        # Generate data once per scale
        embeddings = [random_embedding() for _ in range(scale)]
        q_rng = random.Random(99)
        query_embs = [[q_rng.random() for _ in range(DIM)]
                      for _ in range(QUERIES_PER_SCALE)]

        # 1) Pure Python brute-force
        print(f"  Pure Python scan... ", end="", flush=True)
        py_times = benchmark_bruteforce_python(embeddings, query_embs)
        py_med = sorted(py_times)[len(py_times) // 2]
        all_python_times.append(py_med)
        print(f"median {py_med:.2f}ms")

        # 2) Brute-force via SQLiteStorageAdapter
        print(f"  SQLite brute-force... ", end="", flush=True)
        ins_t, q_times = benchmark_adapter(
            SQLiteStorageAdapter, "brute", scale, embeddings, query_embs)
        q_med = sorted(q_times)[len(q_times) // 2]
        results_brute.append({
            "scale": scale,
            "insert_time_s": ins_t,
            "query_median_ms": q_med,
        })
        print(f"insert={ins_t:.2f}s  query={q_med:.2f}ms")

        # 3) ANN via SQLiteVecStorageAdapter
        print(f"  SQLite-vec ANN... ", end="", flush=True)
        ins2_t, q2_times = benchmark_adapter(
            SQLiteVecStorageAdapter, "vec", scale, embeddings, query_embs)
        q2_med = sorted(q2_times)[len(q2_times) // 2]
        results_vec.append({
            "scale": scale,
            "insert_time_s": ins2_t,
            "query_median_ms": q2_med,
        })
        speedup = q_med / q2_med if q2_med > 0 else float("inf")
        print(f"insert={ins2_t:.2f}s  query={q2_med:.2f}ms")

        # 4) Recall@10
        print(f"  Recall@10... ", end="", flush=True)
        recall = compute_recall(
            SQLiteVecStorageAdapter, "vec", scale, embeddings, query_embs[:5])
        all_recalls.append(recall)
        print(f"{recall*100:.1f}%")

        # Per-scale summary row
        print(f"  {'─' * 50}")
        print(f"  │ {scale:>6} | brute={q_med:>8.2f}ms | ann={q2_med:>8.2f}ms "
              f"| {speedup:>6.1f}x | recall={recall*100:>5.1f}% |")
        print(f"  {'─' * 50}")

    # ── Final results table ──
    print("\n")
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Scale':>8} | {'Brute (ms)':>12} | {'ANN (ms)':>12} | "
          f"{'Speedup':>10} | {'Recall@10':>10} | {'Py-only(ms)':>14}")
    print("  " + "-" * 72)
    for i, scale in enumerate(SCALES):
        b = results_brute[i]
        v = results_vec[i]
        r = all_recalls[i]
        py = all_python_times[i]
        sp = b["query_median_ms"] / v["query_median_ms"] if v["query_median_ms"] > 0 else float("inf")
        print(f"  {scale:>8} | {b['query_median_ms']:>10.2f}ms | "
              f"{v['query_median_ms']:>10.2f}ms | {sp:>8.1f}x | "
              f"{r*100:>8.1f}% | {py:>12.2f}ms")
    print("  " + "-" * 72)

    # Insert times
    print(f"\n  {'Scale':>8} | {'Brute Insert':>15} | {'ANN Insert':>15}")
    print("  " + "-" * 42)
    for i, scale in enumerate(SCALES):
        print(f"  {scale:>8} | {results_brute[i]['insert_time_s']:>13.2f}s | "
              f"{results_vec[i]['insert_time_s']:>13.2f}s")

    # Generate graph
    plot_results(results_brute, results_vec, all_recalls, all_python_times)

    # Save raw data
    data = {
        "scales": SCALES,
        "dim": DIM,
        "top_k": TOP_K,
        "brute_force": results_brute,
        "ann_vec": results_vec,
        "recall_at_10": [r * 100 for r in all_recalls],
        "pure_python_median_ms": all_python_times,
    }
    data_path = Path(__file__).resolve().parent / "benchmark_data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nRaw data saved to {data_path}")


if __name__ == "__main__":
    main()
