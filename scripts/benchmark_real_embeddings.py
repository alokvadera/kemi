#!/usr/bin/env python3
"""Benchmark ANN (sqlite-vec) vs brute-force using real fastembed embeddings.

Loads 1K real embeddings from cache + realistic synthetic embeddings at
5K and 10K scales. Measures query latency and recall@10 at each scale.
Generates a comparison graph.
"""

import gc
import json
import math
import pickle
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter, _SQLITE_VEC_AVAILABLE
from kemi.models import LifecycleState, MemoryObject, MemorySource

DIM = 384
SCALES = [1000, 5000, 10000]
TOP_K = 10
QUERIES_PER_SCALE = 10


def make_memory(memory_id: str, user_id: str, embedding: list[float]):
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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na * nb > 0 else 0


def brute_force_ground_truth(embeddings: list[list[float]], query: list[float],
                              top_k: int) -> set[str]:
    scores = [(cosine_similarity(e, query), i) for i, e in enumerate(embeddings)]
    scores.sort(key=lambda x: -x[0])
    return {f"mem-{idx}" for _, idx in scores[:top_k]}


def benchmark_adapter(adapter_cls, embeddings: list[list[float]],
                      query_embs: list[list[float]]):
    """Benchmark insert + query for one adapter at one scale."""
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
    mems = [make_memory(f"mem-{i}", user_id, emb) for i, emb in enumerate(embeddings)]
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

    # Recall: compare ANN results to brute-force ground truth
    recalls = []
    for qemb in query_embs:
        ann_results = adapter.search(user_id, qemb, top_k=TOP_K)
        ann_ids = set(r.memory_id for r in ann_results)
        truth_ids = brute_force_ground_truth(embeddings, qemb, TOP_K)
        overlap = len(ann_ids & truth_ids)
        recalls.append(overlap / TOP_K)

    adapter.close()
    return insert_time, query_times, recalls


def main():
    if not _SQLITE_VEC_AVAILABLE:
        print("ERROR: sqlite-vec must be installed.")
        sys.exit(1)

    cache_path = Path(__file__).resolve().parent / "real_embeddings_cache.pkl"
    if not cache_path.exists():
        print(f"ERROR: cache not found at {cache_path}")
        print("Run scripts/generate_real_embeddings.py first.")
        sys.exit(1)

    print("Loading cached embeddings...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    real_embs = cache["real_embeddings"]
    real_queries = cache["query_embeddings_real"][:QUERIES_PER_SCALE]
    synthetic = cache["synthetic"]
    syn_queries = cache["query_embeddings_synthetic"][:QUERIES_PER_SCALE]

    structure = cache["structure"]
    print(f"  Real embeddings: {len(real_embs)} ({structure['intra_inter_ratio']:.2f}x intra/inter ratio)")
    print(f"  Synthetic scales: {list(synthetic.keys())}")
    print()

    print("=" * 72)
    print("  REAL EMBEDDING BENCHMARK: ANN (sqlite-vec) vs Brute-Force")
    print(f"  Model: BAAI/bge-small-en-v1.5 ({DIM}-dim)  |  Top-K: {TOP_K}")
    print("=" * 72)

    results = {"scales": [], "brute_force": [], "ann_vec": [], "recall_at_10": [], "speedup": []}

    for scale in SCALES:
        scale_str = str(scale)
        if scale == 1000:
            embeddings = real_embs[:scale]
            query_embs = real_queries
            label = "real"
        else:
            embeddings = synthetic[scale_str]
            query_embs = syn_queries
            label = "synthetic"

        print(f"\n  ─── Scale: {scale:,} vectors ({label}) ───")

        # 1) Brute-force
        print(f"    Brute-force (SQLiteStorageAdapter)... ", end="", flush=True)
        ins_b, q_times_b, _ = benchmark_adapter(
            SQLiteStorageAdapter, embeddings, query_embs)
        q_b_med = statistics.median(q_times_b)
        print(f"insert={ins_b:.3f}s  query={q_b_med:.2f}ms")

        # 2) ANN
        print(f"    ANN (SQLiteVecStorageAdapter)...      ", end="", flush=True)
        ins_v, q_times_v, recalls = benchmark_adapter(
            SQLiteVecStorageAdapter, embeddings, query_embs)
        q_v_med = statistics.median(q_times_v)
        recall_mean = statistics.mean(recalls)
        speedup = q_b_med / q_v_med if q_v_med > 0 else float("inf")
        print(f"insert={ins_v:.3f}s  query={q_v_med:.2f}ms  recall@{TOP_K}={recall_mean*100:.1f}%")

        # 3) Pure Python brute-force
        print(f"    Pure Python scan...                   ", end="", flush=True)
        py_times = []
        for qemb in query_embs:
            t0 = time.perf_counter()
            _ = brute_force_ground_truth(embeddings, qemb, TOP_K)
            t1 = time.perf_counter()
            py_times.append((t1 - t0) * 1000)
        py_med = statistics.median(py_times)
        print(f"query={py_med:.2f}ms")

        # Summary row
        print(f"    {'─' * 50}")
        print(f"    │ {scale:>6} ({label:>9}) | brute={q_b_med:>8.2f}ms | "
              f"ann={q_v_med:>8.2f}ms | {speedup:>6.1f}x | recall={recall_mean*100:>5.1f}% │")
        print(f"    {'─' * 50}")

        results["scales"].append(scale)
        results["brute_force"].append({
            "insert_s": round(ins_b, 3), "query_ms": round(q_b_med, 2)})
        results["ann_vec"].append({
            "insert_s": round(ins_v, 3), "query_ms": round(q_v_med, 2),
            "type": label})
        results["recall_at_10"].append(round(recall_mean * 100, 1))
        results["speedup"].append(round(speedup, 1))
        results["pure_python_ms"] = results.get("pure_python_ms", []) + [round(py_med, 2)]

    # ── Final table ──
    print("\n\n  " + "=" * 72)
    print("  FINAL RESULTS")
    print("  " + "=" * 72)
    print(f"  {'Scale':>8} | {'Type':>10} | {'Brute(ms)':>10} | {'ANN(ms)':>10} | "
          f"{'Speedup':>9} | {'Recall@10':>10}")
    print("  " + "-" * 72)
    for i, scale in enumerate(SCALES):
        b = results["brute_force"][i]
        v = results["ann_vec"][i]
        r = results["recall_at_10"][i]
        sp = results["speedup"][i]
        print(f"  {scale:>8} | {v['type']:>10} | {b['query_ms']:>8.2f}ms | "
              f"{v['query_ms']:>8.2f}ms | {sp:>7.1f}x | {r:>7.1f}%")
    print("  " + "-" * 72)

    # Insert times
    print(f"\n  {'Scale':>8} | {'Type':>10} | {'Brute insert':>15} | {'ANN insert':>15}")
    print("  " + "-" * 52)
    for i, scale in enumerate(SCALES):
        b = results["brute_force"][i]
        v = results["ann_vec"][i]
        print(f"  {scale:>8} | {v['type']:>10} | {b['insert_s']:>13.3f}s | {v['insert_s']:>13.3f}s")

    # ── Save results ──
    data_path = Path(__file__).resolve().parent / "benchmark_real_data.json"
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Raw data saved to {data_path}")

    # ── Graph ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        scales_str = [f"{s}\n({v['type']})" for s, v in zip(SCALES, results["ann_vec"])]
        x = np.arange(len(scales_str))
        width = 0.3

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Query latency
        b_ms = [r["query_ms"] for r in results["brute_force"]]
        v_ms = [r["query_ms"] for r in results["ann_vec"]]
        py_ms = results.get("pure_python_ms", v_ms)

        bars1 = ax1.bar(x - width, b_ms, width, label="Brute-force (SQLite)",
                        color="#e74c3c", alpha=0.85)
        bars2 = ax1.bar(x, v_ms, width, label="ANN (sqlite-vec HNSW)",
                        color="#2ecc71", alpha=0.85)
        bars3 = ax1.bar(x + width, py_ms, width, label="Pure Python",
                        color="#f39c12", alpha=0.85)
        ax1.set_ylabel("Query Latency (ms, lower is better)")
        ax1.set_xlabel("Number of Vectors")
        ax1.set_title("Search Latency: ANN vs Brute-Force\n(Real fastembed embeddings)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scales_str)
        ax1.legend(fontsize=9)
        ax1.set_yscale("log")
        ax1.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bars, vals in [(bars1, b_ms), (bars2, v_ms), (bars3, py_ms)]:
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

        # Right: Recall
        recall_vals = results["recall_at_10"]
        colors = ["#27ae60" if r >= 90 else "#e67e22" if r >= 70 else "#e74c3c"
                  for r in recall_vals]
        bars = ax2.bar(scales_str, recall_vals, color=colors, alpha=0.85, width=0.5)
        ax2.set_ylabel("Recall@10 (%)")
        ax2.set_xlabel("Number of Vectors")
        ax2.set_title("ANN Recall vs Brute-Force Ground Truth")
        ax2.set_ylim(max(70, min(recall_vals) - 5), 100.5)
        ax2.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, recall_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Add speedup annotation
        sp_text = "\n".join([
            f"Speedup: {results['speedup'][i]:.0f}x"
            for i in range(len(SCALES))
        ])
        ax2.text(0.98, 0.98, sp_text, transform=ax2.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent / "benchmark_real_results.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Graph saved to {out_path}")
        plt.close()
    except ImportError:
        print("  matplotlib not available, skipping graph")

    print("Done.")


if __name__ == "__main__":
    main()
