#!/usr/bin/env python3
"""Benchmark lazy vs non-lazy vs brute-force insert and query performance.

Compares three storage modes at multiple scales:
  - Brute-force (SQLiteStorageAdapter)
  - ANN direct (SQLiteVecStorageAdapter, lazy=False)
  - ANN lazy   (SQLiteVecStorageAdapter, lazy=True)

Measures:
  - Bulk insert time
  - Query latency (median)
  - Speedup of lazy vs direct inserts
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
SCALES = [1000, 5000, 10000, 25000]
QUERIES_PER_SCALE = 5
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


def benchmark_adapter(adapter_cls, label, scale, embeddings, query_embs, **kwargs):
    """Benchmark insert + query for one adapter at one scale using :memory: DB."""
    adapter_kwargs = {"db_path": ":memory:"}
    if adapter_cls is SQLiteVecStorageAdapter:
        adapter_kwargs["embedding_dim"] = DIM
        adapter_kwargs.update(kwargs)
    adapter = adapter_cls(**adapter_kwargs)
    user_id = "bench"

    # Warmup
    for i in range(min(3, len(embeddings))):
        adapter.store(make_memory(f"warmup-{i}", user_id, embeddings[i]))
    adapter.delete_by_user(user_id)
    if hasattr(adapter, '_pending_count'):
        adapter._pending_count = None

    mems = [make_memory(f"mem-{label}-{scale}-{i}", user_id, emb)
            for i, emb in enumerate(embeddings)]

    # Bulk insert
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

    try:
        adapter.close()
    except Exception:
        pass
    finally:
        # Clean up to avoid __del__ errors
        import gc as _gc
        _gc.collect()

    # Skip first query (includes flush overhead for lazy mode)
    return insert_time, query_times[1:] if len(query_times) > 1 else query_times


def main():
    if not _SQLITE_VEC_AVAILABLE:
        print("ERROR: sqlite-vec must be installed.")
        sys.exit(1)

    print("=" * 72)
    print("  Insert Performance: Lazy-ANN vs Direct-ANN vs Brute-Force")
    print("=" * 72)
    print(f"  Dim: {DIM}  |  Top-K: {TOP_K}  |  Queries: {QUERIES_PER_SCALE}")
    print()

    results = {"brute_force": [], "ann_direct": [], "ann_lazy": []}

    for scale in SCALES:
        print(f"  ─── Scale: {scale:,} vectors ───")

        embeddings = [random_embedding() for _ in range(scale)]
        q_rng = random.Random(99)
        query_embs = [[q_rng.random() for _ in range(DIM)]
                      for _ in range(QUERIES_PER_SCALE)]

        # 1) Brute-force
        ins_b, q_b = benchmark_adapter(
            SQLiteStorageAdapter, "brute", scale, embeddings, query_embs)
        q_b_med = sorted(q_b)[len(q_b) // 2]
        results["brute_force"].append({
            "scale": scale, "insert_s": ins_b, "query_ms": q_b_med})
        print(f"    Brute-force: insert={ins_b:.3f}s  query={q_b_med:.2f}ms")

        # 2) ANN direct (lazy=False)
        ins_d, q_d = benchmark_adapter(
            SQLiteVecStorageAdapter, "direct", scale, embeddings, query_embs,
            lazy=False)
        q_d_med = sorted(q_d)[len(q_d) // 2]
        results["ann_direct"].append({
            "scale": scale, "insert_s": ins_d, "query_ms": q_d_med})
        print(f"    ANN direct:  insert={ins_d:.3f}s  query={q_d_med:.2f}ms  "
              f"({ins_b/ins_d:.1f}x vs brute)")

        # 3) ANN lazy (lazy=True)
        ins_l, q_l = benchmark_adapter(
            SQLiteVecStorageAdapter, "lazy", scale, embeddings, query_embs,
            lazy=True)
        q_l_med = sorted(q_l)[len(q_l) // 2]
        results["ann_lazy"].append({
            "scale": scale, "insert_s": ins_l, "query_ms": q_l_med})
        print(f"    ANN lazy:    insert={ins_l:.3f}s  query={q_l_med:.2f}ms  "
              f"({ins_d/ins_l:.1f}x faster than direct, {ins_b/ins_l:.1f}x vs brute)")

        print()

    # Summary table
    print("  " + "=" * 72)
    print(f"  {'Scale':>8} | {'Brute ins':>10} | {'Direct ins':>10} | "
          f"{'Lazy ins':>10} | {'Direct q':>10} | {'Lazy q':>10} | {'Speedup':>10}")
    print("  " + "-" * 72)
    for i, scale in enumerate(SCALES):
        b = results["brute_force"][i]
        d = results["ann_direct"][i]
        l = results["ann_lazy"][i]
        sp = d["insert_s"] / l["insert_s"] if l["insert_s"] > 0 else float("inf")
        print(f"  {scale:>8,} | {b['insert_s']:>8.3f}s | {d['insert_s']:>8.3f}s | "
              f"{l['insert_s']:>8.3f}s | {d['query_ms']:>8.2f}ms | "
              f"{l['query_ms']:>8.2f}ms | {sp:>7.1f}x")
    print("  " + "-" * 72)

    # Save results
    data_path = Path(__file__).resolve().parent / "benchmark_lazy_results.json"
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw data saved to {data_path}")

    # Generate graph
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        scales_str = [str(s) for s in SCALES]
        x = np.arange(len(scales_str))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        brute_ins = [r["insert_s"] for r in results["brute_force"]]
        direct_ins = [r["insert_s"] for r in results["ann_direct"]]
        lazy_ins = [r["insert_s"] for r in results["ann_lazy"]]

        ax.bar(x - width, brute_ins, width, label="Brute-force", color="#e74c3c", alpha=0.85)
        ax.bar(x, direct_ins, width, label="ANN direct", color="#2ecc71", alpha=0.85)
        ax.bar(x + width, lazy_ins, width, label="ANN lazy", color="#3498db", alpha=0.85)

        ax.set_ylabel("Bulk Insert Time (seconds, lower is better)")
        ax.set_xlabel("Number of Vectors")
        ax.set_title("Insert Performance: Lazy-ANN vs Direct-ANN vs Brute-Force")
        ax.set_xticks(x)
        ax.set_xticklabels(scales_str)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bars, vals in [(ax.containers[0], brute_ins),
                           (ax.containers[1], direct_ins),
                           (ax.containers[2], lazy_ins)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}s", ha="center", va="bottom", fontsize=7, rotation=45)

        plt.tight_layout()
        out_path = Path(__file__).resolve().parent / "benchmark_lazy_results.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Graph saved to {out_path}")
        plt.close()
    except ImportError:
        print("  matplotlib not available, skipping graph")


if __name__ == "__main__":
    main()
