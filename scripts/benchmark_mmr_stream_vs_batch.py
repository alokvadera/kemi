#!/usr/bin/env python3
"""Benchmark MMR batch (list) vs streaming (generator) overhead.

Compares ``scoring.mmr_rerank`` and ``scoring.mmr_rerank_stream`` at
multiple corpus sizes to verify that the generator extraction introduces
no measurable overhead.

Measures:
  - Wall-clock time for batch (list) recall
  - Wall-clock time for stream (generator) consumption
  - Time-to-first-yield (streaming only)
  - Result consistency (same memory_ids and ordering)

Usage:
    uv run python scripts/benchmark_mmr_stream_vs_batch.py
"""

import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi.memory import scoring
from kemi.memory.model import MemoryObject, MemoryType

# ── Configuration ───────────────────────────────────────────────────
DIM = 64
SCALES = [10, 100, 500, 1000, 5000]
TOP_K = 10
RUNS_PER_SCALE = 20  # averaged per scale for stable micro-benchmarks
RNG = random.Random(42)

RESULTS_FILE = Path(__file__).resolve().parent / "benchmark_mmr_stream_vs_batch_results.json"
PNG_FILE = Path(__file__).resolve().parent / "benchmark_mmr_stream_vs_batch_results.png"

# ── Helpers ──────────────────────────────────────────────────────────


def _make_memory(i: int, dim: int) -> MemoryObject:
    """Create a MemoryObject with a deterministic random embedding and score."""
    emb = [RNG.uniform(-1.0, 1.0) for _ in range(dim)]
    return MemoryObject(
        memory_id=f"mmr-bench-{i}",
        user_id="bench",
        content=f"Benchmark memory number {i} with some realistic filler text.",
        embedding=emb,
        embedding_dim=dim,
        memory_type=MemoryType.EPISODIC,
        score=RNG.uniform(0.3, 1.0),
    )


def _result_ids(memories: list[MemoryObject]) -> list[str]:
    return [m.memory_id for m in memories]


# ── Batch benchmark ──────────────────────────────────────────────────


def bench_batch(
    memories: list[MemoryObject],
    query_embedding: list[float],
    top_k: int,
    runs: int,
) -> tuple[float, list[str]]:
    """Run mmr_rerank multiple times; return (median_time_ms, ids)."""
    gc.collect()
    times: list[float] = []
    result_ids: list[str] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        results = scoring.mmr_rerank(memories, query_embedding, top_k=top_k)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if not result_ids:
            result_ids = _result_ids(results)
    median_ms = sorted(times)[len(times) // 2] * 1000
    return median_ms, result_ids


# ── Stream benchmark ─────────────────────────────────────────────────


def bench_stream(
    memories: list[MemoryObject],
    query_embedding: list[float],
    top_k: int,
    runs: int,
) -> tuple[float, float, list[str]]:
    """Run mmr_rerank_stream multiple times.

    Returns (median_total_ms, median_ttf_ms, ids).
    """
    gc.collect()
    total_times: list[float] = []
    ttf_times: list[float] = []
    result_ids: list[str] = []
    for _ in range(runs):
        ids: list[str] = []
        yielded_first = False
        ttf = 0.0
        t0 = time.perf_counter()
        for memory in scoring.mmr_rerank_stream(
            memories, query_embedding, top_k=top_k
        ):
            if not yielded_first:
                ttf = time.perf_counter()
                yielded_first = True
            ids.append(memory.memory_id)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        ttf_times.append(ttf - t0 if yielded_first else 0.0)
        if not result_ids:
            result_ids = ids
    median_total_ms = sorted(total_times)[len(total_times) // 2] * 1000
    median_ttf_ms = sorted(ttf_times)[len(ttf_times) // 2] * 1000
    return median_total_ms, median_ttf_ms, result_ids


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 74)
    print("  MMR Batch (list) vs Stream (generator) — Overhead Benchmark")
    print("=" * 74)
    print(f"  Dim: {DIM}  |  Top-K: {TOP_K}  |  Runs/scale: {RUNS_PER_SCALE}")
    print()

    results: dict[str, list[Any]] = {
        "batch_median_ms": [],
        "stream_median_ms": [],
        "stream_ttf_ms": [],
        "speedup": [],
        "overhead_pct": [],
        "results_match": [],
    }

    query_embedding = [RNG.uniform(-1.0, 1.0) for _ in range(DIM)]

    for scale in SCALES:
        print(f"  ─── Scale: {scale:,} memories ───")

        memories = [_make_memory(i, DIM) for i in range(scale)]

        # Warm-up: single run of each to avoid cold-start effects
        scoring.mmr_rerank(memories, query_embedding, top_k=TOP_K)
        list(scoring.mmr_rerank_stream(memories, query_embedding, top_k=TOP_K))

        # 1) Batch
        batch_ms, batch_ids = bench_batch(
            memories, query_embedding, TOP_K, RUNS_PER_SCALE
        )

        # 2) Stream
        stream_ms, stream_ttf, stream_ids = bench_stream(
            memories, query_embedding, TOP_K, RUNS_PER_SCALE
        )

        # 3) Compare
        match = batch_ids == stream_ids
        speedup = (batch_ms / stream_ms) if stream_ms > 0 else float("inf")
        overhead = ((stream_ms - batch_ms) / batch_ms * 100) if batch_ms > 0 else 0.0

        results["batch_median_ms"].append(round(batch_ms, 3))
        results["stream_median_ms"].append(round(stream_ms, 3))
        results["stream_ttf_ms"].append(round(stream_ttf, 3))
        results["speedup"].append(round(speedup, 3))
        results["overhead_pct"].append(round(overhead, 3))
        results["results_match"].append(match)

        match_mark = "✓" if match else "✗"
        print(f"    Batch median:  {batch_ms:>10.3f} ms")
        print(f"    Stream median: {stream_ms:>10.3f} ms  (TTF: {stream_ttf:.3f} ms)")
        print(f"    Overhead:      {overhead:>+10.2f}%  Speedup: {speedup:.3f}x")
        print(f"    Match:         {match_mark}")
        print()

    # Summary table
    print("  " + "=" * 74)
    h = (
        f"  {'Scale':>8} | {'Batch ms':>10} | {'Stream ms':>10} | "
        f"{'TTF ms':>10} | {'Overhead %':>10} | {'Match':>8}"
    )
    print(h)
    print("  " + "-" * 74)
    for i, scale in enumerate(SCALES):
        print(
            f"  {scale:>8,} | {results['batch_median_ms'][i]:>10.3f} | "
            f"{results['stream_median_ms'][i]:>10.3f} | "
            f"{results['stream_ttf_ms'][i]:>10.3f} | "
            f"{results['overhead_pct'][i]:>+9.2f}% | "
            f"{'✓' if results['results_match'][i] else '✗':>8}"
        )
    print("  " + "-" * 74)

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(
            {
                "config": {
                    "dim": DIM,
                    "top_k": TOP_K,
                    "runs_per_scale": RUNS_PER_SCALE,
                    "scales": SCALES,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {RESULTS_FILE}")

    # Generate graph
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        scales_str = [str(s) for s in SCALES]
        x = np.arange(len(scales_str))
        width = 0.3

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: median time bars
        ax1.bar(
            x - width / 2,
            results["batch_median_ms"],
            width,
            label="mmr_rerank (list)",
            color="#e74c3c",
            alpha=0.85,
        )
        ax1.bar(
            x + width / 2,
            results["stream_median_ms"],
            width,
            label="mmr_rerank_stream (gen)",
            color="#3498db",
            alpha=0.85,
        )
        ax1.set_ylabel("Median Time (ms, lower is better)")
        ax1.set_xlabel("Number of Candidate Memories")
        ax1.set_title("MMR Batch vs Stream — Median Latency")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scales_str)
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bars, vals in [
            (ax1.containers[0], results["batch_median_ms"]),
            (ax1.containers[1], results["stream_median_ms"]),
        ]:
            for bar, val in zip(bars, vals):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

        # Right: overhead percentage
        colors = ["#2ecc71" if o <= 0 else "#f39c12" for o in results["overhead_pct"]]
        ax2.bar(x, results["overhead_pct"], width * 1.5, color=colors, alpha=0.85)
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_ylabel("Overhead % (stream vs batch)")
        ax2.set_xlabel("Number of Candidate Memories")
        ax2.set_title("MMR Stream Overhead vs Batch")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scales_str)
        ax2.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bar, val in zip(ax2.containers[0], results["overhead_pct"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(PNG_FILE, dpi=150, bbox_inches="tight")
        print(f"  Graph saved to {PNG_FILE}")
        plt.close()

    except ImportError:
        print("  matplotlib not available, skipping graph")


if __name__ == "__main__":
    main()
