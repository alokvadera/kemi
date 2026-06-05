#!/usr/bin/env python3
"""Benchmark streaming vs batch recall latency at multiple scales.

Compares Memory.recall() (batch) and Memory.recall_stream() (async streaming)
at several corpus sizes, measuring:

  - Total wall-clock time to receive all results
  - Time-to-first-result (streaming only — shows progressive delivery benefit)
  - Result consistency (same memory_ids and ordering?)
  - Memory overhead

Usage:
    uv run python scripts/benchmark_stream_vs_batch.py
"""

import asyncio
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi import Memory
from kemi.adapters.embedding.custom import CustomEmbedAdapter
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.models import MemoryObject, MemorySource, MemoryType

# ── Configuration ───────────────────────────────────────────────────
DIM = 64
SCALES = [10, 100, 500, 1000]
TOP_K = 10
QUERIES_PER_SCALE = 5
RNG = random.Random(42)

RESULTS_FILE = Path(__file__).resolve().parent / "benchmark_stream_vs_batch_results.json"
PNG_FILE = Path(__file__).resolve().parent / "benchmark_stream_vs_batch_results.png"

# ── Embedding adapter ────────────────────────────────────────────────


import hashlib


class _BenchEmbed:
    """Hash-based deterministic embedding — same text always → same vector."""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_single(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        # SHA-256 produces 32 bytes → scale to dim
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        return [b / 255.0 for b in expanded[: self._dim]]

    def dimension(self) -> int:
        return self._dim


# ── Bench helpers ────────────────────────────────────────────────────


def _seed_memories(mem: Memory, user_id: str, count: int) -> None:
    """Populate the store with *count* random memories."""
    embed = mem._embed
    store = mem._store
    for i in range(count):
        emb = embed.embed_single(f"seed {i}")
        mo = MemoryObject(
            memory_id=f"bench-{user_id}-{i}",
            user_id=user_id,
            content=f"This is benchmark memory number {i} with some filler text to make it realistic.",
            embedding=emb,
            embedding_dim=embed.dimension(),
            memory_type=MemoryType.EPISODIC,
        )
        store.store(mo)


def _make_query() -> str:
    """Generate a random query string."""
    words = ["python", "hiking", "cooking", "travel", "music", "sports",
             "books", "movies", "coding", "design", "photography", "garden"]
    k = RNG.randint(2, 4)
    return " ".join(RNG.sample(words, k))


def _result_ids(memories: list) -> list[str]:
    """Extract memory_id list from batch results or stream-yielded objects."""
    return [m.memory_id for m in memories]


# ── Sync batch benchmark ─────────────────────────────────────────────


def bench_batch(mem: Memory, user_id: str, queries: list[str],
                top_k: int) -> tuple[float, list[list[str]]]:
    """Run batch recall for each query; return (total_time_s, id_lists)."""
    gc.collect()
    t0 = time.perf_counter()
    all_ids: list[list[str]] = []
    for q in queries:
        results = mem.recall(user_id, q, top_k=top_k)
        all_ids.append(_result_ids(results))
    t1 = time.perf_counter()
    return t1 - t0, all_ids


# ── Async streaming benchmark ────────────────────────────────────────


async def bench_stream(mem: Memory, user_id: str, queries: list[str],
                       top_k: int) -> tuple[float, float, list[list[str]]]:
    """Run streaming recall for each query.
    
    Returns (total_time_s, time_to_first_s, id_lists).
    """
    gc.collect()
    t_total_0 = time.perf_counter()
    first_ttf: list[float] = []
    all_ids: list[list[str]] = []
    for q in queries:
        ids: list[str] = []
        yielded_first = False
        t_q0 = time.perf_counter()
        async for memory in mem.recall_stream(user_id, q, top_k=top_k):
            if not yielded_first:
                t_first = time.perf_counter()
                yielded_first = True
            ids.append(memory.memory_id)
        all_ids.append(ids)
        # TTF relative to this query's start, not cumulative
        first_ttf.append(t_first - t_q0 if yielded_first else 0.0)

    t1 = time.perf_counter()
    # Median TTF across queries
    median_ttf = sorted(first_ttf)[len(first_ttf) // 2] if first_ttf else 0.0
    return t1 - t_total_0, median_ttf, all_ids


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 74)
    print("  Streaming vs Batch Recall — Latency Benchmark")
    print("=" * 74)
    print(f"  Dim: {DIM}  |  Top-K: {TOP_K}  |  Queries/scale: {QUERIES_PER_SCALE}")
    print()

    results: dict[str, list] = {
        "batch_total_ms": [],
        "stream_total_ms": [],
        "stream_ttf_ms": [],
        "speedup_total": [],
        "results_match": [],
    }

    import os
    import tempfile

    tmp_files: list[str] = []

    for scale in SCALES:
        print(f"  ─── Scale: {scale:,} memories ───")

        # Fresh store per scale — use a temp file instead of :memory: so
        # thread-pool connections (asyncio.to_thread inside recall_stream)
        # share the same database.
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name
        tmp_files.append(db_path)
        store = SQLiteStorageAdapter(db_path=db_path)
        embed = _BenchEmbed()
        mem = Memory(embed=embed, store=store)
        user_id = "bench"

        _seed_memories(mem, user_id, scale)
        queries = [_make_query() for _ in range(QUERIES_PER_SCALE)]

        # Warmup: one batch call to avoid cold-start overhead
        mem.recall(user_id, queries[0], top_k=TOP_K)
        mem.recall(user_id, queries[0], top_k=TOP_K)

        # 1) Batch recall
        batch_time, batch_ids = bench_batch(mem, user_id, queries, TOP_K)

        # 2) Streaming recall
        stream_time, stream_ttf, stream_ids = await bench_stream(
            mem, user_id, queries, TOP_K
        )

        # 3) Compare results
        match = all(b == s for b, s in zip(batch_ids, stream_ids, strict=True))

        batch_ms = batch_time * 1000
        stream_ms = stream_time * 1000
        ttf_ms = stream_ttf * 1000
        speedup = (batch_time / stream_time) if stream_time > 0 else float("inf")

        results["batch_total_ms"].append(round(batch_ms, 2))
        results["stream_total_ms"].append(round(stream_ms, 2))
        results["stream_ttf_ms"].append(round(ttf_ms, 2))
        results["speedup_total"].append(round(speedup, 2))
        results["results_match"].append(match)

        match_mark = "✓" if match else "✗"
        print(f"    Batch total:   {batch_ms:>8.2f} ms")
        print(f"    Stream total:  {stream_ms:>8.2f} ms  (TTF: {ttf_ms:.2f} ms)")
        print(f"    Speedup:       {speedup:>7.2f}x  Results match: {match_mark}")
        print()

    # Summary table
    print("  " + "=" * 74)
    h = f"  {'Scale':>8} | {'Batch ms':>10} | {'Stream ms':>10} | {'TTF ms':>10} | {'Speedup':>10} | {'Match':>8}"
    print(h)
    print("  " + "-" * 74)
    for i, scale in enumerate(SCALES):
        print(
            f"  {scale:>8,} | {results['batch_total_ms'][i]:>10.2f} | "
            f"{results['stream_total_ms'][i]:>10.2f} | "
            f"{results['stream_ttf_ms'][i]:>10.2f} | "
            f"{results['speedup_total'][i]:>9.2f}x | "
            f"{'✓' if results['results_match'][i] else '✗':>8}"
        )
    print("  " + "-" * 74)

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "config": {"dim": DIM, "top_k": TOP_K, "queries_per_scale": QUERIES_PER_SCALE,
                       "scales": SCALES},
            "results": results,
        }, f, indent=2)
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: total time bars
        ax1.bar(x - width / 2, results["batch_total_ms"], width,
                label="Batch recall", color="#e74c3c", alpha=0.85)
        ax1.bar(x + width / 2, results["stream_total_ms"], width,
                label="Stream recall", color="#3498db", alpha=0.85)
        ax1.set_ylabel("Total Time (ms, lower is better)")
        ax1.set_xlabel("Number of Memories")
        ax1.set_title("Total Recall Latency")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scales_str)
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bars, vals in [(ax1.containers[0], results["batch_total_ms"]),
                           (ax1.containers[1], results["stream_total_ms"])]:
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

        # Right: TTF vs total stream time
        ax2.bar(x - width / 2, results["stream_ttf_ms"], width,
                label="Time to First", color="#2ecc71", alpha=0.85)
        ax2.bar(x + width / 2, results["stream_total_ms"], width,
                label="Stream Total", color="#3498db", alpha=0.85)
        ax2.set_ylabel("Time (ms)")
        ax2.set_xlabel("Number of Memories")
        ax2.set_title("Streaming: Time-to-First vs Total")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scales_str)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bars, vals in [(ax2.containers[0], results["stream_ttf_ms"]),
                           (ax2.containers[1], results["stream_total_ms"])]:
            for bar, val in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

        plt.tight_layout()
        plt.savefig(PNG_FILE, dpi=150, bbox_inches="tight")
        print(f"  Graph saved to {PNG_FILE}")
        plt.close()

    except ImportError:
        print("  matplotlib not available, skipping graph")

    # Cleanup temp files
    for f in tmp_files:
        try:
            os.unlink(f)
        except OSError:
            pass
    print(f"  Cleaned up {len(tmp_files)} temp DB files")


if __name__ == "__main__":
    asyncio.run(main())
