#!/usr/bin/env python3
"""Benchmark recall latency: cached entities vs on-the-fly extraction.

Seeds a synthetic dataset, then measures how long recall takes when:
1. Entity boost is disabled (baseline)
2. Entity boost is enabled but entities are extracted on-the-fly
3. Entity boost is enabled and entities are cached in metadata

Usage:
    uv run python scripts/benchmark_entity_latency.py
"""

import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi import Memory, MemoryConfig
from kemi.adapters.base import EmbeddingAdapter
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.memory.model import LifecycleState, MemoryObject, MemoryType

# ── Configuration ───────────────────────────────────────────────────
DIM = 64
NUM_MEMORIES = int(os.environ.get("BENCH_LAT_NUM_MEMORIES", "1000"))
NUM_QUERIES = int(os.environ.get("BENCH_LAT_NUM_QUERIES", "100"))
TOP_K = int(os.environ.get("BENCH_LAT_TOP_K", "5"))
WARMUP_RUNS = int(os.environ.get("BENCH_LAT_WARMUP", "3"))
TIMED_RUNS = int(os.environ.get("BENCH_LAT_TIMED", "10"))
RNG = random.Random(42)

_default_dir = Path(__file__).resolve().parent
RESULTS_FILE = Path(
    os.environ.get("BENCH_LAT_RESULTS_FILE", _default_dir / "benchmark_entity_latency_results.json")
)
PNG_FILE = Path(
    os.environ.get("BENCH_LAT_PNG_FILE", _default_dir / "benchmark_entity_latency_results.png")
)


# ── Deterministic embedding adapter ───────────────────────────────────


class _BenchEmbed(EmbeddingAdapter):
    """Hash-based deterministic embedding — same text always → same vector."""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vec(text)

    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        return [b / 255.0 for b in expanded[: self._dim]]


# ── Dataset generators ────────────────────────────────────────────────

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
    "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo",
]
PLACES = [
    "Paris", "Tokyo", "London", "Berlin", "Sydney", "Cairo",
    "Moscow", "Rio", "Dubai", "Seoul", "Mumbai", "Lima",
]
DATES = [
    "2024-01-15", "2024-03-22", "2024-05-10", "2024-07-08",
    "2024-09-01", "2024-11-19", "2025-01-05", "2025-02-14",
]
ORGANIZATIONS = [
    "Acme Corp", "Globex", "Initech", "Umbrella", "Stark Ind",
    "Wayne Ent", "Cyberdyne", "Massive Dynamic", "Hooli", "Pied Piper",
]

FILLERS = [
    "The weather was pleasant throughout the entire day.",
    "They discussed various topics over a warm cup of tea.",
    "A long walk in the park helped clear their minds.",
    "The meeting went smoothly without any interruptions.",
    "Everyone enjoyed the food at the small gathering.",
    "Reading a book before bed is a calming routine.",
    "The project was completed ahead of the deadline.",
    "Music played softly in the background all evening.",
    "A quiet morning set the tone for a productive week.",
    "The garden looked beautiful after the spring rain.",
]


def _pick_entity_subset(rng: random.Random, count: int) -> list[str]:
    pool = NAMES + PLACES + DATES + ORGANIZATIONS
    return rng.sample(pool, min(count, len(pool)))


def _make_memory_content(rng: random.Random, entities: list[str]) -> str:
    filler = rng.choice(FILLERS)
    if not entities:
        return filler
    entity_phrase = ", ".join(entities)
    return f"{filler} {entity_phrase} was part of the experience."


def _seed_memories(mem: Memory, user_id: str, count: int) -> list[str]:
    """Populate store with *count* memories. Returns list of memory IDs."""
    mids: list[str] = []
    for _i in range(count):
        n_entities = RNG.randint(0, 3)
        entities = _pick_entity_subset(RNG, n_entities)
        content = _make_memory_content(RNG, entities)
        emb = mem._embed.embed_single(content)
        mo = MemoryObject(
            memory_id=f"bench-{user_id}-{_i:05d}",
            user_id=user_id,
            content=content,
            embedding=emb,
            embedding_dim=DIM,
            memory_type=MemoryType.EPISODIC,
            importance=RNG.uniform(0.3, 0.7),
        )
        mem._store.store(mo)
        mids.append(mo.memory_id)
    return mids


def _make_queries(count: int) -> list[str]:
    """Create diverse queries that target entity subsets."""
    queries: list[str] = []
    pool = NAMES + PLACES + DATES + ORGANIZATIONS
    for _i in range(count):
        n = RNG.randint(1, 2)
        entities = RNG.sample(pool, n)
        queries.append(f"Tell me about {' and '.join(entities)}.")
    return queries


# ── Timing helpers ────────────────────────────────────────────────────


def _time_recall(mem: Memory, user_id: str, queries: list[str], top_k: int) -> list[float]:
    """Run one pass of all queries and return per-query latencies in ms."""
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        mem.recall(
            user_id=user_id,
            query=q,
            top_k=top_k,
            hybrid_search=True,
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return latencies


# ── Main benchmark ──────────────────────────────────────────────────


def main() -> int:
    print("=" * 74)
    print("  Entity Cache Recall Latency Benchmark")
    print("=" * 74)
    print(f"  Memories: {NUM_MEMORIES}  |  Queries: {NUM_QUERIES}  |  Top-K: {TOP_K}")
    print(f"  Warmup runs: {WARMUP_RUNS}  |  Timed runs: {TIMED_RUNS}")
    print()

    user_id = "bench_user"
    embed = _BenchEmbed()

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    store = SQLiteStorageAdapter(db_path=db_path)

    # ── Seed dataset ──
    print("  Seeding dataset...")
    config_on = MemoryConfig(enable_entity_boost=True, entity_boost_weight=0.3)
    mem = Memory(embed=embed, store=store, config=config_on)
    _seed_memories(mem, user_id, NUM_MEMORIES)
    queries = _make_queries(NUM_QUERIES)

    # ── Baseline: entity boost disabled ──
    print(f"  [1/3] Baseline — entity boost disabled...")
    config_off = MemoryConfig(enable_entity_boost=False)
    mem_off = Memory(embed=embed, store=store, config=config_off)

    for _ in range(WARMUP_RUNS):
        _time_recall(mem_off, user_id, queries, TOP_K)

    latencies_off: list[float] = []
    for _ in range(TIMED_RUNS):
        latencies_off.extend(_time_recall(mem_off, user_id, queries, TOP_K))

    # ── On-the-fly extraction (clear cached entities first) ──
    print(f"  [2/3] On-the-fly extraction (no cache)...")
    # Ensure no cached entities exist
    all_memories = store.get_all_by_user(
        user_id,
        lifecycle_filter=[
            LifecycleState.ACTIVE,
            LifecycleState.DECAYING,
            LifecycleState.ARCHIVED,
        ],
    )
    for m in all_memories:
        if "extracted_entities" in m.metadata:
            del m.metadata["extracted_entities"]
            store.update(m)

    for _ in range(WARMUP_RUNS):
        _time_recall(mem, user_id, queries, TOP_K)

    latencies_uncached: list[float] = []
    for _ in range(TIMED_RUNS):
        latencies_uncached.extend(_time_recall(mem, user_id, queries, TOP_K))

    # ── Cached entities (backfill then benchmark) ──
    print(f"  [3/3] Cached entities (after backfill)...")
    backfilled = mem.backfill_entities(user_id=user_id)
    print(f"        Backfilled {backfilled} memories")

    for _ in range(WARMUP_RUNS):
        _time_recall(mem, user_id, queries, TOP_K)

    latencies_cached: list[float] = []
    for _ in range(TIMED_RUNS):
        latencies_cached.extend(_time_recall(mem, user_id, queries, TOP_K))

    # ── Aggregate ──
    def _stats(data: list[float]) -> dict[str, float]:
        s = sorted(data)
        n = len(s)
        p99_idx = min(int(n * 0.99), n - 1)
        return {
            "mean_ms": statistics.mean(data),
            "stddev_ms": statistics.pstdev(data) if n >= 1 else 0.0,
            "median_ms": statistics.median(data),
            "min_ms": min(data),
            "max_ms": max(data),
            "p99_ms": s[p99_idx],
        }

    stats_off = _stats(latencies_off)
    stats_uncached = _stats(latencies_uncached)
    stats_cached = _stats(latencies_cached)

    # ── Print results ──
    print()
    print("  " + "=" * 74)
    print("  Results (per-query latency)")
    print("  " + "=" * 74)
    print(f"  {'Metric':<20} {'Boost Off':>14} {'On-the-fly':>14} {'Cached':>14}")
    print("  " + "-" * 74)

    def _row(label: str, off_v: float, uncached_v: float, cached_v: float) -> None:
        print(
            f"  {label:<20} {off_v:>13.3f}ms {uncached_v:>13.3f}ms {cached_v:>13.3f}ms"
        )

    _row("Mean", stats_off["mean_ms"], stats_uncached["mean_ms"], stats_cached["mean_ms"])
    _row("Stddev", stats_off["stddev_ms"], stats_uncached["stddev_ms"], stats_cached["stddev_ms"])
    _row("Median", stats_off["median_ms"], stats_uncached["median_ms"], stats_cached["median_ms"])
    _row("Min", stats_off["min_ms"], stats_uncached["min_ms"], stats_cached["min_ms"])
    _row("Max", stats_off["max_ms"], stats_uncached["max_ms"], stats_cached["max_ms"])
    _row("P99", stats_off["p99_ms"], stats_uncached["p99_ms"], stats_cached["p99_ms"])
    print("  " + "-" * 74)

    overhead_uncached = stats_uncached["mean_ms"] - stats_off["mean_ms"]
    overhead_cached = stats_cached["mean_ms"] - stats_off["mean_ms"]
    saved_vs_uncached = stats_uncached["mean_ms"] - stats_cached["mean_ms"]
    print(f"  Baseline mean latency:          {stats_off['mean_ms']:.3f}ms")
    print(f"  On-the-fly mean latency:        {stats_uncached['mean_ms']:.3f}ms  (+{overhead_uncached:.3f}ms overhead)")
    print(f"  Cached mean latency:            {stats_cached['mean_ms']:.3f}ms  (+{overhead_cached:.3f}ms overhead)")
    if saved_vs_uncached > 0:
        print(f"  Cache saves:                    {saved_vs_uncached:.3f}ms per query ({saved_vs_uncached/stats_uncached['mean_ms']*100:.1f}% faster than on-the-fly)")
    print()

    # ── Save JSON ──
    results: dict[str, Any] = {
        "config": {
            "dim": DIM,
            "num_memories": NUM_MEMORIES,
            "num_queries": NUM_QUERIES,
            "top_k": TOP_K,
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
        },
        "baseline_disabled": {
            "per_query_latencies_ms": [round(v, 3) for v in latencies_off],
            "aggregate": stats_off,
        },
        "on_the_fly": {
            "per_query_latencies_ms": [round(v, 3) for v in latencies_uncached],
            "aggregate": stats_uncached,
        },
        "cached": {
            "per_query_latencies_ms": [round(v, 3) for v in latencies_cached],
            "aggregate": stats_cached,
        },
        "summary": {
            "baseline_mean_ms": round(stats_off["mean_ms"], 3),
            "uncached_mean_ms": round(stats_uncached["mean_ms"], 3),
            "cached_mean_ms": round(stats_cached["mean_ms"], 3),
            "overhead_uncached_ms": round(overhead_uncached, 3),
            "overhead_cached_ms": round(overhead_cached, 3),
            "saved_vs_uncached_ms": round(saved_vs_uncached, 3),
            "saved_vs_uncached_pct": round(saved_vs_uncached / stats_uncached["mean_ms"] * 100, 1) if stats_uncached["mean_ms"] > 0 else 0.0,
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_FILE}")

    # ── Generate graph ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Baseline\n(disabled)", "On-the-fly\n(extraction)", "Cached\n(metadata)"]
        means = [stats_off["mean_ms"], stats_uncached["mean_ms"], stats_cached["mean_ms"]]
        colors = ["#95a5a6", "#e74c3c", "#2ecc71"]

        x = np.arange(len(labels))
        width = 0.5

        _fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        # Top-left subplot: mean latency bar chart
        ax1 = axes[0, 0]
        stddevs = [stats_off["stddev_ms"], stats_uncached["stddev_ms"], stats_cached["stddev_ms"]]
        bars = ax1.bar(x, means, width, color=colors, alpha=0.85, yerr=stddevs, capsize=6, error_kw={"linewidth": 1.5, "ecolor": "#2c3e50"})
        ax1.set_ylabel("Mean Latency (ms)")
        ax1.set_title("Recall Latency: Cached Entities vs On-the-fly Extraction")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.grid(axis="y", alpha=0.3)

        for bar, std in zip(bars, stddevs):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height + std + 0.05,
                f"{height:.2f}ms", ha="center", va="bottom", fontsize=10,
            )

        # Top-right subplot: per-query latency distribution (box plot)
        ax2 = axes[0, 1]
        box_data = [latencies_off, latencies_uncached, latencies_cached]
        box_labels = labels

        bp = ax2.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        for mean in bp["means"]:
            mean.set_color("#f39c12")
            mean.set_linewidth(2)

        ax2.set_title("Per-Recall Latency Distribution")
        ax2.set_ylabel("Latency (ms)")
        ax2.grid(axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#95a5a6", alpha=0.6, label="Baseline (disabled)"),
            Patch(facecolor="#e74c3c", alpha=0.6, label="On-the-fly (extraction)"),
            Patch(facecolor="#2ecc71", alpha=0.6, label="Cached (metadata)"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        # Bottom-left subplot: latency density (violin plot)
        ax3 = axes[1, 0]
        violin_data = [latencies_off, latencies_uncached, latencies_cached]
        vp = ax3.violinplot(violin_data, positions=x, widths=0.5, showmeans=True, showmedians=True)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
            vp[partname].set_edgecolor("black")
            vp[partname].set_linewidth(1.2)
        vp["cmeans"].set_edgecolor("#f39c12")
        vp["cmeans"].set_linewidth(2)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_title("Latency Density Shape (Violin)")
        ax3.set_ylabel("Latency (ms)")
        ax3.grid(axis="y", alpha=0.3)

        # Bottom-right subplot: latency CDF
        ax4 = axes[1, 1]
        line_styles = ["-", "--", "-."]
        for latencies, color, label, ls in zip(
            [latencies_off, latencies_uncached, latencies_cached],
            colors,
            ["Baseline (disabled)", "On-the-fly (extraction)", "Cached (metadata)"],
            line_styles,
        ):
            sorted_lat = np.sort(latencies)
            cum_prob = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
            ax4.plot(sorted_lat, cum_prob, color=color, linestyle=ls, linewidth=2, label=label)

        ax4.set_title("Latency CDF (Tail Behavior)")
        ax4.set_xlabel("Latency (ms, log scale)")
        ax4.set_ylabel("Cumulative Probability")
        ax4.set_xscale("log")
        ax4.grid(alpha=0.3, which="both")
        ax4.legend(loc="lower right")

        # Horizontal reference lines at p50, p95, p99
        for p in [0.50, 0.95, 0.99]:
            ax4.axhline(y=p, color="#bdc3c7", linestyle=":", linewidth=1)
            ax4.text(
                ax4.get_xlim()[1] * 0.98, p - 0.02,
                f"p{int(p*100)}", ha="right", va="top",
                fontsize=8, color="#7f8c8d",
            )

        plt.tight_layout()
        plt.savefig(PNG_FILE, dpi=150, bbox_inches="tight")
        print(f"  Graph saved to {PNG_FILE}")
        plt.close()

    except ImportError:
        print("  matplotlib not available, skipping graph")

    # Cleanup
    import os as _os
    try:
        _os.unlink(db_path)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
