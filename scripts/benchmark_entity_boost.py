#!/usr/bin/env python3
"""Benchmark entity-aware retrieval: recall quality with vs without entity boost.

Creates a synthetic dataset where some memories share entities with queries
and others do not. Compares ranking quality (hit rate, MRR, average rank)
when entity boost is enabled vs disabled.

Usage:
    uv run python scripts/benchmark_entity_boost.py
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
from kemi.models import MemoryObject, MemoryType

# ── Configuration ───────────────────────────────────────────────────
DIM = 64
NUM_MEMORIES = int(os.environ.get("BENCH_NUM_MEMORIES", "1000"))
NUM_QUERIES = int(os.environ.get("BENCH_NUM_QUERIES", "20"))
TOP_K = int(os.environ.get("BENCH_TOP_K", "5"))
RNG = random.Random(42)

_default_dir = Path(__file__).resolve().parent
RESULTS_FILE = Path(os.environ.get("BENCH_RESULTS_FILE", _default_dir / "benchmark_entity_boost_results.json"))
PNG_FILE = Path(os.environ.get("BENCH_PNG_FILE", _default_dir / "benchmark_entity_boost_results.png"))

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


# Named entities we will mix into memory content
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

# Generic filler sentences without named entities
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
    """Pick *count* random entities from the pool (names + places + dates + orgs)."""
    pool = NAMES + PLACES + DATES + ORGANIZATIONS
    return rng.sample(pool, min(count, len(pool)))


def _make_memory_content(rng: random.Random, entities: list[str]) -> str:
    """Build a sentence that includes the given entities."""
    filler = rng.choice(FILLERS)
    if not entities:
        return filler
    entity_phrase = ", ".join(entities)
    return f"{filler} {entity_phrase} was part of the experience."


def _make_query_for_entities(entities: list[str]) -> str:
    """Build a query string targeting the given entities."""
    if not entities:
        return "What happened that day?"
    return f"Tell me about {' and '.join(entities)}."


def _seed_memories(mem: Memory, user_id: str, count: int) -> list[dict[str, Any]]:
    """Populate store with *count* memories. Each memory gets 0-3 random entities.

    Returns a list of dicts with keys: memory_id, entities, content.
    """
    records: list[dict[str, Any]] = []
    for i in range(count):
        n_entities = RNG.randint(0, 3)
        entities = _pick_entity_subset(RNG, n_entities)
        content = _make_memory_content(RNG, entities)
        emb = mem._embed.embed_single(content)
        mo = MemoryObject(
            memory_id=f"bench-{user_id}-{i:05d}",
            user_id=user_id,
            content=content,
            embedding=emb,
            embedding_dim=DIM,
            memory_type=MemoryType.EPISODIC,
            importance=RNG.uniform(0.3, 0.7),
        )
        mem._store.store(mo)
        records.append({"memory_id": mo.memory_id, "entities": entities, "content": content})
    return records


def _make_queries(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Create queries by sampling entity sets from existing memories."""
    queries: list[dict[str, Any]] = []
    # Ensure each query has at least 1 entity so we can measure overlap
    for i in range(count):
        target = RNG.choice(records)
        # Pick 1-2 entities from the target memory
        target_entities = target["entities"]
        if not target_entities:
            # Fallback: pick random entities
            target_entities = _pick_entity_subset(RNG, 2)
        query_entities = RNG.sample(target_entities, min(len(target_entities), RNG.randint(1, 2)))
        query = _make_query_for_entities(query_entities)
        queries.append({
            "query_id": f"q{i:03d}",
            "query": query,
            "query_entities": query_entities,
            "relevant_memory_ids": [
                r["memory_id"] for r in records
                if any(e in r["entities"] for e in query_entities)
            ],
        })
    return queries


# ── Ranking metrics ─────────────────────────────────────────────────


def _compute_metrics(
    results: list[MemoryObject],
    relevant_ids: set[str],
    top_k: int,
) -> dict[str, Any]:
    """Compute ranking quality metrics for a single query."""
    result_ids = [r.memory_id for r in results]

    # Hit rate: at least one relevant memory in top_k?
    hits = [mid for mid in result_ids[:top_k] if mid in relevant_ids]
    hit = len(hits) > 0

    # MRR: reciprocal of rank of first relevant memory
    mrr = 0.0
    for rank, mid in enumerate(result_ids, start=1):
        if mid in relevant_ids:
            mrr = 1.0 / rank
            break

    # Average rank of relevant memories (within top_k)
    relevant_ranks = [
        rank for rank, mid in enumerate(result_ids[:top_k], start=1)
        if mid in relevant_ids
    ]
    avg_rank = statistics.mean(relevant_ranks) if relevant_ranks else top_k + 1

    # Count of relevant memories in top_k
    relevant_in_top_k = len(hits)

    return {
        "hit": hit,
        "mrr": mrr,
        "avg_rank": avg_rank,
        "relevant_in_top_k": relevant_in_top_k,
    }


def _agg(data: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hit_rate": statistics.mean(1.0 if d["hit"] else 0.0 for d in data),
        "mrr": statistics.mean(d["mrr"] for d in data),
        "avg_rank": statistics.mean(d["avg_rank"] for d in data),
        "avg_relevant_in_top_k": statistics.mean(d["relevant_in_top_k"] for d in data),
    }


def _norm_rank(r: float, top_k: int) -> float:
    """Normalize average rank to a 0-1 'higher is better' scale."""
    return (top_k + 1 - r) / top_k


def _print_results(
    agg_off: dict[str, Any],
    agg_on: dict[str, Any],
    t_off: float,
    t_on: float,
) -> None:
    print()
    print("  " + "=" * 74)
    print("  Results")
    print("  " + "=" * 74)
    print(f"  {'Metric':<28} {'No Boost':>12} {'Entity Boost':>14} {'Delta':>12}")
    print("  " + "-" * 74)

    def _row(label: str, off_v: float, on_v: float, fmt: str = ".3f") -> None:
        delta = on_v - off_v
        delta_pct = (delta / off_v * 100) if off_v != 0 else float("inf")
        print(
            f"  {label:<28} {off_v:>12.{fmt[1]}} {on_v:>14.{fmt[1]}} "
            f"{delta:>+11.{fmt[1]}} ({delta_pct:+.1f}%)"
        )

    _row("Hit Rate", agg_off["hit_rate"], agg_on["hit_rate"])
    _row("MRR", agg_off["mrr"], agg_on["mrr"])
    _row("Avg Rank (lower=better)", agg_off["avg_rank"], agg_on["avg_rank"])
    _row("Relevant in Top-K", agg_off["avg_relevant_in_top_k"], agg_on["avg_relevant_in_top_k"])
    print("  " + "-" * 74)
    print(f"  {'Total time':<28} {t_off:>12.3f}s {t_on:>14.3f}s")
    print()

    improvement = agg_on["hit_rate"] - agg_off["hit_rate"]
    if improvement > 0:
        print(f"  ✅ Entity boost improved hit rate by {improvement*100:.1f} percentage points")
    else:
        print(f"  ⚠️  No improvement detected (delta = {improvement*100:.1f}pp)")


def _build_results(
    num_memories: int,
    num_queries: int,
    top_k: int,
    agg_off: dict[str, Any],
    agg_on: dict[str, Any],
    metrics_off: list[dict[str, Any]],
    metrics_on: list[dict[str, Any]],
    t_off: float,
    t_on: float,
) -> dict[str, Any]:
    improvement = agg_on["hit_rate"] - agg_off["hit_rate"]
    return {
        "config": {
            "dim": DIM,
            "num_memories": num_memories,
            "num_queries": num_queries,
            "top_k": top_k,
            "entity_boost_weight": 0.3,
        },
        "without_boost": {
            "aggregate": agg_off,
            "per_query": metrics_off,
            "total_time_s": round(t_off, 3),
        },
        "with_boost": {
            "aggregate": agg_on,
            "per_query": metrics_on,
            "total_time_s": round(t_on, 3),
        },
        "summary": {
            "hit_rate_improvement_pp": round(improvement, 4),
            "mrr_improvement": round(agg_on["mrr"] - agg_off["mrr"], 4),
            "avg_rank_improvement": round(agg_off["avg_rank"] - agg_on["avg_rank"], 4),
        },
    }


def _save_graph(
    agg_off: dict[str, Any],
    agg_on: dict[str, Any],
    top_k: int,
    png_file: Path,
    metrics_off: list[dict[str, Any]],
    metrics_on: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Hit Rate", "MRR", "Avg Rank\n(normalized, higher=better)", "Relevant in Top-K"]
        no_boost_vals = [
            agg_off["hit_rate"],
            agg_off["mrr"],
            _norm_rank(agg_off["avg_rank"], top_k),
            agg_off["avg_relevant_in_top_k"],
        ]
        boost_vals = [
            agg_on["hit_rate"],
            agg_on["mrr"],
            _norm_rank(agg_on["avg_rank"], top_k),
            agg_on["avg_relevant_in_top_k"],
        ]

        x = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        ax1 = axes[0]
        bars1 = ax1.bar(
            x - width / 2, no_boost_vals, width,
            label="No Boost", color="#e74c3c", alpha=0.85,
        )
        bars2 = ax1.bar(
            x + width / 2, boost_vals, width,
            label="Entity Boost", color="#2ecc71", alpha=0.85,
        )

        ax1.set_ylabel("Score")
        ax1.set_title("Entity Boost Recall Quality Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Annotate bars (show raw avg_rank for the 3rd metric)
        raw_no_boost = [
            agg_off["hit_rate"],
            agg_off["mrr"],
            agg_off["avg_rank"],
            agg_off["avg_relevant_in_top_k"],
        ]
        raw_boost = [
            agg_on["hit_rate"],
            agg_on["mrr"],
            agg_on["avg_rank"],
            agg_on["avg_relevant_in_top_k"],
        ]
        for bars, raw_vals in [(bars1, raw_no_boost), (bars2, raw_boost)]:
            for bar, raw in zip(bars, raw_vals):
                height = bar.get_height()
                label = f"{raw:.2f}"
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, height + 0.02,
                    label, ha="center", va="bottom", fontsize=8,
                )

        # ── Second subplot: per-query score distributions ──
        ax2 = axes[1]
        mrr_off = [m["mrr"] for m in metrics_off]
        mrr_on = [m["mrr"] for m in metrics_on]
        rank_off = [m["avg_rank"] for m in metrics_off]
        rank_on = [m["avg_rank"] for m in metrics_on]
        rel_off = [m["relevant_in_top_k"] for m in metrics_off]
        rel_on = [m["relevant_in_top_k"] for m in metrics_on]

        box_data = [mrr_off, mrr_on, rank_off, rank_on, rel_off, rel_on]
        box_labels = [
            "MRR\n(No Boost)", "MRR\n(Boost)",
            "Avg Rank\n(No Boost)", "Avg Rank\n(Boost)",
            "Relevant in Top-K\n(No Boost)", "Relevant in Top-K\n(Boost)",
        ]
        colors = [
            "#e74c3c", "#2ecc71",
            "#e74c3c", "#2ecc71",
            "#e74c3c", "#2ecc71",
        ]

        # Normalize avg_rank and relevant_in_top_k to 0-1 scale for comparability
        norm_rank_off = [_norm_rank(r, top_k) for r in rank_off]
        norm_rank_on = [_norm_rank(r, top_k) for r in rank_on]
        norm_rel_off = [r / top_k for r in rel_off]
        norm_rel_on = [r / top_k for r in rel_on]

        box_data = [mrr_off, mrr_on, norm_rank_off, norm_rank_on, norm_rel_off, norm_rel_on]

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

        ax2.set_title("Per-Query Score Distributions (normalized, higher=better)")
        ax2.set_ylabel("Normalized Score")
        ax2.set_ylim(-0.05, 1.15)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", alpha=0.6, label="No Boost"),
            Patch(facecolor="#2ecc71", alpha=0.6, label="Entity Boost"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(png_file, dpi=150, bbox_inches="tight")
        print(f"  Graph saved to {png_file}")
        plt.close()

    except ImportError:
        print("  matplotlib not available, skipping graph")


# ── Main benchmark ──────────────────────────────────────────────────


def run_benchmark(
    *,
    num_memories: int,
    num_queries: int,
    top_k: int,
    results_file: Path,
    png_file: Path,
) -> dict[str, Any]:
    """Run the entity boost benchmark with explicit parameters.

    Returns the results dict.
    """
    print("=" * 74)
    print("  Entity Boost Recall Quality Benchmark")
    print("=" * 74)
    print(f"  Memories: {num_memories}  |  Queries: {num_queries}  |  Top-K: {top_k}")
    print()

    user_id = "bench_user"
    embed = _BenchEmbed()

    # Use temp file DB so async streaming shares state
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    store = SQLiteStorageAdapter(db_path=db_path)

    # ── Without entity boost ──
    print("  [1/2] Running recall WITHOUT entity boost...")
    config_off = MemoryConfig(enable_entity_boost=False, entity_boost_weight=0.0)
    mem_off = Memory(embed=embed, store=store, config=config_off)
    records = _seed_memories(mem_off, user_id, num_memories)
    queries = _make_queries(records, num_queries)

    metrics_off: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for q in queries:
        results = mem_off.recall(
            user_id=user_id,
            query=q["query"],
            top_k=top_k,
            hybrid_search=True,
        )
        m = _compute_metrics(results, set(q["relevant_memory_ids"]), top_k)
        m["query_id"] = q["query_id"]
        metrics_off.append(m)
    t_off = time.perf_counter() - t0

    # ── With entity boost ──
    print("  [2/2] Running recall WITH entity boost (weight=0.3)...")
    config_on = MemoryConfig(enable_entity_boost=True, entity_boost_weight=0.3)
    # Re-use the same store so we don't re-seed
    mem_on = Memory(embed=embed, store=store, config=config_on)

    metrics_on: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for q in queries:
        results = mem_on.recall(
            user_id=user_id,
            query=q["query"],
            top_k=top_k,
            hybrid_search=True,
        )
        m = _compute_metrics(results, set(q["relevant_memory_ids"]), top_k)
        m["query_id"] = q["query_id"]
        metrics_on.append(m)
    t_on = time.perf_counter() - t0

    # ── Aggregate ──
    agg_off = _agg(metrics_off)
    agg_on = _agg(metrics_on)

    _print_results(agg_off, agg_on, t_off, t_on)

    # ── Save JSON ──
    results = _build_results(
        num_memories, num_queries, top_k,
        agg_off, agg_on, metrics_off, metrics_on, t_off, t_on,
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # ── Generate graph ──
    _save_graph(agg_off, agg_on, top_k, png_file, metrics_off, metrics_on)

    # Cleanup
    import os as _os
    try:
        _os.unlink(db_path)
    except OSError:
        pass

    return results


def main() -> int:
    results = run_benchmark(
        num_memories=NUM_MEMORIES,
        num_queries=NUM_QUERIES,
        top_k=TOP_K,
        results_file=RESULTS_FILE,
        png_file=PNG_FILE,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
