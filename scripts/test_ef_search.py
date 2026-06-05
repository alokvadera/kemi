#!/usr/bin/env python3
"""Test PRAGMA vec_ef_search recall/latency tradeoff at various values."""

import math
import random
import statistics
import time
import sqlite3
import sqlite_vec

DIM = 384
SCALE = 10000
QUERIES = 20
TOP_K = 10
RNG = random.Random(42)


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na * nb > 0 else 0


def brute_force_ground_truth(embeddings, query):
    scores = [(cosine_similarity(e, query), i) for i, e in enumerate(embeddings)]
    scores.sort(key=lambda x: -x[0])
    return {f"mem-{idx}" for _, idx in scores[:TOP_K]}


def test_with_clusters():
    """Test with structured data (clustered vectors) for realistic recall."""
    print("\n" + "=" * 70)
    print("  With CLUSTERED data (simulating real embeddings)")
    print("=" * 70)

    # Generate 10 clusters of vectors
    cluster_centers = [[RNG.random() for _ in range(DIM)] for _ in range(10)]
    clustered = []
    for i in range(SCALE):
        center = cluster_centers[i % 10]
        noise = [(RNG.random() - 0.5) * 0.3 for _ in range(DIM)]
        vec = [c + n for c, n in zip(center, noise)]
        vec = [v / math.sqrt(sum(x * x for x in vec)) for v in vec]  # normalize
        clustered.append(vec)

    # Queries near cluster centers
    q_rng = random.Random(99)
    clustered_queries = []
    for cc in cluster_centers[:5]:
        noise = [(q_rng.random() - 0.5) * 0.2 for _ in range(DIM)]
        vec = [c + n for c, n in zip(cc, noise)]
        vec = [v / math.sqrt(sum(x * x for x in vec)) for v in vec]
        clustered_queries.append(vec)

    # Ground truth
    ctruth = [brute_force_ground_truth(clustered, q) for q in clustered_queries]

    # Build vec0
    db2 = sqlite3.connect(":memory:")
    db2.enable_load_extension(True)
    sqlite_vec.load(db2)
    db2.enable_load_extension(False)
    db2.row_factory = sqlite3.Row
    db2.execute(f"CREATE VIRTUAL TABLE cv USING vec0(embedding float[{DIM}], memory_id text)")
    for i, emb in enumerate(clustered):
        db2.execute("INSERT INTO cv(rowid, embedding, memory_id) VALUES (?, ?, ?)",
                     (i + 1, json.dumps(emb), f"mem-{i}"))

    print(f"\n  {'ef_search':>10} | {'Recall@10':>10} | {'Latency(ms)':>12} | {'Slowdown':>10}")
    print("  " + "-" * 48)

    for ef in [1, 10, 50, 100, 200, 400]:
        db2.execute(f"PRAGMA vec_ef_search={ef}")
        recalls, latencies = [], []
        for qi, qemb in enumerate(clustered_queries):
            t0 = time.perf_counter()
            rows = db2.execute(
                "SELECT memory_id, distance FROM cv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                [json.dumps(qemb), TOP_K],
            ).fetchall()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            ann_ids = {r["memory_id"] for r in rows}
            overlap = len(ann_ids & ctruth[qi])
            recalls.append(overlap / TOP_K)

        avg_recall = statistics.mean(recalls) * 100
        avg_latency = statistics.median(latencies)
        slowdown = avg_latency / latencies[0] if avg_latency > 0 else 0
        speed_str = f"{slowdown:>9.1f}x" if ef != ef_values[0] else "         -"
        print(f"  {ef:>10} | {avg_recall:>9.1f}% | {avg_latency:>10.2f}ms | {speed_str}")

    db2.close()


def main():
    print("=" * 70)
    print("  sqlite-vec: ef_search Recall vs Latency Tradeoff")
    print(f"  Scale: {SCALE} vectors, {DIM}-dim, {QUERIES} queries, top-{TOP_K}")
    print("=" * 70)

    # Generate data
    print("\n  Generating data... ", end="", flush=True)
    embeddings = [[RNG.random() for _ in range(DIM)] for _ in range(SCALE)]
    q_rng = random.Random(99)
    queries = [[q_rng.random() for _ in range(DIM)] for _ in range(QUERIES)]
    print("done")

    # Brute-force ground truth (once)
    print("  Computing brute-force ground truth... ", end="", flush=True)
    truth = [brute_force_ground_truth(embeddings, q) for q in queries]
    print("done")

    # Build vec0 table
    print("  Building vec0 index... ", end="", flush=True)
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row

    db.execute(
        f"CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[{DIM}], memory_id text)"
    )
    for i, emb in enumerate(embeddings):
        db.execute(
            "INSERT INTO memories_vec(rowid, embedding, memory_id) VALUES (?, ?, ?)",
            (i + 1, json.dumps(emb), f"mem-{i}"),
        )
    print("done")

    # Test different ef_search values
    ef_values = [1, 10, 50, 100, 200, 400, 800, 1600]

    print(f"\n  {'ef_search':>10} | {'Recall@10':>10} | {'Latency(ms)':>12} | {'Speed vs 1':>10}")
    print("  " + "-" * 48)

    for ef in ef_values:
        db.execute(f"PRAGMA vec_ef_search={ef}")

        recalls = []
        latencies = []

        for qi, qemb in enumerate(queries):
            t0 = time.perf_counter()
            rows = db.execute(
                f"SELECT memory_id, distance FROM memories_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                [json.dumps(qemb), TOP_K],
            ).fetchall()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            ann_ids = {r["memory_id"] for r in rows}
            overlap = len(ann_ids & truth[qi])
            recalls.append(overlap / TOP_K)

        avg_recall = statistics.mean(recalls) * 100
        avg_latency = statistics.median(latencies)
        speed_vs_1 = latencies[0] / avg_latency if avg_latency > 0 else float("inf")

        speed_str = f"{speed_vs_1:>9.1f}x" if ef != 1 else "         -"
        print(f"  {ef:>10} | {avg_recall:>9.1f}% | {avg_latency:>10.2f}ms | {speed_str}")

    print(f"  {'─' * 48}")
    print(f"  * Recall is vs brute-force ground truth (cosine similarity)")
    print(f"  * Latency is median of {QUERIES} queries")
    print(f"  * Speed vs 1: how much slower than ef_search=1 (baseline)")


if __name__ == "__main__":
    import json
    main()
