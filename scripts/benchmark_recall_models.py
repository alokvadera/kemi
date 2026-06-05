#!/usr/bin/env python3
"""Benchmark recall@10 across multiple fastembed models.

Runs at small scale (300 texts, 10 queries per model) so it's fast enough
for CI. Tests ANN vs brute-force recall for each model and reports results.

Exit code: 0 if all tested models meet the recall threshold, 1 otherwise.
"""

import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kemi.adapters.storage.sqlite_vec import SQLiteVecStorageAdapter, _SQLITE_VEC_AVAILABLE
from kemi.models import LifecycleState, MemoryObject, MemorySource

# ── Models to test ────────────────────────────────────────────────────
# Small, fast models suitable for CI. Each tuple: (name, min_recall_threshold)
MODELS = [
    ("BAAI/bge-small-en-v1.5", 0.15),
    ("sentence-transformers/all-MiniLM-L6-v2", 0.15),
]

TEXTS_PER_MODEL = 300
QUERIES_PER_MODEL = 10
TOP_K = 10

# ── Topic-diverse text templates ──────────────────────────────────────
TOPIC_TEMPLATES = {
    "technology": [
        "The database query executed in {n} milliseconds on the GPU cluster",
        "Container orchestration manages {n} microservices across {n} nodes",
        "The API gateway processed {n} requests per second with low latency",
        "Kernel {n} introduced support for new filesystem drivers",
        "Debugging production issues requires structured logging and tracing",
        "The compiler optimized the code by {n} percent using auto-vectorization",
        "Serverless functions scaled from {n} to {n} concurrent invocations",
        "The CI pipeline runs {n} tests across {n} parallel build agents",
        "Memory usage dropped by {n} percent after migrating to a new runtime",
        "Developer experience improved with hot reloading and type checking",
    ],
    "science": [
        "The experiment measured {n} particles at high energy levels",
        "Quantum computing achieved {n} qubit coherence for microsecond durations",
        "The protein folded successfully using molecular dynamics simulation",
        "CRISPR gene editing modified {n} base pairs with high efficiency",
        "The telescope detected {n} new exoplanets in a nearby stellar system",
        "Neural reconstruction mapped {n} synapses from microscopy data",
        "Radioactive decay measured with a half-life of {n} million years",
        "The ecosystem contained {n} species across {n} trophic levels",
        "Ocean currents transported {n} cubic meters of water daily",
        "Atmospheric carbon dioxide levels reached {n} parts per million",
    ],
    "sports": [
        "The forward scored {n} goals in league matches this season",
        "The athlete ran the hundred meters in record time",
        "The defensive strategy reduced opponent scoring by {n} percent",
        "The team won {n} consecutive championships in recent years",
        "The swimmer broke the world record by fractions of a second",
        "The stadium seats tens of thousands of fans across multiple tiers",
        "Basketball player achieved a high free throw rate this season",
        "The marathon course covers many kilometers with elevation gain",
        "The tennis player served many aces at high speed",
        "The goalkeeper made crucial saves in the championship match",
    ],
    "politics": [
        "The senator proposed several amendments to the policy bill",
        "Voter turnout reached record levels in the recent election",
        "The international treaty reduced emissions across participating nations",
        "The budget allocated billions for infrastructure projects",
        "The court ruled on the constitutional challenge this term",
        "Diplomatic negotiations lasted many months across multiple sessions",
        "The reform bill passed by a narrow margin in parliament",
        "Public spending increased over several fiscal quarters",
        "The ambassador met with officials to discuss cooperation",
        "Public opinion polls showed growing support for the initiative",
    ],
    "health": [
        "The clinical trial enrolled patients across multiple medical centers",
        "The vaccine showed strong efficacy against the viral strain",
        "The new surgical technique reduced recovery time significantly",
        "Telomere length decreased measurably in aging cell populations",
        "The diagnostic test detected biomarkers with high accuracy",
        "Regular exercise improved cardiovascular function over time",
        "The therapy reduced symptoms compared to standard treatment",
        "Microbiome diversity increased after dietary intervention",
        "The hospital treated many emergency cases with limited staff",
        "Sleep quality improved using consistent circadian protocols",
    ],
}


def generate_texts(count: int, seed: int = 42) -> list[str]:
    """Generate `count` topic-diverse texts."""
    rng = random.Random(seed)
    topic_names = list(TOPIC_TEMPLATES.keys())
    texts = []
    while len(texts) < count:
        topic = rng.choice(topic_names)
        template = rng.choice(TOPIC_TEMPLATES[topic])
        text = template.format(n=rng.randint(1, 9999))
        texts.append(text)
    return texts[:count]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na * nb > 0 else 0


def brute_force_ground_truth(
    embeddings: list[list[float]], query: list[float], top_k: int
) -> list[int]:
    scores = [(cosine_similarity(e, query), i) for i, e in enumerate(embeddings)]
    scores.sort(key=lambda x: -x[0])
    return [i for _, i in scores[:top_k]]


def compute_recall(embeddings: list[list[float]],
                   queries: list[list[float]],
                   memory_ids: list[str]) -> dict:
    """Benchmark ANN vs brute-force recall using sqlite-vec.

    Returns dict with recall metrics.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return {"error": "sqlite-vec not available", "recall": 0.0}

    dim = len(embeddings[0])
    user_id = "bench"

    # Store memories in ANN adapter
    adapter = SQLiteVecStorageAdapter(db_path=":memory:", embedding_dim=dim)
    for i, emb in enumerate(embeddings):
        store = MemoryObject(
            memory_id=memory_ids[i],
            user_id=user_id,
            content=f"Memory {i}",
            embedding=emb,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=dim,
            tags=[],
        )
        adapter.store(store)

    # For each query, get ANN results and brute-force ground truth
    all_recalls = []
    for qemb in queries:
        ann_results = adapter.search(user_id, qemb, top_k=TOP_K)
        ann_ids = set(r.memory_id for r in ann_results)
        truth_ids = set(memory_ids[i] for i in brute_force_ground_truth(embeddings, qemb, TOP_K))
        overlap = len(ann_ids & truth_ids)
        all_recalls.append(overlap / TOP_K)

    adapter.close()
    return {
        "recall": statistics.mean(all_recalls),
        "recalls_per_query": all_recalls,
        "dim": dim,
        "num_vectors": len(embeddings),
    }


def main() -> int:
    print("=" * 70)
    print("  RECALL BENCHMARK: ANN vs Brute-Force Across Embedding Models")
    print(f"  Texts per model: {TEXTS_PER_MODEL}  |  Queries: {QUERIES_PER_MODEL}"
          f"  |  Top-K: {TOP_K}")
    print("=" * 70)

    results = {}

    for model_name, threshold in MODELS:
        print(f"\n{'─' * 70}")
        print(f"  Model: {model_name}  (min recall threshold: {threshold*100:.0f}%)")
        print(f"{'─' * 70}")

        # Load model
        try:
            from fastembed import TextEmbedding
            print(f"  Loading model... ", end="", flush=True)
            t0 = time.perf_counter()
            model = TextEmbedding(model_name=model_name)
            load_time = time.perf_counter() - t0
            print(f"{load_time:.1f}s")
        except Exception as e:
            msg = f"  FAILED to load model: {e}"
            print(msg)
            results[model_name] = {"error": str(e), "recall": 0.0, "passed": False}
            continue

        # Generate texts
        all_texts = generate_texts(TEXTS_PER_MODEL + QUERIES_PER_MODEL, seed=42)
        train_texts = all_texts[:TEXTS_PER_MODEL]
        query_texts = all_texts[TEXTS_PER_MODEL:]
        print(f"  Generated {len(train_texts)} training + {len(query_texts)} query texts")

        # Embed training texts
        print(f"  Embedding training texts... ", end="", flush=True)
        t0 = time.perf_counter()
        train_embs = []
        for emb in model.embed(train_texts):
            train_embs.append(emb.tolist())
        embed_time = time.perf_counter() - t0
        dim = len(train_embs[0]) if train_embs else 0
        print(f"{embed_time:.2f}s ({TEXTS_PER_MODEL/embed_time:.0f} texts/s)  dim={dim}")

        # Embed query texts
        print(f"  Embedding query texts... ", end="", flush=True)
        query_embs = []
        for emb in model.embed(query_texts):
            query_embs.append(emb.tolist())
        print(f"{len(query_embs)} done")

        # Compute recall
        print(f"  Computing recall@10... ", end="", flush=True)
        t0 = time.perf_counter()
        mem_ids = [f"mem-{i}" for i in range(TEXTS_PER_MODEL)]
        recall_result = compute_recall(train_embs, query_embs, mem_ids)
        recall_time = time.perf_counter() - t0

        if "error" in recall_result:
            print(f"ERROR: {recall_result['error']}")
            results[model_name] = {"error": recall_result["error"], "recall": 0.0, "passed": False}
            continue

        recall = recall_result["recall"]
        passed = recall >= threshold

        # Print per-query breakdown
        print(f"\n  Per-query recall:")
        for i, r in enumerate(recall_result["recalls_per_query"]):
            bar = "█" * int(r * 20) + "░" * (20 - int(r * 20))
            print(f"    Q{i+1:02d}: {r*100:>5.1f}% {bar}")
        print(f"\n  Mean Recall@10: {recall*100:.1f}%  " +
              ("✅ PASS" if passed else "❌ FAIL") +
              f"  (threshold: {threshold*100:.0f}%)")
        print(f"  Benchmark time: {recall_time:.2f}s")
        print(f"  Vectors: {recall_result['num_vectors']}  |  Dim: {recall_result['dim']}")

        results[model_name] = {
            "recall": round(recall, 4),
            "threshold": threshold,
            "passed": passed,
            "dim": dim,
            "num_vectors": TEXTS_PER_MODEL,
            "load_time_s": round(load_time, 2),
            "embed_time_s": round(embed_time, 2),
            "recall_time_s": round(recall_time, 2),
            "embed_speed_texts_per_s": round(TEXTS_PER_MODEL / embed_time, 0),
        }

    # ── Final summary ──
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':55s} {'Recall':>8} {'Threshold':>10} {'Status':>8}")
    print("  " + "-" * 70)
    all_passed = True
    for model_name, result in results.items():
        if "error" in result:
            print(f"  {model_name:55s} {'ERROR':>8} {'N/A':>10} {'⚠️ SKIP':>8}")
            continue
        recall_pct = result["recall"] * 100
        thresh_pct = result["threshold"] * 100
        status = "✅" if result["passed"] else "❌"
        all_passed = all_passed and result["passed"]
        print(f"  {model_name:55s} {recall_pct:>7.1f}% {thresh_pct:>9.0f}% {status:>8}")

    print("  " + "-" * 70)
    models_tested = sum(1 for r in results.values() if "error" not in r)
    if models_tested == 0:
        print("  ⚠️ NO MODELS WERE SUCCESSFULLY TESTED")
        all_passed = False
    elif all_passed:
        print(f"  ✅ ALL {models_tested} MODELS PASSED")
    else:
        print("  ❌ SOME MODELS FAILED (recall below threshold)")
    print()

    # ── Save results ──
    out_path = Path(__file__).resolve().parent / "benchmark_recall_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "models": {
                name: {k: v for k, v in r.items() if k != "recalls_per_query"}
                for name, r in results.items()
            },
            "summary": {
                "all_passed": all_passed,
                "num_models": len(results),
                "texts_per_model": TEXTS_PER_MODEL,
                "top_k": TOP_K,
            },
        }, f, indent=2)
    print(f"  Results saved to {out_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
