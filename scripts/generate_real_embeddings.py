#!/usr/bin/env python3
"""Generate real embeddings from fastembed + realistic synthetic vectors at scale.

Strategy:
1. Generate 1K real embeddings from fastembed (topic-diverse texts)
2. Analyze their statistical structure (cluster centers by topic)
3. Generate realistic synthetic embeddings at larger scales preserving that structure
"""

import math
import pickle
import random
import statistics
import time
from pathlib import Path

DIM = 384
REAL_COUNT = 1000
QUERY_COUNT = 20
TARGET_SCALES = [1000, 5000, 10000, 25000, 50000]

# Topic-tagged templates — each template returns (text, topic_name)
TOPIC_TEMPLATES = [
    # technology
    ("The scalable database query executed in {num} milliseconds on the GPU cluster", "tech"),
    ("Container orchestration manages {num} microservices across {num} nodes", "tech"),
    ("The API gateway processed {num} requests per second with low latency", "tech"),
    ("Kernel version {num} introduced support for new filesystem drivers", "tech"),
    ("Debugging production issues requires structured logging and distributed tracing", "tech"),
    ("The compiler optimized the code by {num} percent using auto-vectorization", "tech"),
    # science
    ("The experiment measured {num} particles at high energy levels in the collider", "science"),
    ("Quantum computing achieved {num} qubit coherence for sustained microseconds", "science"),
    ("The protein folded successfully in milliseconds using molecular simulation", "science"),
    ("CRISPR gene editing modified {num} base pairs with high efficiency", "science"),
    ("The telescope detected {num} new exoplanets in a nearby stellar system", "science"),
    ("Neural reconstruction mapped {num} synapses from high-resolution microscopy data", "science"),
    # sports
    ("The forward scored {num} goals in league matches this season", "sports"),
    ("The athlete ran the hundred meters in record time at the Olympic stadium", "sports"),
    ("The defensive strategy reduced opponent scoring by {num} percent", "sports"),
    ("The team won multiple consecutive championships over several years", "sports"),
    ("The swimmer broke the world record by fractions of a second", "sports"),
    ("The new stadium seats tens of thousands of fans across multiple tiers", "sports"),
    # politics
    ("The senator proposed several amendments to the new policy bill", "politics"),
    ("Voter turnout reached record levels in the recent election", "politics"),
    ("The international treaty reduced carbon emissions across participating nations", "politics"),
    ("The budget allocated billions of dollars for infrastructure projects", "politics"),
    ("The court ruled on the constitutional challenge this term", "politics"),
    ("Diplomatic negotiations lasted many months across multiple sessions", "politics"),
    # health
    ("The clinical trial enrolled hundreds of patients across multiple medical centers", "health"),
    ("The vaccine showed strong efficacy against the viral strain in trials", "health"),
    ("The new surgical technique reduced recovery time significantly for patients", "health"),
    ("Telomere length decreased measurably in aging cell populations", "health"),
    ("The diagnostic test detected multiple biomarkers with high accuracy", "health"),
    ("Regular exercise improved cardiovascular function over several weeks", "health"),
]

rng = random.Random(42)


def generate_texts(count):
    """Generate (text, topic) pairs."""
    samples = []
    while len(samples) < count:
        template, topic = rng.choice(TOPIC_TEMPLATES)
        text = template.format(num=rng.randint(1, 9999))
        samples.append((text, topic))
    return samples


def normalize(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else v


def main():
    cache = Path(__file__).resolve().parent / "real_embeddings_cache.pkl"

    if cache.exists():
        print(f"Loading cached embeddings from {cache}...", flush=True)
        with open(cache, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded: scales={list(data['synthetic'].keys())}, real_embs={len(data.get('real_embeddings', []))}", flush=True)
        return

    print("Generating topic-diverse texts...", flush=True)
    all_samples = generate_texts(REAL_COUNT + QUERY_COUNT)
    train_samples = all_samples[:REAL_COUNT]
    query_samples = all_samples[REAL_COUNT:]

    print(f"Generated {REAL_COUNT} training + {QUERY_COUNT} query texts", flush=True)

    print("\nLoading fastembed model (BAAI/bge-small-en-v1.5)...", flush=True)
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print(f"\nEmbedding {REAL_COUNT} training texts...", flush=True)
    train_texts = [s[0] for s in train_samples]
    real_embs = []
    batch_size = 100
    for i in range(0, REAL_COUNT, batch_size):
        batch = train_texts[i:i+batch_size]
        for emb in model.embed(batch):
            real_embs.append(normalize(emb.tolist()))
        print(f"  {min(i+batch_size, REAL_COUNT)}/{REAL_COUNT}", flush=True)

    print(f"Embedding {QUERY_COUNT} query texts...", flush=True)
    query_texts = [s[0] for s in query_samples]
    query_embs = []
    for i in range(0, QUERY_COUNT, batch_size):
        batch = query_texts[i:i+batch_size]
        for emb in model.embed(batch):
            query_embs.append(normalize(emb.tolist()))

    # Group by topic for cluster analysis
    topic_to_indices = {}
    for i, (_, topic) in enumerate(train_samples):
        topic_to_indices.setdefault(topic, []).append(i)

    print(f"\nTopic distribution: { {t: len(idxs) for t, idxs in topic_to_indices.items()} }", flush=True)

    # Compute centroids per topic
    centroids = {}
    for topic, indices in topic_to_indices.items():
        if len(indices) >= 2:
            centroid = [0.0] * DIM
            for idx in indices:
                for j in range(DIM):
                    centroid[j] += real_embs[idx][j]
            centroid = [v / len(indices) for v in centroid]
            centroids[topic] = normalize(centroid)

    print(f"Computed centroids for topics: {list(centroids.keys())}", flush=True)

    # Analyze intra vs inter-cluster similarity
    intra_sims, inter_sims = [], []
    for i in range(REAL_COUNT):
        topic_i = next(t for t, idxs in topic_to_indices.items() if i in idxs)
        for j in range(i + 1, REAL_COUNT):
            topic_j = next(t for t, idxs in topic_to_indices.items() if j in idxs)
            dot = sum(a*b for a,b in zip(real_embs[i], real_embs[j]))
            if topic_i == topic_j:
                intra_sims.append(dot)
            else:
                inter_sims.append(dot)

    intra_mean = statistics.mean(intra_sims) if intra_sims else 0
    inter_mean = statistics.mean(inter_sims) if inter_sims else 0
    all_sims = intra_sims + inter_sims
    overall_mean = statistics.mean(all_sims)
    overall_std = statistics.stdev(all_sims) if len(all_sims) > 1 else 0

    print(f"\nReal embedding structure:")
    print(f"  Overall mean sim: {overall_mean:.4f} (std: {overall_std:.4f})")
    print(f"  Intra-topic sim: {intra_mean:.4f} ({len(intra_sims)} pairs)")
    print(f"  Inter-topic sim: {inter_mean:.4f} ({len(inter_sims)} pairs)")
    ratio = intra_mean / inter_mean if inter_mean else float('inf')
    print(f"  Intra/Inter ratio: {ratio:.2f}x")

    # Calibrate noise level: avg angular distance from centroid within each topic
    topic_noise = {}
    for topic, indices in topic_to_indices.items():
        c = centroids[topic]
        ang_dists = []
        for idx in indices:
            dot = sum(a*b for a,b in zip(real_embs[idx], c))
            ang_dists.append(math.acos(max(-1, min(1, dot))))
        topic_noise[topic] = statistics.mean(ang_dists) if ang_dists else 0.3

    avg_noise = statistics.mean(topic_noise.values()) if topic_noise else 0.3
    print(f"  Avg angular noise from centroid: {avg_noise:.4f} rad")

    # Generate realistic synthetic vectors at each scale
    print(f"\nGenerating synthetic vectors at scales: {TARGET_SCALES}", flush=True)
    topic_names = list(centroids.keys())
    n_topics = len(topic_names)
    rng_gen = random.Random(42)

    all_synthetic = {}
    for scale in TARGET_SCALES:
        vecs = []
        for i in range(scale):
            topic = topic_names[i % n_topics]
            c = centroids[topic]
            noise = [rng_gen.gauss(0, avg_noise) for _ in range(DIM)]
            v = [c[j] + noise[j] for j in range(DIM)]
            vecs.append(normalize(v))
        all_synthetic[str(scale)] = vecs
        print(f"  Scale {scale}: generated", flush=True)

    # Generate synthetic queries (closer to centroids, as queries tend to be)
    q_rng = random.Random(99)
    synthetic_queries = []
    for i in range(QUERY_COUNT):
        topic = topic_names[i % n_topics]
        c = centroids[topic]
        noise = [q_rng.gauss(0, avg_noise * 0.4) for _ in range(DIM)]
        v = [c[j] + noise[j] for j in range(DIM)]
        synthetic_queries.append(normalize(v))

    # Validate synthetic structure
    all_vecs = all_synthetic[str(TARGET_SCALES[-1])]
    val_intra, val_inter = [], []
    for i in range(min(200, len(all_vecs))):
        ti = topic_names[i % n_topics]
        for j in range(i + 1, min(200, len(all_vecs))):
            tj = topic_names[j % n_topics]
            dot = sum(a*b for a,b in zip(all_vecs[i], all_vecs[j]))
            if ti == tj:
                val_intra.append(dot)
            else:
                val_inter.append(dot)

    print(f"\nSynthetic structure validation (at {TARGET_SCALES[-1]}):")
    print(f"  Intra-topic sim: {statistics.mean(val_intra):.4f}" if val_intra else "  N/A")
    print(f"  Inter-topic sim: {statistics.mean(val_inter):.4f}" if val_inter else "  N/A")

    # Save dataset
    dataset = {
        "real_embeddings": real_embs,
        "query_embeddings_real": query_embs,
        "query_embeddings_synthetic": synthetic_queries,
        "synthetic": all_synthetic,
        "structure": {
            "overall_mean_sim": overall_mean,
            "overall_std": overall_std,
            "intra_topic_sim": intra_mean,
            "inter_topic_sim": inter_mean,
            "intra_inter_ratio": ratio,
            "angular_noise_std": avg_noise,
        }
    }
    with open(cache, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nCached to {cache}")
    print("Ready for benchmark.")


if __name__ == "__main__":
    main()
