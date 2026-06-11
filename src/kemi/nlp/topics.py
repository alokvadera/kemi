"""Topic clustering for memories using local CPU-only methods.

Requires scikit-learn (optional dependency).
"""

import logging
from typing import Any

from kemi.exceptions import ConfigurationError
from kemi.memory.model import LifecycleState, MemoryObject

logger = logging.getLogger(__name__)


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def cluster_memories(
    store: Any,
    user_id: str,
    n_clusters: int = 3,
    namespace: str = "default",
) -> dict[str, list[MemoryObject]]:
    """Cluster a user's memories into topic groups using KMeans on embeddings.

    Args:
        store: StorageAdapter instance.
        user_id: User ID.
        n_clusters: Number of clusters. Auto-capped to number of memories.
        namespace: Memory namespace.

    Returns:
        Dict mapping topic label (e.g. "topic_0") to list of MemoryObjects.
    """
    if not _sklearn_available():
        raise ConfigurationError(
            "scikit-learn is required for topic clustering. Install with: pip install scikit-learn"
        )

    from sklearn.cluster import KMeans

    memories = store.get_all_by_user(
        user_id,
        lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
        namespace=namespace,
    )

    # Filter memories that have embeddings
    memories_with_emb = [m for m in memories if m.embedding is not None]

    if len(memories_with_emb) < 2:
        if memories_with_emb:
            return {"topic_0": memories_with_emb}
        return {}

    effective_k = min(n_clusters, len(memories_with_emb))
    if effective_k < 2:
        effective_k = 2

    embeddings = [m.embedding for m in memories_with_emb]

    try:
        kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
    except Exception as e:
        logger.warning(f"KMeans clustering failed: {e}")
        return {"topic_0": memories_with_emb}

    clusters: dict[str, list[MemoryObject]] = {}
    for mem, label in zip(memories_with_emb, labels, strict=False):
        key = f"topic_{label}"
        clusters.setdefault(key, []).append(mem)

    # Sort clusters by size (largest first) and rename by top keywords
    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))

    # Generate topic labels from top TF-like keywords
    labeled: dict[str, list[MemoryObject]] = {}
    for idx, (_, mems) in enumerate(sorted_clusters.items()):
        label = _generate_topic_label(mems, idx)
        labeled[label] = mems

    return labeled


def _generate_topic_label(memories: list[MemoryObject], index: int) -> str:
    """Generate a human-readable topic label from memory contents."""
    # Simple TF-like keyword extraction
    word_freq: dict[str, int] = {}
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "and",
        "but",
        "or",
        "yet",
        "so",
        "if",
        "because",
        "although",
        "though",
        "while",
        "where",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        "am",
    }

    for mem in memories:
        for word in mem.content.lower().split():
            clean = word.strip(".,!?;:'\"()-")
            if len(clean) > 3 and clean not in stopwords:
                word_freq[clean] = word_freq.get(clean, 0) + 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
    if top_words:
        label = " ".join(w[0].capitalize() for w in top_words)
    else:
        label = f"Topic {index + 1}"

    return label
