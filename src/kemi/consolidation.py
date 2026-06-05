"""Memory consolidation: summarize old episodic memories into semantic ones.

Supports both local extractive summarization (no LLM required) and optional
LLM-powered abstractive summarization via :class:`kemi.summarizer.LLMSummarizer`.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from kemi import lifecycle
from kemi.models import LifecycleState, MemoryObject, MemorySource, MemoryType
from kemi.scoring import cosine_similarity

logger = logging.getLogger(__name__)


def _get_summarizer(
    with_llm_summary: bool,
    summarizer_llm_provider: str | None = None,
    summarizer_llm_model: str | None = None,
    summarizer_prompt_template: str | None = None,
) -> Any | None:
    """Return an LLMSummarizer instance if LLM summarization is requested.

    Returns None if *with_llm_summary* is False or the summarizer module is
    not available.  On import failure the error is logged and None is returned
    so caller can fall back to extractive summarization gracefully.
    """
    if not with_llm_summary:
        return None
    try:
        from kemi.summarizer import LLMSummarizer

        return LLMSummarizer(
            provider=summarizer_llm_provider or "openai",
            model=summarizer_llm_model,
            prompt_template=summarizer_prompt_template,
        )
    except Exception:
        logger.warning(
            "LLM summarizer not available, falling back to extractive summary",
            exc_info=True,
        )
        return None


def consolidate_cluster(
    store: Any,
    embed: Any,
    user_id: str,
    cluster: list[MemoryObject],
    namespace: str = "default",
    summarizer: Any | None = None,
) -> MemoryObject | None:
    """Consolidate a single cluster of related memories into a semantic summary.

    Generates either an extractive or LLM-powered summary (if *summarizer* is
    provided), embeds it, and returns a ``MemoryObject`` ready for storage.

    Args:
        store: StorageAdapter instance (used only for lifecycle transitions).
        embed: EmbeddingAdapter instance.
        user_id: User to consolidate.
        cluster: List of related MemoryObjects to consolidate.
        namespace: Memory namespace.
        summarizer: Optional ``LLMSummarizer`` instance for abstractive summaries.

    Returns:
        A ``MemoryObject`` representing the consolidated summary, or None if
        the cluster is empty.
    """
    if not cluster:
        return None

    # Always generate extractive summary as the canonical content
    summary_text = _extractive_summary(cluster)

    # If LLM summarizer is available, generate abstractive summary as metadata
    metadata: dict[str, Any] = {
        "consolidated_from": [m.memory_id for m in cluster],
        "consolidated_count": len(cluster),
    }
    if summarizer is not None:
        contents = [m.content for m in cluster]
        llm_summary = summarizer.summarize(contents)
        if llm_summary:
            metadata["llm_summary"] = llm_summary

    summary_embedding = embed.embed_single(summary_text)

    summary_memory = MemoryObject(
        memory_id=str(uuid.uuid4()),
        user_id=user_id,
        content=summary_text,
        embedding=summary_embedding,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.SYSTEM_GENERATED,
        importance=0.7,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata=metadata,
        embedding_dim=len(summary_embedding),
        memory_type=MemoryType.SEMANTIC,
        namespace=namespace,
    )

    # Archive old memories
    for mem in cluster:
        archived = lifecycle.transition(mem, LifecycleState.ARCHIVED)
        store.update(archived)

    return summary_memory


def consolidate(
    store: Any,
    embed: Any,
    user_id: str,
    namespace: str = "default",
    min_memories: int = 5,
    max_age_days: float = 30.0,
    with_llm_summary: bool = False,
    summarizer_llm_provider: str | None = None,
    summarizer_llm_model: str | None = None,
    summarizer_prompt_template: str | None = None,
) -> str | None:
    """Consolidate old episodic memories into a semantic summary.

    Algorithm:
    1. Fetch old EPISODIC memories
    2. Cluster them by semantic similarity
    3. For each cluster, generate a summary (extractive or LLM-powered)
    4. Store the summary as a SEMANTIC memory
    5. Mark old memories as ARCHIVED

    Args:
        store: StorageAdapter instance.
        embed: EmbeddingAdapter instance.
        user_id: User to consolidate.
        namespace: Memory namespace.
        min_memories: Minimum memories needed to form a cluster.
        max_age_days: Only consider memories older than this many days.
        with_llm_summary: If True, use LLM-powered abstractive summarization
            instead of extractive. Falls back to extractive on failure.
        summarizer_llm_provider: LLM provider ("openai", "anthropic",
            "ollama", "custom").  Default "openai".
        summarizer_llm_model: Model name override.
        summarizer_prompt_template: Custom prompt template with ``{memories}``.

    Returns:
        Memory ID of the consolidated summary, or None if consolidation did not occur.
    """
    all_memories = store.get_all_by_user(
        user_id,
        lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
        namespace=namespace,
    )

    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - (max_age_days * 86400)

    # Filter old episodic memories
    old_episodic = [
        m
        for m in all_memories
        if m.memory_type == MemoryType.EPISODIC
        and m.created_at.timestamp() < cutoff
        and m.embedding is not None
    ]

    if len(old_episodic) < min_memories:
        logger.info(
            f"Consolidation skipped for {user_id}: only {len(old_episodic)} "
            f"old episodic memories (min={min_memories})"
        )
        return None

    # Simple greedy clustering by similarity
    clusters = _cluster_by_similarity(old_episodic, threshold=0.75)

    best_cluster = max(clusters, key=len)
    if len(best_cluster) < min_memories:
        logger.info(
            f"Consolidation skipped for {user_id}: best cluster has "
            f"{len(best_cluster)} memories (min={min_memories})"
        )
        return None

    # Initialize LLM summarizer if requested
    summarizer = _get_summarizer(
        with_llm_summary,
        summarizer_llm_provider,
        summarizer_llm_model,
        summarizer_prompt_template,
    )

    # Consolidate the best cluster
    summary_memory = consolidate_cluster(
        store=store,
        embed=embed,
        user_id=user_id,
        cluster=best_cluster,
        namespace=namespace,
        summarizer=summarizer,
    )

    if summary_memory is None:
        logger.warning(f"Consolidation failed for {user_id}: cluster processing returned None")
        return None

    # Store the consolidated memory
    store.store(summary_memory)
    logger.info(
        f"Consolidated {len(best_cluster)} memories for {user_id} "
        f"into semantic memory {summary_memory.memory_id}"
        + (" (LLM summary)" if summarizer else "")
    )

    return summary_memory.memory_id


def _cluster_by_similarity(
    memories: list[MemoryObject],
    threshold: float = 0.75,
) -> list[list[MemoryObject]]:
    """Greedy clustering of memories by embedding similarity."""
    clusters: list[list[MemoryObject]] = []
    unassigned = list(memories)

    while unassigned:
        seed = unassigned.pop(0)
        cluster = [seed]
        to_remove: list[int] = []

        for i, candidate in enumerate(unassigned):
            if candidate.embedding is None:
                continue
            sim = cosine_similarity(seed.embedding, candidate.embedding)
            normalized = (sim + 1.0) / 2.0
            if normalized >= threshold:
                cluster.append(candidate)
                to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(to_remove):
            unassigned.pop(i)

        clusters.append(cluster)

    return clusters


def _extractive_summary(memories: list[MemoryObject]) -> str:
    """Generate an extractive summary from a cluster of memories.

    Uses a simple TextRank-like approach: score sentences by their
    average similarity to all other sentences, then pick the top ones.
    """
    # Collect all sentences
    sentences: list[str] = []
    for mem in memories:
        for sent in mem.content.split("."):
            sent = sent.strip()
            if len(sent) > 10:
                sentences.append(sent)

    if not sentences:
        return " ".join(m.content for m in memories[:3])

    if len(sentences) <= 3:
        return ". ".join(sentences) + "."

    # Simple TF-IDF-like scoring: pick sentences with most overlap in keywords
    word_counts: dict[str, int] = {}
    for sent in sentences:
        for word in sent.lower().split():
            clean = word.strip(".,!?;:'\"()-")
            if len(clean) > 2:
                word_counts[clean] = word_counts.get(clean, 0) + 1

    # Score each sentence by sum of word frequencies
    sentence_scores: list[tuple[str, float]] = []
    for sent in sentences:
        score = 0.0
        words = sent.lower().split()
        for word in words:
            clean = word.strip(".,!?;:'\"()-")
            if len(clean) > 2:
                score += word_counts.get(clean, 0)
        sentence_scores.append((sent, score / max(len(words), 1)))

    # Sort by score and pick top sentences (up to 3)
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in sentence_scores[:3]]
    top_set = set(top_sentences)

    # Preserve original order; set lookup is O(1) instead of O(n) per element
    ordered = [s for s in sentences if s in top_set]

    summary = ". ".join(ordered) + "."
    # Cap summary length to avoid blowing up embedding or storage with
    # a giant consolidated memory.
    return summary[:1024]
