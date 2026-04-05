from datetime import datetime
from typing import Callable, Optional

from kemi.models import MemoryObject


def cosine_similarity(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is None or empty to avoid division by zero.
    Never returns NaN.
    """
    if a is None or b is None or not a or not b:
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):
        dot_product += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    norm_a = norm_a**0.5
    norm_b = norm_b**0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def temporal_recency(last_accessed: datetime, half_life_hours: float = 168.0) -> float:
    """Compute temporal recency score using exponential decay.

    A memory accessed now scores 1.0.
    A memory accessed half_life_hours ago scores 0.5.
    A memory accessed 2x half_life_hours ago scores 0.25.

    Default half_life is 168 hours (7 days).
    """
    now = datetime.utcnow()
    hours_elapsed = (now - last_accessed).total_seconds() / 3600.0

    if hours_elapsed <= 0:
        return 1.0

    return 2.0 ** (-hours_elapsed / half_life_hours)


def score_memory(memory: MemoryObject, query_embedding: list[float]) -> float:
    """Compute final relevance score for a memory.

    Formula: (cosine_similarity × 0.5) + (temporal_recency × 0.3) + (importance × 0.2)

    If memory.embedding is None or query_embedding is empty, cosine contribution is 0.0.
    """
    cosine_score = 0.0
    if memory.embedding is not None and query_embedding is not None:
        similarity = cosine_similarity(memory.embedding, query_embedding)
        cosine_score = (similarity + 1.0) / 2.0

    recency_score = temporal_recency(memory.last_accessed_at)

    importance_score = max(0.0, min(1.0, memory.importance))

    return (cosine_score * 0.5) + (recency_score * 0.3) + (importance_score * 0.2)


def rank_memories(memories: list[MemoryObject], query_embedding: list[float]) -> list[MemoryObject]:
    """Rank memories by computed score, highest first.

    Mutates the score field on each MemoryObject in place.
    Returns the sorted list.
    """
    for memory in memories:
        memory.score = score_memory(memory, query_embedding)

    return sorted(memories, key=lambda m: m.score, reverse=True)


def _default_token_counter(text: str) -> int:
    """Default token counter: rough estimate = word_count * 1.3"""
    return int(len(text.split()) * 1.3)


def truncate_by_tokens(
    memories: list[MemoryObject],
    max_tokens: Optional[int],
    token_counter: Optional[Callable[[str], int]] = None,
) -> list[MemoryObject]:
    """Truncate memories by token budget.

    Walks ranked list, sums token counts, stops when budget reached.
    If max_tokens is None, returns all memories.
    If a single memory exceeds budget, includes it anyway.
    Never returns an empty list (if any input, returns at least one).
    """
    if max_tokens is None:
        return memories

    if not memories:
        return memories

    counter = token_counter or _default_token_counter
    result = []
    total_tokens = 0

    for memory in memories:
        memory_tokens = counter(memory.content)

        if result and total_tokens + memory_tokens > max_tokens:
            break

        result.append(memory)
        total_tokens += memory_tokens

    if not result and memories:
        result = [memories[0]]

    return result
