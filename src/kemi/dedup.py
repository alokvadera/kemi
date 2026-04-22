from datetime import datetime, timezone

from kemi.models import MemoryObject
from kemi.scoring import cosine_similarity

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "can't",
    "cannot",
    "hate",
    "dislike",
    "avoid",
    "stop",
    "stopped",
    "quit",
    "quitting",
    "anymore",
    "no longer",
    "ceased",
}

SENTIMENT_SHIFT_PAIRS = [
    ("love", "hate"),
    ("like", "dislike"),
    ("enjoy", "avoid"),
    ("always", "never"),
    ("do", "don't"),
    ("will", "won't"),
    ("can", "can't"),
    ("start", "stop"),
    ("begin", "quit"),
    ("am", "am not"),
    ("was", "wasn't"),
    ("is", "isn't"),
    ("have", "haven't"),
    ("had", "hadn't"),
]


def _extract_nouns(text: str) -> set[str]:
    words = text.lower().split()
    result = set()
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        clean = word.strip(",.!?;:'\"")
        if len(clean) > 2:
            result.add(clean)
        if i + 1 < len(words) and words[i + 1] in ("at", "in", "on", "to", "for"):
            skip_next = True
            result.add(words[i + 1].strip(",.!?;:'\""))
    return result


def has_sentiment_flip(text_a: str, text_b: str) -> bool:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    neg_a = words_a & NEGATION_WORDS
    neg_b = words_b & NEGATION_WORDS
    has_neg_a = bool(neg_a)
    has_neg_b = bool(neg_b)

    nouns_a = _extract_nouns(text_a)
    nouns_b = _extract_nouns(text_b)
    common_nouns = nouns_a & nouns_b

    for pos, neg in SENTIMENT_SHIFT_PAIRS:
        if (pos in words_a and neg in words_b) or (neg in words_a and pos in words_b):
            if common_nouns:
                return True

    if has_neg_a != has_neg_b and common_nouns:
        neg_in_a = bool(neg_a & words_a)
        neg_in_b = bool(neg_b & words_b)
        if neg_in_a != neg_in_b:
            return True

    return False


def find_duplicates(
    new_memory: MemoryObject,
    existing_memories: list[MemoryObject],
    threshold: float = 0.85,
) -> list[MemoryObject]:
    """Find memories that are semantically similar to new_memory.

    Returns memories with cosine similarity strictly above threshold (>= threshold is NOT included).
    Default threshold is 0.85.
    """
    if new_memory.embedding is None or not existing_memories:
        return []

    duplicates = []
    for existing in existing_memories:
        if existing.embedding is None:
            continue

        similarity = cosine_similarity(new_memory.embedding, existing.embedding)
        normalized_sim = (similarity + 1.0) / 2.0

        if normalized_sim > threshold:
            if has_sentiment_flip(new_memory.content, existing.content):
                continue
            duplicates.append(existing)

    return duplicates


def find_conflicts(
    new_memory: MemoryObject,
    existing_memories: list[MemoryObject],
    conflict_threshold: float = 0.65,
    dedup_threshold: float = 0.85,
) -> list[MemoryObject]:
    """Find memories that are potentially conflicting with new_memory.

    Returns memories with similarity strictly between conflict_threshold and
    dedup_threshold. This range excludes duplicates (above dedup_threshold) and
    excludes unrelated (below conflict_threshold). Default: 0.65 < similarity < 0.85
    """
    if new_memory.embedding is None or not existing_memories:
        return []

    conflicts = []
    for existing in existing_memories:
        if existing.embedding is None:
            continue

        similarity = cosine_similarity(new_memory.embedding, existing.embedding)
        normalized_sim = (similarity + 1.0) / 2.0

        if conflict_threshold < normalized_sim < dedup_threshold:
            conflicts.append(existing)

    return conflicts


def resolve_duplicate(
    new_memory: MemoryObject,
    existing: MemoryObject,
) -> MemoryObject:
    """Resolve duplicate using LATEST_WINS strategy.

    Copies content from new_memory into existing, updates last_accessed_at,
    preserves the existing memory_id.

    Does not mutate either input. Returns a new MemoryObject.
    """
    from datetime import datetime

    return MemoryObject(
        memory_id=existing.memory_id,
        user_id=existing.user_id,
        content=new_memory.content,
        embedding=existing.embedding,
        score=0.0,
        created_at=existing.created_at,
        last_accessed_at=datetime.now(timezone.utc),
        source=existing.source,
        importance=existing.importance,
        lifecycle_state=existing.lifecycle_state,
        metadata=existing.metadata.copy() if existing.metadata else {},
        embedding_dim=existing.embedding_dim,
        tags=new_memory.tags,
    )
