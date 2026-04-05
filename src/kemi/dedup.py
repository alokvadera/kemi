from kemi.models import MemoryObject
from kemi.scoring import cosine_similarity


def find_duplicates(
    new_memory: MemoryObject,
    existing_memories: list[MemoryObject],
    threshold: float = 0.85,
) -> list[MemoryObject]:
    """Find memories that are semantically similar to new_memory.

    Returns memories with cosine similarity strictly above threshold (>= threshold is NOT included).
    Default threshold is 0.85.
    """
    if not new_memory.embedding or not existing_memories:
        return []

    duplicates = []
    for existing in existing_memories:
        if existing.embedding is None:
            continue

        similarity = cosine_similarity(new_memory.embedding, existing.embedding)
        normalized_sim = (similarity + 1.0) / 2.0

        if normalized_sim > threshold:
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
    if not new_memory.embedding or not existing_memories:
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
        last_accessed_at=datetime.utcnow(),
        source=existing.source,
        importance=existing.importance,
        lifecycle_state=existing.lifecycle_state,
        metadata=existing.metadata.copy() if existing.metadata else {},
        embedding_dim=existing.embedding_dim,
    )
