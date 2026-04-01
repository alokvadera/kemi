import pytest

from kemi import dedup
from kemi.models import LifecycleState, MemoryObject, MemorySource


def test_find_duplicates_above_threshold() -> None:
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    existing = [
        MemoryObject(
            memory_id="old",
            user_id="user",
            content="I am vegetarian",
            embedding=[1.0] * 64,
            score=0.0,
            created_at=None,
            last_accessed_at=None,
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        )
    ]

    result = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    assert len(result) == 1
    assert result[0].memory_id == "old"


def test_find_duplicates_below_threshold() -> None:
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    existing = [
        MemoryObject(
            memory_id="old",
            user_id="user",
            content="I live in NYC",
            embedding=[1.0 if i % 2 == 0 else -1.0 for i in range(64)],
            score=0.0,
            created_at=None,
            last_accessed_at=None,
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        )
    ]

    result = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    assert len(result) == 0


def test_find_conflicts_in_range() -> None:
    import math

    rad = 50 * math.pi / 180
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I like running",
        embedding=[1.0, 0.0] * 32,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    conflicting = [math.cos(rad), math.sin(rad)] * 32
    existing = [
        MemoryObject(
            memory_id="old",
            user_id="user",
            content="I hate running",
            embedding=conflicting,
            score=0.0,
            created_at=None,
            last_accessed_at=None,
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        )
    ]

    result = dedup.find_conflicts(new_mem, existing, conflict_threshold=0.65, dedup_threshold=0.85)
    assert len(result) == 1


def test_find_duplicates_and_conflicts_no_overlap() -> None:
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    existing = [
        MemoryObject(
            memory_id="dup",
            user_id="user",
            content="I am vegetarian",
            embedding=[1.0] * 64,
            score=0.0,
            created_at=None,
            last_accessed_at=None,
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        )
    ]

    duplicates = dedup.find_duplicates(new_mem, existing, threshold=0.85)
    conflicts = dedup.find_conflicts(
        new_mem, existing, conflict_threshold=0.65, dedup_threshold=0.85
    )

    duplicate_ids = {m.memory_id for m in duplicates}
    conflict_ids = {m.memory_id for m in conflicts}
    assert duplicate_ids.isdisjoint(conflict_ids)


def test_resolve_duplicate_preserves_memory_id() -> None:
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I am vegetarian now",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    existing = MemoryObject(
        memory_id="old-id",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    resolved = dedup.resolve_duplicate(new_mem, existing)
    assert resolved.memory_id == "old-id"


def test_resolve_duplicate_updates_content() -> None:
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I am vegetarian now",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    existing = MemoryObject(
        memory_id="old-id",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    resolved = dedup.resolve_duplicate(new_mem, existing)
    assert resolved.content == "I am vegetarian now"


def test_resolve_duplicate_no_mutation() -> None:
    new_mem = MemoryObject(
        memory_id="new",
        user_id="user",
        content="I am vegetarian now",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    existing = MemoryObject(
        memory_id="old-id",
        user_id="user",
        content="I am vegetarian",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=None,
        last_accessed_at=None,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )

    original_existing_content = existing.content
    original_new_content = new_mem.content

    dedup.resolve_duplicate(new_mem, existing)

    assert existing.content == original_existing_content
    assert new_mem.content == original_new_content
