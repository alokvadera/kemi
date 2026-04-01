import pytest

from datetime import datetime, timedelta

from kemi import lifecycle
from kemi.models import LifecycleState, MemoryObject, MemorySource


def test_evaluate_lifecycle_active() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.ACTIVE


def test_evaluate_lifecycle_decaying() -> None:
    old_time = datetime.utcnow() - timedelta(hours=800)
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=old_time,
        last_accessed_at=old_time,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.evaluate_lifecycle(mem, decay_threshold_hours=720.0)
    assert result == LifecycleState.DECAYING


def test_transition_valid() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.transition(mem, LifecycleState.DECAYING)
    assert result.lifecycle_state == LifecycleState.DECAYING


def test_transition_invalid() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    with pytest.raises(ValueError):
        lifecycle.transition(mem, LifecycleState.ARCHIVED)


def test_transition_no_mutation() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    original_state = mem.lifecycle_state
    lifecycle.transition(mem, LifecycleState.DECAYING)

    assert mem.lifecycle_state == original_state


def test_get_recall_filter() -> None:
    result = lifecycle.get_recall_filter()
    assert LifecycleState.ACTIVE in result
    assert LifecycleState.DECAYING in result
    assert LifecycleState.ARCHIVED not in result
    assert LifecycleState.DELETED not in result


def test_evaluate_lifecycle_deleted_state() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.DELETED,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.DELETED


def test_evaluate_lifecycle_archived_state() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ARCHIVED,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.ARCHIVED


def test_evaluate_lifecycle_future_access() -> None:
    from datetime import timedelta

    future = datetime.utcnow() + timedelta(hours=1)
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=future,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.ACTIVE


def test_transition_decaying_to_active() -> None:
    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.DECAYING,
        metadata={},
        embedding_dim=None,
    )

    result = lifecycle.transition(mem, LifecycleState.ACTIVE)
    assert result.lifecycle_state == LifecycleState.ACTIVE
