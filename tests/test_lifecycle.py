from datetime import datetime, timedelta, timezone

import pytest

from kemi.memory import lifecycle
from kemi.memory.model import LifecycleState
from tests._helpers.factories import make_memory


def test_evaluate_lifecycle_active() -> None:
    mem = make_memory(memory_id="test", user_id="user", content="test", embedding=None)

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.ACTIVE


def test_evaluate_lifecycle_decaying() -> None:
    old_time = datetime.now(timezone.utc) - timedelta(hours=800)
    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        last_accessed_at=old_time,
    )

    result = lifecycle.evaluate_lifecycle(mem, decay_threshold_hours=720.0)
    assert result == LifecycleState.DECAYING


def test_transition_valid() -> None:
    mem = make_memory(memory_id="test", user_id="user", content="test", embedding=None)

    result = lifecycle.transition(mem, LifecycleState.DECAYING)
    assert result.lifecycle_state == LifecycleState.DECAYING


def test_transition_invalid() -> None:
    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        lifecycle_state=LifecycleState.ARCHIVED,
    )

    with pytest.raises(ValueError):
        lifecycle.transition(mem, LifecycleState.ACTIVE)


def test_transition_no_mutation() -> None:
    mem = make_memory(memory_id="test", user_id="user", content="test", embedding=None)

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
    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        lifecycle_state=LifecycleState.DELETED,
    )

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.DELETED


def test_evaluate_lifecycle_archived_state() -> None:
    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        lifecycle_state=LifecycleState.ARCHIVED,
    )

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.ARCHIVED


def test_evaluate_lifecycle_future_access() -> None:
    from datetime import timedelta

    datetime.now(timezone.utc) + timedelta(hours=1)
    mem = make_memory(memory_id="test", user_id="user", content="test", embedding=None)

    result = lifecycle.evaluate_lifecycle(mem)
    assert result == LifecycleState.ACTIVE


def test_transition_decaying_to_active() -> None:
    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        lifecycle_state=LifecycleState.DECAYING,
    )

    result = lifecycle.transition(mem, LifecycleState.ACTIVE)
    assert result.lifecycle_state == LifecycleState.ACTIVE


def test_validate_transition_active_to_decaying() -> None:
    lifecycle.validate_transition(LifecycleState.ACTIVE, LifecycleState.DECAYING)


def test_validate_transition_active_to_deleted() -> None:
    lifecycle.validate_transition(LifecycleState.ACTIVE, LifecycleState.DELETED)


def test_validate_transition_active_to_archived() -> None:
    lifecycle.validate_transition(LifecycleState.ACTIVE, LifecycleState.ARCHIVED)


def test_validate_transition_decaying_to_active() -> None:
    lifecycle.validate_transition(LifecycleState.DECAYING, LifecycleState.ACTIVE)


def test_validate_transition_decaying_to_deleted() -> None:
    lifecycle.validate_transition(LifecycleState.DECAYING, LifecycleState.DELETED)


def test_validate_transition_decaying_to_archived() -> None:
    lifecycle.validate_transition(LifecycleState.DECAYING, LifecycleState.ARCHIVED)


def test_validate_transition_archived_is_terminal() -> None:
    with pytest.raises(ValueError):
        lifecycle.validate_transition(LifecycleState.ARCHIVED, LifecycleState.ACTIVE)


def test_validate_transition_deleted_is_terminal() -> None:
    with pytest.raises(ValueError):
        lifecycle.validate_transition(LifecycleState.DELETED, LifecycleState.ACTIVE)


def test_validate_transition_archived_to_archived() -> None:
    with pytest.raises(ValueError):
        lifecycle.validate_transition(LifecycleState.ARCHIVED, LifecycleState.ARCHIVED)


def test_validate_transition_deleted_to_deleted() -> None:
    with pytest.raises(ValueError):
        lifecycle.validate_transition(LifecycleState.DELETED, LifecycleState.DELETED)
