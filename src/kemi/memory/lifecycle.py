from datetime import datetime, timezone

from kemi.exceptions import ValidationError
from kemi.memory.model import LifecycleState, MemoryObject


def evaluate_lifecycle(
    memory: MemoryObject,
    decay_threshold_hours: float = 720.0,
) -> LifecycleState:
    """Evaluate what lifecycle state a memory should be in.

    Based on last_accessed_at compared to decay_threshold_hours.
    Default: 720 hours = 30 days.

    If last_accessed_at is in the future (clock skew), returns ACTIVE.
    """
    if memory.lifecycle_state == LifecycleState.DELETED:
        return LifecycleState.DELETED

    if memory.lifecycle_state == LifecycleState.ARCHIVED:
        return LifecycleState.ARCHIVED

    now = datetime.now(timezone.utc)
    hours_since_access = (now - memory.last_accessed_at).total_seconds() / 3600.0

    if hours_since_access < 0:
        return LifecycleState.ACTIVE

    if hours_since_access > decay_threshold_hours:
        return LifecycleState.DECAYING

    return LifecycleState.ACTIVE


def transition(memory: MemoryObject, new_state: LifecycleState) -> MemoryObject:
    """Create a new MemoryObject with updated lifecycle state.

    Does not mutate the original. Returns a new instance.

    Raises ValueError if the transition is not allowed.
    """
    validate_transition(memory.lifecycle_state, new_state)

    return MemoryObject(
        memory_id=memory.memory_id,
        user_id=memory.user_id,
        content=memory.content,
        embedding=memory.embedding,
        score=memory.score,
        created_at=memory.created_at,
        last_accessed_at=memory.last_accessed_at,
        source=memory.source,
        importance=memory.importance,
        lifecycle_state=new_state,
        metadata=memory.metadata.copy() if memory.metadata else {},
        embedding_dim=memory.embedding_dim,
        tags=memory.tags,
        confidence=memory.confidence,
        memory_type=memory.memory_type,
        session_id=memory.session_id,
        namespace=memory.namespace,
        version=memory.version,
    )


def get_recall_filter() -> list[LifecycleState]:
    """Return the list of lifecycle states that should be included in recall results.

    Excludes ARCHIVED and DELETED. Includes ACTIVE and DECAYING.
    """
    return [LifecycleState.ACTIVE, LifecycleState.DECAYING]


_VALID_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
    LifecycleState.ACTIVE: {
        LifecycleState.DECAYING,
        LifecycleState.DELETED,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.DECAYING: {
        LifecycleState.ACTIVE,
        LifecycleState.DELETED,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.ARCHIVED: set(),
    LifecycleState.DELETED: set(),
}


def validate_transition(from_state: LifecycleState, to_state: LifecycleState) -> None:
    """Validate a state transition.

    Raises ValueError if the transition is not allowed.
    Valid transitions:
    - ACTIVE → DECAYING
    - ACTIVE → DELETED
    - ACTIVE → ARCHIVED
    - DECAYING → ACTIVE
    - DECAYING → DELETED
    - DECAYING → ARCHIVED

    ARCHIVED and DELETED are terminal.
    """
    if to_state not in _VALID_TRANSITIONS.get(from_state, set()):
        raise ValidationError(
            f"Invalid transition from {from_state.value} to {to_state.value}. "
            f"Valid transitions from {from_state.value}: "
            f"{[s.value for s in _VALID_TRANSITIONS.get(from_state, set())]}"
        )
