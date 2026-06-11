"""Procedural memory helpers for kemi.

Procedural memories represent *how-to* knowledge — step-by-step instructions,
workflows, recipes, or standard operating procedures. They are distinct from
*episodic* (event-based) and *semantic* (fact-based) memories.

Use procedural memory when you want an agent to recall reusable action sequences
rather than isolated facts or past events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kemi.exceptions import ValidationError
from kemi.memory.model import LifecycleState, MemoryType

if TYPE_CHECKING:
    from kemi.core import Memory
    from kemi.memory.model import MemoryObject


def remember_procedure(
    memory: Memory,
    *,
    user_id: str | None = None,
    name: str,
    steps: list[str],
    metadata: dict[str, Any] | None = None,
    namespace: str = "default",
    importance: float = 0.7,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Store a step-by-step procedure as a procedural memory.

    Joins *steps* into a concise content string, tags the memory with
    ``["procedure", <name>]``, and sets ``memory_type=MemoryType.PROCEDURAL``.

    Args:
        memory: A kemi :class:`~kemi.core.Memory` instance.
        user_id: User to associate the procedure with. If ``None``, a generic
            system user (``"_system"``) is used.
        name: Short identifier for the procedure (e.g. ``"onboarding_flow"``).
        steps: Ordered list of action strings.
        metadata: Optional extra metadata dict.
        namespace: Memory namespace.
        importance: Importance score (0.0-1.0). Defaults to 0.7 because
            procedures are usually high-value reusable knowledge.
        agent_id: Optional agent identifier.
        session_id: Optional session identifier.

    Returns:
        The memory ID of the stored procedure.

    Example:
        >>> procedure_id = remember_procedure(
        ...     memory,
        ...     user_id="alice",
        ...     name="password_reset",
        ...     steps=[
        ...         "Ask the user for their email address",
        ...         "Send a reset link to the verified email",
        ...         "Confirm the reset was initiated",
        ...     ],
        ... )
    """
    if not steps:
        raise ValidationError("steps cannot be empty — there is nothing to remember")
    if not name or not name.strip():
        raise ValidationError("name cannot be empty")

    uid = user_id if user_id is not None else "_system"
    # Join steps into a concise, readable block
    step_lines = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
    content = f"Procedure: {name}\n{step_lines}"

    merged_meta = metadata.copy() if metadata else {}
    merged_meta["procedure_name"] = name
    merged_meta["step_count"] = len(steps)

    return memory.remember(
        user_id=uid,
        content=content,
        memory_type=MemoryType.PROCEDURAL,
        tags=["procedure", name],
        metadata=merged_meta,
        namespace=namespace,
        importance=importance,
        agent_id=agent_id,
        session_id=session_id,
    )


def recall_procedures(
    memory: Memory,
    query: str,
    *,
    user_id: str | None = None,
    namespace: str = "default",
    top_k: int = 10,
    lifecycle_filter: list[LifecycleState] | None = None,
    session_id: str | None = None,
) -> list[MemoryObject]:
    """Recall procedural memories relevant to a query.

    Performs a semantic search and filters the results to only
    ``memory_type=PROCEDURAL``.

    Args:
        memory: A kemi :class:`~kemi.core.Memory` instance.
        query: Natural-language query (e.g. ``"how do I reset a password?"``).
        user_id: If provided, scope the search to this user. If ``None``,
            the search is scoped to the generic ``"_system"`` user.
        namespace: Memory namespace.
        top_k: Maximum number of procedures to return.
        lifecycle_filter: Optional lifecycle states to include. Defaults to
            ``[ACTIVE, DECAYING, ARCHIVED]``.
        session_id: Optional session ID to scope the search.

    Returns:
        List of :class:`~kemi.memory.model.MemoryObject` instances with
        ``memory_type=PROCEDURAL``, sorted by relevance score.

    Example:
        >>> results = recall_procedures(
        ...     memory,
        ...     "password reset",
        ...     user_id="alice",
        ...     top_k=3,
        ... )
        >>> for proc in results:
        ...     print(proc.content)
    """
    if not query or not query.strip():
        raise ValidationError("query cannot be empty — what procedure should kemi search for?")
    if top_k < 1:
        raise ValidationError(f"top_k must be at least 1, got {top_k}")

    uid = user_id if user_id is not None else "_system"

    if lifecycle_filter is None:
        lifecycle_filter = [
            LifecycleState.ACTIVE,
            LifecycleState.DECAYING,
            LifecycleState.ARCHIVED,
        ]

    results = memory.recall(
        user_id=uid,
        query=query,
        top_k=top_k * 3,  # fetch extra to survive the post-hoc type filter
        lifecycle_filter=lifecycle_filter,
        namespace=namespace,
        session_id=session_id,
    )

    procedures = [m for m in results if m.memory_type == MemoryType.PROCEDURAL]
    return procedures[:top_k]
