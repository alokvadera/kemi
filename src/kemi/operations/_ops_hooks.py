"""Event hook operations: add_event_hook, remove_event_hook, _run_hooks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from kemi._memory_impl import Memory

logger = logging.getLogger(__name__)


def add(memory: "Memory", phase: str, callback: Callable[..., Any]) -> None:
    """Register an event hook callback.

    Args:
        phase: "pre" or "post" — called before or after the operation.
        callback: Callable that receives (operation, **kwargs).
    """
    if phase not in ("pre", "post"):
        raise ValueError("phase must be 'pre' or 'post'")
    memory._event_hooks[phase].append(callback)


def remove(
    memory: "Memory", phase: str, callback: Callable[..., Any]
) -> bool:
    """Remove a previously registered event hook callback.

    Returns True if removed, False if not found.
    """
    if phase in memory._event_hooks and callback in memory._event_hooks[phase]:
        memory._event_hooks[phase].remove(callback)
        return True
    return False


def run(
    memory: "Memory",
    phase: str,
    operation: str,
    *,
    raise_on_error: bool | None = None,
    **kwargs: Any,
) -> None:
    """Run all hooks registered for a phase/operation.

    Args:
        phase: "pre" or "post".
        operation: Name of the operation triggering the hook.
        raise_on_error: If True, exceptions from hooks are re-raised so
            a failing pre-hook can abort the operation. If None (default),
            the value is taken from ``memory._config.hooks_raise_on_error``.
        **kwargs: Passed through to each callback.
    """
    if raise_on_error is None:
        raise_on_error = memory._config.hooks_raise_on_error
    for hook in memory._event_hooks.get(phase, []):
        try:
            hook(operation, **kwargs)
        except Exception:
            # Broad catch: hooks are user code, so any error is their fault.
            # Honour the configured raise_on_error flag.
            if raise_on_error:
                raise
            logger.warning(
                f"Event hook failed for {phase}:{operation}", exc_info=True
            )
