"""Event hook operations: add_event_hook, remove_event_hook, _run_hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kemi.exceptions import ValidationError

if TYPE_CHECKING:
    from kemi.memory.service import MemoryService

logger = logging.getLogger(__name__)


def add(memory: MemoryService, phase: str, callback: Callable[..., Any]) -> None:
    """Register an event hook callback.

    Adds the callback to the default :class:`HookSink` (which shares the
    legacy ``_event_hooks`` dict), so legacy and registry paths see the
    same hooks.

    Args:
        phase: "pre" or "post" — called before or after the operation.
        callback: Callable that receives (operation, **kwargs).
    """
    if phase not in ("pre", "post"):
        raise ValidationError("phase must be 'pre' or 'post'")
    memory._event_hooks[phase].append(callback)


def remove(
    memory: MemoryService, phase: str, callback: Callable[..., Any]
) -> bool:
    """Remove a previously registered event hook callback.

    Returns True if removed, False if not found.
    """
    if phase in memory._event_hooks and callback in memory._event_hooks[phase]:
        memory._event_hooks[phase].remove(callback)
        return True
    return False


def run(
    memory: MemoryService,
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
    for sink in memory._plugins.hook_sinks:
        sink.run(
            phase,
            operation,
            raise_on_error=bool(raise_on_error),
            **kwargs,
        )
