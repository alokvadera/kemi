"""Backwards-compat shim — :class:`Memory` is now a thin alias for :class:`MemoryService`.

The implementation has been split out of this module into:
  - :mod:`kemi.memory_service` — the new :class:`MemoryService` class that
    composes the ingestion/retrieval pipelines and the I/O module.
  - :mod:`kemi.operations` — extracted free functions for I/O, versioning,
    webhooks, hooks, cache, metrics, and audit.
  - :mod:`kemi.operations._query_cache` — the LRU query cache.

:class:`Memory` is kept as a subclass of :class:`MemoryService` so every
existing import (``from kemi import Memory``, ``from kemi.core import Memory``,
``from kemi.memory.facade import Memory``) continues to work identically.
A single :class:`DeprecationWarning` is emitted on construction to nudge
long-lived code toward the new :class:`MemoryService` entry point.
"""

from __future__ import annotations

import warnings
from typing import Any

from kemi.memory.service import MemoryService
from kemi.operations._query_cache import _QueryCache

__all__ = ["Memory", "_QueryCache"]

# Only warn once per process so that documented ``from kemi import Memory``
# usage doesn't spam logs in every agent loop.
_memory_warned: bool = False


class Memory(MemoryService):
    """Backwards-compatible alias for :class:`MemoryService`.

    The full implementation lives in :mod:`kemi.memory_service`. This subclass
    exists only to emit a single :class:`DeprecationWarning` on construction
    so users know to migrate to :class:`MemoryService` directly.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _memory_warned
        if not _memory_warned:
            _memory_warned = True
            warnings.warn(
                "Memory is a backwards-compatible alias for MemoryService; "
                "import and use MemoryService directly in new code. "
                "The Memory class will be removed in a future major release.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(*args, **kwargs)
