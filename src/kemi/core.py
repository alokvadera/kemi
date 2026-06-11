"""Public re-export of the Memory / MemoryService classes.

The implementation has been split out of a monolithic ``core.py`` into:
  - :mod:`kemi.memory_service` — the ``MemoryService`` class that composes
    the ingestion / retrieval pipelines and the I/O module.
  - :mod:`kemi._memory_impl` — a thin ``Memory`` subclass kept for backwards
    compatibility (emits a ``DeprecationWarning`` on construction).
  - :mod:`kemi.operations` — extracted free functions for I/O, versioning,
    webhooks, hooks, cache, metrics, and audit.
  - :mod:`kemi.operations._query_cache` — the LRU query cache.
"""

from kemi.memory.facade import Memory
from kemi.memory.service import MemoryService
from kemi.operations._query_cache import _QueryCache

__all__ = ["Memory", "MemoryService", "_QueryCache"]
