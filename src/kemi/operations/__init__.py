"""Internal package: extracted operations from the monolithic core.py.

Each module here contains free functions that the :class:`kemi.Memory` class
delegates to. The split is purely organisational — the public API is
unchanged.
"""

from kemi.operations._query_cache import _QueryCache

__all__ = ["_QueryCache"]
