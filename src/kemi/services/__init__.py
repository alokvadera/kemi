"""Service facades split out of the monolithic :class:`MemoryService`.

The public :class:`MemoryService` API is preserved by delegating to one
of the three facades here.  Each facade operates on a shared
:class:`_MemoryCore` that holds the mutable state, so facades can be
constructed and tested independently.
"""

from kemi.services.admin_service import MemoryAdminService
from kemi.services.read_service import MemoryReadService
from kemi.services.write_service import MemoryWriteService

__all__ = ["MemoryReadService", "MemoryWriteService", "MemoryAdminService"]
