try:
    from importlib.metadata import version as _version

    __version__ = _version("kemi")
except (ImportError, AttributeError):  # pragma: no cover
    __version__ = "0.4.0"

from kemi.core import Memory, MemoryService
from kemi.exceptions import CompatibilityError
from kemi.memory.entities import (
    EntityLinker,
    NoopEntityLinker,
    RegexEntityLinker,
    SpacyEntityLinker,
)
from kemi.memory.formation import (
    CandidateMemory,
    LLMMemoryExtractor,
    OpenAIMemoryExtractor,
    RegexMemoryExtractor,
    StaticMemoryExtractor,
    extract_memories,
    remember_from_conversation,
)
from kemi.memory.model import LifecycleState, MemoryConfig, MemoryObject, MemorySource, MemoryType
from kemi.memory.procedures import recall_procedures, remember_procedure
from kemi.plugins import (
    KEMI_PROTOCOL_VERSION,
    AuditSink,
    AuditTrailSink,
    CallbackHookSink,
    HookSink,
    LruQueryCache,
    PluginRegistry,
    QueryCacheProvider,
    WebhookDispatcherSink,
    WebhookSink,
)

__all__ = [
    "Memory",
    "MemoryService",
    "MemoryConfig",
    "MemoryObject",
    "MemorySource",
    "LifecycleState",
    "MemoryType",
    "PluginRegistry",
    "WebhookSink",
    "AuditSink",
    "QueryCacheProvider",
    "HookSink",
    "WebhookDispatcherSink",
    "AuditTrailSink",
    "LruQueryCache",
    "CallbackHookSink",
    "KEMI_PROTOCOL_VERSION",
    "CompatibilityError",
    "CandidateMemory",
    "extract_memories",
    "remember_from_conversation",
    "LLMMemoryExtractor",
    "RegexMemoryExtractor",
    "OpenAIMemoryExtractor",
    "StaticMemoryExtractor",
    "remember_procedure",
    "recall_procedures",
    "EntityLinker",
    "NoopEntityLinker",
    "RegexEntityLinker",
    "SpacyEntityLinker",
]
