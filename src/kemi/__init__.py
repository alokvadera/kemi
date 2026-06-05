try:
    from importlib.metadata import version as _version

    __version__ = _version("kemi")
except (ImportError, AttributeError):  # pragma: no cover
    __version__ = "0.3.0"

from kemi.core import Memory
from kemi.memory_formation import (
    CandidateMemory,
    LLMMemoryExtractor,
    OpenAIMemoryExtractor,
    RegexMemoryExtractor,
    StaticMemoryExtractor,
    extract_memories,
    remember_from_conversation,
)
from kemi.entities import EntityLinker, NoopEntityLinker, RegexEntityLinker, SpacyEntityLinker
from kemi.models import LifecycleState, MemoryConfig, MemoryObject, MemorySource, MemoryType
from kemi.procedures import remember_procedure, recall_procedures

__all__ = [
    "Memory",
    "MemoryConfig",
    "MemoryObject",
    "MemorySource",
    "LifecycleState",
    "MemoryType",
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
