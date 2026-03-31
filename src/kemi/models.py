from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemorySource(Enum):
    USER_STATED = "user_stated"
    AGENT_INFERRED = "agent_inferred"
    SYSTEM_GENERATED = "system_generated"


class LifecycleState(Enum):
    ACTIVE = "active"
    DECAYING = "decaying"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class MemoryObject:
    memory_id: str
    user_id: str
    content: str
    embedding: Optional[List[float]] = None
    score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = field(default_factory=datetime.utcnow)
    source: MemorySource = MemorySource.USER_STATED
    importance: float = 0.5
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_dim: Optional[int] = None


@dataclass
class MemoryConfig:
    dedup_threshold: float = 0.85
    conflict_threshold: float = 0.65
    decay_half_life_hours: float = 168.0
    decay_threshold_hours: float = 720.0
    default_importance: float = 0.5
    sanitize: bool = False
    default_top_k: int = 5
    max_tokens_default: Optional[int] = None
