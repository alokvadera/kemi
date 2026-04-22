from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


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
    embedding: list[float] | None = None
    score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: MemorySource = MemorySource.USER_STATED
    importance: float = 0.5
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_dim: int | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class MemoryConfig:
    dedup_threshold: float = 0.85
    conflict_threshold: float = 0.65
    decay_half_life_hours: float = 168.0
    decay_threshold_hours: float = 720.0
    default_importance: float = 0.5
    sanitize: bool = False
    default_top_k: int = 5
    max_tokens_default: int | None = None
    hybrid_search: bool = True

    def __post_init__(self):
        if not 0.0 <= self.dedup_threshold <= 1.0:
            raise ValueError(
                f"dedup_threshold must be between 0.0 and 1.0, got {self.dedup_threshold}"
            )
        if not 0.0 <= self.conflict_threshold <= 1.0:
            raise ValueError(
                f"conflict_threshold must be between 0.0 and 1.0, got {self.conflict_threshold}"
            )
        if self.decay_half_life_hours <= 0:
            raise ValueError(f"decay_half_life_hours must be > 0, got {self.decay_half_life_hours}")
        if self.decay_threshold_hours <= 0:
            raise ValueError(f"decay_threshold_hours must be > 0, got {self.decay_threshold_hours}")
        if not 0.0 <= self.default_importance <= 1.0:
            raise ValueError(
                f"default_importance must be between 0.0 and 1.0, got {self.default_importance}"
            )
        if self.default_top_k < 1:
            raise ValueError(f"default_top_k must be >= 1, got {self.default_top_k}")
        if self.max_tokens_default is not None and self.max_tokens_default < 1:
            raise ValueError(f"max_tokens_default must be >= 1, got {self.max_tokens_default}")
