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


class MemoryType(Enum):
    """Type of memory: episodic (event-based), semantic (fact-based), or procedural (how-to)."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


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
    confidence: float = 1.0
    memory_type: MemoryType = MemoryType.EPISODIC
    session_id: str | None = None
    namespace: str = "default"
    version: int = 1
    agent_id: str | None = None
    run_id: str | None = None
    app_id: str | None = None
    expires_at: datetime | None = None


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
    hooks_raise_on_error: bool = True
    # Scoring weights for hybrid search
    weight_semantic: float = 0.6
    weight_recency: float = 0.25
    weight_bm25: float = 0.15
    # Summarizer configuration for LLM-powered consolidation
    summarizer_llm_provider: str | None = None
    summarizer_llm_model: str | None = None
    summarizer_prompt_template: str | None = None
    # Scoring weights for non-hybrid search (when query_embedding is empty or hybrid_search=False)
    weight_semantic_no_embed: float = 0.5
    weight_recency_no_embed: float = 0.3
    weight_importance: float = 0.2
    # Entity-aware retrieval
    enable_entity_boost: bool = False
    entity_boost_weight: float = 0.1

    def __post_init__(self) -> None:
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
        if not 0.0 <= self.entity_boost_weight <= 1.0:
            raise ValueError(
                f"entity_boost_weight must be between 0.0 and 1.0, got {self.entity_boost_weight}"
            )
        # Validate scoring weights sum to ~1.0 (with some tolerance)
        total_hybrid = self.weight_semantic + self.weight_recency + self.weight_bm25
        if abs(total_hybrid - 1.0) > 0.01:
            raise ValueError(
                "Scoring weights (weight_semantic + weight_recency + weight_bm25) "
                f"must sum to 1.0, got {total_hybrid}"
            )
        total_no_embed = (
            self.weight_semantic_no_embed + self.weight_recency_no_embed + self.weight_importance
        )
        if abs(total_no_embed - 1.0) > 0.01:
            raise ValueError(
                "Scoring weights (weight_semantic_no_embed + weight_recency_no_embed + "
                f"weight_importance) must sum to 1.0, got {total_no_embed}"
            )
