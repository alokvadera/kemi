import pytest

from kemi.memory.model import MemoryConfig
from tests._helpers.factories import make_memory


def test_memoryconfig_invalid_dedup_threshold() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(dedup_threshold=1.5)


def test_memoryconfig_invalid_conflict_threshold() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(conflict_threshold=-0.1)


def test_memoryconfig_invalid_decay_half_life() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(decay_half_life_hours=0)


def test_memoryconfig_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(default_top_k=0)


def test_memoryconfig_invalid_max_tokens() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(max_tokens_default=0)


def test_memoryconfig_valid_defaults() -> None:
    config = MemoryConfig()
    assert config.dedup_threshold == 0.85
    assert config.conflict_threshold == 0.65
    assert config.decay_half_life_hours == 168.0
    assert config.decay_threshold_hours == 720.0
    assert config.default_importance == 0.5
    assert config.default_top_k == 5
    assert config.hybrid_search is True


def test_memoryconfig_invalid_decay_threshold() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(decay_threshold_hours=0)


def test_memoryconfig_invalid_importance() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(default_importance=1.5)


def test_memoryconfig_invalid_entity_boost_weight() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(entity_boost_weight=1.5)


def test_memoryconfig_invalid_scoring_weights_hybrid() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(weight_semantic=0.5, weight_recency=0.25, weight_bm25=0.10)


def test_memoryconfig_invalid_scoring_weights_no_embed() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(
            weight_semantic_no_embed=0.5,
            weight_recency_no_embed=0.3,
            weight_importance=0.10,
        )


def test_memoryconfig_valid_scoring_weights() -> None:
    config = MemoryConfig(
        weight_semantic=0.55,
        weight_recency=0.25,
        weight_bm25=0.20,
    )
    assert config.weight_semantic == 0.55


def test_memoryconfig_memory_source_enum() -> None:
    from kemi.memory.model import MemorySource
    assert MemorySource.USER_STATED.value == "user_stated"
    assert MemorySource.AGENT_INFERRED.value == "agent_inferred"
    assert MemorySource.SYSTEM_GENERATED.value == "system_generated"


def test_memoryconfig_lifecycle_state_enum() -> None:
    from kemi.memory.model import LifecycleState
    assert LifecycleState.ACTIVE.value == "active"
    assert LifecycleState.DECAYING.value == "decaying"
    assert LifecycleState.ARCHIVED.value == "archived"
    assert LifecycleState.DELETED.value == "deleted"


def test_memoryconfig_memory_type_enum() -> None:
    from kemi.memory.model import MemoryType
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.SEMANTIC.value == "semantic"
    assert MemoryType.PROCEDURAL.value == "procedural"


def test_memoryobject_defaults() -> None:
    from kemi.memory.model import LifecycleState, MemorySource, MemoryType
    mo = make_memory(memory_id="test", user_id="user", content="hello")
    assert mo.score == 0.0
    assert mo.source == MemorySource.USER_STATED
    assert mo.importance == 0.5
    assert mo.lifecycle_state == LifecycleState.ACTIVE
    assert mo.confidence == 1.0
    assert mo.memory_type == MemoryType.EPISODIC
    assert mo.namespace == "default"
    assert mo.version == 1
    assert mo.embedding is None
    assert mo.tags == []
    assert mo.metadata == {}

