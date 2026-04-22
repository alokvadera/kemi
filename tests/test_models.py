import pytest

from kemi.models import MemoryConfig


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
