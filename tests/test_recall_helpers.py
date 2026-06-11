"""Tests for multi-level recall helpers: recall_user_profile, recall_session_context, recall_agent_knowledge."""  # noqa: E501

import pytest

from kemi import Memory
from kemi.memory.model import LifecycleState, MemoryType


@pytest.fixture
def no_dedup_memory(mock_embedding, mock_storage) -> Memory:
    """Memory fixture with dedup disabled so unrelated texts don't merge."""
    mem = Memory(embed=mock_embedding(), store=mock_storage())
    mem._config.dedup_threshold = 1.0
    mem._config.conflict_threshold = 1.0
    return mem


def test_recall_user_profile_filters_semantic(no_dedup_memory) -> None:
    """Only SEMANTIC memories are returned, sorted by importance descending."""
    no_dedup_memory.remember(
        "alice",
        "I am vegetarian",
        importance=0.9,
        memory_type=MemoryType.SEMANTIC,
    )
    no_dedup_memory.remember(
        "alice",
        "I live in Mumbai",
        importance=0.7,
        memory_type=MemoryType.SEMANTIC,
    )
    no_dedup_memory.remember(
        "alice",
        "I ordered pizza yesterday",
        importance=0.8,
        memory_type=MemoryType.EPISODIC,
    )

    results = no_dedup_memory.recall_user_profile("alice", top_k=20)
    assert len(results) == 2
    assert results[0].memory_type == MemoryType.SEMANTIC
    assert results[1].memory_type == MemoryType.SEMANTIC
    assert results[0].content == "I am vegetarian"
    assert results[1].content == "I live in Mumbai"
    assert results[0].importance >= results[1].importance


def test_recall_user_profile_respects_top_k(no_dedup_memory) -> None:
    """top_k limits the number of returned profile memories."""
    for i in range(5):
        no_dedup_memory.remember(
            "alice",
            f"fact {i}",
            importance=0.5 + i * 0.1,
            memory_type=MemoryType.SEMANTIC,
        )

    results = no_dedup_memory.recall_user_profile("alice", top_k=3)
    assert len(results) == 3


def test_recall_user_profile_empty_user(no_dedup_memory) -> None:
    """Empty user raises ValueError."""
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        no_dedup_memory.recall_user_profile("")


def test_recall_user_profile_top_k_validation(no_dedup_memory) -> None:
    """top_k < 1 raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        no_dedup_memory.recall_user_profile("alice", top_k=0)


def test_recall_user_profile_excludes_deleted(no_dedup_memory) -> None:
    """DELETED semantic memories are not included."""
    no_dedup_memory.remember(
        "alice",
        "I am vegetarian",
        importance=0.9,
        memory_type=MemoryType.SEMANTIC,
    )
    all_mem = no_dedup_memory._store.get_all_by_user("alice")
    mem = all_mem[0]
    mem.lifecycle_state = LifecycleState.DELETED
    no_dedup_memory._store.update(mem)

    results = no_dedup_memory.recall_user_profile("alice")
    assert len(results) == 0


def test_recall_session_context_filters_episodic(no_dedup_memory) -> None:
    """Only EPISODIC memories for the given session are returned, sorted by recency."""
    no_dedup_memory.remember(
        "alice",
        "session start",
        memory_type=MemoryType.EPISODIC,
        session_id="sess_123",
    )
    no_dedup_memory.remember(
        "alice",
        "I ordered pizza",
        memory_type=MemoryType.EPISODIC,
        session_id="sess_123",
    )
    no_dedup_memory.remember(
        "alice",
        "I am vegetarian",
        memory_type=MemoryType.SEMANTIC,
        session_id="sess_123",
    )
    no_dedup_memory.remember(
        "alice",
        "unrelated session",
        memory_type=MemoryType.EPISODIC,
        session_id="sess_456",
    )

    results = no_dedup_memory.recall_session_context("alice", "sess_123", top_k=20)
    assert len(results) == 2
    assert all(r.memory_type == MemoryType.EPISODIC for r in results)
    assert all(r.session_id == "sess_123" for r in results)
    # Most recent first
    assert results[0].content == "I ordered pizza"
    assert results[1].content == "session start"


def test_recall_session_context_respects_top_k(no_dedup_memory) -> None:
    """top_k limits the number of returned session memories."""
    for i in range(5):
        no_dedup_memory.remember(
            "alice",
            f"event {i}",
            memory_type=MemoryType.EPISODIC,
            session_id="sess_abc",
        )

    results = no_dedup_memory.recall_session_context("alice", "sess_abc", top_k=2)
    assert len(results) == 2


def test_recall_session_context_empty_user(no_dedup_memory) -> None:
    """Empty user_id raises ValueError."""
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        no_dedup_memory.recall_session_context("", "sess_123")


def test_recall_session_context_empty_session(no_dedup_memory) -> None:
    """Empty session_id raises ValueError."""
    with pytest.raises(ValueError, match="session_id cannot be empty"):
        no_dedup_memory.recall_session_context("alice", "")


def test_recall_session_context_top_k_validation(no_dedup_memory) -> None:
    """top_k < 1 raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        no_dedup_memory.recall_session_context("alice", "sess_123", top_k=0)


def test_recall_session_context_excludes_deleted(no_dedup_memory) -> None:
    """DELETED episodic memories are not included."""
    no_dedup_memory.remember(
        "alice",
        "I ordered pizza",
        memory_type=MemoryType.EPISODIC,
        session_id="sess_123",
    )
    all_mem = no_dedup_memory._store.get_all_by_user("alice")
    mem = all_mem[0]
    mem.lifecycle_state = LifecycleState.DELETED
    no_dedup_memory._store.update(mem)

    results = no_dedup_memory.recall_session_context("alice", "sess_123")
    assert len(results) == 0


def test_recall_agent_knowledge_filters_by_agent(no_dedup_memory) -> None:
    """Only memories with matching agent_id are returned, sorted by importance."""
    no_dedup_memory.remember(
        "alice",
        "Agent rule: be polite",
        importance=0.9,
        memory_type=MemoryType.SEMANTIC,
        agent_id="agent_1",
    )
    no_dedup_memory.remember(
        "alice",
        "Agent rule: be concise",
        importance=0.7,
        memory_type=MemoryType.SEMANTIC,
        agent_id="agent_1",
    )
    no_dedup_memory.remember(
        "alice",
        "User preference: vegetarian",
        importance=0.8,
        memory_type=MemoryType.SEMANTIC,
    )
    no_dedup_memory.remember(
        "bob",
        "Agent rule: use emojis",
        importance=0.6,
        memory_type=MemoryType.SEMANTIC,
        agent_id="agent_1",
    )

    results = no_dedup_memory.recall_agent_knowledge("agent_1", top_k=20)
    assert len(results) == 3
    assert all(r.agent_id == "agent_1" for r in results)
    # Sorted by importance descending
    assert results[0].content == "Agent rule: be polite"
    assert results[1].content == "Agent rule: be concise"
    assert results[2].content == "Agent rule: use emojis"


def test_recall_agent_knowledge_respects_namespace(no_dedup_memory) -> None:
    """Only memories in the requested namespace are returned."""
    no_dedup_memory.remember(
        "alice",
        "Agent rule: be polite",
        importance=0.9,
        memory_type=MemoryType.SEMANTIC,
        agent_id="agent_1",
        namespace="default",
    )
    no_dedup_memory.remember(
        "alice",
        "Agent rule: be funny",
        importance=0.8,
        memory_type=MemoryType.SEMANTIC,
        agent_id="agent_1",
        namespace="other",
    )

    results = no_dedup_memory.recall_agent_knowledge("agent_1", namespace="default")
    assert len(results) == 1
    assert results[0].content == "Agent rule: be polite"


def test_recall_agent_knowledge_respects_top_k(no_dedup_memory) -> None:
    """top_k limits the number of returned agent memories."""
    for i in range(5):
        no_dedup_memory.remember(
            "alice",
            f"rule {i}",
            importance=0.5 + i * 0.1,
            memory_type=MemoryType.SEMANTIC,
            agent_id="agent_1",
        )

    results = no_dedup_memory.recall_agent_knowledge("agent_1", top_k=3)
    assert len(results) == 3


def test_recall_agent_knowledge_empty_agent(no_dedup_memory) -> None:
    """Empty agent_id raises ValueError."""
    with pytest.raises(ValueError, match="agent_id cannot be empty"):
        no_dedup_memory.recall_agent_knowledge("")


def test_recall_agent_knowledge_top_k_validation(no_dedup_memory) -> None:
    """top_k < 1 raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        no_dedup_memory.recall_agent_knowledge("agent_1", top_k=0)


def test_recall_agent_knowledge_excludes_deleted(no_dedup_memory) -> None:
    """DELETED agent memories are not included."""
    no_dedup_memory.remember(
        "alice",
        "Agent rule: be polite",
        importance=0.9,
        memory_type=MemoryType.SEMANTIC,
        agent_id="agent_1",
    )
    all_mem = no_dedup_memory._store.get_all_by_user("alice")
    mem = all_mem[0]
    mem.lifecycle_state = LifecycleState.DELETED
    no_dedup_memory._store.update(mem)

    results = no_dedup_memory.recall_agent_knowledge("agent_1")
    assert len(results) == 0


def test_recall_user_profile_updates_last_accessed(no_dedup_memory) -> None:
    """recall_user_profile updates last_accessed_at on returned memories."""
    no_dedup_memory.remember(
        "alice",
        "I am vegetarian",
        memory_type=MemoryType.SEMANTIC,
    )
    before = no_dedup_memory._store.get_all_by_user("alice")[0].last_accessed_at

    results = no_dedup_memory.recall_user_profile("alice")
    after = results[0].last_accessed_at

    assert after > before


def test_recall_session_context_updates_last_accessed(no_dedup_memory) -> None:
    """recall_session_context updates last_accessed_at on returned memories."""
    no_dedup_memory.remember(
        "alice",
        "I ordered pizza",
        memory_type=MemoryType.EPISODIC,
        session_id="sess_123",
    )
    before = no_dedup_memory._store.get_all_by_user("alice")[0].last_accessed_at

    results = no_dedup_memory.recall_session_context("alice", "sess_123")
    after = results[0].last_accessed_at

    assert after > before
