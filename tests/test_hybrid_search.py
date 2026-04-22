import pytest

from kemi.scoring import bm25_score, rank_memories
from kemi.models import MemoryObject, MemorySource, LifecycleState
from datetime import datetime, timezone


def test_bm25_score_exact_keyword_match() -> None:
    """BM25 returns higher score for exact keyword match."""
    query = "pizza"
    doc1 = "I love pizza with cheese"
    doc2 = "I love pasta with tomato"

    score1 = bm25_score(query, doc1)
    score2 = bm25_score(query, doc2)

    assert score1 > score2
    assert score1 > 0.0


def test_bm25_score_empty_query() -> None:
    """BM25 returns 0.0 for empty query."""
    score = bm25_score("", "some document")
    assert score == 0.0

    score = bm25_score(None, "some document")  # type: ignore
    assert score == 0.0


def test_bm25_score_case_insensitive() -> None:
    """BM25 is case insensitive."""
    query_lower = "pizza"
    query_upper = "PIZZA"
    query_mixed = "Pizza"
    document = "I love PIZZA with cheese"

    score_lower = bm25_score(query_lower, document)
    score_upper = bm25_score(query_upper, document)
    score_mixed = bm25_score(query_mixed, document)

    assert score_lower == score_upper
    assert score_lower == score_mixed


def test_bm25_score_no_match() -> None:
    """BM25 returns 0.0 when there's no keyword match."""
    query = "basketball"
    document = "I love pizza and pasta"

    score = bm25_score(query, document)
    assert score == 0.0


def test_bm25_score_multiple_terms() -> None:
    """BM25 scores higher when multiple query terms match."""
    query = "pizza italian food"
    doc1 = "I love pizza and italian food"
    doc2 = "I love pizza"

    score1 = bm25_score(query, doc1)
    score2 = bm25_score(query, doc2)

    assert score1 > score2
    assert score1 > 0.0


def test_recall_with_hybrid_search_true(mock_memory) -> None:
    """recall with hybrid_search=True returns results."""
    mock_memory.remember("user123", "I love pizza with cheese", importance=0.5)

    results = mock_memory.recall("user123", "pizza", hybrid_search=True)

    assert len(results) > 0
    assert results[0].content == "I love pizza with cheese"


def test_recall_with_hybrid_search_false(mock_memory) -> None:
    """recall with hybrid_search=False returns results."""
    mock_memory.remember("user123", "I love pizza with cheese", importance=0.5)

    results = mock_memory.recall("user123", "pizza", hybrid_search=False)

    assert len(results) > 0


def test_hybrid_search_ranks_keyword_higher() -> None:
    """rank_memories with hybrid_search adds keyword boosting."""
    from kemi.scoring import rank_memories

    memories = [
        MemoryObject(
            memory_id="1",
            user_id="user1",
            content="pizza is my favorite food",
            embedding=[0.1] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
        MemoryObject(
            memory_id="2",
            user_id="user1",
            content="random weather content here",
            embedding=[0.1] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
    ]

    ranked = rank_memories(memories, [0.1] * 64, query="pizza", hybrid_search=True)

    assert ranked[0].content == "pizza is my favorite food"


def test_rank_memories_with_hybrid(mock_memory) -> None:
    """rank_memories with hybrid_search uses BM25 scoring."""
    memories = [
        MemoryObject(
            memory_id="1",
            user_id="user1",
            content="I love pizza and pasta",
            embedding=[0.1] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
        MemoryObject(
            memory_id="2",
            user_id="user1",
            content="The weather is nice today",
            embedding=[0.1] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
    ]

    ranked = rank_memories(memories, [0.1] * 64, query="pizza", hybrid_search=True)

    assert ranked[0].memory_id == "1"
    assert "pizza" in ranked[0].content.lower()


def test_rank_memories_without_hybrid(mock_memory) -> None:
    """rank_memories without hybrid_search uses semantic + recency + importance."""
    memories = [
        MemoryObject(
            memory_id="1",
            user_id="user1",
            content="I love pizza and pasta",
            embedding=[0.1] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
        MemoryObject(
            memory_id="2",
            user_id="user1",
            content="Random content about something else entirely different",
            embedding=[0.9] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.9,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
    ]

    ranked = rank_memories(memories, [0.1] * 64, query="pizza", hybrid_search=False)

    assert len(ranked) == 2


def test_hybrid_search_with_empty_query_uses_semantic(mock_memory) -> None:
    """Hybrid search with empty query falls back to semantic scoring."""
    memories = [
        MemoryObject(
            memory_id="1",
            user_id="user1",
            content="I love pizza and pasta",
            embedding=[0.1] * 64,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
        ),
    ]

    ranked = rank_memories(memories, [0.1] * 64, query="", hybrid_search=True)

    assert ranked[0].memory_id == "1"


@pytest.mark.asyncio
async def test_arecall_with_hybrid_search(mock_memory) -> None:
    """arecall with hybrid_search=True returns results."""
    await mock_memory.aremember("user123", "I love pizza with cheese")

    results = await mock_memory.arecall("user123", "pizza", hybrid_search=True)

    assert len(results) > 0
    assert "pizza" in results[0].content.lower()


def test_memory_config_hybrid_search_default() -> None:
    """MemoryConfig has hybrid_search defaulting to True."""
    from kemi.models import MemoryConfig

    config = MemoryConfig()
    assert config.hybrid_search is True
