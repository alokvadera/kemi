import pytest

from datetime import datetime, timedelta, timezone

from kemi import scoring
from kemi.models import LifecycleState, MemoryObject, MemorySource


def test_cosine_similarity_identical_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert scoring.cosine_similarity(a, b) == 1.0


def test_cosine_similarity_zero_vector() -> None:
    a = [0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert scoring.cosine_similarity(a, b) == 0.0


def test_cosine_similarity_opposite_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    b = [-1.0, 0.0, 0.0]
    result = scoring.cosine_similarity(a, b)
    assert result == pytest.approx(-1.0)
    normalized = (result + 1.0) / 2.0
    assert normalized == pytest.approx(0.0)


def test_temporal_recency_now() -> None:
    from datetime import datetime

    now = datetime.now(timezone.utc)
    result = scoring.temporal_recency(now)
    assert result == pytest.approx(1.0, abs=0.01)


def test_temporal_recency_old() -> None:
    from datetime import datetime, timedelta

    old = datetime.now(timezone.utc) - timedelta(hours=1000)
    result = scoring.temporal_recency(old)
    assert result < 0.1


def test_score_memory_weights() -> None:
    from datetime import datetime

    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.8,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )
    query = [1.0] * 64

    result = scoring.score_memory(mem, query)

    cosine = (1.0 + 1.0) / 2.0
    recency = 1.0
    importance = 0.8
    expected = cosine * 0.5 + recency * 0.3 + importance * 0.2

    assert result == pytest.approx(expected)


def test_rank_memories_sorted() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0, 0.0] * 32,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[0.0, 1.0] * 32,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    query = [1.0, 0.0] * 32
    ranked = scoring.rank_memories(memories, query)

    assert ranked[0].memory_id == "a"
    assert ranked[1].memory_id == "b"


def test_truncate_by_tokens_none() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="test",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    result = scoring.truncate_by_tokens(memories, max_tokens=None)
    assert len(result) == 1


def test_truncate_by_tokens_never_empty() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="a long content " * 100,
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    result = scoring.truncate_by_tokens(memories, max_tokens=1)
    assert len(result) >= 1


def test_cosine_similarity_empty_vectors() -> None:
    result = scoring.cosine_similarity([], [1.0, 2.0])
    assert result == 0.0


def test_temporal_recency_exact_now() -> None:
    from datetime import datetime

    now = datetime.now(timezone.utc)
    result = scoring.temporal_recency(now, half_life_hours=168.0)
    assert result == pytest.approx(1.0, abs=0.01)


def test_score_memory_no_embedding() -> None:
    from datetime import datetime

    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=None,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=None,
    )
    result = scoring.score_memory(mem, [1.0] * 64)
    assert 0.0 <= result <= 1.0


def test_score_memory_importance_clamped() -> None:
    from datetime import datetime

    mem = MemoryObject(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=[1.0] * 64,
        score=0.0,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        importance=1.5,
        lifecycle_state=LifecycleState.ACTIVE,
        metadata={},
        embedding_dim=64,
    )
    result = scoring.score_memory(mem, [1.0] * 64)
    assert result <= 1.0


def test_truncate_by_tokens_with_custom_counter() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="word",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    def custom_counter(text):
        return 10

    result = scoring.truncate_by_tokens(memories, max_tokens=5, token_counter=custom_counter)
    assert len(result) == 1


def test_truncate_by_tokens_empty_list() -> None:
    result = scoring.truncate_by_tokens([], max_tokens=10)
    assert result == []


def test_temporal_recency_negative_hours() -> None:
    from datetime import datetime, timedelta

    future = datetime.now(timezone.utc) - timedelta(hours=-1)
    result = scoring.temporal_recency(future)
    assert result == 1.0


def test_truncate_edge_case_single_memory_exceeds_budget() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="test word " * 50,
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]
    result = scoring.truncate_by_tokens(memories, max_tokens=1)
    assert len(result) == 1


def test_truncate_stops_when_budget_exceeded() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="short",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="b",
            user_id="user",
            content="another short text here",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]
    result = scoring.truncate_by_tokens(memories, max_tokens=2)
    assert len(result) == 1


def test_mmr_rerank_returns_diverse_results() -> None:
    from datetime import datetime

    similar_emb = [1.0, 0.0] * 32
    diverse_emb = [0.0, 1.0] * 32

    memories = [
        MemoryObject(
            memory_id="sim1",
            user_id="user",
            content="similar1",
            embedding=similar_emb,
            score=0.9,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="sim2",
            user_id="user",
            content="similar2",
            embedding=similar_emb,
            score=0.85,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="sim3",
            user_id="user",
            content="similar3",
            embedding=similar_emb,
            score=0.8,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="div1",
            user_id="user",
            content="diverse1",
            embedding=diverse_emb,
            score=0.7,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="div2",
            user_id="user",
            content="diverse2",
            embedding=diverse_emb,
            score=0.6,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0, 0.0] * 32, top_k=3, lambda_param=0.7)

    assert len(result) == 3
    has_diverse = any(m.memory_id in ("div1", "div2") for m in result)
    assert has_diverse, "Result should contain at least one diverse memory"


def test_mmr_rerank_top_k_larger_than_memories() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.5,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[0.0, 1.0] * 32,
            score=0.3,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0, 0.0] * 32, top_k=5)
    assert len(result) == 2


def test_mmr_rerank_lambda_1_is_pure_relevance() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.9,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[1.0] * 64,
            score=0.7,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
        MemoryObject(
            memory_id="c",
            user_id="user",
            content="c",
            embedding=[1.0] * 64,
            score=0.5,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0, 0.0] * 32, top_k=2, lambda_param=1.0)
    assert result[0].score == 0.9


def test_cosine_similarity_numpy_path() -> None:
    result = scoring.cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert result == pytest.approx(1.0)

    result = scoring.cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert result == pytest.approx(0.0, abs=0.01)


def test_bm25_empty_query() -> None:
    result = scoring.bm25_score("", "some document")
    assert result == 0.0


def test_bm25_empty_document() -> None:
    result = scoring.bm25_score("some query", "")
    assert result == 0.0


def test_bm25_empty_both() -> None:
    result = scoring.bm25_score("", "")
    assert result == 0.0


def test_bm25_corpus_empty_query() -> None:
    result = scoring.bm25_score_corpus("", "doc", ["corpus"])
    assert result == 0.0


def test_bm25_corpus_empty_document() -> None:
    result = scoring.bm25_score_corpus("query", "", ["corpus"])
    assert result == 0.0


def test_bm25_corpus_empty_corpus() -> None:
    result = scoring.bm25_score_corpus("query", "doc", [])
    assert result >= 0.0


def test_bm25_corpus_zero_docs() -> None:
    result = scoring.bm25_score_corpus("query", "doc", [])
    assert result >= 0.0


def test_mmr_rerank_no_embeddings() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=None,
            score=0.9,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=None,
        ),
        MemoryObject(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=None,
            score=0.7,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=None,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0] * 32, top_k=2, lambda_param=0.7)
    assert len(result) == 2


def test_mmr_rerank_top_k_zero() -> None:
    from datetime import datetime

    memories = [
        MemoryObject(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.9,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=64,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0] * 64, top_k=0)
    assert result == []
