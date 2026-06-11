from datetime import datetime, timedelta, timezone

import pytest

from kemi.memory import scoring
from tests._helpers.factories import make_memory


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

    now = datetime.now(timezone.utc)
    result = scoring.temporal_recency(now)
    assert result == pytest.approx(1.0, abs=0.01)


def test_temporal_recency_old() -> None:

    old = datetime.now(timezone.utc) - timedelta(hours=1000)
    result = scoring.temporal_recency(old)
    assert result < 0.1


def test_score_memory_weights() -> None:

    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=[1.0] * 64,
        importance=0.8,
    )
    query = [1.0] * 64

    result = scoring.score_memory(mem, query)

    cosine = (1.0 + 1.0) / 2.0
    recency = 1.0
    importance = 0.8
    expected = cosine * 0.5 + recency * 0.3 + importance * 0.2

    assert result == pytest.approx(expected)


def test_rank_memories_sorted() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0, 0.0] * 32,
        ),
        make_memory(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[0.0, 1.0] * 32,
        ),
    ]

    query = [1.0, 0.0] * 32
    ranked = scoring.rank_memories(memories, query)

    assert ranked[0].memory_id == "a"
    assert ranked[1].memory_id == "b"


def test_truncate_by_tokens_none() -> None:

    memories = [
        make_memory(memory_id="a", user_id="user", content="test", embedding=[0.1] * 64),
    ]

    result = scoring.truncate_by_tokens(memories, max_tokens=None)
    assert len(result) == 1


def test_truncate_by_tokens_never_empty() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a long content " * 100,
            embedding=[0.1] * 64,
        ),
    ]

    result = scoring.truncate_by_tokens(memories, max_tokens=1)
    assert len(result) >= 1


def test_cosine_similarity_empty_vectors() -> None:
    result = scoring.cosine_similarity([], [1.0, 2.0])
    assert result == 0.0


def test_temporal_recency_exact_now() -> None:

    now = datetime.now(timezone.utc)
    result = scoring.temporal_recency(now, half_life_hours=168.0)
    assert result == pytest.approx(1.0, abs=0.01)


def test_score_memory_no_embedding() -> None:

    mem = make_memory(memory_id="test", user_id="user", content="test", embedding=None)
    result = scoring.score_memory(mem, [1.0] * 64)
    assert 0.0 <= result <= 1.0


def test_score_memory_importance_clamped() -> None:

    mem = make_memory(
        memory_id="test",
        user_id="user",
        content="test",
        embedding=[1.0] * 64,
        importance=1.5,
    )
    result = scoring.score_memory(mem, [1.0] * 64)
    assert result <= 1.0


def test_truncate_by_tokens_with_custom_counter() -> None:

    memories = [
        make_memory(memory_id="a", user_id="user", content="word", embedding=[0.1] * 64),
    ]

    def custom_counter(text):
        return 10

    result = scoring.truncate_by_tokens(memories, max_tokens=5, token_counter=custom_counter)
    assert len(result) == 1


def test_truncate_by_tokens_empty_list() -> None:
    result = scoring.truncate_by_tokens([], max_tokens=10)
    assert result == []


def test_temporal_recency_negative_hours() -> None:

    future = datetime.now(timezone.utc) - timedelta(hours=-1)
    result = scoring.temporal_recency(future)
    assert result == 1.0


def test_truncate_edge_case_single_memory_exceeds_budget() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="test word " * 50,
            embedding=[0.1] * 64,
        ),
    ]
    result = scoring.truncate_by_tokens(memories, max_tokens=1)
    assert len(result) == 1


def test_truncate_stops_when_budget_exceeded() -> None:

    memories = [
        make_memory(memory_id="a", user_id="user", content="short", embedding=[0.1] * 64),
        make_memory(
            memory_id="b",
            user_id="user",
            content="another short text here",
            embedding=[0.1] * 64,
        ),
    ]
    result = scoring.truncate_by_tokens(memories, max_tokens=2)
    assert len(result) == 1


def test_mmr_rerank_returns_diverse_results() -> None:

    similar_emb = [1.0, 0.0] * 32
    diverse_emb = [0.0, 1.0] * 32

    memories = [
        make_memory(
            memory_id="sim1",
            user_id="user",
            content="similar1",
            embedding=similar_emb,
            score=0.9,
        ),
        make_memory(
            memory_id="sim2",
            user_id="user",
            content="similar2",
            embedding=similar_emb,
            score=0.85,
        ),
        make_memory(
            memory_id="sim3",
            user_id="user",
            content="similar3",
            embedding=similar_emb,
            score=0.8,
        ),
        make_memory(
            memory_id="div1",
            user_id="user",
            content="diverse1",
            embedding=diverse_emb,
            score=0.7,
        ),
        make_memory(
            memory_id="div2",
            user_id="user",
            content="diverse2",
            embedding=diverse_emb,
            score=0.6,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0, 0.0] * 32, top_k=3, lambda_param=0.7)

    assert len(result) == 3
    has_diverse = any(m.memory_id in ("div1", "div2") for m in result)
    assert has_diverse, "Result should contain at least one diverse memory"


def test_mmr_rerank_top_k_larger_than_memories() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.5,
        ),
        make_memory(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[0.0, 1.0] * 32,
            score=0.3,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0, 0.0] * 32, top_k=5)
    assert len(result) == 2


def test_mmr_rerank_lambda_1_is_pure_relevance() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.9,
        ),
        make_memory(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[1.0] * 64,
            score=0.7,
        ),
        make_memory(
            memory_id="c",
            user_id="user",
            content="c",
            embedding=[1.0] * 64,
            score=0.5,
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


def test_bm25_corpus_avgdl_zero() -> None:
    """Corpus with empty strings should not cause division by zero (avgdl=0 → 1.0)."""
    result = scoring.bm25_score_corpus("query", "doc", [""])
    assert result >= 0.0
    assert isinstance(result, float)


def test_mmr_rerank_no_embeddings() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=None,
            score=0.9,
        ),
        make_memory(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=None,
            score=0.7,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0] * 32, top_k=2, lambda_param=0.7)
    assert len(result) == 2


def test_mmr_rerank_top_k_zero() -> None:

    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.9,
        ),
    ]

    result = scoring.mmr_rerank(memories, [1.0] * 64, top_k=0)
    assert result == []


def test_bm25_score_corpus_with_corpus() -> None:
    corpus = [
        "the quick brown fox",
        "the lazy dog sleeps",
        "the quick dog jumps",
    ]
    result = scoring.bm25_score_corpus("quick fox", "the quick brown fox", corpus)
    assert result > 0.0
    assert isinstance(result, float)


def test_bm25_score_corpus_empty_corpus_falls_back() -> None:
    result = scoring.bm25_score_corpus("quick fox", "the quick brown fox", [])
    assert result >= 0.0
    assert isinstance(result, float)


def test_mmr_rerank_stream_yields_same_as_rerank() -> None:
    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0, 0.0] * 32,
            score=0.9,
        ),
        make_memory(
            memory_id="b",
            user_id="user",
            content="b",
            embedding=[0.0, 1.0] * 32,
            score=0.7,
        ),
        make_memory(
            memory_id="c",
            user_id="user",
            content="c",
            embedding=[1.0, 0.0] * 32,
            score=0.5,
        ),
    ]

    stream_result = list(
        scoring.mmr_rerank_stream(memories, [1.0, 0.0] * 32, top_k=2, lambda_param=0.7)
    )
    list_result = scoring.mmr_rerank(memories, [1.0, 0.0] * 32, top_k=2, lambda_param=0.7)

    assert len(stream_result) == len(list_result)
    for s, rhs in zip(stream_result, list_result, strict=False):
        assert s.memory_id == rhs.memory_id


def test_mmr_rerank_stream_empty() -> None:
    result = list(scoring.mmr_rerank_stream([], [1.0] * 64, top_k=2))
    assert result == []


def test_mmr_rerank_stream_top_k_zero() -> None:
    memories = [
        make_memory(
            memory_id="a",
            user_id="user",
            content="a",
            embedding=[1.0] * 64,
            score=0.9,
        ),
    ]
    result = list(scoring.mmr_rerank_stream(memories, [1.0] * 64, top_k=0))
    assert result == []
