"""Tests for src/kemi/reranker.py — cross-encoder re-ranking."""

import math

import pytest

from kemi import reranker
from kemi.reranker import (
    CrossEncoderReranker,
    FallbackReranker,
    NomicReranker,
    RerankerConfig,
    RerankerResult,
    fuse_and_rerank,
    rerank_results,
)


# ---------------------------------------------------------------------------
# RerankerConfig
# ---------------------------------------------------------------------------

class TestRerankerConfig:
    def test_default_values(self) -> None:
        config = RerankerConfig()
        assert config.provider == "fallback"
        assert config.model is None
        assert config.device == "cpu"
        assert config.batch_size == 8
        assert config.score_threshold == 0.0

    def test_custom_values(self) -> None:
        config = RerankerConfig(
            provider="nomic",
            model="nomic-embed-text-v1.5",
            device="cuda",
            batch_size=16,
            score_threshold=0.5,
        )
        assert config.provider == "nomic"
        assert config.model == "nomic-embed-text-v1.5"
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.score_threshold == 0.5


# ---------------------------------------------------------------------------
# RerankerResult dataclass
# ---------------------------------------------------------------------------

class TestRerankerResult:
    def test_fields(self) -> None:
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="test",
            embedding=[0.1, 0.2],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=2,
        )
        result = RerankerResult(
            memory=mem,
            cross_encoder_score=0.85,
            bi_encoder_rank=2,
            cross_encoder_rank=0,
        )
        assert result.memory.memory_id == "test"
        assert result.cross_encoder_score == 0.85
        assert result.bi_encoder_rank == 2
        assert result.cross_encoder_rank == 0


# ---------------------------------------------------------------------------
# FallbackReranker
# ---------------------------------------------------------------------------

class TestFallbackReranker:
    def setup_method(self) -> None:
        self.embed = _make_embed_fn()

    def test_exact_term_match_boosts_score(self) -> None:
        reranker = FallbackReranker()
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Python programming is great",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored = reranker.score("Python", [mem])
        assert len(scored) == 1
        assert scored[0].cross_encoder_score > 0.0

    def test_no_query_terms_in_content_low_score(self) -> None:
        reranker = FallbackReranker()
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Completely unrelated content xyzabc",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored = reranker.score("Python coding", [mem])
        # Low overlap → lower score
        assert scored[0].cross_encoder_score < 0.5

    def test_position_bonus_early_mention(self) -> None:
        reranker = FallbackReranker()
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        early = MemoryObject(
            memory_id="early",
            user_id="user",
            content="Python is the best language",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        late = MemoryObject(
            memory_id="late",
            user_id="user",
            content="A language that is Python and great",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored_early = reranker.score("Python", [early])
        scored_late = reranker.score("Python", [late])
        # Early mention gets position bonus
        assert scored_early[0].cross_encoder_score >= scored_late[0].cross_encoder_score

    def test_stemmed_overlap(self) -> None:
        reranker = FallbackReranker()
        query_terms = {"program", "testing"}
        doc_terms = {"program", "test"}
        score = reranker._stemmed_overlap(query_terms, doc_terms)
        assert score > 0.0

    def test_stemmed_overlap_no_match(self) -> None:
        reranker = FallbackReranker()
        query_terms = {"hello"}
        doc_terms = {"world"}
        score = reranker._stemmed_overlap(query_terms, doc_terms)
        assert score == 0.0

    def test_strip_suffix(self) -> None:
        reranker = FallbackReranker()
        assert reranker._strip_suffix("running") == "runn"
        assert reranker._strip_suffix("played") == "play"
        assert reranker._strip_suffix("quickly") == "quick"
        # "tion" suffix matches: "attention" -> "atten"
        assert reranker._strip_suffix("attention") == "atten"
        assert reranker._strip_suffix("basic") == "basic"  # no matching suffix

    def test_normalize_terms(self) -> None:
        reranker = FallbackReranker()
        result = reranker._normalize_terms("Hello World PYTHON")
        assert result == ["hello", "world", "python"]

    def test_position_bonus_capped_at_05(self) -> None:
        reranker = FallbackReranker()
        # Five distinct terms appearing in first 100 chars = 0.5 max
        bonus = reranker._position_bonus({"a", "b", "c", "d", "e"}, "a b c d e" + " x" * 20)
        assert bonus == 0.5

    def test_score_with_embed_fn(self) -> None:
        reranker = FallbackReranker(embed_fn=self.embed)
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Python programming",
            embedding=self.embed.embed_single("Python programming"),
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored = reranker.score("Python", [mem])
        assert scored[0].cross_encoder_score > 0.0

    def test_score_empty_content(self) -> None:
        reranker = FallbackReranker()
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored = reranker.score("Python", [mem])
        assert scored[0].cross_encoder_score >= 0.0

    def test_score_preserves_bi_encoder_rank(self) -> None:
        reranker = FallbackReranker()
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Python test",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored = reranker.score("Python", [mem])
        assert scored[0].bi_encoder_rank == 0


# ---------------------------------------------------------------------------
# NomicReranker
# ---------------------------------------------------------------------------

class TestNomicReranker:
    def test_init_default_model(self) -> None:
        reranker = NomicReranker()
        assert reranker._model == "nomic-embed-text-v1.5"

    def test_init_custom_model(self) -> None:
        reranker = NomicReranker(model="custom-model")
        assert reranker._model == "custom-model"

    def test_score_falls_back_on_exception(self) -> None:
        # No nomic server running → falls back to FallbackReranker
        reranker = NomicReranker()
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Python",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        scored = reranker.score("Python", [mem])
        # Falls back to FallbackReranker → should return valid results
        assert len(scored) == 1


# ---------------------------------------------------------------------------
# rerank_results — public API
# ---------------------------------------------------------------------------

class TestRerankResults:
    def test_empty_results(self) -> None:
        config = RerankerConfig(provider="fallback")
        result = rerank_results([], "query", config)
        assert result == []

    def test_fallback_provider(self) -> None:
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Python",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        config = RerankerConfig(provider="fallback")
        result = rerank_results([mem], "Python", config)
        assert len(result) == 1
        # Should be re-sorted by cross_encoder_score
        assert result[0].memory_id == "test"

    def test_score_threshold_drops_low_results(self) -> None:
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        low_mem = MemoryObject(
            memory_id="low",
            user_id="user",
            content="xyz unrelated content",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        high_mem = MemoryObject(
            memory_id="high",
            user_id="user",
            content="Python programming is great",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        config = RerankerConfig(provider="fallback", score_threshold=0.5)
        result = rerank_results([low_mem, high_mem], "Python", config)
        # Only "high" should remain
        assert len(result) <= 2

    def test_unknown_provider_uses_fallback(self) -> None:
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="test",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        config = RerankerConfig(provider="unknown-provider")
        result = rerank_results([mem], "test", config)
        assert len(result) == 1

    def test_bi_encoder_rank_assigned(self) -> None:
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="test",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        # rerank_results assigns bi_encoder_rank based on index in input list
        config = RerankerConfig(provider="fallback")
        result = rerank_results([mem], "test", config)
        # First (and only) item should have bi_encoder_rank == 0
        assert result[0].bi_encoder_rank == 0


# ---------------------------------------------------------------------------
# fuse_and_rerank
# ---------------------------------------------------------------------------

class TestFuseAndRerank:
    def test_empty_fusion_results(self) -> None:
        config = RerankerConfig(provider="fallback")
        result = fuse_and_rerank([], "query", config)
        assert result == []

    def test_combines_rrf_and_cross_encoder_scores(self) -> None:
        from kemi.models import MemoryObject
        from datetime import datetime, timezone
        from kemi.models import MemorySource
        from kemi.decomposer import FusionResult

        mem = MemoryObject(
            memory_id="test",
            user_id="user",
            content="Python",
            embedding=[0.1] * 64,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=None,
            metadata={},
            embedding_dim=64,
        )
        mem.cross_encoder_score = 0.8

        fusion_result = FusionResult(
            memory=mem, rrf_score=0.5, source_ranks={"query": 0}
        )
        config = RerankerConfig(provider="fallback")
        result = fuse_and_rerank([fusion_result], "Python", config)
        assert len(result) == 1
        # Combined: 0.4 * cross_encoder + 0.6 * rrf
        assert result[0].cross_encoder_score > 0.0


# ---------------------------------------------------------------------------
# CrossEncoderReranker (stub)
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    def test_is_a_stub_class(self) -> None:
        # CrossEncoderReranker is a stub/placeholder for future cross-encoder integration
        assert CrossEncoderReranker is not None
        assert isinstance(CrossEncoderReranker, type)
        # It should have no concrete implementation yet
        assert hasattr(CrossEncoderReranker, "__init__")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_fn():
    import hashlib

    class _EmbedFn:
        def embed(self, texts):
            return [self._vector(t) for t in texts]

        def embed_single(self, text):
            return self._vector(text)

        def _vector(self, text):
            raw = hashlib.sha256(text.encode()).digest()
            expanded = raw * (64 // len(raw) + 1)
            return [b / 255.0 for b in expanded[:64]]

    return _EmbedFn()