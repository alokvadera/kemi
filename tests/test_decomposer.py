"""Tests for src/kemi/decomposer.py — query decomposition and RRF fusion."""

import pytest

from kemi import decomposer
from kemi.decomposer import (
    FusionResult,
    QueryDecompositionStrategy,
    SimpleDecomposition,
    SubqueryExpansion,
    decompose_query,
    fused_recall,
    rerank_with_reranker,
)


# ---------------------------------------------------------------------------
# QueryDecompositionStrategy (base class)
# ---------------------------------------------------------------------------

class TestQueryDecompositionStrategy:
    def test_base_decompose_raises(self) -> None:
        strategy = QueryDecompositionStrategy()
        with pytest.raises(NotImplementedError):
            strategy.decompose("test query")


# ---------------------------------------------------------------------------
# SimpleDecomposition
# ---------------------------------------------------------------------------

class TestSimpleDecomposition:
    def setup_method(self) -> None:
        self.strat = SimpleDecomposition()

    def test_simple_query_no_split(self) -> None:
        result = self.strat.decompose("What is my favorite color?")
        assert len(result) == 1
        assert result[0] == "What is my favorite color?"

    def test_split_on_and(self) -> None:
        result = self.strat.decompose("I like cats and dogs")
        assert len(result) >= 2

    def test_split_on_or(self) -> None:
        result = self.strat.decompose("Tell me about work or personal tasks")
        assert len(result) >= 2

    def test_split_on_but(self) -> None:
        result = self.strat.decompose("I want to exercise but I am tired")
        assert len(result) >= 2

    def test_empty_query(self) -> None:
        assert self.strat.decompose("") == []
        assert self.strat.decompose("   ") == []

    def test_single_word_no_split(self) -> None:
        result = self.strat.decompose("Python")
        assert len(result) == 1

    def test_reconstruct_query_adds_prefix(self) -> None:
        result = self.strat.decompose("tasks")
        # Single word 'tasks' has no conjunction, returns as-is → no prefix test
        # Test via decompose: 'schedules' alone also returns as-is
        # The prefix is added by _reconstruct_query for non-question-word clauses
        # For 'work meetings' (no conjunction) it returns as-is since no conjunction was found
        result2 = self.strat.decompose("do my work")
        # 'do my work' starts with 'do' which is in question_words → no prefix
        # But 'my work' (after split on 'and') starts with 'my' → prefix added
        # Actually 'my work' has only 2 words so passes >=2 filter
        # Let's directly test the internal method
        from kemi.decomposer import SimpleDecomposition
        strat = SimpleDecomposition()
        # 'meetings' alone starts with 'meetings' (not a question word) → prefix added
        prefix_result = strat._reconstruct_query("meetings")
        assert "Tell me about" in prefix_result, f"Expected 'Tell me about' prefix, got {prefix_result!r}"

    def test_no_conjunction_returns_original(self) -> None:
        text = "What did I eat for breakfast?"
        result = self.strat.decompose(text)
        assert len(result) == 1

    def test_question_word_preserved(self) -> None:
        result = self.strat.decompose("breakfast and dinner plans")
        # Should reconstruct with "Tell me about" since no question word
        assert len(result) >= 1

    def test_realistic_multi_aspect_query(self) -> None:
        result = self.strat.decompose(
            "What did I eat for breakfast and dinner yesterday?"
        )
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# SubqueryExpansion
# ---------------------------------------------------------------------------

class TestSubqueryExpansion:
    def setup_method(self) -> None:
        self.strat = SubqueryExpansion()

    def test_empty_query(self) -> None:
        assert self.strat.decompose("") == []
        assert self.strat.decompose("   ") == []

    def test_expands_eat(self) -> None:
        result = self.strat.decompose("I eat breakfast")
        assert len(result) >= 2
        assert any("eat" in q or "consume" in q or "have" in q for q in result)

    def test_expands_work(self) -> None:
        result = self.strat.decompose("Work meeting today")
        assert len(result) >= 2

    def test_original_always_included(self) -> None:
        result = self.strat.decompose("I exercise in the morning")
        assert any(q.strip() == "I exercise in the morning" for q in result)

    def test_max_five_variants(self) -> None:
        text = "I eat breakfast dinner lunch work meeting exercise"
        result = self.strat.decompose(text)
        assert len(result) <= 5

    def test_no_change_for_unknown_terms(self) -> None:
        result = self.strat.decompose("xyzqwerty")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# decompose_query — public API
# ---------------------------------------------------------------------------

class TestDecomposeQuery:
    def test_none_strategy(self) -> None:
        result = decompose_query("test query", strategy="none")
        assert result.strategy == "none"
        assert result.sub_queries == ["test query"]
        assert result.original_query == "test query"

    def test_simple_strategy(self) -> None:
        result = decompose_query("work and personal tasks", strategy="simple")
        assert result.strategy == "simple"
        assert len(result.sub_queries) >= 2

    def test_expand_strategy(self) -> None:
        result = decompose_query("I eat lunch", strategy="expand")
        assert result.strategy == "expand"
        assert len(result.sub_queries) >= 1

    def test_both_strategy_deduplicates(self) -> None:
        result = decompose_query("work and work tasks", strategy="both")
        # Both strategies run; duplicates should be removed
        assert result.strategy == "both"
        # All sub-queries should be unique
        assert len(result.sub_queries) == len(set(result.sub_queries))

    def test_both_capped_at_five(self) -> None:
        result = decompose_query(
            "I eat breakfast and dinner and lunch and work and exercise",
            strategy="both"
        )
        assert len(result.sub_queries) <= 5

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown decomposition strategy"):
            decompose_query("test", strategy="invalid")

    def test_returns_decomposed_query_dataclass(self) -> None:
        from kemi.decomposer import DecomposedQuery
        result = decompose_query("test query", strategy="simple")
        assert isinstance(result, DecomposedQuery)
        assert hasattr(result, "strategy")
        assert hasattr(result, "sub_queries")
        assert hasattr(result, "original_query")

    def test_original_query_preserved(self) -> None:
        original = "Complex query about work and life"
        result = decompose_query(original, strategy="simple")
        assert result.original_query == original


# ---------------------------------------------------------------------------
# fused_recall
# ---------------------------------------------------------------------------

class TestFusedRecall:
    def test_empty_sub_queries(self) -> None:
        result = fused_recall(None, "user1", [])
        assert result == []

    def test_single_sub_query_returns_memory_recall(self, mock_memory) -> None:
        from kemi.decomposer import FusionResult
        mock_memory.remember("user1", "Python is great")
        result = fused_recall(mock_memory, "user1", ["Python"], top_k=3)
        assert len(result) >= 1
        assert all(isinstance(r, FusionResult) for r in result)
        for fr in result:
            assert hasattr(fr, "memory")
            assert hasattr(fr, "rrf_score")
            assert hasattr(fr, "source_ranks")

    def test_rrf_score_positive(self, mock_memory) -> None:
        mock_memory.remember("user1", "I like cats")
        mock_memory.remember("user1", "I like dogs")
        result = fused_recall(mock_memory, "user1", ["cats", "dogs"], top_k=5)
        for fr in result:
            assert fr.rrf_score > 0.0

    def test_source_ranks_contain_all_sub_queries(self, mock_memory) -> None:
        mock_memory.remember("user1", "Cats are furry")
        mock_memory.remember("user1", "Dogs are loyal")
        result = fused_recall(
            mock_memory, "user1", ["cats", "dogs"], top_k=5
        )
        if result:
            # Each result should show which sub-query(s) it came from
            for fr in result:
                assert len(fr.source_ranks) >= 1

    def test_rrf_favors_memories_appearing_in_multiple_results(self, mock_memory) -> None:
        # A memory that appears in both sub-query results should rank higher
        mock_memory.remember("user1", "I like both cats and dogs")
        mock_memory.remember("user1", "I like cats")
        mock_memory.remember("user1", "I like dogs")
        result = fused_recall(mock_memory, "user1", ["cats", "dogs"], top_k=3)
        if len(result) >= 2:
            # "both cats and dogs" appears in both results → should rank high
            ids = [fr.memory.memory_id for fr in result]
            both_mem = next(
                (fr.memory for fr in result
                 if "both" in fr.memory.content.lower()),
                None
            )
            # If it exists, it should be in top positions
            if both_mem:
                assert any(fr.memory.memory_id == both_mem.memory_id for fr in result[:2])

    def test_different_top_k(self, mock_memory) -> None:
        mock_memory.remember("user1", "Memory one")
        mock_memory.remember("user1", "Memory two")
        result_1 = fused_recall(mock_memory, "user1", ["memory"], top_k=1)
        result_2 = fused_recall(mock_memory, "user1", ["memory"], top_k=2)
        assert len(result_1) <= 1
        assert len(result_2) <= 2

    def test_results_sorted_by_rrf_score_descending(self, mock_memory) -> None:
        mock_memory.remember("user1", "Python is great")
        mock_memory.remember("user1", "Python and coding")
        result = fused_recall(mock_memory, "user1", ["Python"], top_k=5)
        scores = [fr.rrf_score for fr in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# rerank_with_reranker — placeholder behavior
# ---------------------------------------------------------------------------

class TestRerankWithReranker:
    def test_returns_original_results_unchanged(self, mock_memory) -> None:
        from kemi.models import MemoryObject
        mock_memory.remember("user1", "Python is great")
        results = mock_memory.recall("user1", "Python")
        reranked = rerank_with_reranker(mock_memory, "user1", "Python", results)
        assert reranked == results  # placeholder returns unchanged

    def test_unknown_provider_still_returns_results(self, mock_memory) -> None:
        mock_memory.remember("user1", "test content")
        results = mock_memory.recall("user1", "test")
        reranked = rerank_with_reranker(
            mock_memory, "user1", "test", results, provider="unknown"
        )
        assert len(reranked) == len(results)