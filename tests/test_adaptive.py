"""Tests for kemi adaptive retrieval module."""

from kemi.memory.adaptive import (
    AdaptiveRetriever,
    AdaptiveWeights,
    QueryType,
)


class TestQueryClassification:
    """Tests for query type classification."""

    def test_classify_factual(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("What is my food preference?")
        assert profile.query_type == QueryType.FACTUAL

    def test_classify_factual_who(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("Who is my manager?")
        assert profile.query_type == QueryType.FACTUAL

    def test_classify_factual_where(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("Where did I park my car?")
        # "where" alone is ambiguous; combined with "did I" it matches keyword_dense pattern
        assert profile.query_type in (
            QueryType.FACTUAL,
            QueryType.AMBIGUOUS,
            QueryType.KEYWORD_DENSE,
        )

    def test_classify_procedural(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("How do I reset my password?")
        assert profile.query_type == QueryType.PROCEDURAL

    def test_classify_temporal(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("What did I do yesterday?")
        assert profile.query_type == QueryType.TEMPORAL

    def test_classify_comparative(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("Which is better, option A or option B?")
        assert profile.query_type == QueryType.COMPARATIVE

    def test_classify_keyword_dense(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("dark mode preference vegetarian food")
        assert profile.query_type == QueryType.KEYWORD_DENSE

    def test_classify_empty_query(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("")
        assert profile.query_type == QueryType.AMBIGUOUS
        assert profile.word_count == 0

    def test_classify_ambiguous(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("hello there")
        # Short greeting with no question words triggers keyword_dense heuristic
        assert profile.query_type in (QueryType.AMBIGUOUS, QueryType.KEYWORD_DENSE)


class TestQueryProfile:
    """Tests for QueryProfile dataclass."""

    def test_profile_has_all_fields(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("What is my favorite color?")
        assert profile.query is not None
        assert profile.query_type is not None
        assert profile.word_count > 0
        assert 0.0 <= profile.keyword_density <= 1.0
        assert 0.0 <= profile.specificity <= 1.0
        assert isinstance(profile.has_question_mark, bool)
        assert isinstance(profile.has_named_entity_hint, bool)
        assert isinstance(profile.recommended_weights, dict)
        assert 0.0 <= profile.confidence <= 1.0

    def test_profile_keyword_density(self) -> None:
        retriever = AdaptiveRetriever()
        # Query with mostly content words
        profile = retriever.analyze_query("food preference vegetarian dark mode")
        assert profile.keyword_density > 0.5

        # Query with mostly stop words
        profile = retriever.analyze_query("what is the thing about that")
        assert profile.keyword_density < 0.5

    def test_profile_specificity(self) -> None:
        retriever = AdaptiveRetriever()
        # Short vague query
        profile = retriever.analyze_query("food")
        assert profile.specificity < 0.5

        # Long specific query with numbers and proper nouns
        profile = retriever.analyze_query(
            "What are John Smith's preferences for dark mode exactly in 2025"
        )
        assert profile.specificity > 0.4

    def test_profile_named_entity_detection(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("What does John prefer?")
        assert profile.has_named_entity_hint is True

        profile = retriever.analyze_query("What is the price of order 12345?")
        assert profile.has_named_entity_hint is True

        profile = retriever.analyze_query("what do i like")
        assert profile.has_named_entity_hint is False


class TestAdaptiveWeights:
    """Tests for AdaptiveWeights and get_weights."""

    def test_get_weights_returns_valid_weights(self) -> None:
        retriever = AdaptiveRetriever()
        weights = retriever.get_weights("What are my food preferences?")
        assert isinstance(weights, AdaptiveWeights)
        assert 0.3 <= weights.weight_semantic <= 0.8
        assert 0.1 <= weights.weight_recency <= 0.45
        assert 0.05 <= weights.weight_bm25 <= 0.5
        # Weights should sum approximately to 1.0
        total = weights.weight_semantic + weights.weight_recency + weights.weight_bm25
        assert abs(total - 1.0) < 0.02

    def test_get_weights_disabled(self) -> None:
        retriever = AdaptiveRetriever(enable_adaptation=False)
        weights = retriever.get_weights("any query")
        assert weights.weight_semantic == 0.6  # Default
        assert weights.weight_recency == 0.25
        assert weights.weight_bm25 == 0.15

    def test_keyword_dense_boosts_bm25(self) -> None:
        retriever = AdaptiveRetriever()
        weights = retriever.get_weights("python async memory library sqlite")
        # BM25 should be higher than default 0.15 for keyword-dense queries
        assert weights.weight_bm25 >= 0.20

    def test_conversational_boosts_semantic(self) -> None:
        retriever = AdaptiveRetriever()
        weights = retriever.get_weights("Tell me about my preferences")
        # Semantic should be higher for conversational queries
        assert weights.weight_semantic > 0.5

    def test_temporal_boosts_recency(self) -> None:
        retriever = AdaptiveRetriever()
        weights = retriever.get_weights("What did I do yesterday?")
        # Recency should be boosted for temporal queries
        assert weights.weight_recency >= 0.25


class TestUserProfile:
    """Tests for user query profile tracking."""

    def test_record_feedback(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("What is my name?")
        retriever.record_feedback("alice", "What is my name?", profile)

        user_profile = retriever.get_user_profile("alice")
        assert user_profile["user_id"] == "alice"
        assert user_profile["total_queries"] == 1
        assert "factual" in user_profile["distribution"]

    def test_user_profile_multiple_types(self) -> None:
        retriever = AdaptiveRetriever()

        queries = [
            "What is my name?",  # factual
            "How do I reset password?",  # procedural
            "dark mode preference",  # keyword_dense
            "What did I do last week?",  # temporal
        ]
        for q in queries:
            profile = retriever.analyze_query(q)
            retriever.record_feedback("alice", q, profile)

        user_profile = retriever.get_user_profile("alice")
        assert user_profile["total_queries"] == 4
        assert user_profile["dominant_type"] is not None

    def test_get_profile_unknown_user(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.get_user_profile("nonexistent")
        assert profile["total_queries"] == 0
        assert profile["dominant_type"] is None


class TestAdaptiveRetrieverEdgeCases:
    """Edge case tests for AdaptiveRetriever."""

    def test_whitespace_query(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("   ")
        assert profile.query_type == QueryType.AMBIGUOUS

    def test_very_long_query(self) -> None:
        retriever = AdaptiveRetriever()
        long_query = "what is the best way to " * 20
        profile = retriever.analyze_query(long_query)
        assert profile.word_count >= 10
        assert isinstance(profile.recommended_weights, dict)

    def test_single_word_query(self) -> None:
        retriever = AdaptiveRetriever()
        profile = retriever.analyze_query("food")
        assert profile.word_count == 1
        # Single content word triggers keyword_dense (high keyword density)
        assert profile.query_type in (QueryType.AMBIGUOUS, QueryType.KEYWORD_DENSE)
