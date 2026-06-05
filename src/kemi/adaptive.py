"""Adaptive retrieval for kemi memory.

Auto-tunes hybrid search weights based on query characteristics.
Provides query analysis, classification, and dynamic weight adjustment.

Features:
- Query classification (factual, conversational, procedural, keyword-dense)
- Dynamic weight adjustment for semantic vs BM25 vs recency
- Query length impact assessment
- Feedback-driven continuous improvement
- Query specificity scoring

Usage:
    from kemi.adaptive import AdaptiveRetriever

    retriever = AdaptiveRetriever()
    weights = retriever.analyze_query("What are my food preferences?")
    # weights = {"weight_semantic": 0.65, "weight_recency": 0.20, "weight_bm25": 0.15}
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Classification of query types for adaptive retrieval."""

    FACTUAL = "factual"  # "What is X?", "Who is Y?"
    CONVERSATIONAL = "conversational"  # "How are you?", "Tell me about..."
    PROCEDURAL = "procedural"  # "How do I...", "Steps to..."
    KEYWORD_DENSE = "keyword_dense"  # "dark mode preference vegetarian food"
    TEMPORAL = "temporal"  # "What did I do yesterday?", "Last week's..."
    COMPARATIVE = "comparative"  # "X vs Y", "better option"
    AMBIGUOUS = "ambiguous"  # Unclear query intent


# Keyword patterns for query classification
_FACTUAL_PATTERNS = [
    r"\bwhat (is|are|was|were)\b",
    r"\bwho (is|are|was|were)\b",
    r"\bwhen (is|was|did)\b",
    r"\bwhere (is|are|was|were)\b",
    r"\bwhich (is|are|was|were)\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bmeaning of\b",
]

_CONVERSATIONAL_PATTERNS = [
    r"\bhow are you\b",
    r"\btell me about\b",
    r"\bcan you\b",
    r"\bplease\b",
    r"\bthanks?\b",
    r"\bhelp me\b",
    r"\bexplain\b",
    r"\bdescribe\b",
]

_PROCEDURAL_PATTERNS = [
    r"\bhow (do|can|would|should|to)\b",
    r"\bsteps?\b",
    r"\bguide\b",
    r"\btutorial\b",
    r"\bprocess\b",
    r"\binstruction\b",
    r"\bwalkthrough\b",
]

_TEMPORAL_PATTERNS = [
    r"\b(yesterday|today|tomorrow)\b",
    r"\blast (week|month|year|night|time)\b",
    r"\bthis (week|month|year)\b",
    r"\b(ago|recently|lately|earlier)\b",
    r"\bwhen (did|was|were)\b",
    r"\bwhat (happened|occurred)\b",
]

_COMPARATIVE_PATTERNS = [
    r"\b(vs|versus|compared)\b",
    r"\b(better|worse|best|worst)\b",
    r"\b(difference|similar)\b",
    r"\b(option|choice|alternative)\b",
    r"\b(prefer|rather)\b",
]


@dataclass
class QueryProfile:
    """Analysis result for a query."""

    query: str
    query_type: QueryType
    word_count: int
    keyword_density: float = 0.0  # Ratio of content words to total words
    specificity: float = 0.0  # 0.0 = vague, 1.0 = highly specific
    has_question_mark: bool = False
    has_named_entity_hint: bool = False  # Has capitalized words or numbers
    recommended_weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5  # Confidence in the classification


@dataclass
class AdaptiveWeights:
    """Dynamically computed retrieval weights."""

    weight_semantic: float = 0.6
    weight_recency: float = 0.25
    weight_bm25: float = 0.15
    weight_semantic_no_embed: float = 0.5
    weight_recency_no_embed: float = 0.3
    weight_importance: float = 0.2
    query_type: QueryType = QueryType.AMBIGUOUS
    analysis_confidence: float = 0.5


class AdaptiveRetriever:
    """Auto-tunes retrieval weights based on query characteristics.

    Uses heuristic analysis of the query text to determine the best
    hybrid search weight configuration. No ML models required.

    Limitations:
    - Classification is based on keyword/regex pattern matching and
      may misclassify unusual or ambiguous queries.
    - When confidence is low, weights fall back to defaults.
    - For production use with very diverse query types, consider
      training a small classifier or using LLM-based classification.
    """

    # Stop words to filter out for keyword density calculation
    _STOP_WORDS: set[str] = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "about",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "when",
        "where",
        "if",
        "then",
        "else",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "we",
        "you",
        "me",
        "my",
        "your",
        "our",
        "i",
        "him",
        "her",
        "us",
    }

    # Base weight configurations for each query type
    _TYPE_WEIGHTS: dict[QueryType, dict[str, float]] = {
        QueryType.FACTUAL: {
            "weight_semantic": 0.55,
            "weight_recency": 0.20,
            "weight_bm25": 0.25,
            "weight_semantic_no_embed": 0.45,
            "weight_recency_no_embed": 0.25,
            "weight_importance": 0.30,
        },
        QueryType.CONVERSATIONAL: {
            "weight_semantic": 0.70,
            "weight_recency": 0.20,
            "weight_bm25": 0.10,
            "weight_semantic_no_embed": 0.60,
            "weight_recency_no_embed": 0.25,
            "weight_importance": 0.15,
        },
        QueryType.PROCEDURAL: {
            "weight_semantic": 0.50,
            "weight_recency": 0.15,
            "weight_bm25": 0.35,
            "weight_semantic_no_embed": 0.40,
            "weight_recency_no_embed": 0.25,
            "weight_importance": 0.35,
        },
        QueryType.KEYWORD_DENSE: {
            "weight_semantic": 0.40,
            "weight_recency": 0.15,
            "weight_bm25": 0.45,
            "weight_semantic_no_embed": 0.35,
            "weight_recency_no_embed": 0.20,
            "weight_importance": 0.45,
        },
        QueryType.TEMPORAL: {
            "weight_semantic": 0.45,
            "weight_recency": 0.40,
            "weight_bm25": 0.15,
            "weight_semantic_no_embed": 0.35,
            "weight_recency_no_embed": 0.45,
            "weight_importance": 0.20,
        },
        QueryType.COMPARATIVE: {
            "weight_semantic": 0.60,
            "weight_recency": 0.15,
            "weight_bm25": 0.25,
            "weight_semantic_no_embed": 0.50,
            "weight_recency_no_embed": 0.20,
            "weight_importance": 0.30,
        },
        QueryType.AMBIGUOUS: {
            "weight_semantic": 0.60,
            "weight_recency": 0.25,
            "weight_bm25": 0.15,
            "weight_semantic_no_embed": 0.50,
            "weight_recency_no_embed": 0.30,
            "weight_importance": 0.20,
        },
    }

    def __init__(
        self,
        enable_adaptation: bool = True,
        feedback_weight: float = 0.1,
    ) -> None:
        """Initialize adaptive retriever.

        Args:
            enable_adaptation: If False, always returns default weights.
            feedback_weight: How much to adjust weights from feedback (0.0-1.0).
        """
        self._enable_adaptation = enable_adaptation
        self._feedback_weight = max(0.0, min(1.0, feedback_weight))
        # Track per-user query type distribution for better adaptation
        self._user_query_history: dict[str, dict[str, int]] = {}

    def analyze_query(self, query: str) -> QueryProfile:
        """Analyze a query and return its profile.

        Args:
            query: The search query string.

        Returns:
            QueryProfile with classification and recommended weights.
        """
        if not query or not query.strip():
            return QueryProfile(
                query="",
                query_type=QueryType.AMBIGUOUS,
                word_count=0,
                recommended_weights=self._TYPE_WEIGHTS[QueryType.AMBIGUOUS],
            )

        words = query.strip().split()
        word_count = len(words)

        # Classify query type
        query_lower = query.lower()
        query_type, confidence = self._classify_query(query_lower)

        # Calculate keyword density
        keyword_density = self._compute_keyword_density(words)

        # Calculate specificity
        specificity = self._compute_specificity(query, words)

        # Check for named entity hints
        has_named_entity_hint = bool(
            re.search(r"[A-Z][a-z]{2,}", query) or re.search(r"\d+", query)
        )

        # Get base weights for this query type
        base_weights = dict(self._TYPE_WEIGHTS[query_type])

        # Adjust weights based on query characteristics
        adjusted_weights = self._adjust_weights(
            base_weights,
            keyword_density,
            specificity,
            word_count,
        )

        return QueryProfile(
            query=query,
            query_type=query_type,
            word_count=word_count,
            keyword_density=keyword_density,
            specificity=specificity,
            has_question_mark=query.rstrip().endswith("?"),
            has_named_entity_hint=has_named_entity_hint,
            recommended_weights=adjusted_weights,
            confidence=confidence,
        )

    def get_weights(self, query: str) -> AdaptiveWeights:
        """Get adaptive retrieval weights for a query.

        This is the main entry point for integration with the recall pipeline.

        Args:
            query: The search query string.

        Returns:
            AdaptiveWeights with the recommended weight configuration.
        """
        if not self._enable_adaptation:
            return AdaptiveWeights()

        profile = self.analyze_query(query)

        return AdaptiveWeights(
            weight_semantic=profile.recommended_weights["weight_semantic"],
            weight_recency=profile.recommended_weights["weight_recency"],
            weight_bm25=profile.recommended_weights["weight_bm25"],
            weight_semantic_no_embed=profile.recommended_weights["weight_semantic_no_embed"],
            weight_recency_no_embed=profile.recommended_weights["weight_recency_no_embed"],
            weight_importance=profile.recommended_weights["weight_importance"],
            query_type=profile.query_type,
            analysis_confidence=profile.confidence,
        )

    def record_feedback(
        self,
        user_id: str,
        query: str,
        profile: QueryProfile,
    ) -> None:
        """Record query type for this user to improve future adaptation.

        Args:
            user_id: User who made the query.
            query: The original query.
            profile: The QueryProfile that was used.
        """
        if user_id not in self._user_query_history:
            self._user_query_history[user_id] = {}

        qtype = profile.query_type.value
        self._user_query_history[user_id][qtype] = (
            self._user_query_history[user_id].get(qtype, 0) + 1
        )

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Get the query type distribution for a user.

        Args:
            user_id: User to get profile for.

        Returns:
            Dict with query type distribution and dominant type.
        """
        history = self._user_query_history.get(user_id, {})
        total = sum(history.values()) if history else 0

        if total == 0:
            return {
                "user_id": user_id,
                "total_queries": 0,
                "distribution": {},
                "dominant_type": None,
            }

        distribution = {k: v / total for k, v in history.items()}
        dominant = max(history, key=history.get)

        return {
            "user_id": user_id,
            "total_queries": total,
            "distribution": distribution,
            "dominant_type": dominant,
        }

    def _classify_query(self, query_lower: str) -> tuple[QueryType, float]:
        """Classify query into a type using pattern matching.

        Returns:
            Tuple of (QueryType, confidence).
        """
        scores: dict[QueryType, int] = {
            QueryType.FACTUAL: 0,
            QueryType.CONVERSATIONAL: 0,
            QueryType.PROCEDURAL: 0,
            QueryType.TEMPORAL: 0,
            QueryType.COMPARATIVE: 0,
        }

        for pattern in _FACTUAL_PATTERNS:
            if re.search(pattern, query_lower):
                scores[QueryType.FACTUAL] += 1
        for pattern in _CONVERSATIONAL_PATTERNS:
            if re.search(pattern, query_lower):
                scores[QueryType.CONVERSATIONAL] += 1
        for pattern in _PROCEDURAL_PATTERNS:
            if re.search(pattern, query_lower):
                scores[QueryType.PROCEDURAL] += 1
        for pattern in _TEMPORAL_PATTERNS:
            if re.search(pattern, query_lower):
                scores[QueryType.TEMPORAL] += 1
        for pattern in _COMPARATIVE_PATTERNS:
            if re.search(pattern, query_lower):
                scores[QueryType.COMPARATIVE] += 1

        # Check for keyword-dense: no question structure, short, many nouns
        words = query_lower.split()
        has_question_word = any(w in words for w in ("what", "who", "when", "where", "why", "how"))
        if not has_question_word and len(words) <= 6:
            content_words = [w for w in words if w not in self._STOP_WORDS]
            if len(content_words) >= len(words) * 0.6:
                scores[QueryType.KEYWORD_DENSE] = 3 if len(content_words) >= 2 else 1

        # Find the highest scoring type
        if not scores or max(scores.values()) == 0:
            return QueryType.AMBIGUOUS, 0.3

        best_type = max(scores, key=lambda k: scores[k])  # type: ignore[arg-type]
        max_score = scores[best_type]
        total_score = sum(scores.values()) if scores else 1
        confidence = max_score / max(total_score, 1)

        return best_type, min(confidence, 0.95)

    def _compute_keyword_density(self, words: list[str]) -> float:
        """Compute ratio of content words to total words."""
        if not words:
            return 0.0
        content_words = [w for w in words if w.lower() not in self._STOP_WORDS]
        return len(content_words) / len(words)

    def _compute_specificity(self, query: str, words: list[str]) -> float:
        """Estimate query specificity (0.0 = vague, 1.0 = highly specific).

        Factors:
        - Query length (longer = more specific)
        - Unique words ratio
        - Presence of numbers, proper nouns, dates
        - Specificity modifiers ("exactly", "specifically", "precise")
        """
        if not words:
            return 0.0

        score = 0.0

        # Length factor
        if len(words) <= 2:
            score += 0.1
        elif len(words) <= 4:
            score += 0.3
        elif len(words) <= 8:
            score += 0.5
        else:
            score += 0.7

        # Unique words ratio
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        score += unique_ratio * 0.3

        # Numbers and proper nouns
        if re.search(r"\d+", query):
            score += 0.15
        if re.search(r"[A-Z][a-z]{2,}", query):
            score += 0.15

        # Specificity modifiers
        if re.search(r"\b(exactly|specifically|precisely|particular)\b", query.lower()):
            score += 0.1

        return min(1.0, score)

    def _adjust_weights(
        self,
        base_weights: dict[str, float],
        keyword_density: float,
        specificity: float,
        word_count: int,
    ) -> dict[str, float]:
        """Fine-tune weights based on query characteristics.

        Rules:
        - Higher keyword density → boost BM25, reduce semantic
        - Higher specificity → boost semantic (more precise semantic match)
        - Longer queries → slight boost to BM25 (more keywords)
        - Very short queries → boost semantic (more likely conceptual)
        """
        weights = dict(base_weights)

        # Keyword density adjustment (max ±0.1)
        # High keyword density = good for BM25 keyword matching
        bm25_adjust = (keyword_density - 0.5) * 0.2
        weights["weight_bm25"] = max(0.05, min(0.55, weights["weight_bm25"] + bm25_adjust))
        weights["weight_semantic"] = max(
            0.30, min(0.80, weights["weight_semantic"] - bm25_adjust * 0.5)
        )

        # Specificity adjustment
        # High specificity = better semantic matching possible
        sem_adjust = (specificity - 0.5) * 0.1
        weights["weight_semantic"] = max(0.30, min(0.80, weights["weight_semantic"] + sem_adjust))

        # Word count adjustment
        if word_count <= 2:
            # Very short: boost semantic slightly
            weights["weight_semantic"] = min(0.80, weights["weight_semantic"] + 0.05)
            weights["weight_bm25"] = max(0.05, weights["weight_bm25"] - 0.03)
        elif word_count >= 10:
            # Very long: boost BM25
            weights["weight_bm25"] = min(0.55, weights["weight_bm25"] + 0.05)

        # Ensure all weights sum approximately to 1.0
        total = weights["weight_semantic"] + weights["weight_recency"] + weights["weight_bm25"]
        if total > 0:
            scale = 1.0 / total
            weights["weight_semantic"] = round(weights["weight_semantic"] * scale, 4)
            weights["weight_recency"] = round(weights["weight_recency"] * scale, 4)
            weights["weight_bm25"] = round(weights["weight_bm25"] * scale, 4)

        # No-embed weights also adjust proportionally
        no_embed_total = (
            weights["weight_semantic_no_embed"]
            + weights["weight_recency_no_embed"]
            + weights["weight_importance"]
        )
        if no_embed_total > 0:
            scale_ne = 1.0 / no_embed_total
            weights["weight_semantic_no_embed"] = round(
                weights["weight_semantic_no_embed"] * scale_ne,
                4,
            )
            weights["weight_recency_no_embed"] = round(
                weights["weight_recency_no_embed"] * scale_ne,
                4,
            )
            weights["weight_importance"] = round(
                weights["weight_importance"] * scale_ne,
                4,
            )

        return weights
