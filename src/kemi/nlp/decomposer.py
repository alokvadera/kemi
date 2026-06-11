"""Query decomposition and result fusion for improved recall.

Breaks complex multi-aspect queries into simpler sub-queries, executes each
sub-query independently against the memory store, then fuses the ranked
results using Reciprocal Rank Fusion (RRF).

RRF is parameter-free and works with any retrieval method (vector, keyword,
or hybrid). It uses only ranks (not scores) so it is robust to score scale
differences between retrieval methods.

Usage:
    sub_queries = decompose_query(
        "What did I eat for breakfast and dinner yesterday?",
        strategy="simple"
    )
    # ["What did I eat for breakfast yesterday?",
    #  "What did I eat for dinner yesterday?"]

    results = fused_recall(memory, user_id, sub_queries, top_k=5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kemi.exceptions import ConfigurationError
from kemi.memory.model import LifecycleState, MemoryObject

if TYPE_CHECKING:
    from kemi import Memory
    __all__ = [
    "decompose_query",
    "fused_recall",
    "QueryDecompositionStrategy",
    "DecomposedQuery",
    "FusionResult",
]


# ---------------------------------------------------------------------------
# Query decomposition strategies
# ---------------------------------------------------------------------------


class QueryDecompositionStrategy:
    """Base class for query decomposition strategies."""

    def decompose(self, query: str) -> list[str]:
        raise NotImplementedError


class SimpleDecomposition(QueryDecompositionStrategy):
    """Split on conjunctions (and, or, but) and question words.

    Handles queries like:
    - "What did I eat for breakfast and dinner?"
      → ["What did I eat for breakfast?", "What did I eat for dinner?"]
    - "Tell me about my work meetings and personal tasks"
      → ["Tell me about my work meetings", "Tell me about my personal tasks"]
    """

    CONJUNCTION_PATTERN = re.compile(
        r"\b(?:and|or|but|however|additionally|also|plus|while)\b",
        re.IGNORECASE,
    )
    QUESTION_STARTS = {"what", "when", "where", "who", "whom", "whose", "why", "how", "which"}

    def decompose(self, query: str) -> list[str]:
        if not query or not query.strip():
            return []

        # Single-sentence, no conjunction — return as-is
        if not self.CONJUNCTION_PATTERN.search(query):
            return [query.strip()]

        # Split on conjunctions
        parts = self.CONJUNCTION_PATTERN.split(query)
        if len(parts) <= 1:
            return [query.strip()]

        sub_queries: list[str] = []
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue

            if self._starts_with_question_word(cleaned):
                sub_queries.append(cleaned)
            else:
                reconstructed = self._reconstruct_query(cleaned)
                sub_queries.append(reconstructed)

        return [q for q in sub_queries if len(q.split()) >= 2]

    def _starts_with_question_word(self, text: str) -> bool:
        first_word = text.split()[0].lower().rstrip("?") if text.split() else ""
        return first_word in self.QUESTION_STARTS

    def _reconstruct_query(self, part: str) -> str:
        """Try to build a self-standing query from a clause."""
        first_word = part.split()[0].lower() if part.split() else ""
        if first_word and first_word not in {"i", "my", "me", "the", "a", "an", "that", "this"}:
            return f"Tell me about {part.strip().rstrip('?')}"
        return part.strip().rstrip("?")


class SubqueryExpansion(QueryDecompositionStrategy):
    """Expand a query with synonyms and related terms to improve recall coverage.

    Generates multiple variants using synonym substitution (no external library required).
    """

    SYNONYMS: dict[str, list[str]] = {
        "eat": ["consume", "have", "dined", "food"],
        "breakfast": ["morning meal", "breakfast"],
        "dinner": ["evening meal", "supper", "dinner"],
        "lunch": ["midday meal", "lunch"],
        "work": ["job", "profession", "career", "task"],
        "meeting": ["discussion", "standup", "sync", "call"],
        "exercise": ["workout", "gym", "run", "fitness"],
        "travel": ["trip", "visit", "journey", "flight"],
        "buy": ["purchase", "shop", "acquire"],
        "learn": ["study", "understand", "discover", "explore"],
        "remember": ["recall", "note", "record"],
        "important": ["significant", "crucial", "priority"],
        "happy": ["glad", "pleased", "delighted", "joyful"],
        "sad": ["unhappy", "upset", "depressed", "melancholy"],
    }

    def decompose(self, query: str) -> list[str]:
        if not query or not query.strip():
            return []

        results = [query.strip()]

        for term, synonyms in self.SYNONYMS.items():
            if term.lower() in query.lower():
                for syn in synonyms[:2]:
                    variant = re.sub(
                        re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE),
                        syn,
                        query,
                        count=1,
                    )
                    if variant != query and variant.strip() not in results:
                        results.append(variant.strip())

        return results[:5]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    strategy: str
    sub_queries: list[str]
    original_query: str


@dataclass
class FusionResult:
    """A single fused result with its RRF score and source rankings."""
    memory: MemoryObject
    rrf_score: float
    source_ranks: dict[str, int]  # sub_query → rank in that result set


def decompose_query(
    query: str,
    strategy: str = "simple",
) -> DecomposedQuery:
    """Decompose a complex query into simpler sub-queries.

    Args:
        query: The original search query (may contain multiple aspects/conjunctions).
        strategy: Decomposition strategy. Options:
            - "simple": Split on conjunctions (and, or, but) and question words.
            - "expand": Generate synonym-expanded variants.
            - "both": Run both strategies and combine (deduplicated).
            - "none": Return the original query unchanged.

    Returns:
        A DecomposedQuery with the strategy used and the list of sub-queries.
    """
    if strategy == "none":
        return DecomposedQuery(strategy="none", sub_queries=[query.strip()], original_query=query)

    if strategy == "simple":
        strat = SimpleDecomposition()
    elif strategy == "expand":
        strat = SubqueryExpansion()
    elif strategy == "both":
        simple_strat = SimpleDecomposition()
        expand_strat = SubqueryExpansion()
        simple_queries = simple_strat.decompose(query)
        expand_queries = expand_strat.decompose(query)
        seen = set()
        combined: list[str] = []
        for q in simple_queries + expand_queries:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                combined.append(q)
        return DecomposedQuery(
            strategy="both",
            sub_queries=combined[:5],
            original_query=query,
        )
    else:
        raise ConfigurationError(
            f"Unknown decomposition strategy: {strategy!r}. "
            "Options: 'simple', 'expand', 'both', 'none'."
        )

    return DecomposedQuery(
        strategy=strategy,
        sub_queries=strat.decompose(query),
        original_query=query,
    )


def fused_recall(
    memory: Memory,
    user_id: str,
    sub_queries: list[str],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
    namespace: str = "default",
    session_id: str | None = None,
    lifecycle_filter: list[LifecycleState] | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[FusionResult]:
    """Execute multiple sub-queries and fuse results using RRF.

    Each sub-query is executed via :meth:`Memory.recall`. Results are ranked
    using Reciprocal Rank Fusion:

        RRF_score(d) = Σ 1 / (k + rank(d)_i)

    where k is a constant (default 60, as recommended by literature) and
    rank(d)_i is the rank of document d in the i-th result list.

    Args:
        memory: A Memory instance.
        user_id: User ID to recall memories for.
        sub_queries: List of sub-queries to execute.
        top_k: Number of results to retrieve per sub-query (before fusion).
        rrf_k: RRF constant; higher = more weight to lower-ranked results (default 60).
        namespace: Memory namespace.
        session_id: Optional session ID filter.
        lifecycle_filter: Optional lifecycle state filter.
        metadata_filter: Optional metadata filter.

    Returns:
        List of FusionResult objects, sorted by RRF score descending.
        Each contains the memory, its RRF score, and which ranks it appeared at
        in each sub-query result.
    """
    if not sub_queries:
        return []

    if len(sub_queries) == 1:
        results = memory.recall(
            user_id,
            sub_queries[0],
            top_k=top_k,
            namespace=namespace,
            session_id=session_id,
            lifecycle_filter=lifecycle_filter,
            metadata_filter=metadata_filter,
        )
        return [
            FusionResult(
                memory=r,
                rrf_score=1.0,
                source_ranks={sub_queries[0]: 0},
            )
            for r in results
        ]

    per_query_results: list[list[MemoryObject]] = []
    for sq in sub_queries:
        hits = memory.recall(
            user_id,
            sq,
            top_k=top_k,
            namespace=namespace,
            session_id=session_id,
            lifecycle_filter=lifecycle_filter,
            metadata_filter=metadata_filter,
        )
        per_query_results.append(hits)

    memory_ranks: dict[str, dict[int, int]] = {}
    memory_objects: dict[str, MemoryObject] = {}

    for sq_idx, results in enumerate(per_query_results):
        for rank, mem in enumerate(results):
            mem_id = mem.memory_id
            memory_objects[mem_id] = mem
            if mem_id not in memory_ranks:
                memory_ranks[mem_id] = {}
            memory_ranks[mem_id][sq_idx] = rank

    rrf_scores: dict[str, float] = {}
    for mem_id, ranks in memory_ranks.items():
        score = sum(1.0 / (rrf_k + rank) for rank in ranks.values())
        rrf_scores[mem_id] = score

    sorted_mem_ids = sorted(rrf_scores, key=lambda mid: rrf_scores[mid], reverse=True)

    fusion_results: list[FusionResult] = []
    for mem_id in sorted_mem_ids:
        mem = memory_objects[mem_id]
        ranks = memory_ranks[mem_id]
        source_ranks = {sub_queries[sq_idx]: rank for sq_idx, rank in ranks.items()}
        fusion_results.append(
            FusionResult(
                memory=mem,
                rrf_score=round(rrf_scores[mem_id], 4),
                source_ranks=source_ranks,
            )
        )

    return fusion_results


def rerank_with_reranker(
    memory: Memory,
    user_id: str,
    query: str,
    results: list[MemoryObject],
    *,
    provider: str = "cross-encoder",
    model: str | None = None,
) -> list[MemoryObject]:
    """Placeholder for future cross-encoder reranking.

    When a cross-encoder model is configured, this re-orders the results by
    scoring (query, document) pairs jointly rather than independently.

    Args:
        memory: Memory instance (used to access embed adapter).
        user_id: User ID (for context).
        query: The original query string.
        results: List of MemoryObjects from initial retrieval.
        provider: Cross-encoder provider (future: "cross-encoder", "bge-reranker").
        model: Model name override.

    Returns:
        Re-ranked list of MemoryObjects (same set, different order).
    """
    _ = memory, user_id, query, provider, model
    return results
