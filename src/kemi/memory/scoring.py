import math
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:  # pragma: no cover
    import numpy as np
    _numpy_available = True
except ImportError:
    np = None  # type: ignore[assignment]
    _numpy_available = False  # pragma: no cover

from kemi.memory.model import MemoryObject

if TYPE_CHECKING:
    from kemi.memory.model import MemoryConfig


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration for :func:`score_memory` and :func:`rank_memories`.

    Encapsulates the scoring weights and behaviour flags so callers
    don't have to pass 8+ individual keyword arguments.
    """

    hybrid_search: bool = True
    weight_semantic: float = 0.6
    weight_recency: float = 0.25
    weight_bm25: float = 0.15
    weight_semantic_no_embed: float = 0.5
    weight_recency_no_embed: float = 0.3
    weight_importance: float = 0.2
    weight_entity: float = 0.1

    @classmethod
    def from_memory_config(
        cls, config: "MemoryConfig", *, hybrid_search: bool | None = None
    ) -> "ScoreConfig":
        """Build a :class:`ScoreConfig` from a :class:`MemoryConfig`.

        Args:
            config: The source memory configuration.
            hybrid_search: Override the hybrid-search flag.  If ``None``,
                ``config.hybrid_search`` is used.

        Returns:
            A fully populated :class:`ScoreConfig`.
        """
        return cls(
            hybrid_search=hybrid_search if hybrid_search is not None else config.hybrid_search,
            weight_semantic=config.weight_semantic,
            weight_recency=config.weight_recency,
            weight_bm25=config.weight_bm25,
            weight_semantic_no_embed=config.weight_semantic_no_embed,
            weight_recency_no_embed=config.weight_recency_no_embed,
            weight_importance=config.weight_importance,
            weight_entity=config.entity_boost_weight,
        )


def _bm25_tf_component(
    query_terms: list[str],
    doc_terms: list[str],
    doc_length: int,
    avg_doc_length: float,
    k1: float,
    b: float,
) -> dict[str, float]:
    """Build term frequencies and return the normalized BM25 TF component per query term.

    For each query term that appears in the document, computes:
        tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))

    Returns a dict mapping query_term -> normalized_tf_score.
    """
    term_freqs: dict[str, int] = {}
    for term in doc_terms:
        term_freqs[term] = term_freqs.get(term, 0) + 1

    result: dict[str, float] = {}
    for query_term in query_terms:
        tf = term_freqs.get(query_term, 0)
        denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
        if denominator != 0:
            result[query_term] = tf * (k1 + 1) / denominator
    return result


def bm25_score(query: str, document: str) -> float:
    """Compute a BM25-inspired keyword score using term frequency only.

    This is a simplified TF-only scoring function. It does **not**
    include the IDF (Inverse Document Frequency) component of true
    BM25, so scores are less discriminative across a large corpus.
    For full BM25 with IDF, use :func:`bm25_score_corpus` when a
    document corpus is available.

    Normalizes query and document to lowercase.
    Returns score between 0.0 and 1.0.

    Args:
        query: Search query string.
        document: Document to score against.

    Returns:
        Normalized TF score in [0.0, 1.0] range.
    """
    if not query or not query.strip():
        return 0.0

    if not document or not document.strip():
        return 0.0

    query_terms = query.lower().split()
    doc_terms = document.lower().split()

    if not query_terms or not doc_terms:  # pragma: no cover (unreachable)
        return 0.0

    doc_length = len(doc_terms)
    if doc_length == 0:  # pragma: no cover (unreachable)
        return 0.0

    avg_doc_length = max(doc_length, 1)
    k1 = 1.5
    b = 0.75

    tf_components = _bm25_tf_component(query_terms, doc_terms, doc_length, avg_doc_length, k1, b)
    score = sum(tf_components.values())

    max_score = len(query_terms) * (k1 + 1) / k1
    if max_score > 0:
        score = min(1.0, score / max_score)

    return score


def bm25_score_corpus(
    query: str,
    document: str,
    corpus: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score with IDF from a corpus.

    Uses Inverse Document Frequency to weight terms based on how rare they are
    across the corpus.

    Args:
        query: Search query string.
        document: Document to score against.
        corpus: List of document strings to compute IDF from.
        k1: Term frequency saturation parameter.
        b: Document length normalization parameter.

    Returns:
        BM25 score as float.
    """
    if not query or not query.strip():
        return 0.0

    if not document or not document.strip():
        return 0.0

    if not corpus:
        return bm25_score(query, document)

    query_terms = query.lower().split()
    doc_terms = document.lower().split()

    if not query_terms or not doc_terms:  # pragma: no cover (unreachable)
        return 0.0

    n_docs = len(corpus)
    if n_docs == 0:  # pragma: no cover (unreachable)
        return 0.0

    doc_length = len(doc_terms)
    avgdl: float = sum(len(d.lower().split()) for d in corpus) / n_docs

    if avgdl == 0:
        avgdl = 1.0

    df_counts: dict[str, int] = {}
    for doc in corpus:
        doc_words = set(doc.lower().split())
        for term in query_terms:
            if term in doc_words:
                df_counts[term] = df_counts.get(term, 0) + 1

    tf_components = _bm25_tf_component(query_terms, doc_terms, doc_length, avgdl, k1, b)

    score = 0.0
    for query_term in query_terms:
        df = df_counts.get(query_term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        score += idf * tf_components.get(query_term, 0.0)

    return score


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Compute cosine similarity between two vectors.

    Handles dimension mismatches by computing over the minimum dimension.
    Returns 0.0 if either vector is None or empty to avoid division by zero.
    Never returns NaN.
    """
    if a is None or b is None or not a or not b:
        return 0.0

    # Handle dimension mismatch gracefully — truncate to min dim so numpy
    # doesn't raise ValueError on mismatched vector lengths.
    min_dim = min(len(a), len(b))

    if np is not None:  # pragma: no cover
        a_arr = np.array(a[:min_dim])
        b_arr = np.array(b[:min_dim])
        norm: float = float(np.linalg.norm(a_arr)) * float(np.linalg.norm(b_arr))
        dot: float = float(np.dot(a_arr, b_arr))
        return dot / norm if norm != 0 else 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(min_dim):
        dot_product += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    norm_a = norm_a**0.5
    norm_b = norm_b**0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def temporal_recency(last_accessed: datetime, half_life_hours: float = 168.0) -> float:
    """Compute temporal recency score using exponential decay.

    A memory accessed now scores 1.0.
    A memory accessed half_life_hours ago scores 0.5.
    A memory accessed 2x half_life_hours ago scores 0.25.

    Default half_life is 168 hours (7 days).
    """
    now = datetime.now(timezone.utc)
    hours_elapsed = (now - last_accessed).total_seconds() / 3600.0

    if hours_elapsed <= 0:
        return 1.0

    return 2.0 ** (-hours_elapsed / half_life_hours)  # type: ignore[no-any-return]


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets of strings.

    Returns 0.0 if either set is empty.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def score_memory(
    memory: MemoryObject,
    query_embedding: list[float] | None,
    query: str | None = None,
    corpus: list[str] | None = None,
    query_entities: set[str] | None = None,
    memory_entities: set[str] | None = None,
    config: ScoreConfig | None = None,
) -> float:
    """Compute final relevance score for a memory.

    When ``config.hybrid_search`` is True and *query* is provided:
        Formula: (semantic × weight_semantic) + (recency × weight_recency) + (bm25 × weight_bm25)

    When ``config.hybrid_search`` is False or *query* is absent:
        Formula: (semantic × weight_semantic_no_embed)
                 + (recency × weight_recency_no_embed)
                 + (importance × weight_importance)

    If entity parameters are provided, an additional boost is applied:
        + (jaccard(query_entities, memory_entities) × weight_entity)

    If ``memory.embedding`` is None or ``query_embedding`` is empty, semantic
    contribution is 0.0.

    Args:
        memory: The memory object to score.
        query_embedding: Embedding vector for semantic search.
        query: Optional query string for keyword search.
        corpus: List of document strings to compute IDF from for BM25.
        query_entities: Optional set of entities extracted from the query.
        memory_entities: Optional set of entities extracted from the memory content.
        config: Scoring weights and behaviour.  Defaults to :class:`ScoreConfig`.
    """
    cfg = config or ScoreConfig()

    semantic_score = 0.0
    if memory.embedding is not None and query_embedding is not None:
        similarity = cosine_similarity(memory.embedding, query_embedding)
        semantic_score = (similarity + 1.0) / 2.0

    recency_score = temporal_recency(memory.last_accessed_at)

    if cfg.hybrid_search and query:
        if corpus and len(corpus) > 1:
            bm25_keyword_score = bm25_score_corpus(query, memory.content, corpus)
        else:
            bm25_keyword_score = bm25_score(query, memory.content)

        final_score = (
            semantic_score * cfg.weight_semantic
            + recency_score * cfg.weight_recency
            + bm25_keyword_score * cfg.weight_bm25
        )
    else:
        importance_score = max(0.0, min(1.0, memory.importance))
        final_score = (
            semantic_score * cfg.weight_semantic_no_embed
            + recency_score * cfg.weight_recency_no_embed
            + importance_score * cfg.weight_importance
        )

    # Entity-aware boost
    if query_entities is not None and memory_entities is not None:
        entity_score = jaccard_similarity(query_entities, memory_entities)
        final_score += entity_score * cfg.weight_entity

    return final_score


def rank_memories(
    memories: list[MemoryObject],
    query_embedding: list[float] | None,
    query: str | None = None,
    corpus: list[str] | None = None,
    query_entities: set[str] | None = None,
    memory_entities_map: dict[str, set[str]] | None = None,
    config: ScoreConfig | None = None,
) -> list[MemoryObject]:
    """Rank memories by computed score, highest first.

    Mutates the score field on each MemoryObject in place.
    Returns the sorted list.

    Args:
        memories: List of MemoryObjects to rank.
        query_embedding: Embedding vector for semantic search.
        query: Optional query string for keyword search.
        corpus: List of document strings to compute IDF from for BM25.
        query_entities: Optional set of entities extracted from the query.
        memory_entities_map: Optional dict mapping memory_id -> set of entities.
        config: Scoring weights and behaviour.  Defaults to :class:`ScoreConfig`.
    """
    if corpus is None:
        corpus = [m.content for m in memories] if len(memories) > 1 else None

    for memory in memories:
        mem_entities = None
        if memory_entities_map is not None:
            mem_entities = memory_entities_map.get(memory.memory_id)
        memory.score = score_memory(
            memory,
            query_embedding,
            query,
            corpus,
            query_entities,
            mem_entities,
            config,
        )

    return sorted(memories, key=lambda m: m.score, reverse=True)


def _mmr_generator(
    memories: list[MemoryObject],
    query_embedding: list[float] | None,
    top_k: int,
    lambda_param: float,
) -> Generator[MemoryObject, None, None]:
    """Core MMR selection loop (generator).

    Shared implementation for :func:`mmr_rerank` and
    :func:`mmr_rerank_stream`.  Yields each selected memory in the
    order MMR picks them.
    """
    if top_k <= 0 or not memories:
        return

    candidates = list(memories)
    selected: list[MemoryObject] = []

    while len(selected) < top_k and candidates:
        best_idx = -1
        best_mmr = float("-inf")

        for i, candidate in enumerate(candidates):
            relevance = candidate.score

            if candidate.embedding is not None and query_embedding:
                max_sim_to_selected = 0.0
                for sel in selected:
                    if sel.embedding is not None:
                        sim = cosine_similarity(candidate.embedding, sel.embedding)
                        max_sim_to_selected = max(max_sim_to_selected, sim)
            else:
                max_sim_to_selected = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx == -1:
            return

        memory = candidates.pop(best_idx)
        selected.append(memory)
        yield memory


def mmr_rerank(
    memories: list[MemoryObject],
    query_embedding: list[float] | None,
    top_k: int,
    lambda_param: float = 0.7,
) -> list[MemoryObject]:
    """Rerank memories using Maximal Marginal Relevance.

    Balances relevance (similarity to query) with diversity
    (dissimilarity to already selected memories).

    lambda_param controls the tradeoff:
      1.0 = pure relevance (same as no MMR)
      0.0 = pure diversity
      0.7 = default, slightly favors relevance

    Algorithm:
    - Start with empty selected list
    - At each step, pick the candidate that maximizes:
        lambda * relevance_score - (1 - lambda) * max_similarity_to_selected
      where relevance_score = memory.score (already computed)
      and max_similarity_to_selected = max cosine_similarity between
      candidate embedding and each already-selected memory embedding
    - Skip candidates with no embedding (embedding is None)
      by treating their relevance as memory.score and similarity as 0.0
    - Stop when top_k memories are selected or candidates exhausted
    - Return selected list in order selected
    """
    return list(_mmr_generator(memories, query_embedding, top_k, lambda_param))


def _default_token_counter(text: str) -> int:
    """Default token counter: rough estimate = word_count * 1.3"""
    result: float = len(text.split()) * 1.3
    return int(result)


def mmr_rerank_stream(
    memories: list[MemoryObject],
    query_embedding: list[float] | None,
    top_k: int,
    lambda_param: float = 0.7,
) -> Generator[MemoryObject, None, None]:
    """Yield memories one at a time as MMR selects them.

    Same algorithm as :func:`mmr_rerank` but yields each selected memory
    immediately rather than collecting them into a list.

    Yields:
        MemoryObject, each selected by the MMR criterion.
    """
    yield from _mmr_generator(memories, query_embedding, top_k, lambda_param)


def truncate_by_tokens(
    memories: list[MemoryObject],
    max_tokens: int | None,
    token_counter: Callable[[str], int] | None = None,
) -> list[MemoryObject]:
    """Truncate memories by token budget.

    Walks ranked list, sums token counts, stops when budget reached.
    If max_tokens is None, returns all memories.
    If a single memory exceeds budget, includes it anyway.
    Never returns an empty list (if any input, returns at least one).
    """
    if max_tokens is None:
        return memories

    if not memories:
        return memories

    counter = token_counter or _default_token_counter
    result: list[MemoryObject] = []
    total_tokens = 0

    for memory in memories:
        memory_tokens = counter(memory.content)

        if result and total_tokens + memory_tokens > max_tokens:
            break

        result.append(memory)
        total_tokens += memory_tokens

    if not result and memories:  # pragma: no cover (unreachable)
        result = [memories[0]]

    return result
