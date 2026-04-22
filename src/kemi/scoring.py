from datetime import datetime, timezone
from typing import Callable
import math

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from kemi.models import MemoryObject


def bm25_score(query: str, document: str) -> float:
    """Compute simple BM25-style keyword score.

    Uses term frequency approach without external libraries.
    Normalizes query and document to lowercase.
    Returns score between 0.0 and 1.0.

    Args:
        query: Search query string.
        document: Document to score against.

    Returns:
        BM25 score normalized to [0.0, 1.0] range.
    """
    if not query or not query.strip():
        return 0.0

    if not document or not document.strip():
        return 0.0

    query_terms = query.lower().split()
    doc_terms = document.lower().split()

    if not query_terms or not doc_terms:
        return 0.0

    doc_length = len(doc_terms)
    if doc_length == 0:
        return 0.0

    avg_doc_length = max(doc_length, 1)

    k1 = 1.5
    b = 0.75

    term_freqs = {}
    for term in doc_terms:
        term_freqs[term] = term_freqs.get(term, 0) + 1

    score = 0.0
    for query_term in query_terms:
        if query_term in term_freqs:
            tf = term_freqs[query_term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
            score += numerator / denominator

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

    if not query_terms or not doc_terms:
        return 0.0

    N = len(corpus)
    if N == 0:
        return 0.0

    doc_length = len(doc_terms)
    avgdl = sum(len(d.lower().split()) for d in corpus) / N

    if avgdl == 0:
        avgdl = 1.0

    df_counts = {}
    for doc in corpus:
        doc_words = set(doc.lower().split())
        for term in query_terms:
            if term in doc_words:
                df_counts[term] = df_counts.get(term, 0) + 1

    term_freqs = {}
    for term in doc_terms:
        term_freqs[term] = term_freqs.get(term, 0) + 1

    score = 0.0
    for query_term in query_terms:
        df = df_counts.get(query_term, 0)

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        tf = term_freqs.get(query_term, 0)

        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avgdl))

        score += idf * tf_norm

    return score


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is None or empty to avoid division by zero.
    Never returns NaN.
    """
    if a is None or b is None or not a or not b:
        return 0.0

    if _NUMPY_AVAILABLE:
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(np.dot(a_arr, b_arr) / norm) if norm != 0 else 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):
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

    return 2.0 ** (-hours_elapsed / half_life_hours)


def score_memory(
    memory: MemoryObject,
    query_embedding: list[float],
    query: str | None = None,
    hybrid_search: bool = True,
    corpus: list[str] | None = None,
) -> float:
    """Compute final relevance score for a memory.

    When hybrid_search=True:
        Formula: (semantic × 0.6) + (recency × 0.25) + (bm25 × 0.15)

    When hybrid_search=False:
        Formula: (semantic × 0.5) + (recency × 0.3) + (importance × 0.2)

    If memory.embedding is None or query_embedding is empty, semantic contribution is 0.0.
    """
    semantic_score = 0.0
    if memory.embedding is not None and query_embedding is not None:
        similarity = cosine_similarity(memory.embedding, query_embedding)
        semantic_score = (similarity + 1.0) / 2.0

    recency_score = temporal_recency(memory.last_accessed_at)

    if hybrid_search and query:
        if corpus and len(corpus) > 1:
            bm25_keyword_score = bm25_score_corpus(query, memory.content, corpus)
        else:
            bm25_keyword_score = bm25_score(query, memory.content)

        final_score = semantic_score * 0.6 + recency_score * 0.25 + bm25_keyword_score * 0.15
    else:
        importance_score = max(0.0, min(1.0, memory.importance))
        final_score = semantic_score * 0.5 + recency_score * 0.3 + importance_score * 0.2

    return final_score


def rank_memories(
    memories: list[MemoryObject],
    query_embedding: list[float],
    query: str | None = None,
    hybrid_search: bool = True,
) -> list[MemoryObject]:
    """Rank memories by computed score, highest first.

    Mutates the score field on each MemoryObject in place.
    Returns the sorted list.

    Args:
        memories: List of MemoryObjects to rank.
        query_embedding: Embedding vector for semantic search.
        query: Optional query string for keyword search.
        hybrid_search: Use hybrid scoring (default True).
    """
    corpus = [m.content for m in memories] if len(memories) > 1 else None

    for memory in memories:
        memory.score = score_memory(memory, query_embedding, query, hybrid_search, corpus)

    return sorted(memories, key=lambda m: m.score, reverse=True)


def mmr_rerank(
    memories: list[MemoryObject],
    query_embedding: list[float],
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
    if top_k <= 0 or not memories:
        return []

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
            break

        selected.append(candidates.pop(best_idx))

    return selected


def _default_token_counter(text: str) -> int:
    """Default token counter: rough estimate = word_count * 1.3"""
    return int(len(text.split()) * 1.3)


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
    result = []
    total_tokens = 0

    for memory in memories:
        memory_tokens = counter(memory.content)

        if result and total_tokens + memory_tokens > max_tokens:
            break

        result.append(memory)
        total_tokens += memory_tokens

    if not result and memories:
        result = [memories[0]]

    return result
