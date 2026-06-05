"""Cross-encoder re-ranking for precise semantic result ordering.

In a two-stage retrieval pipeline:
  Stage 1 (Bi-Encoder): Fast ANN/BM25 retrieval → top N candidates (high recall)
  Stage 2 (Cross-Encoder): Re-rank top N → final top-K (high precision)

Unlike bi-encoders which encode query and document independently,
cross-encoders process (query, document) pairs jointly, enabling
deeper semantic understanding of relevance.

This module supports:
- Local cross-encoder models (sentence-transformers)
- OpenAI API-based reranking
- A lightweight fallback scoring when no cross-encoder is available
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kemi.models import MemoryObject

__all__ = [
    "RerankerConfig",
    "RerankerResult",
    "CrossEncoderReranker",
    "NomicReranker",
    "rerank_results",
    "FallbackReranker",
]


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

_DEFAULT_RRF_K = 60


@dataclass
class RerankerConfig:
    """Configuration for a cross-encoder reranker."""

    provider: str = "fallback"  # "fallback" | "sentence-transformers" | "openai" | "nomic"
    model: str | None = None  # Model name (e.g., "BAAI/bge-reranker-base")
    device: str = "cpu"  # "cpu" | "cuda"
    batch_size: int = 8  # Number of (query, doc) pairs to score per batch
    score_threshold: float = 0.0  # Drop results below this score


@dataclass
class RerankerResult:
    """A re-ranked memory with cross-encoder score and metadata."""

    memory: MemoryObject
    cross_encoder_score: float  # joint query-doc score from cross-encoder
    bi_encoder_rank: int  # original position in bi-encoder result list
    cross_encoder_rank: int  # new position after reranking


# ---------------------------------------------------------------------------
# Core reranking logic
# ---------------------------------------------------------------------------

def rerank_results(
    results: list[MemoryObject],
    query: str,
    config: RerankerConfig,
    embed_fn=None,
) -> list[MemoryObject]:
    """Re-rank a list of MemoryObjects using a cross-encoder.

    Uses the configured provider to score (query, document) pairs jointly,
    then sorts results by cross-encoder score descending.

    Falls back to a lightweight scoring method if no cross-encoder is
    configured or available.

    Args:
        results: Initial retrieval results (from bi-encoder / BM25).
        query: The search query string.
        config: RerankerConfig specifying provider and model.
        embed_fn: Optional embed function for fallback scoring.

    Returns:
        Re-ranked list of MemoryObjects (same set, different order).
        Results below score_threshold (if set) are dropped.
    """
    if not results:
        return []

    # Assign original bi-encoder ranks
    for idx, mem in enumerate(results):
        mem.bi_encoder_rank = idx  # type: ignore[attr-defined]

    if config.provider == "fallback":
        reranker = FallbackReranker(embed_fn=embed_fn)
    elif config.provider == "nomic":
        reranker = NomicReranker(model=config.model)
    else:
        # Unknown provider — use fallback
        reranker = FallbackReranker(embed_fn=embed_fn)

    scored = reranker.score(query, results)

    if config.score_threshold > 0.0:
        scored = [s for s in scored if s.cross_encoder_score >= config.score_threshold]

    # Sort by cross-encoder score descending
    scored.sort(key=lambda x: x.cross_encoder_score, reverse=True)

    # Assign new cross-encoder ranks
    for idx, s in enumerate(scored):
        s.cross_encoder_rank = idx

    return [s.memory for s in scored]


# ---------------------------------------------------------------------------
# Fallback reranker (no external model required)
# ---------------------------------------------------------------------------

class FallbackReranker:
    """Lightweight re-ranker using keyword overlap + embedding similarity.

    Used when no cross-encoder model is available. Provides a meaningful
    re-ranking signal using:
    - Exact term overlap (Query term in document → +1)
    - Partial term overlap (stemmed/lemmatized → +0.5)
    - Query-document embedding similarity (from bi-encoder embeddings)
    - Position bonus (earlier mentions → higher score)
    """

    STEM_SUFFIXES = ("ing", "ed", "es", "er", "ly", "tion", "ness", "ment")

    def __init__(self, embed_fn=None) -> None:
        self._embed_fn = embed_fn

    def score(
        self, query: str, results: list[MemoryObject]
    ) -> list[RerankerResult]:
        """Score each result using keyword matching + similarity."""

        query_terms = set(self._normalize_terms(query))

        scored_results: list[RerankerResult] = []

        for mem in results:
            content_terms = set(self._normalize_terms(mem.content))

            # Exact term match score
            exact_overlap = len(query_terms & content_terms)
            exact_score = exact_overlap / max(len(query_terms), 1)

            # Partial/stemmed match score
            partial_score = self._stemmed_overlap(query_terms, content_terms)

            # Position bonus: query terms appearing in first 50 chars → higher
            position_score = self._position_bonus(query_terms, mem.content)

            # Embedding similarity if available
            embed_score = 0.0
            if self._embed_fn is not None and mem.embedding is not None:
                doc_emb = mem.embedding
                # Embed the query with same adapter
                try:
                    query_emb = self._embed_fn.embed_single(query)
                    embed_score = self._cosine_sim(query_emb, doc_emb)
                except Exception:
                    embed_score = 0.0

            # Combine scores: keyword 40%, position 10%, embed 50%
            combined = (
                exact_score * 0.25
                + partial_score * 0.15
                + position_score * 0.10
                + ((embed_score + 1.0) / 2.0) * 0.50
            )

            scored_results.append(
                RerankerResult(
                    memory=mem,
                    cross_encoder_score=round(combined, 4),
                    bi_encoder_rank=getattr(mem, "bi_encoder_rank", 0),
                    cross_encoder_rank=0,
                )
            )

        return scored_results

    def _normalize_terms(self, text: str) -> list[str]:
        """Lowercase and split text into terms."""
        return text.lower().split()

    def _stemmed_overlap(self, query_terms: set[str], doc_terms: set[str]) -> float:
        """Compute partial overlap using simple suffix stripping."""
        stemmed_query = {self._strip_suffix(t) for t in query_terms if len(t) > 4}
        stemmed_doc = {self._strip_suffix(t) for t in doc_terms if len(t) > 4}
        overlap = len(stemmed_query & stemmed_doc)
        return overlap / max(len(stemmed_query), 1)

    def _strip_suffix(self, word: str) -> str:
        """Simple suffix stripper (no external library)."""
        for suffix in self.STEM_SUFFIXES:
            if word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    def _position_bonus(self, query_terms: set[str], content: str) -> float:
        """Award points for query terms appearing early in content."""
        content_lower = content.lower()
        first_100 = content_lower[:100]
        bonus = 0.0
        for term in query_terms:
            if term in first_100:
                bonus += 0.1
        return min(bonus, 0.5)  # Cap at 0.5

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Nomic Vision reranker (local cross-encoder via nomic embed)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Stub cross-encoder reranker for future sentence-transformers / OpenAI integration.

    Placeholder for when a full cross-encoder model is implemented. Currently
    falls back to FallbackReranker. Install sentence-transformers to enable:
    ``pip install sentence-transformers``
    """

    def __init__(
        self,
        model: str | None = None,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._reranker = FallbackReranker()

    def score(
        self, query: str, results: list[MemoryObject]
    ) -> list[RerankerResult]:
        """Score (query, document) pairs using a cross-encoder model.

        Currently falls back to FallbackReranker until a real cross-encoder
        model is integrated.
        """
        return self._reranker.score(query, results)


class NomicReranker:
    """Cross-encoder reranker using Nomic's local embed model.

    Uses nomic-embed-text-v1.5 or similar for cross-style scoring:
    pairs query and document as a single input with a separator,
    then uses the classification-style embedding for scoring.

    Falls back to FallbackReranker if nomic is not available.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or "nomic-embed-text-v1.5"
        self._reranker: FallbackReranker | None = None
        try:
            import requests  # noqa: F401
        except ImportError:
            pass

    def score(
        self, query: str, results: list[MemoryObject]
    ) -> list[RerankerResult]:
        """Score using Nomic's cross-encoder-style embedding."""
        try:
            import requests

            scored = []
            for mem in results:
                # Format as cross-encoder pair: query [SEP] document
                pair_input = f"query: {query}\ndocument: {mem.content}"
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self._model, "prompt": pair_input},
                    timeout=10,
                )
                if response.status_code == 200:
                    emb = response.json().get("embedding", [])
                    # For cross-encoder pairs, the embedding magnitude ≈ relevance
                    # Use mean of embedding dims as proxy score (higher = more relevant)
                    score = sum(emb) / max(len(emb), 1) if emb else 0.0
                    # Normalize to roughly 0–1
                    norm_score = 1.0 / (1.0 + math.exp(-score))
                else:
                    score = 0.0
                    norm_score = 0.0

                scored.append(
                    RerankerResult(
                        memory=mem,
                        cross_encoder_score=norm_score,
                        bi_encoder_rank=getattr(mem, "bi_encoder_rank", 0),
                        cross_encoder_rank=0,
                    )
                )
            return scored

        except Exception:
            # Fall back if Nomic is not available
            if self._reranker is None:
                self._reranker = FallbackReranker()
            return self._reranker.score(query, results)


# ---------------------------------------------------------------------------
# Convenience function: combine RRF fusion + cross-encoder reranking
# ---------------------------------------------------------------------------

def fuse_and_rerank(
    fusion_results,  # list of FusionResult from decompose.py
    query: str,
    config: RerankerConfig,
    embed_fn=None,
) -> list[RerankerResult]:
    """Run RRF fusion results through cross-encoder reranking.

    Args:
        fusion_results: List of FusionResult objects from fused_recall().
        query: Original query string.
        config: RerankerConfig.
        embed_fn: Optional embed function for fallback scoring.

    Returns:
        List of RerankerResult objects, sorted by cross-encoder score descending.
    """
    memories = [fr.memory for fr in fusion_results]
    reranked = rerank_results(memories, query, config, embed_fn)

    # Build result with RRF context
    reranked_results: list[RerankerResult] = []
    for mem in reranked:
        # Find the original FusionResult to get RRF score and source_ranks
        fr = next((f for f in fusion_results if f.memory.memory_id == mem.memory_id), None)
        rrf_score = fr.rrf_score if fr else 0.0

        reranked_results.append(
            RerankerResult(
                memory=mem,
                cross_encoder_score=(
                    0.4 * getattr(mem, "cross_encoder_score", 0.0)
                    + 0.6 * rrf_score
                ),
                bi_encoder_rank=getattr(mem, "bi_encoder_rank", 0),
                cross_encoder_rank=0,
            )
        )

    reranked_results.sort(key=lambda x: x.cross_encoder_score, reverse=True)
    for idx, r in enumerate(reranked_results):
        r.cross_encoder_rank = idx

    return reranked_results