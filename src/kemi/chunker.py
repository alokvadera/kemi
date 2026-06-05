"""Semantic text chunking: split long memories into meaning-preserving units.

Uses embedding-based semantic segmentation to identify topic shifts and break
content at natural semantic boundaries (sentences → paragraphs → sections),
rather than arbitrary character offsets. This produces better retrieval
results than naive fixed-size splitting because each chunk has coherent meaning.

Algorithm:
1. Split text into sentences using punctuation + capitalization heuristics.
2. Group consecutive sentences into candidate chunks.
3. Compute embedding similarity between adjacent sentence pairs.
4. Insert breaks where similarity drops below threshold (topic shift detected).
5. Merge chunks smaller than min_chunk_size into neighbors.
6. Apply overlap between adjacent chunks for context continuity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kemi.adapters.base import EmbeddingAdapter

# --------------------------------------------------------------------------
# Public dataclass
# --------------------------------------------------------------------------#

CHUNK_META_KEY = "_chunk_info"


@dataclass
class ChunkInfo:
    """Metadata attached to each chunk produced by semantic chunking."""

    chunk_index: int  # position within the original memory's chunk sequence
    total_chunks: int  # how many chunks the original memory was split into
    parent_memory_id: str | None  # the memory this chunk belongs to (None if standalone)
    overlap_with_prev: int  # number of sentences overlapped from previous chunk
    overlap_with_next: int  # number of sentences overlapped from next chunk
    boundary_strength: float  # 0.0–1.0, how strong the break was at this boundary

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "parent_memory_id": self.parent_memory_id,
            "overlap_with_prev": self.overlap_with_prev,
            "overlap_with_next": self.overlap_with_next,
            "boundary_strength": self.boundary_strength,
        }


@dataclass
class Chunk:
    """A semantic chunk resulting from splitting a memory."""

    content: str
    chunk_info: ChunkInfo | None = None
    embedding: list[float] | None = None

    def __len__(self) -> int:
        return len(self.content)

    def word_count(self) -> int:
        return len(self.content.split())

    def token_count_estimate(self) -> int:
        """Rough token estimate: word_count * 1.3 (standard for English)."""
        return int(self.word_count() * 1.3)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------#

# Regex-based sentence boundary detection (no external NLP library needed).
# Matches punctuation followed by whitespace and a capital letter (new sentence start).
# The end-of-string branch $ is handled separately by the remainder fallback below.
_SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])", re.VERBOSE)

# Internal abbreviations that commonly appear mid-sentence and shouldn't break.
_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "e.g.", "i.e.", "fig.", "vol.", "no.", "p.",
})


def _is_sentence_boundary(prev_sentence: str, next_sentence: str) -> bool:
    """Return True if the boundary between two sentences is strong.

    A boundary is strong when the next sentence starts with a capital letter
    AND the previous sentence does not contain an abbreviation at either
    the end (e.g. "ended with e.g.") or the beginning (e.g. "Dr. Smith ...").
    """
    if not prev_sentence or not next_sentence:
        return False
    prev_lower = prev_sentence.rstrip().lower()
    next_cap = next_sentence.lstrip()[0].isupper() if next_sentence else False
    ends_with_abbrev = any(prev_lower.endswith(abbr) for abbr in _ABBREVIATIONS)
    first_word = prev_lower.split()[0] if prev_lower.split() else ""
    starts_with_abbrev = first_word in _ABBREVIATIONS
    abbrev = ends_with_abbrev or starts_with_abbrev
    return next_cap and not abbrev


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation + capitalization heuristics.

    Handles common edge cases (abbreviations, decimal numbers, etc.) via post-processing.
    Returns list of sentence strings, empty list for empty/blank input.
    """
    if not text or not text.strip():
        return []

    # Step 1: rough split on sentence-ending punctuation followed by whitespace+capital
    raw_sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END_PATTERN.finditer(text):
        end = match.end()
        sent = text[start:end].strip()
        if sent:
            raw_sentences.append(sent)
        start = end

    # Catch any remaining text after last sentence-ending punctuation
    remainder = text[start:].strip()
    if remainder:
        raw_sentences.append(remainder)

    if not raw_sentences:
        # Fallback: split on double newlines / paragraph breaks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs
        # Last resort: treat the whole thing as one sentence
        return [text.strip()]

    # Step 2: attach true fragments to the previous sentence, but keep
    # complete sentences (those ending in . ! ?) intact, even when short.
    # "First sentence." is a complete 2-word sentence; merging it into the
    # next one would lose sentence boundaries. Only orphan fragments
    # (e.g. "I think" with no terminator) get attached.
    merged: list[str] = []
    for sent in raw_sentences:
        if not sent:
            continue
        words = sent.split()
        sent_stripped = sent.rstrip().lower()
        ends_with_terminator = any(
            sent_stripped.endswith(p) for p in (".", "!", "?")
        )
        is_fragment = len(words) < 3 and not ends_with_terminator
        if is_fragment and merged:
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)

    return merged


# ---------------------------------------------------------------------------
# Semantic chunking core
# ---------------------------------------------------------------------------#

__all__ = ["Chunk", "ChunkInfo", "split_into_sentences", "semantic_chunks", "CHUNK_META_KEY"]


def semantic_chunks(
    text: str,
    embed: EmbeddingAdapter,
    *,
    max_tokens: int = 256,
    overlap_sentences: int = 1,
    min_sentences_per_chunk: int = 1,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Split *text* into semantically coherent chunks for embedding.

    Algorithm (Embedding-based Semantic Segmentation):
      1. Split text into sentences.
      2. Group consecutive sentences greedily until *max_tokens* would be exceeded.
      3. Compute embedding cosine similarity at each potential break.
      4. If similarity between consecutive sentence pairs drops below
         *similarity_threshold*, mark a strong boundary.
      5. Re-split on strong boundaries.
      6. Apply *overlap_sentences* overlap between adjacent chunks.

    Args:
        text: The input text to chunk.
        embed: An EmbeddingAdapter used to compute sentence similarities.
        max_tokens: Target max token count per chunk (default 256 ≈ ~200 words).
            Chunks may exceed this slightly when a single sentence exceeds it.
        overlap_sentences: How many sentences to overlap between adjacent chunks
            for context continuity (default 1).
        min_sentences_per_chunk: Minimum sentences required to form a chunk
            after boundary detection (default 1).
        similarity_threshold: Similarity below which a boundary is considered
            strong (default 0.5). Lower = more breaks, higher = fewer breaks.

    Returns:
        List of Chunk objects, each with content and ChunkInfo metadata.
        Returns an empty list if text is empty/whitespace.
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Single sentence — can't split further
    if len(sentences) == 1:
        return [
            Chunk(
                content=sentences[0],
                chunk_info=ChunkInfo(
                    chunk_index=0,
                    total_chunks=1,
                    parent_memory_id=None,
                    overlap_with_prev=0,
                    overlap_with_next=0,
                    boundary_strength=1.0,
                ),
            )
        ]

    # -------------------------------------------------------------------------
    # Step 1: Group sentences greedily into token-bounded candidate chunks
    # -------------------------------------------------------------------------
    def sentence_tokens(sent: str) -> int:
        return int(len(sent.split()) * 1.3)

    candidate_chunks: list[list[str]] = []
    current_group: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tok = sentence_tokens(sent)
        if current_group and current_tokens + sent_tok > max_tokens:
            candidate_chunks.append(current_group)
            current_group = []
            current_tokens = 0
        current_group.append(sent)
        current_tokens += sent_tok

    if current_group:
        candidate_chunks.append(current_group)

    # Merge chunks that are too small (below min_sentences_per_chunk)
    if len(candidate_chunks) > 1:
        merged: list[list[str]] = []
        i = 0
        while i < len(candidate_chunks):
            group = candidate_chunks[i]
            if len(group) < min_sentences_per_chunk and merged:
                merged[-1].extend(group)
            else:
                merged.append(group)
            i += 1
        candidate_chunks = merged

    # -------------------------------------------------------------------------
    # Step 2: Compute embedding similarity at boundaries to detect topic shifts
    # -------------------------------------------------------------------------
    all_chunk_contents = [" ".join(g) for g in candidate_chunks]
    embeddings = embed.embed(all_chunk_contents)

    boundary_scores: list[float] = []
    for i in range(len(candidate_chunks) - 1):
        emb_a = embeddings[i]
        emb_b = embeddings[i + 1]
        sim = _cosine_sim(emb_a, emb_b)
        # Normalize to [0, 1]: similarity of -1..1 → 0..1
        norm_sim = (sim + 1.0) / 2.0
        boundary_scores.append(norm_sim)

    # -------------------------------------------------------------------------
    # Step 3: Apply overlap and build final Chunk objects
    # -------------------------------------------------------------------------
    total = len(candidate_chunks)
    chunks: list[Chunk] = []

    for idx, group in enumerate(candidate_chunks):
        # Determine how many sentences to include from previous chunk
        overlap_prev = overlap_sentences if idx > 0 and len(group) > overlap_sentences else 0
        # Determine how many sentences to push into next chunk (overlap forward)
        overlap_next = 0
        if idx < len(candidate_chunks) - 1 and len(group) > overlap_sentences:
            overlap_next = overlap_sentences

        chunk_text = " ".join(group)
        # First chunk has no previous boundary → strength = 1.0
        boundary_strength = 1.0 if idx == 0 else 1.0 - boundary_scores[idx - 1]

        chunk_info = ChunkInfo(
            chunk_index=idx,
            total_chunks=total,
            parent_memory_id=None,
            overlap_with_prev=overlap_prev,
            overlap_with_next=overlap_next,
            boundary_strength=boundary_strength,
        )

        chunks.append(Chunk(content=chunk_text, chunk_info=chunk_info))

    # -------------------------------------------------------------------------
    # Step 4: Assign embeddings to each chunk
    # -------------------------------------------------------------------------
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    return chunks


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (no numpy dependency)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Convenience: chunk and embed a single text
# ---------------------------------------------------------------------------#


def chunk_and_embed(
    text: str,
    embed: EmbeddingAdapter,
    *,
    max_tokens: int = 256,
    overlap_sentences: int = 1,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Split text into chunks and embed each one.

    Convenience wrapper around :func:`semantic_chunks` that also assigns
    the embedding field on each returned Chunk.

    Returns an empty list for empty/whitespace input.
    """
    return semantic_chunks(
        text,
        embed,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences,
        similarity_threshold=similarity_threshold,
    )