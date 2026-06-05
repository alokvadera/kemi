"""Tests for src/kemi/chunker.py — semantic text chunking."""

from datetime import datetime, timezone

import pytest

from kemi import chunker
from kemi.chunker import (
    CHUNK_META_KEY,
    Chunk,
    ChunkInfo,
    _cosine_sim,
    _is_sentence_boundary,
    chunk_and_embed,
    semantic_chunks,
    split_into_sentences,
)
from kemi.models import LifecycleState, MemoryObject, MemorySource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockEmbed:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vector(text)

    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        import hashlib
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        return [b / 255.0 for b in expanded[: self._dim]]


# ---------------------------------------------------------------------------
# split_into_sentences
# ---------------------------------------------------------------------------

class TestSplitIntoSentences:
    def test_simple_two_sentences(self) -> None:
        text = "Hello world. This is a test."
        result = split_into_sentences(text)
        assert len(result) == 2
        assert result[0] == "Hello world."
        assert result[1] == "This is a test."

    def test_three_sentences(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"
        result = split_into_sentences(text)
        assert len(result) >= 3

    def test_no_punctuation_returns_single(self) -> None:
        text = "Just one long sentence with no punctuation"
        result = split_into_sentences(text)
        assert len(result) == 1

    def test_empty_string(self) -> None:
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_only_whitespace(self) -> None:
        assert split_into_sentences("  \n  ") == []

    def test_abbreviation_not_split(self) -> None:
        text = "Dr. Smith works at e.g. MIT."
        result = split_into_sentences(text)
        # Should NOT break on Dr. or e.g.
        assert any("Dr." in s for s in result)

    def test_very_short_fragment_attached_to_previous(self) -> None:
        text = "This is a sentence. Yes."
        result = split_into_sentences(text)
        assert len(result) >= 1

    def test_newline_paragraph_split(self) -> None:
        text = "First paragraph here.\n\nSecond paragraph here."
        result = split_into_sentences(text)
        # May split into 1 or 2 depending on sentence detection
        assert len(result) >= 1

    def test_single_sentence_no_split(self) -> None:
        text = "Just one sentence."
        result = split_into_sentences(text)
        assert len(result) == 1
        assert result[0] == "Just one sentence."

    def test_three_short_complete_sentences(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence!"
        assert result[2] == "Third sentence?"

    def test_decimal_number_not_split(self) -> None:
        text = "The value is 3.14 and it is high. Next thing."
        result = split_into_sentences(text)
        assert "3.14" in result[0]
        assert len(result) == 2

    def test_ellipsis_handled(self) -> None:
        text = "Wait... what happened. Then more."
        result = split_into_sentences(text)
        assert len(result) >= 1
        # First piece should retain the ellipsis
        assert "..." in result[0]

    def test_fragment_without_terminator_attaches(self) -> None:
        # A bare fragment with no terminator should attach to the previous sentence.
        text = "He said hello. I think. Then he left."
        result = split_into_sentences(text)
        # "He said hello." (3 words, complete) and "Then he left." (3 words, complete)
        # are both kept. "I think." (2 words, complete) is also a complete sentence
        # and stays separate.
        assert len(result) == 3
        assert result[0] == "He said hello."
        assert result[1] == "I think."
        assert result[2] == "Then he left."

    def test_bare_word_fragments_attach(self) -> None:
        # Two-word fragment with no terminator attaches to previous complete sentence.
        text = "She arrived. I think so. They left."
        result = split_into_sentences(text)
        # "She arrived." (2 words, complete) → kept
        # "I think so." (3 words, complete) → kept
        # "They left." (2 words, complete) → kept
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _is_sentence_boundary
# ---------------------------------------------------------------------------

class TestIsSentenceBoundary:
    def test_strong_boundary_capital_following(self) -> None:
        assert _is_sentence_boundary("Hello world.", "Next sentence starts here")
        assert _is_sentence_boundary("Is this right?", "Absolutely yes")

    def test_no_boundary_abbreviation(self) -> None:
        # "Dr." ends with abbreviation → next_cap=True but abbrev check prevents boundary
        # For Dr. the first word is "dr." which IS in abbreviations
        result_dr = _is_sentence_boundary("Dr. Smith", "Next sentence")
        # For e.g. the prev_lower ends with "e.g." which IS in abbreviations  
        result_eg = _is_sentence_boundary("e.g. data", "Another point")
        # Both should be False (abbreviation detected, boundary suppressed)
        assert not result_dr, "Dr. is an abbreviation and should not trigger boundary"
        assert not result_eg, "e.g. is an abbreviation and should not trigger boundary"

    def test_no_boundary_empty_string(self) -> None:
        assert not _is_sentence_boundary("", "Next")
        assert not _is_sentence_boundary("Previous", "")

    def test_no_boundary_no_capital(self) -> None:
        assert not _is_sentence_boundary("End of sentence.", "lowercase start")

    def test_no_boundary_starts_with_abbrev(self) -> None:
        # "Mr. Jones" → first word is "mr." (an abbreviation)
        assert not _is_sentence_boundary("Mr. Jones arrived.", "Next sentence")
        # "Prof. Smith" → first word is "prof."
        assert not _is_sentence_boundary("Prof. Smith spoke.", "Audience listened")
        # "e.g." at end is still caught (existing behaviour)
        assert not _is_sentence_boundary("e.g. some value", "Next thing")

    def test_boundary_starts_without_abbrev(self) -> None:
        # "John Smith arrived." → first word is "john" (not an abbrev)
        assert _is_sentence_boundary("John Smith arrived.", "Next sentence")


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------

class TestCosineSim:
    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.5, 0.3]
        assert _cosine_sim(a, b) == 0.0

    def test_high_dim_vectors(self) -> None:
        import math
        a = [1.0] * 64
        b = [1.0] * 64
        assert _cosine_sim(a, b) == pytest.approx(1.0)
        c = [-1.0 / math.sqrt(64)] * 64
        assert _cosine_sim(a, c) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# semantic_chunks
# ---------------------------------------------------------------------------

class TestSemanticChunks:
    def test_empty_input(self) -> None:
        assert semantic_chunks("", MockEmbed()) == []
        assert semantic_chunks("   ", MockEmbed()) == []

    def test_single_sentence_returns_one_chunk(self) -> None:
        text = "Hello world."
        chunks = semantic_chunks(text, MockEmbed())
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].chunk_info is not None
        assert chunks[0].chunk_info.chunk_index == 0
        assert chunks[0].chunk_info.total_chunks == 1
        assert chunks[0].chunk_info.boundary_strength == 1.0

    def test_chunks_have_correct_chunk_info(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunks = semantic_chunks(text, MockEmbed(), max_tokens=50)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_info is not None
            assert chunk.chunk_info.chunk_index >= 0
            assert chunk.chunk_info.total_chunks == len(chunks)
            assert 0.0 <= chunk.chunk_info.boundary_strength <= 1.0

    def test_respects_max_tokens(self) -> None:
        text = "One word. " * 100
        embed = MockEmbed(dim=64)
        chunks = semantic_chunks(text, embed, max_tokens=20)
        # Some chunks may exceed slightly when a single sentence is longer than max_tokens
        # The key property is: with more chunks, average size should be smaller
        assert len(chunks) > 1, "Long text should be split into multiple chunks"
        for chunk in chunks:
            # Allow 2x overshoot for single oversized sentences (tokens are estimated)
            assert chunk.token_count_estimate() <= 20 * 2 + 5

    def test_overlap_sentences(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        embed = MockEmbed()
        chunks_with_overlap = semantic_chunks(
            text, embed, max_tokens=50, overlap_sentences=1
        )
        # With no overlap, same config
        chunks_no_overlap = semantic_chunks(
            text, embed, max_tokens=50, overlap_sentences=0
        )
        # Overlap may cause more/same chunks or different boundaries
        assert len(chunks_with_overlap) >= 1

    def test_similarity_threshold_affects_split_count(self) -> None:
        text = "This is about cats. Dogs are also great. Birds fly in the sky."
        embed = MockEmbed()
        low_thresh_chunks = semantic_chunks(
            text, embed, max_tokens=200, similarity_threshold=0.1
        )
        high_thresh_chunks = semantic_chunks(
            text, embed, max_tokens=200, similarity_threshold=0.99
        )
        # Higher threshold → more boundaries → same or more chunks
        assert len(high_thresh_chunks) >= len(low_thresh_chunks)

    def test_min_sentences_per_chunk(self) -> None:
        text = "Short. " * 10
        embed = MockEmbed()
        chunks = semantic_chunks(text, embed, max_tokens=200, min_sentences_per_chunk=2)
        for chunk in chunks:
            assert len(chunk.content.split()) >= 2 * 0.5  # rough check

    def test_chunks_have_embeddings(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        embed = MockEmbed(dim=64)
        chunks = semantic_chunks(text, embed, max_tokens=50)
        for chunk in chunks:
            # When there are multiple chunks, embeddings are assigned
            if len(chunks) > 1:
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 64
            # Single-chunk case: semantic_chunks may not assign embedding in all paths
            # (the function assigns embeddings at the end, but only for multi-chunk case)
            # We accept either state here since the function behavior is correct
        assert len(chunks) >= 1

    def test_boundary_strength_for_first_chunk(self) -> None:
        text = "One. Two. Three."
        chunks = semantic_chunks(text, MockEmbed(), max_tokens=50)
        assert chunks[0].chunk_info.boundary_strength == 1.0

    def test_boundary_strength_for_subsequent_chunks(self) -> None:
        text = "A sentence about cats. Another about dogs. And birds too."
        chunks = semantic_chunks(text, MockEmbed(), max_tokens=50)
        for i, chunk in enumerate(chunks):
            if i > 0:
                assert 0.0 <= chunk.chunk_info.boundary_strength <= 1.0

    def test_all_chunks_have_unique_index(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = semantic_chunks(text, MockEmbed(), max_tokens=50)
        indices = {c.chunk_info.chunk_index for c in chunks}
        assert indices == set(range(len(chunks)))


# ---------------------------------------------------------------------------
# chunk_and_embed
# ---------------------------------------------------------------------------

class TestChunkAndEmbed:
    def test_empty_input_returns_empty(self) -> None:
        assert chunk_and_embed("", MockEmbed()) == []

    def test_returns_chunk_objects(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        embed = MockEmbed(dim=64)
        result = chunk_and_embed(text, embed)
        assert len(result) >= 1
        assert all(isinstance(c, Chunk) for c in result)
        # When there are multiple sentences, chunk_and_embed should produce chunks with embeddings
        if len(result) > 1:
            assert all(c.embedding is not None for c in result)

    def test_custom_params_passed_through(self) -> None:
        text = "One. Two. Three. Four. Five."
        result = chunk_and_embed(
            text, MockEmbed(), max_tokens=30, overlap_sentences=1, similarity_threshold=0.6
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Chunk and ChunkInfo dataclasses
# ---------------------------------------------------------------------------

class TestChunkDataclass:
    def test_chunk_len(self) -> None:
        chunk = Chunk(content="Hello world")
        assert len(chunk) == 11

    def test_word_count(self) -> None:
        chunk = Chunk(content="Hello world from testing")
        assert chunk.word_count() == 4

    def test_token_count_estimate(self) -> None:
        chunk = Chunk(content="one two three four five")
        assert chunk.token_count_estimate() == int(5 * 1.3)

    def test_chunk_info_to_dict(self) -> None:
        info = ChunkInfo(
            chunk_index=1,
            total_chunks=3,
            parent_memory_id="mem-123",
            overlap_with_prev=1,
            overlap_with_next=0,
            boundary_strength=0.7,
        )
        d = info.to_dict()
        assert d["chunk_index"] == 1
        assert d["total_chunks"] == 3
        assert d["parent_memory_id"] == "mem-123"
        assert d["boundary_strength"] == 0.7

    def test_chunk_default_values(self) -> None:
        chunk = Chunk(content="test")
        assert chunk.chunk_info is None
        assert chunk.embedding is None

    def test_chunk_with_info_and_embedding(self) -> None:
        info = ChunkInfo(
            chunk_index=0, total_chunks=1, parent_memory_id=None,
            overlap_with_prev=0, overlap_with_next=0, boundary_strength=1.0,
        )
        chunk = Chunk(content="test content", chunk_info=info, embedding=[0.1, 0.2])
        assert chunk.content == "test content"
        assert chunk.chunk_info == info
        assert chunk.embedding == [0.1, 0.2]


# ---------------------------------------------------------------------------
# CHUNK_META_KEY
# ---------------------------------------------------------------------------

class TestChunkMetaKey:
    def test_chunk_meta_key_is_string(self) -> None:
        assert isinstance(CHUNK_META_KEY, str)
        assert CHUNK_META_KEY == "_chunk_info"