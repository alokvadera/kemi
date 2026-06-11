"""Tests for src/kemi/consolidation.py — memory consolidation."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from kemi.memory.consolidation import (
    _cluster_by_similarity,
    _extractive_summary,
    _get_summarizer,
    consolidate,
    consolidate_cluster,
)
from kemi.memory.model import MemorySource, MemoryType
from tests._helpers.factories import make_memory


def _make_memory(
    content: str,
    memory_id: str = "mem-1",
    user_id: str = "alice",
    memory_type: MemoryType = MemoryType.EPISODIC,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
    importance: float = 0.5,
) -> MemoryObject:
    return make_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=([0.1, 0.2] if embedding is None else embedding),
        memory_type=memory_type,
        created_at=created_at or datetime(2020, 1, 1, tzinfo=timezone.utc),
        importance=importance,
        source=MemorySource.USER_STATED,
    )


class TestGetSummarizer:
    def test_false_returns_none(self) -> None:
        assert _get_summarizer(False) is None

    def test_true_but_unavailable_returns_none(self) -> None:
        # When LLM summarizer module is not available, should gracefully fall back
        with patch("kemi.memory.consolidation.logger"):
            result = _get_summarizer(
                True,
                summarizer_llm_provider="openai",
                summarizer_llm_model="gpt-4",
            )
            assert result is None


class TestConsolidateCluster:
    def test_empty_cluster_returns_none(self) -> None:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1, 0.2]
        result = consolidate_cluster(mock_store, mock_embed, "alice", [])
        assert result is None

    def test_single_memory_consolidation(self) -> None:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1, 0.2]

        mem = _make_memory(content="Python is great.")
        result = consolidate_cluster(mock_store, mock_embed, "alice", [mem])

        assert result is not None
        assert result.memory_type == MemoryType.SEMANTIC
        assert result.user_id == "alice"
        assert result.metadata["consolidated_count"] == 1
        assert result.metadata["consolidated_from"] == ["mem-1"]
        mock_store.update.assert_called_once()

    def test_multiple_memories_consolidation(self) -> None:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.3, 0.4]

        mem1 = _make_memory(content="Python is great.", memory_id="mem-1")
        mem2 = _make_memory(content="Java is powerful.", memory_id="mem-2")
        result = consolidate_cluster(mock_store, mock_embed, "alice", [mem1, mem2])

        assert result is not None
        assert result.memory_type == MemoryType.SEMANTIC
        assert result.metadata["consolidated_count"] == 2
        assert result.metadata["consolidated_from"] == ["mem-1", "mem-2"]
        assert mock_store.update.call_count == 2

    def test_importance_set_to_0_7(self) -> None:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1, 0.2]

        mem = _make_memory(content="Test content.")
        result = consolidate_cluster(mock_store, mock_embed, "alice", [mem])
        assert result.importance == 0.7


class TestClusterBySimilarity:
    def test_empty_list(self) -> None:
        assert _cluster_by_similarity([]) == []

    def test_single_memory(self) -> None:
        mem = _make_memory(content="A")
        clusters = _cluster_by_similarity([mem])
        assert len(clusters) == 1
        assert clusters[0] == [mem]

    def test_similar_memories_clustered(self) -> None:
        # Two memories with identical embeddings → same cluster
        mem1 = _make_memory(content="Python is great.", embedding=[0.1] * 64)
        mem2 = _make_memory(content="Python is awesome.", embedding=[0.1] * 64)
        clusters = _cluster_by_similarity([mem1, mem2], threshold=0.75)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_dissimilar_memories_separate(self) -> None:
        # Two memories with opposite embeddings → separate clusters
        mem1 = _make_memory(content="Python.", embedding=[1.0] * 64)
        mem2 = _make_memory(content="Cooking.", embedding=[-1.0] * 64)
        clusters = _cluster_by_similarity([mem1, mem2], threshold=0.75)
        assert len(clusters) == 2
        assert len(clusters[0]) == 1
        assert len(clusters[1]) == 1

    def test_none_embedding_skipped(self) -> None:
        mem1 = _make_memory(content="A", embedding=[0.1] * 64)
        # Construct mem2 directly so embedding can truly be None
        mem2 = make_memory(
            memory_id="mem-2",
            user_id="alice",
            content="B",
            embedding=None,
            memory_type=MemoryType.EPISODIC,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            source=MemorySource.USER_STATED,
        )
        clusters = _cluster_by_similarity([mem1, mem2], threshold=0.75)
        # mem2 has no embedding so similarity can't be computed → stays unassigned
        assert len(clusters) == 2
        assert clusters[0] == [mem1]
        assert clusters[1] == [mem2]


class TestExtractiveSummary:
    def test_empty_sentences(self) -> None:
        mem = _make_memory(content="")
        result = _extractive_summary([mem])
        assert result == ""

    def test_short_content(self) -> None:
        mem = _make_memory(content="One sentence only.")
        result = _extractive_summary([mem])
        assert "One sentence only" in result

    def test_long_content_summarizes(self) -> None:
        mem = _make_memory(
            content=" "
            + ". ".join(f"Sentence number {i} about python and machine learning" for i in range(20))
            + "."
        )
        result = _extractive_summary([mem])
        assert len(result) <= 1024
        assert "python" in result.lower() or "machine" in result.lower()

    def test_multiple_memories_combined(self) -> None:
        mem1 = _make_memory(content="Python is great for data science.")
        mem2 = _make_memory(content="Machine learning is a subset of data science.")
        result = _extractive_summary([mem1, mem2])
        assert "python" in result.lower() or "machine" in result.lower()

    def test_capped_at_1024_chars(self) -> None:
        # Use multiple short sentences so the scoring logic runs and the cap applies
        content = (
            ". ".join(f"Sentence number {i} about python programming" for i in range(50)) + "."
        )
        mem = _make_memory(content=content)
        result = _extractive_summary([mem])
        assert len(result) <= 1024


class TestConsolidate:
    def test_no_memories_returns_none(self) -> None:
        mock_store = MagicMock()
        mock_store.get_all_by_user.return_value = []
        mock_embed = MagicMock()
        result = consolidate(mock_store, mock_embed, "alice")
        assert result is None

    def test_only_new_memories_returns_none(self) -> None:
        """Memories newer than max_age_days should be ignored."""
        mock_store = MagicMock()
        now = datetime.now(timezone.utc)
        mem = _make_memory(
            content="Recent memory.",
            created_at=now,
        )
        mock_store.get_all_by_user.return_value = [mem]
        mock_embed = MagicMock()
        result = consolidate(mock_store, mock_embed, "alice", max_age_days=30.0)
        assert result is None

    def test_below_min_memories_returns_none(self) -> None:
        mock_store = MagicMock()
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        mem1 = _make_memory(content="Old memory one.", created_at=old)
        mem2 = _make_memory(content="Old memory two.", created_at=old)
        mock_store.get_all_by_user.return_value = [mem1, mem2]
        mock_embed = MagicMock()
        result = consolidate(mock_store, mock_embed, "alice", min_memories=5)
        assert result is None

    def test_successful_consolidation(self) -> None:
        mock_store = MagicMock()
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        memories = [
            _make_memory(content=f"Memory {i} about python.", created_at=old, memory_id=f"mem-{i}")
            for i in range(5)
        ]
        # Give them similar embeddings so they cluster together
        for i, mem in enumerate(memories):
            mem.embedding = [0.1 + i * 0.01] * 64

        mock_store.get_all_by_user.return_value = memories
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1] * 64

        result = consolidate(mock_store, mock_embed, "alice", min_memories=3, max_age_days=30.0)

        assert result is not None
        assert isinstance(result, str)
        mock_store.store.assert_called_once()
        # All old memories should be archived
        assert mock_store.update.call_count == 5

    def test_only_episodic_memories_considered(self) -> None:
        mock_store = MagicMock()
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        episodic = _make_memory(
            content="Episodic memory.", created_at=old, memory_type=MemoryType.EPISODIC
        )
        semantic = _make_memory(
            content="Semantic memory.", created_at=old, memory_type=MemoryType.SEMANTIC
        )
        mock_store.get_all_by_user.return_value = [episodic, semantic]
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1, 0.2]

        result = consolidate(mock_store, mock_embed, "alice", min_memories=1, max_age_days=30.0)
        # Only episodic memories are considered; with 1 episodic memory the
        # best cluster has size 1 which is >= min_memories=1, so it should consolidate.
        assert result is not None

    def test_with_llm_summary_false(self) -> None:
        mock_store = MagicMock()
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        memories = [
            _make_memory(content=f"Memory {i}.", created_at=old, memory_id=f"mem-{i}")
            for i in range(5)
        ]
        for i, mem in enumerate(memories):
            mem.embedding = [0.1 + i * 0.01] * 64

        mock_store.get_all_by_user.return_value = memories
        mock_embed = MagicMock()
        mock_embed.embed_single.return_value = [0.1] * 64

        result = consolidate(
            mock_store,
            mock_embed,
            "alice",
            min_memories=3,
            max_age_days=30.0,
            with_llm_summary=False,
        )
        assert result is not None
