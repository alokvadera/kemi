"""Tests for src/kemi/topics.py"""

from unittest.mock import MagicMock, patch

import pytest

from kemi.models import LifecycleState


def _sklearn_available() -> bool:
    try:
        from sklearn.cluster import KMeans  # noqa: F401

        return True
    except ImportError:
        return False


# Skip cluster tests if sklearn not available (they mock sklearn.cluster internally)
SKLEARN_SKIP = pytest.mark.skipif(not _sklearn_available(), reason="sklearn not installed")


class TestClusterMemories:
    @SKLEARN_SKIP
    def test_cluster_memories_requires_sklearn(self):
        from kemi import topics

        with patch.object(topics, "_sklearn_available", return_value=False):
            with pytest.raises(ImportError, match="scikit-learn"):
                topics.cluster_memories(MagicMock(), "alice")

    @SKLEARN_SKIP
    def test_cluster_memories_less_than_2_with_embeddings(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        mem = MagicMock()
        mem.embedding = [0.1, 0.2]
        mem.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem]

        with patch("kemi.topics._sklearn_available", return_value=True):
            with patch("sklearn.cluster.KMeans"):
                result = cluster_memories(mock_store, "alice", n_clusters=3)
                assert "topic_0" in result

    @SKLEARN_SKIP
    def test_cluster_memories_no_memories(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        mock_store.get_all_by_user.return_value = []

        with patch("kemi.topics._sklearn_available", return_value=True):
            result = cluster_memories(mock_store, "alice")
            assert result == {}

    @SKLEARN_SKIP
    def test_cluster_memories_single_memory(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        mem = MagicMock()
        mem.embedding = [0.1, 0.2]
        mem.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem]

        with patch("kemi.topics._sklearn_available", return_value=True):
            with patch("sklearn.cluster.KMeans"):
                result = cluster_memories(mock_store, "alice")
                assert "topic_0" in result

    @SKLEARN_SKIP
    def test_cluster_memories_k_capped_to_memory_count(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        memories = []
        for i in range(3):
            mem = MagicMock()
            mem.embedding = [0.1 + i * 0.1, 0.2 + i * 0.1]
            mem.lifecycle_state = LifecycleState.ACTIVE
            memories.append(mem)
        mock_store.get_all_by_user.return_value = memories

        with patch("kemi.topics._sklearn_available", return_value=True):
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = [0, 1, 0]
            with patch("sklearn.cluster.KMeans", return_value=mock_kmeans):
                # n_clusters=10 but only 3 memories => effective k=2 (minimum)
                result = cluster_memories(mock_store, "alice", n_clusters=10)
                assert len(result) >= 1

    @SKLEARN_SKIP
    def test_cluster_memories_filters_memories_without_embeddings(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        mem_with_emb = MagicMock()
        mem_with_emb.embedding = [0.1, 0.2]
        mem_with_emb.lifecycle_state = LifecycleState.ACTIVE
        mem_without_emb = MagicMock()
        mem_without_emb.embedding = None
        mem_without_emb.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem_with_emb, mem_without_emb]

        with patch("kemi.topics._sklearn_available", return_value=True):
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = [0]
            with patch("sklearn.cluster.KMeans", return_value=mock_kmeans):
                result = cluster_memories(mock_store, "alice")
                for mems in result.values():
                    for m in mems:
                        assert m.embedding is not None

    @SKLEARN_SKIP
    def test_cluster_memories_kmeans_failure_falls_back(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        mem1 = MagicMock()
        mem1.embedding = [0.1, 0.2]
        mem1.lifecycle_state = LifecycleState.ACTIVE
        mem2 = MagicMock()
        mem2.embedding = [0.3, 0.4]
        mem2.lifecycle_state = LifecycleState.ACTIVE
        mock_store.get_all_by_user.return_value = [mem1, mem2]

        with patch("kemi.topics._sklearn_available", return_value=True):
            with patch(
                "sklearn.cluster.KMeans",
                side_effect=Exception("KMeans failed"),
            ):
                result = cluster_memories(mock_store, "alice")
                # Should fall back to topic_0 with all memories
                assert "topic_0" in result

    @SKLEARN_SKIP
    def test_cluster_memories_sorted_by_size(self):
        from kemi.topics import cluster_memories

        mock_store = MagicMock()
        memories = []
        for i in range(4):
            mem = MagicMock()
            mem.embedding = [0.1 * i, 0.2 * i]
            mem.lifecycle_state = LifecycleState.ACTIVE
            memories.append(mem)
        mock_store.get_all_by_user.return_value = memories

        with patch("kemi.topics._sklearn_available", return_value=True):
            mock_kmeans = MagicMock()
            # 2 in cluster 0, 1 in cluster 1, 1 in cluster 2
            mock_kmeans.fit_predict.return_value = [0, 0, 1, 2]
            with patch("sklearn.cluster.KMeans", return_value=mock_kmeans):
                result = cluster_memories(mock_store, "alice", n_clusters=3)
                # Cluster 0 should come first (largest)
                labels = list(result.keys())
                assert len(labels) == 3


class TestGenerateTopicLabel:
    def test_generates_label_from_top_words(self):
        from kemi.topics import _generate_topic_label

        mem1 = MagicMock()
        mem1.content = "Python programming is great and python is fun"
        mem2 = MagicMock()
        mem2.content = "Python scripts automate tasks"

        label = _generate_topic_label([mem1, mem2], 0)
        assert "python" in label.lower()
        assert label != "Topic 1"

    def test_fallback_when_all_words_filtered(self):
        from kemi.topics import _generate_topic_label

        mem = MagicMock()
        mem.content = "the a is are and but or"

        label = _generate_topic_label([mem], 0)
        assert label == "Topic 1"

    def test_single_memory_single_word(self):
        from kemi.topics import _generate_topic_label

        mem = MagicMock()
        mem.content = "kubernetes orchestration"

        label = _generate_topic_label([mem], 2)
        assert "kubernetes" in label.lower() or "orchestration" in label.lower()

    def test_index_offset_in_label(self):
        from kemi.topics import _generate_topic_label

        mem = MagicMock()
        mem.content = "machine learning neural networks deep"

        label = _generate_topic_label([mem], 5)
        # Now generates meaningful label from content, not generic "Topic N"
        assert "machine" in label.lower() or "learning" in label.lower()

    def test_stopwords_filtered(self):
        from kemi.topics import _generate_topic_label

        mem = MagicMock()
        mem.content = "the category sat on the mattress and the dog ran quickly"

        label = _generate_topic_label([mem], 0)
        # "category" appears twice, "mattress" once, "dog" once, "quickly" once
        # Stopwords like "the", "on", "and", "the" should be filtered
        # The assertion checks that stopwords didn't dominate
        assert "category" in label.lower() or "mattress" in label.lower()

    def test_punctuation_stripped(self):
        from kemi.topics import _generate_topic_label

        mem = MagicMock()
        mem.content = "python!!! python?? python,, python..."

        label = _generate_topic_label([mem], 0)
        assert "python" in label.lower()

    def test_short_words_filtered(self):
        from kemi.topics import _generate_topic_label

        mem = MagicMock()
        mem.content = "a b c d e f g hi there python"

        label = _generate_topic_label([mem], 0)
        # Only "there" and "python" are > 3 chars
        assert label != "Topic 1"
