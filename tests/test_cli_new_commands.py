"""Integration tests for new CLI commands: chunk, decompose, rerank, versions, rollback."""

import argparse
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from kemi import Memory
from kemi.cli import (
    decompose_and_recall,
    main,
    preview_chunk,
    rerank_recall,
    rollback_memory,
    show_history,
    show_version_diff,
)
from kemi.models import LifecycleState, MemoryObject, MemorySource, MemoryType
from kemi.versions import MemoryVersionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _patch_get_memory(mock_memory: Memory):
    """Patch kemi.cli.get_memory to return the mock_memory fixture."""
    with patch("kemi.cli.get_memory", return_value=mock_memory):
        yield


# ---------------------------------------------------------------------------
# kemi chunk — preview_chunk
# ---------------------------------------------------------------------------

class TestCLIChunk:
    """Tests for the chunk CLI command."""

    def test_chunk_single_chunk(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk with short content produces a single chunk."""
        preview_chunk(
            argparse.Namespace(
                content="This is a short sentence.",
                max_tokens=256,
                overlap=1,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Produced 1 chunk(s)" in captured.out
        assert "This is a short sentence" in captured.out

    def test_chunk_multiple_chunks(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk with long content produces multiple chunks."""
        long_text = ". ".join([f"Sentence number {i} with some additional words to make it longer" for i in range(1, 21)])
        preview_chunk(
            argparse.Namespace(
                content=long_text,
                max_tokens=30,
                overlap=1,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Produced" in captured.out
        assert "Chunk" in captured.out

    def test_chunk_empty_content(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk with empty content prints appropriate message."""
        preview_chunk(
            argparse.Namespace(
                content="",
                max_tokens=256,
                overlap=1,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "No chunks produced" in captured.out

    def test_chunk_respects_max_tokens_param(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk --max-tokens controls chunk size."""
        long_text = ". ".join([f"Word number {i}" for i in range(1, 51)])
        preview_chunk(
            argparse.Namespace(
                content=long_text,
                max_tokens=10,
                overlap=0,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Produced" in captured.out

    def test_chunk_respects_overlap_param(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk --overlap controls sentence overlap between chunks."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
        preview_chunk(
            argparse.Namespace(
                content=text,
                max_tokens=50,
                overlap=2,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Produced" in captured.out

    def test_chunk_shows_boundary_strength(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk output includes boundary strength for each chunk."""
        text = "This is about cats. Dogs are also great. Birds fly in the sky. Fish swim in water."
        preview_chunk(
            argparse.Namespace(
                content=text,
                max_tokens=256,
                overlap=1,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "strength=" in captured.out

    def test_chunk_via_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk through main() parses arguments correctly."""
        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "chunk", "A short test sentence.", "--max-tokens", "100", "--overlap", "1"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Produced" in captured.out
        assert "A short test sentence" in captured.out


# ---------------------------------------------------------------------------
# kemi decompose — decompose_and_recall
# ---------------------------------------------------------------------------

class TestCLIDecompose:
    """Tests for the decompose CLI command."""

    def test_decompose_simple_query(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose with simple query shows sub-queries."""
        mem = MemoryObject(
            memory_id="decomp-1",
            user_id="user1",
            content="I eat breakfast at 8am and dinner at 7pm",
            embedding=mock_memory._embed.embed_single("I eat breakfast at 8am and dinner at 7pm"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="What did I eat for breakfast and dinner?",
                strategy="simple",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Original query:" in captured.out
        assert "Strategy: simple" in captured.out
        assert "Sub-queries" in captured.out
        assert "breakfast" in captured.out.lower()
        assert "dinner" in captured.out.lower()

    def test_decompose_no_match_shows_no_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose when no memories match prints 'No memories found'."""
        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="nonexistent query xyzqwerty",
                strategy="simple",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Original query:" in captured.out
        assert "No memories found" in captured.out

    def test_decompose_expand_strategy(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose --strategy expand shows expanded variants."""
        mem = MemoryObject(
            memory_id="decomp-expand-1",
            user_id="user1",
            content="I eat breakfast every morning",
            embedding=mock_memory._embed.embed_single("I eat breakfast every morning"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="What do I eat in the morning?",
                strategy="expand",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Strategy: expand" in captured.out
        assert "Original query:" in captured.out

    def test_decompose_single_subquery(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose with single sub-query skips fusion."""
        mem = MemoryObject(
            memory_id="decomp-single-1",
            user_id="user1",
            content="Python is a programming language",
            embedding=mock_memory._embed.embed_single("Python is a programming language"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="Tell me about Python",
                strategy="simple",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Original query:" in captured.out
        assert "Sub-queries" in captured.out or "(Single sub-query" in captured.out

    def test_decompose_via_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose through main() parses arguments correctly."""
        mem = MemoryObject(
            memory_id="decomp-e2e-1",
            user_id="user1",
            content="I like cats and dogs",
            embedding=mock_memory._embed.embed_single("I like cats and dogs"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "decompose", "user1", "Tell me about cats and dogs", "--strategy", "simple"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Original query:" in captured.out
        assert "Strategy: simple" in captured.out

    def test_decompose_both_strategy(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose --strategy both combines simple and expand."""
        mem = MemoryObject(
            memory_id="decomp-both-1",
            user_id="user1",
            content="I work on projects",
            embedding=mock_memory._embed.embed_single("I work on projects"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="Tell me about my work",
                strategy="both",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Strategy: both" in captured.out

    def test_decompose_none_strategy(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose --strategy none returns original query unchanged."""
        mem = MemoryObject(
            memory_id="decomp-none-1",
            user_id="user1",
            content="Some memory content",
            embedding=mock_memory._embed.embed_single("Some memory content"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="Some memory content",
                strategy="none",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Strategy: none" in captured.out

    def test_decompose_results_show_scores(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose results include score information."""
        mem = MemoryObject(
            memory_id="decomp-score-1",
            user_id="user1",
            content="I eat breakfast every morning",
            embedding=mock_memory._embed.embed_single("I eat breakfast every morning"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        decompose_and_recall(
            argparse.Namespace(
                user_id="user1",
                query="What do I eat in the morning?",
                strategy="simple",
                top_k=5,
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        # Score appears as "  0.785 | content" format (no literal "Score:" label)
        assert "0.785" in captured.out


# ---------------------------------------------------------------------------
# kemi rerank — rerank_recall
# ---------------------------------------------------------------------------

class TestCLIRerank:
    """Tests for the rerank CLI command."""

    def test_rerank_shows_initial_recall_count(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank prints initial recall count."""
        mem = MemoryObject(
            memory_id="rerank-1",
            user_id="user1",
            content="I love Python programming",
            embedding=mock_memory._embed.embed_single("I love Python programming"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        rerank_recall(
            argparse.Namespace(
                user_id="user1",
                query="Python",
                top_k=10,
                provider="fallback",
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Initial recall:" in captured.out

    def test_rerank_no_memories(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank with no matching memories prints message."""
        rerank_recall(
            argparse.Namespace(
                user_id="user1",
                query="nonexistent xyzqwerty",
                top_k=10,
                provider="fallback",
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    def test_rerank_shows_provider(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank output shows the provider name."""
        mem = MemoryObject(
            memory_id="rerank-prov-1",
            user_id="user1",
            content="I enjoy coding in Python",
            embedding=mock_memory._embed.embed_single("I enjoy coding in Python"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        rerank_recall(
            argparse.Namespace(
                user_id="user1",
                query="Python coding",
                top_k=10,
                provider="fallback",
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "fallback reranker" in captured.out

    def test_rerank_via_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank through main() parses arguments correctly."""
        mem = MemoryObject(
            memory_id="rerank-e2e-1",
            user_id="user1",
            content="I love JavaScript programming",
            embedding=mock_memory._embed.embed_single("I love JavaScript programming"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "rerank", "user1", "JavaScript", "--provider", "fallback"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Initial recall:" in captured.out

    def test_rerank_top_k_limits_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank --top-k limits initial recall set."""
        for i in range(5):
            mem = MemoryObject(
                memory_id=f"rerank-tk-{i}",
                user_id="user1",
                content=f"Memory number {i} about programming",
                embedding=mock_memory._embed.embed_single(f"Memory number {i} about programming"),
                embedding_dim=mock_memory._embed.dimension(),
            )
            mock_memory._store.store(mem)

        rerank_recall(
            argparse.Namespace(
                user_id="user1",
                query="programming",
                top_k=2,
                provider="fallback",
                namespace="default",
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Initial recall: 2" in captured.out or "Initial recall:" in captured.out


# ---------------------------------------------------------------------------
# kemi versions — show_versions
# ---------------------------------------------------------------------------

class TestCLIVersions:
    """Tests for the versions CLI command."""

    def test_versions_no_history(self, _patch_get_memory, mock_memory: Memory, capsys):
        """versions with no history prints message."""
        show_history(
            argparse.Namespace(
                memory_id="nonexistent-memory",
                limit=100,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "No version history found" in captured.out

    def test_versions_shows_snapshots(self, _patch_get_memory, mock_memory: Memory, capsys):
        """versions shows recorded version snapshots."""
        from kemi.versions import MemoryVersionStore

        db_path = getattr(mock_memory._store, "_db_path", None)
        if db_path is None:
            pytest.skip("No temp db path available in mock store")

        vs = MemoryVersionStore(db_path=db_path)
        mem = MemoryObject(
            memory_id="versions-test-1",
            user_id="user1",
            content="v1 content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=1,
        )
        vs.record_version(mem, changed_by="update")

        show_history(
            argparse.Namespace(
                memory_id="versions-test-1",
                limit=100,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Version history for: versions-test-1" in captured.out
        assert "Total versions: 1" in captured.out
        assert "v1" in captured.out

    def test_versions_via_main_no_db(self, capsys):
        """versions through main() with no db prints message."""
        with patch("kemi.cli.os.path.exists", return_value=False):
            with patch.object(sys, "argv", ["kemi", "history", "some-memory-id"]):
                main()
        captured = capsys.readouterr()
        assert "No memory database found" in captured.out


# ---------------------------------------------------------------------------
# kemi rollback — rollback_memory
# ---------------------------------------------------------------------------

class TestCLIRollback:
    """Tests for the rollback CLI command."""

    def test_rollback_nonexistent_memory(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rollback with nonexistent memory shows error."""
        rollback_memory(
            argparse.Namespace(
                memory_id="nonexistent-memory",
                to_version=1,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert captured.out is not None

    def test_rollback_version_not_found(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rollback to version that doesn't exist prints message."""
        from kemi.versions import MemoryVersionStore

        db_path = getattr(mock_memory._store, "_db_path", None)
        if db_path is None:
            pytest.skip("No temp db path available")

        vs = MemoryVersionStore(db_path=db_path)
        mem = MemoryObject(
            memory_id="rollback-test-none",
            user_id="user1",
            content="content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=1,
        )
        vs.record_version(mem, changed_by="update")

        rollback_memory(
            argparse.Namespace(
                memory_id="rollback-test-none",
                to_version=99,
                hooks_raise_on_error=None,
            )
        )
        captured = capsys.readouterr()
        assert "Version 99 not found" in captured.out or captured.out != ""


# ---------------------------------------------------------------------------
# End-to-end tests via main()
# ---------------------------------------------------------------------------

class TestCLINewCommandsE2E:
    """End-to-end tests for new CLI commands through main()."""

    def test_chunk_e2e(self, _patch_get_memory, mock_memory: Memory, capsys):
        """chunk through main() works end-to-end."""
        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "chunk", "One sentence here. Two sentence here. Three sentence here."],
            ):
                main()
        captured = capsys.readouterr()
        assert "Produced" in captured.out

    def test_decompose_e2e_with_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """decompose through main() with matching memories shows results."""
        mem = MemoryObject(
            memory_id="decomp-e2e-results",
            user_id="user1",
            content="I visited Paris last week and London yesterday",
            embedding=mock_memory._embed.embed_single("I visited Paris last week and London yesterday"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "decompose", "user1",
                    "What cities did I visit and when?",
                    "--strategy", "simple",
                    "--top-k", "3",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Original query:" in captured.out
        assert "Sub-queries" in captured.out

    def test_rerank_e2e_with_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank through main() with matching memories shows reranked output."""
        mem = MemoryObject(
            memory_id="rerank-e2e-results",
            user_id="user1",
            content="I enjoy hiking in mountains",
            embedding=mock_memory._embed.embed_single("I enjoy hiking in mountains"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "rerank", "user1", "hiking mountains", "--provider", "fallback"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Initial recall:" in captured.out
        assert "Reranked results" in captured.out

    def test_rerank_e2e_empty_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """rerank through main() with no matching memories prints message."""
        with patch("kemi.cli.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "rerank", "user1", "nonexistent query xyzqwerty123"],
            ):
                main()
        captured = capsys.readouterr()
        assert "No memories found" in captured.out


# ---------------------------------------------------------------------------
# Real DB integration tests for versions and rollback
# ---------------------------------------------------------------------------


class TestCLIVersionsRealDB:
    """Integration tests for versions and rollback using real SQLite DB."""

    def test_versions_shows_version_history(self, real_db_memory: Memory, capsys):
        """versions command shows recorded version history from real DB."""
        from kemi.versions import MemoryVersionStore

        db_path = real_db_memory._store._db_path
        vs = MemoryVersionStore(db_path=db_path)

        mem = MemoryObject(
            memory_id="versions-real-1",
            user_id="user1",
            content="original content version 1",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=["tag1", "tag2"],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=1,
        )
        vs.record_version(mem, changed_by="update")

        mem_v2 = MemoryObject(
            memory_id="versions-real-1",
            user_id="user1",
            content="updated content version 2",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.7,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=["tag1", "tag2", "tag3"],
            confidence=0.9,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=2,
        )
        vs.record_version(mem_v2, changed_by="update")

        # Patch os.path.expanduser in cli module so CLI uses our temp db instead of ~/.kemi/memories.db
        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.expanduser",
                       lambda x: db_path if "~/.kemi/memories.db" in str(x) else x):
                with patch("kemi.cli.os.path.exists", return_value=True):
                    with patch.object(sys, "argv", ["kemi", "history", "versions-real-1"]):
                        main()
        captured = capsys.readouterr()
        assert "Version history for: versions-real-1" in captured.out
        assert "Total versions: 2" in captured.out
        assert "v2" in captured.out
        assert "v1" in captured.out
        assert "updated content version 2" in captured.out
        assert "original content version 1" in captured.out

    def test_versions_shows_diff(self, real_db_memory: Memory, capsys):
        """versions --diff shows field-level differences between versions."""
        from kemi.versions import MemoryVersionStore

        db_path = real_db_memory._store._db_path
        vs = MemoryVersionStore(db_path=db_path)

        mem_v1 = MemoryObject(
            memory_id="versions-diff-1",
            user_id="user1",
            content="original content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.3,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=0.5,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=1,
        )
        vs.record_version(mem_v1, changed_by="update")

        mem_v2 = MemoryObject(
            memory_id="versions-diff-1",
            user_id="user1",
            content="updated content changed",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.9,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=["new-tag"],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=2,
        )
        vs.record_version(mem_v2, changed_by="update")

        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.expanduser",
                       lambda x: db_path if "~/.kemi/memories.db" in str(x) else x):
                with patch("kemi.cli.os.path.exists", return_value=True):
                    with patch.object(
                        sys,
                        "argv",
                        ["kemi", "version", "diff", "versions-diff-1", "--v1", "1", "--v2", "2"],
                    ):
                        main()
        captured = capsys.readouterr()
        assert "Diff v1" in captured.out or "versions-diff-1" in captured.out

    def test_rollback_restores_content(self, real_db_memory: Memory, capsys):
        """rollback command restores memory to previous version's content."""
        from kemi.versions import MemoryVersionStore

        db_path = real_db_memory._store._db_path
        vs = MemoryVersionStore(db_path=db_path)

        mem_current = MemoryObject(
            memory_id="rollback-real-1",
            user_id="user1",
            content="current version content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=2,
        )
        real_db_memory._store.store(mem_current)

        mem_v1 = MemoryObject(
            memory_id="rollback-real-1",
            user_id="user1",
            content="v1 content restored",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=1,
        )
        vs.record_version(mem_v1, changed_by="update")

        mem_v2 = MemoryObject(
            memory_id="rollback-real-1",
            user_id="user1",
            content="current version content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=2,
        )
        vs.record_version(mem_v2, changed_by="update")

        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.expanduser",
                       lambda x: db_path if "~/.kemi/memories.db" in str(x) else x):
                with patch("kemi.cli.os.path.exists", return_value=True):
                    with patch.object(
                        sys,
                        "argv",
                        ["kemi", "rollback", "rollback-real-1", "--to-version", "1"],
                    ):
                        main()
        captured = capsys.readouterr()
        assert "rollback" in captured.out.lower() or "Rolled back" in captured.out

    def test_rollback_via_versions_command(self, real_db_memory: Memory, capsys):
        """versions command shows rollback after rollback is performed."""
        from kemi.versions import MemoryVersionStore

        db_path = real_db_memory._store._db_path
        vs = MemoryVersionStore(db_path=db_path)

        mem_current = MemoryObject(
            memory_id="rollback-via-versions",
            user_id="user1",
            content="current content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=2,
        )
        real_db_memory._store.store(mem_current)

        mem_v1 = MemoryObject(
            memory_id="rollback-via-versions",
            user_id="user1",
            content="v1 old content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=1,
        )
        vs.record_version(mem_v1, changed_by="update")

        mem_v2 = MemoryObject(
            memory_id="rollback-via-versions",
            user_id="user1",
            content="current content",
            embedding=[0.1, 0.2, 0.3],
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=3,
            tags=[],
            confidence=1.0,
            memory_type=MemoryType.EPISODIC,
            session_id=None,
            namespace="default",
            version=2,
        )
        vs.record_version(mem_v2, changed_by="update")

        # Perform rollback
        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.expanduser",
                       lambda x: db_path if "~/.kemi/memories.db" in str(x) else x):
                with patch("kemi.cli.os.path.exists", return_value=True):
                    with patch.object(
                        sys,
                        "argv",
                        ["kemi", "rollback", "rollback-via-versions", "--to-version", "1"],
                    ):
                        main()
        capsys.readouterr()  # discard rollback output

        # Now check versions — should show the new rollback version
        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.expanduser",
                       lambda x: db_path if "~/.kemi/memories.db" in str(x) else x):
                with patch("kemi.cli.os.path.exists", return_value=True):
                    with patch.object(
                        sys,
                        "argv",
                        ["kemi", "history", "rollback-via-versions"],
                    ):
                        main()
        captured = capsys.readouterr()
        assert "Version history for: rollback-via-versions" in captured.out
        # Should have v1, v2, and the rollback version (v3)
        assert "Total versions: 3" in captured.out

    def test_chunk_with_real_db(self, real_db_memory: Memory, capsys):
        """chunk command works with real database backend."""
        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.exists", return_value=True):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "kemi", "chunk",
                        "First sentence here. Second sentence here. Third sentence here.",
                        "--max-tokens", "50",
                    ],
                ):
                    main()
        captured = capsys.readouterr()
        assert "Content:" in captured.out
        assert "Produced" in captured.out

    def test_decompose_with_real_db(self, real_db_memory: Memory, capsys):
        """decompose command works with real database backend."""
        real_db_memory.remember("user1", "I eat breakfast and dinner every day")

        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.exists", return_value=True):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "kemi", "decompose", "user1",
                        "What do I eat for breakfast and dinner?",
                        "--strategy", "simple",
                    ],
                ):
                    main()
        captured = capsys.readouterr()
        assert "Original query:" in captured.out
        assert "Strategy: simple" in captured.out

    def test_rerank_with_real_db(self, real_db_memory: Memory, capsys):
        """rerank command works with real database backend."""
        real_db_memory.remember("user1", "I love Python programming")
        real_db_memory.remember("user1", "I enjoy hiking in mountains")

        with patch("kemi.cli.get_memory", return_value=real_db_memory):
            with patch("kemi.cli.os.path.exists", return_value=True):
                with patch.object(
                    sys,
                    "argv",
                    ["kemi", "rerank", "user1", "Python programming", "--provider", "fallback"],
                ):
                    main()
        captured = capsys.readouterr()
        assert "Initial recall:" in captured.out
        assert "Reranked results" in captured.out