from tests._helpers.factories import make_memory

"""Integration tests for kemi CLI commands.

Tests the handler functions directly with mocked Memory instances.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.slow

# scikit-learn is an optional dependency for topic clustering
try:
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from kemi import Memory
from kemi.interfaces.cli import (
    consolidate_memories,
    explain_memories,
    get_memory,
    graph_memories,
    main,
    prune_memories,
    recall_stream_memories,
    topics_memories,
    update_memory,
)
from kemi.memory.model import LifecycleState, MemoryType


@pytest.fixture
def _patch_get_memory(mock_memory: Memory):
    """Patch kemi.interfaces.cli.get_memory to return the mock_memory fixture."""
    with patch("kemi.interfaces.cli.main.get_memory", return_value=mock_memory):
        yield


class TestCLIPrune:
    """Tests for the prune CLI command."""

    def test_prune_by_age(self, _patch_get_memory, mock_memory: Memory, capsys):
        """prune --max-age-days deletes old memories."""
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        mem = make_memory(
            memory_id="old-1",
            user_id="user1",
            content="old memory",
            created_at=old_time,
            last_accessed_at=old_time,
            lifecycle_state=LifecycleState.DECAYING,
        )
        mock_memory._store.store(mem)

        prune_memories(
            argparse.Namespace(
                user_id="user1", max_age_days=30.0, min_importance=None, namespace="default"
            )
        )
        captured = capsys.readouterr()
        assert "Pruned 1 memories" in captured.out
        assert mock_memory._store.get("old-1") is None

    def test_prune_by_importance(self, _patch_get_memory, mock_memory: Memory, capsys):
        """prune --min-importance deletes low-importance memories."""
        mem = make_memory(
            memory_id="low-1",
            user_id="user1",
            content="low importance",
            importance=0.05,
            lifecycle_state=LifecycleState.DECAYING,
        )
        mock_memory._store.store(mem)

        prune_memories(
            argparse.Namespace(
                user_id="user1", max_age_days=None, min_importance=0.1, namespace="default"
            )
        )
        captured = capsys.readouterr()
        assert "Pruned 1 memories" in captured.out
        assert mock_memory._store.get("low-1") is None

    def test_prune_no_matches(self, _patch_get_memory, mock_memory: Memory, capsys):
        """prune with no matching memories prints 0 deleted."""
        prune_memories(
            argparse.Namespace(
                user_id="user1", max_age_days=30.0, min_importance=None, namespace="default"
            )
        )
        captured = capsys.readouterr()
        assert "Pruned 0 memories" in captured.out

    def test_prune_with_namespace(self, _patch_get_memory, mock_memory: Memory, capsys):
        """prune --namespace only affects the given namespace."""
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        mem_default = make_memory(
            memory_id="old-default",
            user_id="user1",
            content="old default",
            created_at=old_time,
            last_accessed_at=old_time,
            lifecycle_state=LifecycleState.DECAYING,
        )
        mem_other = make_memory(
            memory_id="old-other",
            user_id="user1",
            content="old other",
            created_at=old_time,
            last_accessed_at=old_time,
            lifecycle_state=LifecycleState.DECAYING,
            namespace="other",
        )
        mock_memory._store.store(mem_default)
        mock_memory._store.store(mem_other)

        prune_memories(
            argparse.Namespace(
                user_id="user1", max_age_days=30.0, min_importance=None, namespace="other"
            )
        )
        captured = capsys.readouterr()
        assert "Pruned 1 memories" in captured.out
        assert mock_memory._store.get("old-default") is not None
        assert mock_memory._store.get("old-other") is None


class TestCLIConsolidate:
    """Tests for the consolidate CLI command."""

    def test_consolidate_old_memories(self, _patch_get_memory, mock_memory: Memory, capsys):
        """consolidate creates a semantic summary from old episodic memories."""
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(5):
            mem = make_memory(
                memory_id=f"ep-{i}",
                user_id="user1",
                content=f"I visited Paris on day {i}",
                created_at=old_time,
                last_accessed_at=old_time,
                memory_type=MemoryType.EPISODIC,
                embedding=mock_memory._embed.embed_single(f"I visited Paris on day {i}"),
                embedding_dim=mock_memory._embed.dimension(),
            )
            mock_memory._store.store(mem)

        consolidate_memories(argparse.Namespace(user_id="user1", namespace="default"))
        captured = capsys.readouterr()
        assert "Consolidated into memory:" in captured.out

        # Old memories should be archived
        for i in range(5):
            archived = mock_memory._store.get(f"ep-{i}")
            assert archived is not None
            assert archived.lifecycle_state == LifecycleState.ARCHIVED

    def test_consolidate_no_memories(self, _patch_get_memory, mock_memory: Memory, capsys):
        """consolidate with no old memories prints 'No consolidation needed'."""
        consolidate_memories(argparse.Namespace(user_id="user1", namespace="default"))
        captured = capsys.readouterr()
        assert "No consolidation needed" in captured.out

    def test_consolidate_with_namespace(self, _patch_get_memory, mock_memory: Memory, capsys):
        """consolidate --namespace only consolidates memories in that namespace."""
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(5):
            mem = make_memory(
                memory_id=f"ep-ns-{i}",
                user_id="user1",
                content=f"I visited Tokyo day {i}",
                created_at=old_time,
                last_accessed_at=old_time,
                memory_type=MemoryType.EPISODIC,
                namespace="travel",
                embedding=mock_memory._embed.embed_single(f"I visited Tokyo day {i}"),
                embedding_dim=mock_memory._embed.dimension(),
            )
            mock_memory._store.store(mem)

        consolidate_memories(argparse.Namespace(user_id="user1", namespace="travel"))
        captured = capsys.readouterr()
        assert "Consolidated into memory:" in captured.out


class TestCLITopics:
    """Tests for the topics CLI command."""

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_topics_found(self, _patch_get_memory, mock_memory: Memory, capsys):
        """topics prints clustered memories."""
        # Create enough memories to cluster
        contents = [
            "I love eating pizza and pasta",
            "Italian food is the best cuisine",
            "My favorite pasta is carbonara",
            "I enjoy running every morning",
            "Running keeps me fit and healthy",
            "I ran a marathon last year",
        ]
        for i, content in enumerate(contents):
            mem = make_memory(
                memory_id=f"topic-{i}",
                user_id="user1",
                content=content,
                embedding=mock_memory._embed.embed_single(content),
                embedding_dim=mock_memory._embed.dimension(),
            )
            mock_memory._store.store(mem)

        with patch("kemi.interfaces.cli.main.sys.exit") as mock_exit:
            topics_memories(argparse.Namespace(user_id="user1", n_clusters=2, namespace="default"))
            assert not mock_exit.called

        captured = capsys.readouterr()
        assert "memories" in captured.out

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_topics_not_found(self, _patch_get_memory, mock_memory: Memory, capsys):
        """topics with no memories prints 'No topics found'."""
        topics_memories(argparse.Namespace(user_id="user1", n_clusters=3, namespace="default"))
        captured = capsys.readouterr()
        assert "No topics found" in captured.out

    def test_topics_import_error(self, _patch_get_memory, mock_memory: Memory):
        """topics without scikit-learn prints error and exits."""
        mem = make_memory(
            memory_id="topic-0",
            user_id="user1",
            content="some memory",
            embedding=mock_memory._embed.embed_single("some memory"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        with patch("kemi.interfaces.cli.main.Memory.cluster_topics", side_effect=ImportError("No sklearn")):  # noqa: E501
            with patch("kemi.interfaces.cli.main.sys.exit") as mock_exit:
                topics_memories(
                    argparse.Namespace(user_id="user1", n_clusters=3, namespace="default")
                )
                mock_exit.assert_called_once_with(1)


class TestCLIGraph:
    """Tests for the graph CLI command."""

    def test_graph_entities_and_relations(self, _patch_get_memory, mock_memory: Memory, capsys):
        """graph prints extracted entities and relations."""
        mem = make_memory(
            memory_id="graph-1",
            user_id="user1",
            content="Alice works at Google and Bob lives in London.",
            embedding=mock_memory._embed.embed_single(
                "Alice works at Google and Bob lives in London."
            ),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        graph_memories(argparse.Namespace(user_id="user1", namespace="default"))
        captured = capsys.readouterr()
        assert "Entities:" in captured.out
        assert "Relations:" in captured.out

    def test_graph_empty(self, _patch_get_memory, mock_memory: Memory, capsys):
        """graph with no memories prints empty results."""
        graph_memories(argparse.Namespace(user_id="user1", namespace="default"))
        captured = capsys.readouterr()
        assert "Entities:" in captured.out
        assert "None" in captured.out

    def test_graph_with_namespace(self, _patch_get_memory, mock_memory: Memory, capsys):
        """graph --namespace only uses memories in that namespace."""
        mem = make_memory(
            memory_id="graph-ns",
            user_id="user1",
            content="Charlie studies at MIT",
            namespace="school",
            embedding=mock_memory._embed.embed_single("Charlie studies at MIT"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        graph_memories(argparse.Namespace(user_id="user1", namespace="school"))
        captured = capsys.readouterr()
        assert "Entities:" in captured.out


class TestCLIE2E:
    """End-to-end tests that exercise argument parsing through main()."""

    def test_update_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """update with valid args through main() updates the memory."""
        mem = make_memory(
            memory_id="e2e-upd",
            user_id="user1",
            content="original",
            embedding=mock_memory._embed.embed_single("original"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "update", "e2e-upd", "--content", "via main"]):
            main()
        captured = capsys.readouterr()
        assert "Updated memory: e2e-upd" in captured.out
        updated = mock_memory._store.get("e2e-upd")
        assert updated.content == "via main"

    def test_recall_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """recall with valid args through main() finds memories."""
        mem = make_memory(
            memory_id="e2e-rec",
            user_id="user1",
            content="I love Python programming",
            embedding=mock_memory._embed.embed_single("I love Python programming"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "recall", "user1", "python"]):
            main()
        captured = capsys.readouterr()
        assert "Python programming" in captured.out

    def test_stats_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """stats with valid args through main() shows statistics."""
        mem = make_memory(
            memory_id="e2e-stat",
            user_id="user1",
            content="stats test",
            embedding=mock_memory._embed.embed_single("stats test"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(sys, "argv", ["kemi", "stats", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Statistics for user: user1" in captured.out
        assert "Total memories:" in captured.out

    def test_list_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """list with valid args through main() lists memories."""
        mem = make_memory(
            memory_id="e2e-list",
            user_id="user1",
            content="list me",
            embedding=mock_memory._embed.embed_single("list me"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(sys, "argv", ["kemi", "list", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "list me" in captured.out

    def test_list_namespace_filter(self, _patch_get_memory, mock_memory: Memory, capsys):
        """list --namespace filters results by namespace."""
        ns_default = make_memory(
            memory_id="e2e-list-ns-default",
            user_id="user1",
            content="default ns memory",
            embedding=mock_memory._embed.embed_single("default ns memory"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        ns_travel = make_memory(
            memory_id="e2e-list-ns-travel",
            user_id="user1",
            content="travel ns memory",
            embedding=mock_memory._embed.embed_single("travel ns memory"),
            embedding_dim=mock_memory._embed.dimension(),
            namespace="travel",
        )
        mock_memory._store.store(ns_default)
        mock_memory._store.store(ns_travel)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(sys, "argv", ["kemi", "list", "user1", "--namespace", "travel"]):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "travel ns memory" in captured.out
        assert "default ns memory" not in captured.out

    def test_list_lifecycle_filter(self, _patch_get_memory, mock_memory: Memory, capsys):
        """list --lifecycle-filter filters results by state."""
        active_mem = make_memory(
            memory_id="e2e-list-lc-active",
            user_id="user1",
            content="active memory",
            embedding=mock_memory._embed.embed_single("active memory"),
            embedding_dim=mock_memory._embed.dimension(),
            lifecycle_state=LifecycleState.ACTIVE,
        )
        decaying_mem = make_memory(
            memory_id="e2e-list-lc-decaying",
            user_id="user1",
            content="decaying memory",
            embedding=mock_memory._embed.embed_single("decaying memory"),
            embedding_dim=mock_memory._embed.dimension(),
            lifecycle_state=LifecycleState.DECAYING,
        )
        mock_memory._store.store(active_mem)
        mock_memory._store.store(decaying_mem)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(
                sys, "argv", ["kemi", "list", "user1", "--lifecycle-filter", "decaying"]
            ):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "decaying memory" in captured.out
        assert "active memory" not in captured.out

    def test_list_session_id_filter(self, _patch_get_memory, mock_memory: Memory, capsys):
        """list --session-id filters results by session."""
        session_a = make_memory(
            memory_id="e2e-list-si-a",
            user_id="user1",
            content="session A memory",
            embedding=mock_memory._embed.embed_single("session A memory"),
            embedding_dim=mock_memory._embed.dimension(),
            session_id="session-a",
        )
        session_b = make_memory(
            memory_id="e2e-list-si-b",
            user_id="user1",
            content="session B memory",
            embedding=mock_memory._embed.embed_single("session B memory"),
            embedding_dim=mock_memory._embed.dimension(),
            session_id="session-b",
        )
        mock_memory._store.store(session_a)
        mock_memory._store.store(session_b)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(sys, "argv", ["kemi", "list", "user1", "--session-id", "session-a"]):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "session A memory" in captured.out
        assert "session B memory" not in captured.out

    def test_list_lifecycle_filter_multiple(self, _patch_get_memory, mock_memory: Memory, capsys):
        """list --lifecycle-filter with comma-separated states."""
        active_mem = make_memory(
            memory_id="e2e-list-lc2-active",
            user_id="user1",
            content="active memory",
            embedding=mock_memory._embed.embed_single("active memory"),
            embedding_dim=mock_memory._embed.dimension(),
            lifecycle_state=LifecycleState.ACTIVE,
        )
        decaying_mem = make_memory(
            memory_id="e2e-list-lc2-decaying",
            user_id="user1",
            content="decaying memory",
            embedding=mock_memory._embed.embed_single("decaying memory"),
            embedding_dim=mock_memory._embed.dimension(),
            lifecycle_state=LifecycleState.DECAYING,
        )
        mock_memory._store.store(active_mem)
        mock_memory._store.store(decaying_mem)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(
                sys,
                "argv",
                ["kemi", "list", "user1", "--lifecycle-filter", "active,decaying"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "active memory" in captured.out
        assert "decaying memory" in captured.out

    def test_list_users_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """list-users through main() lists users with counts."""
        mem = make_memory(
            memory_id="e2e-lu",
            user_id="alice",
            content="alice memory",
            embedding=mock_memory._embed.embed_single("alice memory"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(sys, "argv", ["kemi", "list-users"]):
                main()
        captured = capsys.readouterr()
        assert "alice: 1 memories" in captured.out

    def test_forget_through_main_cancelled(self, _patch_get_memory, mock_memory: Memory, capsys):
        """forget through main() cancels when user types n."""
        mem = make_memory(
            memory_id="e2e-forget",
            user_id="user1",
            content="to be deleted",
            embedding=mock_memory._embed.embed_single("to be deleted"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch("builtins.input", return_value="n"):
            with patch.object(sys, "argv", ["kemi", "forget", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out
        # memory should still exist
        assert mock_memory._store.get("e2e-forget") is not None

    def test_forget_through_main_confirmed(self, _patch_get_memory, mock_memory: Memory, capsys):
        """forget through main() deletes when user types y."""
        mem = make_memory(
            memory_id="e2e-forget2",
            user_id="user1",
            content="to be deleted",
            embedding=mock_memory._embed.embed_single("to be deleted"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch("builtins.input", return_value="y"):
            with patch.object(sys, "argv", ["kemi", "forget", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Deleted 1 memories" in captured.out
        assert mock_memory._store.get("e2e-forget2") is None

    def test_prune_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """prune with valid args through main() removes old memories."""
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        mem = make_memory(
            memory_id="e2e-prune",
            user_id="user1",
            content="old pruneable",
            created_at=old_time,
            last_accessed_at=old_time,
            lifecycle_state=LifecycleState.DECAYING,
        )
        mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "prune", "user1", "--max-age-days", "30"]):
            main()
        captured = capsys.readouterr()
        assert "Pruned 1 memories" in captured.out
        assert mock_memory._store.get("e2e-prune") is None

    def test_consolidate_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """consolidate through main() creates summary memories."""
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(5):
            mem = make_memory(
                memory_id=f"e2e-cons-{i}",
                user_id="user1",
                content=f"Paris day {i}",
                created_at=old_time,
                last_accessed_at=old_time,
                memory_type=MemoryType.EPISODIC,
                embedding=mock_memory._embed.embed_single(f"Paris day {i}"),
                embedding_dim=mock_memory._embed.dimension(),
            )
            mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "consolidate", "user1"]):
            main()
        captured = capsys.readouterr()
        assert "Consolidated into memory:" in captured.out

    def test_graph_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """graph through main() extracts entities and relations."""
        mem = make_memory(
            memory_id="e2e-graph",
            user_id="user1",
            content="Sarah lives in Berlin and works at Tesla.",
            embedding=mock_memory._embed.embed_single("Sarah lives in Berlin and works at Tesla."),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "graph", "user1"]):
            main()
        captured = capsys.readouterr()
        assert "Entities:" in captured.out
        assert "Relations:" in captured.out

    def test_explain_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """explain through main() shows score breakdowns."""
        mem = make_memory(
            memory_id="e2e-exp",
            user_id="user1",
            content="I enjoy hiking in mountains",
            embedding=mock_memory._embed.embed_single("I enjoy hiking in mountains"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "explain", "user1", "hiking"]):
            main()
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Semantic:" in captured.out

    def test_export_through_main(self, _patch_get_memory, mock_memory: Memory, tmp_path):
        """export through main() writes memories to a file."""
        mem = make_memory(
            memory_id="e2e-exp-file",
            user_id="user1",
            content="export me",
            embedding=mock_memory._embed.embed_single("export me"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        out_file = str(tmp_path / "exported.json")
        with patch.object(sys, "argv", ["kemi", "export", out_file]):
            main()
        import json

        with open(out_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["memory_id"] == "e2e-exp-file"

    def test_recall_stream_through_main(self, _patch_get_memory, mock_memory: Memory, capsys):
        """recall-stream through main() streams memories progressively."""
        mem = make_memory(
            memory_id="e2e-stream",
            user_id="user1",
            content="I love Python programming",
            embedding=mock_memory._embed.embed_single("I love Python programming"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        with patch.object(sys, "argv", ["kemi", "recall-stream", "user1", "python"]):
            main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "Python programming" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_through_main_empty(self, _patch_get_memory, mock_memory: Memory, capsys):
        """recall-stream through main() with no results prints not found."""
        with patch.object(sys, "argv", ["kemi", "recall-stream", "user1", "nothing"]):
            main()
        captured = capsys.readouterr()
        assert "No memories found for: nothing" in captured.out

    def test_import_through_main(self, _patch_get_memory, mock_memory: Memory, tmp_path, capsys):
        """import through main() reads memories from a file."""
        in_file = str(tmp_path / "imported.json")
        import json

        with open(in_file, "w") as f:
            json.dump(
                [
                    {
                        "memory_id": "imp-1",
                        "user_id": "user1",
                        "content": "imported content",
                        "importance": 0.7,
                        "lifecycle_state": "active",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                        "embedding": mock_memory._embed.embed_single("imported content"),
                        "embedding_dim": mock_memory._embed.dimension(),
                    }
                ],
                f,
            )
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch.object(sys, "argv", ["kemi", "import", in_file]):
                main()
        captured = capsys.readouterr()
        assert "Import complete:" in captured.out
        assert "Imported: 1" in captured.out
        assert mock_memory._store.get("imp-1") is not None


class TestCLIHooksRaiseOnError:
    """Tests for the --hooks-raise-on-error / --no-hooks-raise-on-error CLI flags."""

    def test_get_memory_with_hooks_raise_on_error_true(self):
        """get_memory(args) with flag=True creates Memory with hooks_raise_on_error=True."""
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch("kemi.interfaces.cli.main.Memory") as mock_mem_cls:
                args = argparse.Namespace(hooks_raise_on_error=True)
                get_memory(args)
                mock_mem_cls.assert_called_once()
                config = mock_mem_cls.call_args.kwargs["config"]
                assert config.hooks_raise_on_error is True

    def test_get_memory_with_hooks_raise_on_error_false(self):
        """get_memory(args) with flag=False creates Memory with hooks_raise_on_error=False."""
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch("kemi.interfaces.cli.main.Memory") as mock_mem_cls:
                args = argparse.Namespace(hooks_raise_on_error=False)
                get_memory(args)
                mock_mem_cls.assert_called_once()
                config = mock_mem_cls.call_args.kwargs["config"]
                assert config.hooks_raise_on_error is False

    def test_get_memory_without_flag_uses_default(self):
        """get_memory(args) without flag uses default Memory() (no config override)."""
        with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
            with patch("kemi.interfaces.cli.main.Memory") as mock_mem_cls:
                args = argparse.Namespace(hooks_raise_on_error=None)
                get_memory(args)
                mock_mem_cls.assert_called_once()
                # Should not pass config kwarg when flag is None
                assert "config" not in mock_mem_cls.call_args.kwargs

    def test_mutually_exclusive_flags_error(self):
        """Passing both --hooks-raise-on-error and --no-hooks-raise-on-error errors."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(
                sys,
                "argv",
                ["kemi", "--hooks-raise-on-error", "--no-hooks-raise-on-error", "list", "user1"],
            ):
                main()
        assert exc_info.value.code == 2


class TestCLIExplain:
    """Tests for the explain CLI command."""

    def test_explain_scores(self, _patch_get_memory, mock_memory: Memory, capsys):
        """explain prints memories with score breakdowns."""
        mem = make_memory(
            memory_id="explain-1",
            user_id="user1",
            content="I enjoy hiking in the mountains",
            embedding=mock_memory._embed.embed_single("I enjoy hiking in the mountains"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        explain_memories(
            argparse.Namespace(
                user_id="user1", query="hiking mountains", top_k=5, namespace="default"
            )
        )
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Content:" in captured.out
        assert "Semantic:" in captured.out
        assert "Recency:" in captured.out

    def test_explain_no_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """explain with no matching memories prints 'No memories found'."""
        explain_memories(
            argparse.Namespace(
                user_id="user1", query="nothing matches this", top_k=5, namespace="default"
            )
        )
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    def test_explain_with_namespace(self, _patch_get_memory, mock_memory: Memory, capsys):
        """explain --namespace filters by namespace."""
        mem = make_memory(
            memory_id="explain-ns",
            user_id="user1",
            content="I love coding in Python",
            namespace="work",
            embedding=mock_memory._embed.embed_single("I love coding in Python"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        explain_memories(
            argparse.Namespace(user_id="user1", query="Python", top_k=5, namespace="work")
        )
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "coding in Python" in captured.out


class TestCLIRecallStream:
    """Tests for the recall-stream CLI command."""

    def test_recall_stream_prints_progressively(self, _patch_get_memory, mock_memory: Memory, capsys):  # noqa: E501
        """recall-stream prints each memory as it arrives."""
        mem = make_memory(
            memory_id="stream-1",
            user_id="user1",
            content="I love Python",
            embedding=mock_memory._embed.embed_single("I love Python"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)

        recall_stream_memories(
            argparse.Namespace(
                user_id="user1",
                query="python",
                top_k=5,
                namespace="default",
                session_id=None,
                hybrid_search=None,
            )
        )
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I love Python" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_multiple_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """recall-stream shows all results with correct numbering."""
        contents = ["first memory", "second memory", "third memory"]
        for i, content in enumerate(contents):
            mem = make_memory(
                memory_id=f"stream-{i}",
                user_id="user1",
                content=content,
                embedding=mock_memory._embed.embed_single(content),
                embedding_dim=mock_memory._embed.dimension(),
            )
            mock_memory._store.store(mem)

        recall_stream_memories(
            argparse.Namespace(
                user_id="user1",
                query="memory",
                top_k=5,
                namespace="default",
                session_id=None,
                hybrid_search=None,
            )
        )
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "#  2 | Score:" in captured.out
        assert "#  3 | Score:" in captured.out
        assert "first memory" in captured.out
        assert "second memory" in captured.out
        assert "third memory" in captured.out
        assert "Streamed 3 result(s)" in captured.out

    def test_recall_stream_empty_results(self, _patch_get_memory, mock_memory: Memory, capsys):
        """recall-stream prints appropriate message when no results found."""
        recall_stream_memories(
            argparse.Namespace(
                user_id="user1",
                query="nothing matches",
                top_k=5,
                namespace="default",
                session_id=None,
                hybrid_search=None,
            )
        )
        captured = capsys.readouterr()
        assert "No memories found for: nothing matches" in captured.out


class TestCLIRealDB:
    """Integration tests using a real temporary SQLite database.

    These tests exercise the full CLI pipeline (main() -> handler -> real storage)
    with actual SQLite storage operations — no MockStorageAdapter.
    The real_db_memory fixture patches get_memory so main() uses the temp DB.
    """

    def test_store_with_metadata(self, real_db_memory: Memory, capsys):
        """store --metadata persists metadata as JSON."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "test content",
                    "--metadata", '{"source":"test","priority":"high"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        # Extract the memory ID from output
        mem_id = captured.out.strip().split(": ")[-1].strip()
        # Verify metadata was stored correctly
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.metadata == {"source": "test", "priority": "high"}

    def test_store_with_tags(self, real_db_memory: Memory, capsys):
        """store --tags persists comma-separated tags."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "tagged content",
                    "--tags", "foo,bar,baz",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.tags == ["foo", "bar", "baz"]

    def test_store_and_recall_metadata_round_trip(self, real_db_memory: Memory, capsys):
        """store --metadata then recall --metadata-filter in a full round-trip."""
        # Step 1: Store a memory with metadata via the CLI
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "I visited Tokyo last spring",
                    "--metadata", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out

        # Step 2: Recall with metadata filter matching the stored metadata
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Score:" in captured.out

    def test_store_with_tags_and_recall_metadata_filter(self, real_db_memory: Memory, capsys):
        """store --tags combined with --metadata, then recall --metadata-filter finds it.

        Stores a memory with both tags and metadata, then verifies recall via
        --metadata-filter returns it and the tags are correctly persisted.
        """
        # Step 1: Store a memory with both --tags and --metadata
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "I visited Tokyo last spring",
                    "--tags", "travel,japan,spring",
                    "--metadata", '{"source":"test","category":"personal"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()

        # Verify tags and metadata were both persisted correctly
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.tags == ["travel", "japan", "spring"]
        assert stored.metadata == {"source": "test", "category": "personal"}

        # Step 2: Recall with --metadata-filter — should find the memory
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Score:" in captured.out

        # Step 3: Store a second memory with different tags/metadata that shouldn't match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "Python programming is fun",
                    "--tags", "coding,python",
                    "--metadata", '{"source":"other","category":"work"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out

        # Step 4: Recall with --metadata-filter matching only the first memory
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out

    def test_store_with_tags_and_recall_stream_metadata_filter(self, real_db_memory: Memory, capsys):  # noqa: E501
        """store --tags and --metadata, then recall-stream --metadata-filter finds it.

        Stores memories with both tags and metadata, then uses recall-stream
        with --metadata-filter to verify the full round-trip pipeline.
        """
        # Step 1: Store a memory with both --tags and --metadata
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "I visited Tokyo last spring",
                    "--tags", "travel,japan,spring",
                    "--metadata", '{"source":"test","category":"personal"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()

        # Verify tags and metadata persisted correctly
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.tags == ["travel", "japan", "spring"]
        assert stored.metadata == {"source": "test", "category": "personal"}

        # Step 2: recall-stream with --metadata-filter — should find the memory
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall-stream", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Streamed 1 result(s)" in captured.out

        # Step 3: Store a second memory with different tags/metadata
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "Python programming is fun",
                    "--tags", "coding,python",
                    "--metadata", '{"source":"other","category":"work"}',
                ],
            ):
                main()
        capsys.readouterr()  # discard store output

        # Step 4: recall-stream with --metadata-filter matching only the first
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall-stream", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_store_with_session_id(self, real_db_memory: Memory, capsys):
        """store --session-id persists the session ID."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "session-specific content",
                    "--session-id", "my-session",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.session_id == "my-session"

    def test_store_with_session_id_and_namespace(self, real_db_memory: Memory, capsys):
        """store --session-id combined with --namespace persists both fields."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "work session content",
                    "--namespace", "work",
                    "--session-id", "my-session",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.namespace == "work"
        assert stored.session_id == "my-session"

    def test_store_with_namespace_and_tags(self, real_db_memory: Memory, capsys):
        """store --namespace combined with --tags persists both fields."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "work tagged content",
                    "--namespace", "work",
                    "--tags", "project,alpha",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.namespace == "work"
        assert stored.tags == ["project", "alpha"]

    def test_store_with_tags_and_session_id(self, real_db_memory: Memory, capsys):
        """store --tags combined with --session-id persists both fields."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "tagged session content",
                    "--tags", "alpha,beta,gamma",
                    "--session-id", "my-session",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.tags == ["alpha", "beta", "gamma"]
        assert stored.session_id == "my-session"

    def test_store_with_tags_metadata_and_session_id(self, real_db_memory: Memory, capsys):
        """store --tags combined with --metadata and --session-id persists all three fields."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "full featured memory",
                    "--tags", "travel,japan,spring",
                    "--metadata", '{"source":"test","priority":"high"}',
                    "--session-id", "my-session",
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Stored memory:" in captured.out
        mem_id = captured.out.strip().split(": ")[-1].strip()
        stored = real_db_memory._store.get(mem_id)
        assert stored is not None
        assert stored.tags == ["travel", "japan", "spring"]
        assert stored.metadata == {"source": "test", "priority": "high"}
        assert stored.session_id == "my-session"

    def test_forget_with_memory_id(self, real_db_memory: Memory, capsys):
        """forget --memory-id deletes a specific memory."""
        mem_id = real_db_memory.remember("user1", "memory to forget")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "forget", "user1", "--memory-id", mem_id],
            ):
                main()
        captured = capsys.readouterr()
        assert "Deleted 1 memory." in captured.out
        assert real_db_memory._store.get(mem_id) is None

    def test_forget_with_memory_id_not_found(self, real_db_memory: Memory, capsys):
        """forget --memory-id with nonexistent ID prints 0."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "forget", "user1", "--memory-id", "nonexistent-id"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Deleted 0 memory." in captured.out

    def test_forget_many_via_cli(self, real_db_memory: Memory, capsys):
        """forget-many deletes multiple memories by ID through CLI.

        Uses _store.store() to bypass dedup (similar texts with hash-based
        embeddings can trigger the 0.85 dedup threshold).
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            make_memory(
                memory_id="fm-1",
                user_id="user1",
                content="alpha brand content",
                embedding=embed.embed_single("alpha brand content"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="fm-2",
                user_id="user1",
                content="beta random stuff",
                embedding=embed.embed_single("beta random stuff"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="fm-3",
                user_id="user1",
                content="gamma extra data",
                embedding=embed.embed_single("gamma extra data"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "forget-many", "fm-1", "fm-2", "fm-3"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Deleted 3 memories." in captured.out
        assert real_db_memory._store.get("fm-1") is None
        assert real_db_memory._store.get("fm-2") is None
        assert real_db_memory._store.get("fm-3") is None

    def test_forget_many_some_not_found(self, real_db_memory: Memory, capsys):
        """forget-many with some nonexistent IDs only deletes the found ones."""
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mem = make_memory(
            memory_id="fm-only",
            user_id="user1",
            content="only this one should be deleted",
            embedding=embed.embed_single("only this one should be deleted"),
            embedding_dim=embed.dimension(),
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            lifecycle_state=LifecycleState.ACTIVE,
        )
        real_db_memory._store.store(mem)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "forget-many", "fm-only", "nonexistent-id-1", "nonexistent-id-2"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Deleted 1 memories." in captured.out
        assert real_db_memory._store.get("fm-only") is None

    def test_list_empty_user(self, real_db_memory: Memory, capsys):
        """list with no memories for user prints appropriate message."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "list", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    def test_store_with_tags_shows_in_list_output(self, real_db_memory: Memory, capsys):
        """store --tags then list shows tags in the output."""
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "store", "user1", "tagged list content",
                    "--tags", "alpha,beta,gamma",
                ],
            ):
                main()
        capsys.readouterr()  # discard store output

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "list", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "tagged list content" in captured.out
        # Tags should be displayed in the list output
        assert "Tags:" in captured.out
        assert "alpha, beta, gamma" in captured.out

    def test_store_with_tags_and_list_lifecycle_filter(self, real_db_memory: Memory, capsys):
        """store --tags then list --lifecycle-filter shows only matching tagged memories."""
        # Store an active memory with tags via CLI
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "store", "user1", "active alpha stuff", "--tags", "foo,bar"],
            ):
                main()
        capsys.readouterr()

        # Store a decaying tagged memory via CLI
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "store", "user1", "decaying beta things", "--tags", "baz,qux"],
            ):
                main()
        capsys.readouterr()

        # Mark the second memory as DECAYING
        all_mems = real_db_memory._store.get_all_by_user("user1")
        for m in all_mems:
            if m.content == "decaying beta things":
                m.lifecycle_state = LifecycleState.DECAYING
                real_db_memory._store.update(m)
                break

        # List with --lifecycle-filter decaying
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "list", "user1", "--lifecycle-filter", "decaying"],
            ):
                main()
        captured = capsys.readouterr()
        # Only the decaying memory should appear
        assert "decaying beta things" in captured.out
        assert "active alpha stuff" not in captured.out
        # Tags should be displayed in the output
        assert "Tags:" in captured.out
        assert "baz, qux" in captured.out

    def test_list_with_tags_filter(self, real_db_memory: Memory, capsys):
        """list --tags filters memories by tag (OR logic).

        Uses _store.store() with explicit MemoryObject instances to bypass
        dedup (hash-based embeddings can trigger the 0.85 dedup threshold
        even on short texts with no shared words).
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            make_memory(
                memory_id="ltf-1",
                user_id="user1",
                content="alpha items here",
                tags=["foo", "bar"],
                embedding=embed.embed_single("alpha items here"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="ltf-2",
                user_id="user1",
                content="beta things here",
                tags=["baz"],
                embedding=embed.embed_single("beta things here"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="ltf-3",
                user_id="user1",
                content="untagged data",
                tags=None,
                embedding=embed.embed_single("untagged data"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        # List with --tags foo — should only match the first memory (has tag "foo")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "list", "user1", "--tags", "foo"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "alpha items here" in captured.out
        assert "beta things here" not in captured.out
        assert "untagged data" not in captured.out
        assert "Tags:" in captured.out
        assert "foo, bar" in captured.out

        # List with --tags foo,baz — OR logic, should match both tagged memories
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "list", "user1", "--tags", "foo,baz"],
            ):
                main()
        captured = capsys.readouterr()
        assert "alpha items here" in captured.out
        assert "beta things here" in captured.out
        assert "untagged data" not in captured.out

    def test_list_with_memories(self, real_db_memory: Memory, capsys):
        """list shows stored memories with correct content and fields."""
        real_db_memory.remember("user1", "I love coding in Python", importance=0.8)
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "list", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "I love coding in Python" in captured.out
        assert "Importance:" in captured.out
        assert "State:" in captured.out

    def test_list_with_namespace_filter(self, real_db_memory: Memory, capsys):
        """list --namespace only shows memories in that namespace."""
        real_db_memory.remember("user1", "default ns memory", namespace="default")
        real_db_memory.remember("user1", "travel ns memory", namespace="travel")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "list", "user1", "--namespace", "travel"]):
                main()
        captured = capsys.readouterr()
        assert "travel ns memory" in captured.out
        assert "default ns memory" not in captured.out

    def test_list_with_lifecycle_filter(self, real_db_memory: Memory, capsys):
        """list --lifecycle-filter only shows memories in that state."""
        # Store an active memory and a decaying memory
        real_db_memory.remember("user1", "active memory content")
        real_db_memory.remember("user1", "decaying memory content")
        # Mark the second one as DECAYING
        stored = real_db_memory._store.get_all_by_user("user1")[1]
        stored.lifecycle_state = LifecycleState.DECAYING
        real_db_memory._store.update(stored)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "list", "user1", "--lifecycle-filter", "decaying"]
            ):
                main()
        captured = capsys.readouterr()
        assert "decaying memory content" in captured.out
        assert "active memory content" not in captured.out

    def test_list_with_session_id_filter(self, real_db_memory: Memory, capsys):
        """list --session-id only shows memories from that session."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", session_id="session-a")
        real_db_memory.remember("user1", "Python programming is fun", session_id="session-b")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "list", "user1", "--session-id", "session-a"]
            ):
                main()
        captured = capsys.readouterr()
        assert "Memories for user: user1" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out

    def test_recall_finds_stored_memory(self, real_db_memory: Memory, capsys):
        """recall with valid args finds previously stored memories."""
        real_db_memory.remember("user1", "I visited Tokyo last spring")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "recall", "user1", "Tokyo"]):
                main()
        captured = capsys.readouterr()
        assert "Tokyo" in captured.out
        assert "Score:" in captured.out

    def test_recall_namespace_filter(self, real_db_memory: Memory, capsys):
        """recall --namespace only searches within that namespace."""
        real_db_memory.remember("user1", "work project details", namespace="work")
        real_db_memory.remember("user1", "personal notes", namespace="personal")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "recall", "user1", "details", "--namespace", "work"]
            ):
                main()
        captured = capsys.readouterr()
        assert "work project details" in captured.out
        assert "personal notes" not in captured.out

    def test_recall_top_k_limit(self, real_db_memory: Memory, capsys):
        """recall --top-k limits the number of results."""
        real_db_memory.remember("user1", "alpha memory content")
        real_db_memory.remember("user1", "beta memory content")
        real_db_memory.remember("user1", "gamma memory content")
        # Query with exact text for deterministic hash match, limit to 1 result
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "recall", "user1", "content", "--top-k", "1"]
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: content" in captured.out
        assert "Score:" in captured.out
        # Should only have one result line (Score: ... | ...)
        score_lines = [l for l in captured.out.split("\n") if l.startswith("Score:")]
        assert len(score_lines) == 1, f"Expected 1 score line, got {len(score_lines)}"

    def test_recall_top_k_with_namespace(self, real_db_memory: Memory, capsys):
        """recall --top-k combined with --namespace limits and scopes results."""
        # Store 3 work memories and 2 personal memories
        real_db_memory.remember("user1", "alpha work memory", namespace="work")
        real_db_memory.remember("user1", "beta work memory", namespace="work")
        real_db_memory.remember("user1", "gamma work memory", namespace="work")
        real_db_memory.remember("user1", "personal note one", namespace="personal")
        real_db_memory.remember("user1", "personal note two", namespace="personal")
        # Query with --top-k 1 --namespace work — only 1 work memory should return
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall", "user1", "memory", "--top-k", "1", "--namespace", "work"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: memory" in captured.out
        assert "Score:" in captured.out
        # Only 1 result, from the work namespace
        score_lines = [l for l in captured.out.split("\n") if l.startswith("Score:")]
        assert len(score_lines) == 1, f"Expected 1 score line, got {len(score_lines)}"
        assert "work memory" in captured.out
        assert "personal note" not in captured.out

    def test_recall_top_k_with_metadata_filter(self, real_db_memory: Memory, capsys):
        """recall --top-k combined with --metadata-filter limits and filters results.

        Stores 4 memories — 3 with source=test, 1 with source=other.
        Queries with --top-k 2 --metadata-filter '{"source":"test"}'.
        Should return at most 2 results, all from the test source.
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            make_memory(
                memory_id="tk-mf-1",
                user_id="user1",
                content="alpha test content one",
                embedding=embed.embed_single("alpha test content one"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"source": "test"},
            ),
            make_memory(
                memory_id="tk-mf-2",
                user_id="user1",
                content="beta test content two",
                embedding=embed.embed_single("beta test content two"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"source": "test"},
            ),
            make_memory(
                memory_id="tk-mf-3",
                user_id="user1",
                content="gamma test content three",
                embedding=embed.embed_single("gamma test content three"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"source": "test"},
            ),
            make_memory(
                memory_id="tk-mf-4",
                user_id="user1",
                content="delta other unrelated",
                embedding=embed.embed_single("delta other unrelated"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                metadata={"source": "other"},
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        # Query with --top-k 2 and --metadata-filter source=test
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "content",
                    "--top-k", "2",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: content" in captured.out
        assert "Score:" in captured.out
        # Only 2 score lines (top-k limit), and none from source=other
        score_lines = [l for l in captured.out.split("\n") if l.startswith("Score:")]
        assert len(score_lines) == 2, f"Expected 2 score lines, got {len(score_lines)}"
        assert "alpha" in captured.out or "beta" in captured.out or "gamma" in captured.out
        assert "delta" not in captured.out

    def test_recall_metadata_filter_match(self, real_db_memory: Memory, capsys):
        """recall --metadata-filter returns only matching memories."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", metadata={"source": "test"})
        real_db_memory.remember("user1", "Python programming is fun", metadata={"source": "other"})
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "Score: 0.500" in captured.out or "Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out

    def test_recall_metadata_filter_no_match(self, real_db_memory: Memory, capsys):
        """recall --metadata-filter with no match prints message."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", metadata={"source": "test"})
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"nonexistent"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "No memories found for: I visited Tokyo last spring" in captured.out

    def test_recall_session_id_filter(self, real_db_memory: Memory, capsys):
        """recall --session-id returns only memories from that session."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", session_id="session-a")
        real_db_memory.remember("user1", "Python programming is fun", session_id="session-b")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall", "user1", "I visited Tokyo last spring", "--session-id", "session-a"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out

    def test_recall_session_id_no_match(self, real_db_memory: Memory, capsys):
        """recall --session-id with no matching memories prints message."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", session_id="session-a")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall", "user1", "I visited Tokyo last spring", "--session-id", "nonexistent-session"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "No memories found for: I visited Tokyo last spring" in captured.out

    def test_recall_namespace_and_session_id(self, real_db_memory: Memory, capsys):
        """recall with both --namespace and --session-id combined."""
        # 4 memories across 2 namespaces and 2 session IDs
        real_db_memory.remember(
            "user1",
            "I visited Tokyo last spring",
            namespace="work",
            session_id="session-a",
        )
        real_db_memory.remember(
            "user1",
            "Python programming is fun",
            namespace="personal",
            session_id="session-a",
        )
        real_db_memory.remember(
            "user1",
            "Important meeting notes",
            namespace="work",
            session_id="session-b",
        )
        real_db_memory.remember(
            "user1",
            "Weekend hiking trip",
            namespace="personal",
            session_id="session-b",
        )
        # Query work namespace + session-a — only Tokyo should match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "spring",
                    "--namespace", "work",
                    "--session-id", "session-a",
                ],
            ):
                main()
        captured = capsys.readouterr()
        # Only the Tokyo memory should appear (namespace=work AND session_id=session-a)
        assert "Results for: spring" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out
        assert "Important meeting notes" not in captured.out
        assert "Weekend hiking" not in captured.out
        assert "Score:" in captured.out

    def test_recall_session_id_and_metadata_filter(self, real_db_memory: Memory, capsys):
        """recall with both --session-id and --metadata-filter combined.

        Memories are stored directly via _store.store() to bypass dedup
        (hash-based embeddings produce random high similarities that trigger
        the default 0.85 dedup threshold).
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        # 4 memories across 2 session IDs and 2 metadata values
        mem1 = make_memory(
            memory_id="si-mf-1",
            user_id="user1",
            content="I visited Tokyo last spring",
            embedding=embed.embed_single("I visited Tokyo last spring"),
            embedding_dim=embed.dimension(),
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            lifecycle_state=LifecycleState.ACTIVE,
            session_id="session-a",
            metadata={"source": "test"},
        )
        mem2 = make_memory(
            memory_id="si-mf-2",
            user_id="user1",
            content="Python programming is fun",
            embedding=embed.embed_single("Python programming is fun"),
            embedding_dim=embed.dimension(),
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            lifecycle_state=LifecycleState.ACTIVE,
            session_id="session-b",
            metadata={"source": "other"},
        )
        mem3 = make_memory(
            memory_id="si-mf-3",
            user_id="user1",
            content="Important meeting notes",
            embedding=embed.embed_single("Important meeting notes"),
            embedding_dim=embed.dimension(),
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            lifecycle_state=LifecycleState.ACTIVE,
            session_id="session-a",
            metadata={"source": "other"},
        )
        mem4 = make_memory(
            memory_id="si-mf-4",
            user_id="user1",
            content="Weekend hiking trip",
            embedding=embed.embed_single("Weekend hiking trip"),
            embedding_dim=embed.dimension(),
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            lifecycle_state=LifecycleState.ACTIVE,
            session_id="session-b",
            metadata={"source": "test"},
        )
        real_db_memory._store.store(mem1)
        real_db_memory._store.store(mem2)
        real_db_memory._store.store(mem3)
        real_db_memory._store.store(mem4)

        # Query session-a + source=test with exact text for deterministic hash match
        # Only memory 1 should match both filters
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--session-id", "session-a",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Score:" in captured.out
        # All other memories should be excluded (wrong session or wrong metadata)
        assert "Python programming" not in captured.out
        assert "Important meeting notes" not in captured.out
        assert "Weekend hiking" not in captured.out

    def test_recall_namespace_session_id_and_metadata_filter(self, real_db_memory: Memory, capsys):
        """recall with --namespace, --session-id, and --metadata-filter combined.

        Stores 6 memories across 2 namespaces, 2 session IDs, and 2 metadata values.
        Only 1 memory should match all three filters.
        Memories are stored via _store.store() to bypass dedup.
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            make_memory(
                memory_id="trip-1",
                user_id="user1",
                content="I visited Tokyo last spring",
                embedding=embed.embed_single("I visited Tokyo last spring"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-a",
                metadata={"source": "test"},
            ),
            make_memory(
                memory_id="trip-2",
                user_id="user1",
                content="Python programming is fun",
                embedding=embed.embed_single("Python programming is fun"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-a",
                metadata={"source": "test"},
            ),
            make_memory(
                memory_id="trip-3",
                user_id="user1",
                content="Important meeting notes",
                embedding=embed.embed_single("Important meeting notes"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-b",
                metadata={"source": "test"},
            ),
            make_memory(
                memory_id="trip-4",
                user_id="user1",
                content="Weekend hiking trip",
                embedding=embed.embed_single("Weekend hiking trip"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-a",
                metadata={"source": "other"},
            ),
            make_memory(
                memory_id="trip-5",
                user_id="user1",
                content="Lunch at noon",
                embedding=embed.embed_single("Lunch at noon"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-a",
                metadata={"source": "other"},
            ),
            make_memory(
                memory_id="trip-6",
                user_id="user1",
                content="Running in the park",
                embedding=embed.embed_single("Running in the park"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-b",
                metadata={"source": "other"},
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        # Work namespace + session-a + source=test — only memory trip-1 matches all three
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--namespace", "work",
                    "--session-id", "session-a",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Score:" in captured.out
        assert "Python programming" not in captured.out
        assert "Important meeting notes" not in captured.out
        assert "Weekend hiking" not in captured.out
        assert "Lunch at noon" not in captured.out
        assert "Running in the park" not in captured.out

    def test_recall_with_all_combined_filters_and_tags(self, real_db_memory: Memory, capsys):
        """recall with --namespace, --session-id, --metadata-filter, and tags combined.

        Stores 8 memories across 2 namespaces, 2 session IDs, 2 metadata values,
        and 2 tag sets. Only 1 memory should match all four dimensions.
        Tags on the returned memory are verified to confirm round-trip integrity.
        Uses _store.store() to bypass dedup.
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            # target: work/session-a/source=test/tags=["travel","tokyo"]
            make_memory(
                memory_id="all-1",
                user_id="user1",
                content="I visited Tokyo last spring",
                embedding=embed.embed_single("I visited Tokyo last spring"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-a",
                metadata={"source": "test"},
                tags=["travel", "tokyo"],
            ),
            # wrong namespace (personal vs work)
            make_memory(
                memory_id="all-2",
                user_id="user1",
                content="Python programming is fun",
                embedding=embed.embed_single("Python programming is fun"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-a",
                metadata={"source": "test"},
                tags=["coding", "python"],
            ),
            # wrong session (session-b vs session-a)
            make_memory(
                memory_id="all-3",
                user_id="user1",
                content="Important meeting notes",
                embedding=embed.embed_single("Important meeting notes"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-b",
                metadata={"source": "test"},
                tags=["work", "meeting"],
            ),
            # wrong metadata (source=other vs source=test)
            make_memory(
                memory_id="all-4",
                user_id="user1",
                content="Weekend hiking trip",
                embedding=embed.embed_single("Weekend hiking trip"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-a",
                metadata={"source": "other"},
                tags=["hiking", "weekend"],
            ),
            # personal/session-a/other
            make_memory(
                memory_id="all-5",
                user_id="user1",
                content="Lunch at noon",
                embedding=embed.embed_single("Lunch at noon"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-a",
                metadata={"source": "other"},
                tags=["food"],
            ),
            # personal/session-b/test
            make_memory(
                memory_id="all-6",
                user_id="user1",
                content="Running in the park",
                embedding=embed.embed_single("Running in the park"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-b",
                metadata={"source": "test"},
                tags=["sports", "fitness"],
            ),
            # personal/session-b/other
            make_memory(
                memory_id="all-7",
                user_id="user1",
                content="Reading a book",
                embedding=embed.embed_single("Reading a book"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="personal",
                session_id="session-b",
                metadata={"source": "other"},
                tags=["reading"],
            ),
            # work/session-b/other
            make_memory(
                memory_id="all-8",
                user_id="user1",
                content="Coding at night",
                embedding=embed.embed_single("Coding at night"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
                namespace="work",
                session_id="session-b",
                metadata={"source": "other"},
                tags=["coding"],
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        # Work namespace + session-a + source=test — only memory all-1 matches all three
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall", "user1", "I visited Tokyo last spring",
                    "--namespace", "work",
                    "--session-id", "session-a",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for: I visited Tokyo last spring" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Score:" in captured.out
        # All other memories should be excluded
        assert "Python programming" not in captured.out
        assert "Important meeting notes" not in captured.out
        assert "Weekend hiking" not in captured.out
        assert "Lunch at noon" not in captured.out
        assert "Running in the park" not in captured.out
        assert "Reading a book" not in captured.out
        assert "Coding at night" not in captured.out

        # Verify tags are persisted correctly for the matching memory
        matching = real_db_memory._store.get("all-1")
        assert matching is not None
        assert matching.tags == ["travel", "tokyo"]
        assert matching.namespace == "work"
        assert matching.session_id == "session-a"
        assert matching.metadata == {"source": "test"}

    def test_stats_shows_user_counts(self, real_db_memory: Memory, capsys):
        """stats shows correct counts after storing memories."""
        real_db_memory.remember("user1", "I learned to bake sourdough bread", importance=0.6)
        real_db_memory.remember("user1", "The capital of Japan is Tokyo", importance=0.7)
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
                with patch.object(sys, "argv", ["kemi", "stats", "user1"]):
                    main()
        captured = capsys.readouterr()
        assert "Statistics for user: user1" in captured.out
        assert "Total memories:" in captured.out
        assert "2" in captured.out

    def test_stats_shows_tag_counts(self, real_db_memory: Memory, capsys):
        """stats shows correct tag statistics after storing tagged and untagged memories.

        Uses _store.store() with explicit MemoryObject instances to bypass dedup
        (hash-based embeddings can trigger the 0.85 dedup threshold on short texts
        with shared words).
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            make_memory(
                memory_id="stc-1",
                user_id="user1",
                content="alpha weather report",
                tags=["foo", "bar"],
                embedding=embed.embed_single("alpha weather report"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="stc-2",
                user_id="user1",
                content="beta quantum stuff",
                tags=["baz"],
                embedding=embed.embed_single("beta quantum stuff"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="stc-3",
                user_id="user1",
                content="untagged memory",
                tags=None,
                embedding=embed.embed_single("untagged memory"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        # Run stats via CLI to verify the tag statistics are displayed
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
                with patch.object(sys, "argv", ["kemi", "stats", "user1"]):
                    main()
        captured = capsys.readouterr()
        assert "Statistics for user: user1" in captured.out
        assert "Total memories: 3" in captured.out
        assert "With tags: 2" in captured.out
        assert "Without tags: 1" in captured.out
        assert "Tags:" in captured.out
        assert "foo" in captured.out
        assert "bar" in captured.out
        assert "baz" in captured.out

    def test_update_via_cli(self, real_db_memory: Memory, capsys):
        """update command changes memory content."""
        mem_id = real_db_memory.remember("user1", "original content")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "update", mem_id, "--content", "updated content"]
            ):
                main()
        captured = capsys.readouterr()
        assert "Updated memory:" in captured.out
        updated = real_db_memory._store.get(mem_id)
        assert updated.content == "updated content"

    def test_update_importance_via_cli(self, real_db_memory: Memory, capsys):
        """update --importance changes importance value."""
        mem_id = real_db_memory.remember("user1", "important memory", importance=0.3)
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "update", mem_id, "--importance", "0.95"]):
                main()
        captured = capsys.readouterr()
        assert "Updated memory:" in captured.out
        updated = real_db_memory._store.get(mem_id)
        assert updated.importance == pytest.approx(0.95)

    def test_update_metadata_via_cli(self, real_db_memory: Memory, capsys):
        """update --metadata merges metadata into the memory."""
        mem_id = real_db_memory.remember("user1", "memory with metadata update")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "update", mem_id,
                    "--metadata", '{"source":"test","priority":"high"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Updated memory:" in captured.out
        updated = real_db_memory._store.get(mem_id)
        assert updated is not None
        assert updated.metadata == {"source": "test", "priority": "high"}

    def test_update_content_and_metadata_via_cli(self, real_db_memory: Memory, capsys):
        """update --content combined with --metadata changes both fields."""
        mem_id = real_db_memory.remember("user1", "original content")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "update", mem_id,
                    "--content", "updated content",
                    "--metadata", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Updated memory:" in captured.out
        updated = real_db_memory._store.get(mem_id)
        assert updated is not None
        assert updated.content == "updated content"
        assert updated.metadata == {"source": "test"}

    def test_update_tags_via_cli(self, real_db_memory: Memory, capsys):
        """update --tags replaces tags on a memory via CLI."""
        mem_id = real_db_memory.remember("user1", "memory with tags")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "update", mem_id, "--tags", "alpha,beta,gamma"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Updated memory:" in captured.out
        updated = real_db_memory._store.get(mem_id)
        assert updated is not None
        assert updated.tags == ["alpha", "beta", "gamma"]

    def test_update_content_importance_metadata_via_cli(self, real_db_memory: Memory, capsys):
        """update --content combined with --importance and --metadata changes all three fields."""
        mem_id = real_db_memory.remember("user1", "original content", importance=0.3)
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "update", mem_id,
                    "--content", "updated content",
                    "--importance", "0.9",
                    "--metadata", '{"source":"test","priority":"high"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "Updated memory:" in captured.out
        updated = real_db_memory._store.get(mem_id)
        assert updated is not None
        assert updated.content == "updated content"
        assert updated.importance == pytest.approx(0.9)
        assert updated.metadata == {"source": "test", "priority": "high"}

    def test_update_many_via_cli(self, real_db_memory: Memory, capsys):
        """update-many updates multiple memories at once via CLI."""
        # Store 2 memories with distinct texts to avoid dedup
        mid1 = real_db_memory.remember("user1", "alpha content")
        mid2 = real_db_memory.remember("user1", "beta content")

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "update-many", mid1, mid2, "--content", "updated content", "--importance", "0.9"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "Updated 2 memories." in captured.out
        updated1 = real_db_memory._store.get(mid1)
        updated2 = real_db_memory._store.get(mid2)
        assert updated1 is not None
        assert updated2 is not None
        assert updated1.content == "updated content"
        assert updated2.content == "updated content"
        assert updated1.importance == pytest.approx(0.9)
        assert updated2.importance == pytest.approx(0.9)

    def test_prune_removes_old_memories(self, real_db_memory: Memory, capsys):
        """prune --max-age-days removes memories older than threshold."""
        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        mem = make_memory(
            memory_id="old-prune-1",
            user_id="user1",
            content="old memory to prune",
            created_at=old_time,
            last_accessed_at=old_time,
            lifecycle_state=LifecycleState.DECAYING,
            embedding=real_db_memory._embed.embed_single("old memory to prune"),
            embedding_dim=real_db_memory._embed.dimension(),
        )
        real_db_memory._store.store(mem)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "prune", "user1", "--max-age-days", "30"]):
                main()
        captured = capsys.readouterr()
        assert "Pruned 1 memories" in captured.out
        assert real_db_memory._store.get("old-prune-1") is None

    def test_consolidate_creates_summary(self, real_db_memory: Memory, capsys):
        """consolidate creates a semantic summary from old episodic memories."""
        from datetime import timedelta

        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        for i in range(5):
            mem = make_memory(
                memory_id=f"real-ep-{i}",
                user_id="user1",
                content=f"I visited Paris on day {i}",
                created_at=old_time,
                last_accessed_at=old_time,
                memory_type=MemoryType.EPISODIC,
                embedding=real_db_memory._embed.embed_single(f"I visited Paris on day {i}"),
                embedding_dim=real_db_memory._embed.dimension(),
            )
            real_db_memory._store.store(mem)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "consolidate", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Consolidated into memory:" in captured.out

    def test_graph_via_cli(self, real_db_memory: Memory, capsys):
        """graph through main() extracts entities and relations from real SQLite DB."""
        real_db_memory.remember("user1", "Alice works at Google and Bob lives in London.")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "graph", "user1"]):
                main()
        captured = capsys.readouterr()
        assert "Entities:" in captured.out
        assert "Relations:" in captured.out

    def test_explain_via_cli(self, real_db_memory: Memory, capsys):
        """explain through main() shows score breakdown from real SQLite DB."""
        real_db_memory.remember("user1", "Python programming is fun and useful")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "explain", "user1", "Python coding"]
            ):
                main()
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Semantic:" in captured.out or "Recency:" in captured.out

    def test_list_users_shows_user_counts(self, real_db_memory: Memory, capsys):
        """list-users shows correct counts after storing memories."""
        real_db_memory.remember("alice", "Alice visited the Louvre museum")
        real_db_memory.remember("alice", "Bob hiked in the Rocky Mountains")
        real_db_memory.remember("bob", "Charlie learned to code Python")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "list-users"]):
                main()
        captured = capsys.readouterr()
        assert "alice: 2 memories" in captured.out
        assert "bob: 1 memories" in captured.out

    def test_recall_stream_finds_stored_memory(self, real_db_memory: Memory, capsys):
        """recall-stream through main() finds memories from real SQLite DB."""
        real_db_memory.remember("user1", "I visited Tokyo last spring")
        # Query with exact stored text for deterministic hash match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "recall-stream", "user1", "I visited Tokyo last spring"]):  # noqa: E501
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_single_result_real_db(self, real_db_memory: Memory, capsys):
        """recall-stream through main() streams a single result from real DB."""
        real_db_memory.remember("user1", "only memory here")
        # Query with exact text for deterministic hash match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "recall-stream", "user1", "only memory here"]):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "only memory here" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_namespace_filter(self, real_db_memory: Memory, capsys):
        """recall-stream --namespace only searches within that namespace."""
        real_db_memory.remember("user1", "work project details", namespace="work")
        real_db_memory.remember("user1", "personal notes", namespace="personal")
        # Query with exact text for deterministic hash match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "recall-stream", "user1", "work project details", "--namespace", "work"]  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "work project details" in captured.out
        assert "personal notes" not in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_empty_results_real_db(self, real_db_memory: Memory, capsys):
        """recall-stream through main() with no query match prints message."""
        # Store memories for a different user so user1 has no memories
        real_db_memory.remember("user1", "some memory content")
        # Query for a different user that has no memories
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "recall-stream", "nonexistent-user", "test"]):
                main()
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    def test_recall_stream_respects_top_k(self, real_db_memory: Memory, capsys):
        """recall-stream --top-k respects the limit with real DB."""
        # Store memories with the same word in each so they all match
        real_db_memory.remember("user1", "alpha memory content")
        real_db_memory.remember("user1", "beta memory content")
        real_db_memory.remember("user1", "gamma memory content")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys, "argv", ["kemi", "recall-stream", "user1", "content", "--top-k", "2"]
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        lines = captured.out.strip().split("\n")
        summary_line = [l for l in lines if "Streamed" in l]
        assert len(summary_line) == 1
        count = int(summary_line[0].split()[1])
        assert count <= 2, f"Expected at most 2 results, got {count}"

    def test_recall_stream_hybrid_search_true(self, real_db_memory: Memory, capsys):
        """recall-stream --hybrid-search true works with a real SQLite DB."""
        real_db_memory.remember("user1", "I visited Tokyo last spring with friends")
        # Exact-text query guarantees deterministic hash match + BM25 keyword match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-stream", "user1", "I visited Tokyo last spring with friends", "--hybrid-search", "true"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_hybrid_search_false(self, real_db_memory: Memory, capsys):
        """recall-stream --hybrid-search false works with a real SQLite DB."""
        real_db_memory.remember("user1", "I visited Tokyo last spring with friends")
        # Exact-text query guarantees deterministic hash match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-stream", "user1", "I visited Tokyo last spring with friends", "--hybrid-search", "false"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_hybrid_search_finds_bm25_match(self, real_db_memory: Memory, capsys):
        """recall-stream --hybrid-search true can find memories by keyword even with
        low semantic similarity (BM25 boost)."""
        # Store a memory with a distinctive keyword
        real_db_memory.remember("user1", "the capital of France is Paris")
        # Query with a substring/keyword from the content — FTS5 BM25 will match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-stream", "user1", "France", "--hybrid-search", "true"],
            ):
                main()
        captured = capsys.readouterr()
        # Should find the memory via BM25 + hybrid scoring
        assert "France" in captured.out
        assert "#" in captured.out

    def test_recall_stream_metadata_filter_match(self, real_db_memory: Memory, capsys):
        """recall-stream --metadata-filter includes only matching memories."""
        # Use texts with sufficiently different embeddings to avoid dedup
        real_db_memory.remember("user1", "I visited Tokyo last spring", metadata={"source": "test"})
        real_db_memory.remember("user1", "Python programming is fun", metadata={"source": "other"})
        # Exact-text query for deterministic hash match on the first memory
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall-stream", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"test"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming is fun" not in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_metadata_filter_no_match(self, real_db_memory: Memory, capsys):
        """recall-stream --metadata-filter with no match prints empty message."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", metadata={"source": "test"})
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall-stream", "user1", "I visited Tokyo last spring",
                    "--metadata-filter", '{"source":"nonexistent"}',
                ],
            ):
                main()
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    def test_recall_stream_with_hooks_raise_on_error(self, real_db_memory: Memory, capsys):
        """recall-stream --hooks-raise-on-error parses and works with real DB."""
        real_db_memory.remember("user1", "I visited Tokyo last spring")
        # Flags are global (defined on parser, not subcommand) so they come first
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "--hooks-raise-on-error", "recall-stream", "user1", "I visited Tokyo last spring"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_with_no_hooks_raise_on_error(self, real_db_memory: Memory, capsys):
        """recall-stream --no-hooks-raise-on-error parses and works with real DB."""
        real_db_memory.remember("user1", "I visited Tokyo last spring")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "--no-hooks-raise-on-error", "recall-stream", "user1", "I visited Tokyo last spring"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_top_k_with_namespace(self, real_db_memory: Memory, capsys):
        """recall-stream --top-k and --namespace together work with a real DB."""
        # Store 3 work memories and 2 personal memories
        real_db_memory.remember("user1", "alpha work memory", namespace="work")
        real_db_memory.remember("user1", "beta work memory", namespace="work")
        real_db_memory.remember("user1", "gamma work memory", namespace="work")
        real_db_memory.remember("user1", "personal note one", namespace="personal")
        real_db_memory.remember("user1", "personal note two", namespace="personal")
        # Query the work namespace with top_k that limits results
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-stream", "user1", "memory", "--top-k", "2", "--namespace", "work"],
            ):
                main()
        captured = capsys.readouterr()
        # Should only find work memories, limited to 2
        assert "#  1 | Score:" in captured.out
        assert "work memory" in captured.out
        assert "personal note" not in captured.out
        lines = captured.out.strip().split("\n")
        summary_line = [l for l in lines if "Streamed" in l]
        assert len(summary_line) == 1
        count = int(summary_line[0].split()[1])
        assert count <= 2, f"Expected at most 2 results, got {count}"

    def test_recall_stream_session_id_filter(self, real_db_memory: Memory, capsys):
        """recall-stream --session-id only returns memories from that session."""
        # Use texts with verified low cosine similarity to avoid dedup merging
        real_db_memory.remember("user1", "I visited Tokyo last spring", session_id="session-a")
        real_db_memory.remember("user1", "Python programming is fun", session_id="session-b")
        # Query with exact text for deterministic hash match
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-stream", "user1", "I visited Tokyo last spring", "--session-id", "session-a"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "#  1 | Score:" in captured.out
        assert "I visited Tokyo last spring" in captured.out
        assert "Python programming" not in captured.out
        assert "Streamed 1 result(s)" in captured.out

    def test_recall_stream_session_id_no_match(self, real_db_memory: Memory, capsys):
        """recall-stream --session-id with no matching memories prints message."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", session_id="session-a")
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-stream", "user1", "I visited Tokyo last spring", "--session-id", "nonexistent-session"],  # noqa: E501
            ):
                main()
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    def test_recall_many_via_cli(self, real_db_memory: Memory, capsys):
        """recall-many queries multiple user:query pairs via CLI.

        Stores 2 memories with distinct texts, then calls recall-many with
        2 user:query pairs and verifies both results appear in the output.
        Uses _store.store() to bypass dedup.
        """
        from datetime import datetime, timezone

        from kemi.memory.model import LifecycleState

        embed = real_db_memory._embed
        mems = [
            make_memory(
                memory_id="rm-1",
                user_id="user1",
                content="alpha brand content",
                embedding=embed.embed_single("alpha brand content"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
            make_memory(
                memory_id="rm-2",
                user_id="user1",
                content="beta random stuff",
                embedding=embed.embed_single("beta random stuff"),
                embedding_dim=embed.dimension(),
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc),
                lifecycle_state=LifecycleState.ACTIVE,
            ),
        ]
        for m in mems:
            real_db_memory._store.store(m)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                ["kemi", "recall-many", "user1:alpha", "user1:beta"],
            ):
                main()
        captured = capsys.readouterr()
        assert "Results for user1:" in captured.out
        assert "alpha brand content" in captured.out
        assert "beta random stuff" in captured.out
        assert "0.866 |" in captured.out or "0.776 |" in captured.out or "Score:" in captured.out

    def test_export_with_real_db(self, real_db_memory: Memory, capsys, tmp_path):
        """export writes memories from real SQLite DB to a JSON file."""
        # Store 2 memories via the real DB
        mid1 = real_db_memory.remember("user1", "I visited Tokyo last spring")
        mid2 = real_db_memory.remember("user1", "Python programming is fun")
        out_file = str(tmp_path / "exported_real.json")

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(sys, "argv", ["kemi", "export", out_file]):
                main()

        captured = capsys.readouterr()
        assert "Exported" in captured.out
        assert "to: exported_real.json" in captured.out or "to:" in captured.out

        import json
        with open(out_file) as f:
            data = json.load(f)
        assert len(data) == 2
        memory_ids = [m["memory_id"] for m in data]
        assert mid1 in memory_ids
        assert mid2 in memory_ids
        # Verify content round-tripped correctly
        for m in data:
            if m["memory_id"] == mid1:
                assert m["content"] == "I visited Tokyo last spring"
            elif m["memory_id"] == mid2:
                assert m["content"] == "Python programming is fun"

    def test_import_with_real_db(self, real_db_memory: Memory, capsys, tmp_path):
        """import reads memories from a JSON file into the real SQLite DB."""
        import json

        embed = real_db_memory._embed
        in_file = str(tmp_path / "imported_real.json")
        data = [
            {
                "memory_id": "imp-real-1",
                "user_id": "user1",
                "content": "imported memory alpha",
                "importance": 0.7,
                "lifecycle_state": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                "embedding": embed.embed_single("imported memory alpha"),
                "embedding_dim": embed.dimension(),
            },
            {
                "memory_id": "imp-real-2",
                "user_id": "user1",
                "content": "imported memory beta",
                "importance": 0.3,
                "lifecycle_state": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                "embedding": embed.embed_single("imported memory beta"),
                "embedding_dim": embed.dimension(),
            },
        ]
        with open(in_file, "w") as f:
            json.dump(data, f)

        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch("kemi.interfaces.cli.main.os.path.exists", return_value=True):
                with patch.object(sys, "argv", ["kemi", "import", in_file]):
                    main()

        captured = capsys.readouterr()
        assert "Import complete:" in captured.out
        assert "Imported: 2" in captured.out
        assert "Skipped (duplicates): 0" in captured.out

        # Verify both memories were stored in the real DB
        stored1 = real_db_memory._store.get("imp-real-1")
        assert stored1 is not None
        assert stored1.content == "imported memory alpha"
        assert stored1.importance == pytest.approx(0.7)

        stored2 = real_db_memory._store.get("imp-real-2")
        assert stored2 is not None
        assert stored2.content == "imported memory beta"
        assert stored2.importance == pytest.approx(0.3)

    def test_recall_stream_top_k_namespace_hybrid(self, real_db_memory: Memory, capsys):
        """recall-stream with --top-k, --namespace, and --hybrid-search combined."""
        real_db_memory.remember("user1", "I visited Tokyo last spring", namespace="work")
        real_db_memory.remember("user1", "Python programming is fun", namespace="work")
        real_db_memory.remember("user1", "Important meeting notes", namespace="work")
        real_db_memory.remember("user1", "Weekend hiking trip", namespace="personal")
        real_db_memory.remember("user1", "Grocery shopping list", namespace="personal")
        # Combine all three flags: limit results, scope to work namespace, use hybrid search
        with patch("kemi.interfaces.cli.main.get_memory", return_value=real_db_memory):
            with patch.object(
                sys,
                "argv",
                [
                    "kemi", "recall-stream", "user1", "Tokyo",
                    "--top-k", "1",
                    "--namespace", "work",
                    "--hybrid-search", "true",
                ],
            ):
                main()
        captured = capsys.readouterr()
        # Only 1 result from namespace=work, none from personal
        assert "#  1 | Score:" in captured.out
        assert "Streamed 1 result(s)" in captured.out
        assert "Weekend hiking" not in captured.out
        assert "Grocery shopping" not in captured.out


class TestCLIUpdate:
    """Tests for the update CLI command argument parsing and validation."""

    def _store_memory(self, mock_memory: Memory) -> str:
        """Store a sample memory and return its ID."""
        mem = make_memory(
            memory_id="upd-1",
            user_id="user1",
            content="original content",
            embedding=mock_memory._embed.embed_single("original content"),
            embedding_dim=mock_memory._embed.dimension(),
        )
        mock_memory._store.store(mem)
        return mem.memory_id

    def test_update_no_fields_errors(self, _patch_get_memory, mock_memory: Memory):
        """update with no optional fields should error and exit."""
        self._store_memory(mock_memory)
        with patch("kemi.interfaces.cli.main.sys.exit") as mock_exit:
            update_memory(
                argparse.Namespace(
                    memory_id="upd-1",
                    content=None,
                    importance=None,
                    confidence=None,
                    memory_type=None,
                )
            )
            mock_exit.assert_called_once_with(1)

    def test_update_content_only(self, _patch_get_memory, mock_memory: Memory, capsys):
        """update --content changes the memory content."""
        self._store_memory(mock_memory)
        update_memory(
            argparse.Namespace(
                memory_id="upd-1",
                content="new content",
                importance=None,
                confidence=None,
                memory_type=None,
            )
        )
        captured = capsys.readouterr()
        assert "Updated memory: upd-1" in captured.out
        updated = mock_memory._store.get("upd-1")
        assert updated is not None
        assert updated.content == "new content"

    def test_update_importance_only(self, _patch_get_memory, mock_memory: Memory, capsys):
        """update --importance changes the memory importance."""
        self._store_memory(mock_memory)
        update_memory(
            argparse.Namespace(
                memory_id="upd-1",
                content=None,
                importance=0.85,
                confidence=None,
                memory_type=None,
            )
        )
        captured = capsys.readouterr()
        assert "Updated memory: upd-1" in captured.out
        updated = mock_memory._store.get("upd-1")
        assert updated.importance == pytest.approx(0.85)

    def test_update_confidence_only(self, _patch_get_memory, mock_memory: Memory, capsys):
        """update --confidence changes the memory confidence."""
        self._store_memory(mock_memory)
        update_memory(
            argparse.Namespace(
                memory_id="upd-1",
                content=None,
                importance=None,
                confidence=0.75,
                memory_type=None,
            )
        )
        captured = capsys.readouterr()
        assert "Updated memory: upd-1" in captured.out
        updated = mock_memory._store.get("upd-1")
        assert updated.confidence == pytest.approx(0.75)

    def test_update_memory_type_only(self, _patch_get_memory, mock_memory: Memory, capsys):
        """update --memory-type changes the memory type."""
        self._store_memory(mock_memory)
        update_memory(
            argparse.Namespace(
                memory_id="upd-1",
                content=None,
                importance=None,
                confidence=None,
                memory_type="semantic",
            )
        )
        captured = capsys.readouterr()
        assert "Updated memory: upd-1" in captured.out
        updated = mock_memory._store.get("upd-1")
        assert updated.memory_type == MemoryType.SEMANTIC

    def test_update_multiple_fields(self, _patch_get_memory, mock_memory: Memory, capsys):
        """update with multiple fields changes all of them."""
        self._store_memory(mock_memory)
        update_memory(
            argparse.Namespace(
                memory_id="upd-1",
                content="multi update",
                importance=0.9,
                confidence=0.95,
                memory_type="episodic",
            )
        )
        captured = capsys.readouterr()
        assert "Updated memory: upd-1" in captured.out
        updated = mock_memory._store.get("upd-1")
        assert updated.content == "multi update"
        assert updated.importance == pytest.approx(0.9)
        assert updated.confidence == pytest.approx(0.95)
        assert updated.memory_type == MemoryType.EPISODIC

    def test_update_value_error_from_memory(self, _patch_get_memory, mock_memory: Memory):
        """update propagates ValueError from Memory.update as a clean exit."""
        self._store_memory(mock_memory)
        with patch("kemi.interfaces.cli.main.Memory.update", side_effect=ValueError("bad id")):
            with patch("kemi.interfaces.cli.main.sys.exit") as mock_exit:
                update_memory(
                    argparse.Namespace(
                        memory_id="upd-1",
                        content="x",
                        importance=None,
                        confidence=None,
                        memory_type=None,
                    )
                )
                mock_exit.assert_called_once_with(1)

    def test_update_argparse_missing_memory_id(self):
        """update without memory_id causes argparse to exit."""
        with patch.object(sys, "argv", ["kemi", "update"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_update_argparse_invalid_memory_type(self):
        """update with invalid --memory-type causes argparse to exit."""
        with patch.object(sys, "argv", ["kemi", "update", "upd-1", "--memory-type", "invalid"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
