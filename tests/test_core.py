from datetime import datetime, timezone

import pytest

from kemi.models import LifecycleState, MemoryObject, MemorySource


def test_remember_returns_string_id(mock_memory) -> None:
    result = mock_memory.remember("user123", "I am vegetarian")
    assert isinstance(result, str)
    assert len(result) > 0


def test_remember_dedup(mock_memory) -> None:
    id1 = mock_memory.remember("user123", "I am vegetarian")
    id2 = mock_memory.remember("user123", "I am vegetarian")
    assert id1 == id2


def test_remember_different_content(mock_memory) -> None:
    id1 = mock_memory.remember("user123", "I am vegetarian")
    id2 = mock_memory.remember("user123", "I live in Mumbai")
    assert id1 != id2


def test_recall_returns_list(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    result = mock_memory.recall("user123", "food preferences")
    assert isinstance(result, list)


def test_recall_empty_user(mock_memory) -> None:
    result = mock_memory.recall("newuser", "any query")
    assert result == []


def test_context_block_format(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")
    result = mock_memory.context_block("user123", "user preferences")
    assert result.startswith("Relevant context from memory:")
    assert "- I am vegetarian" in result
    assert "- I live in Mumbai" in result


def test_context_block_empty(mock_memory) -> None:
    result = mock_memory.context_block("user123", "query")
    assert result == ""


def test_forget_by_id(mock_memory) -> None:
    mem_id = mock_memory.remember("user123", "I am vegetarian")
    result = mock_memory.forget("user123", mem_id)
    assert result == 1

    result = mock_memory.forget("user123", mem_id)
    assert result == 0


def test_forget_by_user(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")
    result = mock_memory.forget("user123")
    assert result == 2


def test_upgrade(mock_memory) -> None:
    mock_memory.upgrade()


def test_migrate(mock_memory) -> None:
    from kemi.adapters.embedding.custom import CustomEmbedAdapter

    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")

    new_adapter = CustomEmbedAdapter(embed_fn=lambda texts: [[0.1] * 32 for _ in texts], dim=32)

    result = mock_memory.migrate("user123", new_adapter)
    assert result == 2


def test_remember_with_sanitize_input(mock_memory) -> None:
    result = mock_memory.remember("user123", "normal content", sanitize_input=True)
    assert isinstance(result, str)


def test_recall_with_lifecycle_filter(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    from kemi.models import LifecycleState

    result = mock_memory.recall("user123", "food", lifecycle_filter=[LifecycleState.ACTIVE])
    assert isinstance(result, list)


def test_recall_updates_lifecycle(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.recall("user123", "food")
    from kemi.models import LifecycleState

    all_mem = mock_memory._store.get_all_by_user(
        "user123", lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING]
    )
    assert len(all_mem) > 0


def test_context_block_custom_prefix(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    result = mock_memory.context_block("user123", "food", prefix="Custom:")
    assert result.startswith("Custom:")


def test_remember_with_metadata(mock_memory) -> None:
    result = mock_memory.remember("user123", "I am vegetarian", metadata={"source": "form"})
    assert isinstance(result, str)


def test_remember_with_source(mock_memory) -> None:
    from kemi.models import MemorySource

    result = mock_memory.remember("user123", "I am vegetarian", source=MemorySource.AGENT_INFERRED)
    assert isinstance(result, str)


def test_remember_with_importance(mock_memory) -> None:
    result = mock_memory.remember("user123", "I am vegetarian", importance=0.9)
    assert isinstance(result, str)


def test_migrate_empty_user(mock_memory) -> None:
    from kemi.adapters.embedding.custom import CustomEmbedAdapter

    new_adapter = CustomEmbedAdapter(embed_fn=lambda texts: [[0.1] * 32 for _ in texts], dim=32)
    result = mock_memory.migrate("nonexistent_user", new_adapter)
    assert result == 0


@pytest.mark.asyncio
async def test_aremember_returns_string_id(mock_memory) -> None:
    result = await mock_memory.aremember("user123", "I am vegetarian")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_arecall_returns_list(mock_memory) -> None:
    await mock_memory.aremember("user123", "I am vegetarian")
    result = await mock_memory.arecall("user123", "food preferences")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_aforget_by_id(mock_memory) -> None:
    mem_id = await mock_memory.aremember("user123", "I am vegetarian")
    result = await mock_memory.aforget("user123", mem_id)
    assert result == 1

    result = await mock_memory.aforget("user123", mem_id)
    assert result == 0


@pytest.mark.asyncio
async def test_acontext_block_format(mock_memory) -> None:
    await mock_memory.aremember("user123", "I am vegetarian")
    await mock_memory.aremember("user123", "I live in Mumbai")
    result = await mock_memory.acontext_block("user123", "user preferences")
    assert result.startswith("Relevant context from memory:")
    assert "- I am vegetarian" in result
    assert "- I live in Mumbai" in result


def test_remember_empty_content(mock_memory) -> None:
    with pytest.raises(ValueError, match="content cannot be empty"):
        mock_memory.remember("user123", "")


def test_remember_empty_user_id(mock_memory) -> None:
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        mock_memory.remember("", "I am vegetarian")


def test_recall_empty_query(mock_memory) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        mock_memory.recall("user123", "")


def test_recall_top_k_zero(mock_memory) -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        mock_memory.recall("user123", "test", top_k=0)


def test_forget_empty_user_id(mock_memory) -> None:
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        mock_memory.forget("")


def test_stats_empty_user(mock_memory) -> None:
    result = mock_memory.stats("newuser123")
    assert result["total"] == 0
    assert result["avg_importance"] == 0.0
    assert result["tag_counts"] == {}


def test_stats_with_memories(mock_memory) -> None:
    mock_memory.remember("user123", "I love pizza", tags=["food"])
    mock_memory.remember("user456", "I live in Delhi", tags=[])

    result = mock_memory.stats("user123")

    assert result["total"] == 1
    assert result["total_with_tags"] == 1
    assert result["total_without_tags"] == 0
    assert "food" in result["tag_counts"]


def test_remember_invalid_importance_type(mock_memory) -> None:
    with pytest.raises(TypeError, match="importance must be a number"):
        mock_memory.remember("user123", "test", importance="high")


def test_recall_top_k_less_than_1(mock_memory) -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        mock_memory.recall("user123", "test", top_k=0)


def test_recall_with_dimension_mismatch(mock_memory) -> None:
    from kemi.adapters.embedding.custom import CustomEmbedAdapter

    mock_memory._store.store(
        MemoryObject(
            memory_id="id1",
            user_id="user123",
            content="I am vegetarian",
            embedding=[0.1] * 32,
            score=0.0,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
            embedding_dim=32,
        )
    )

    alt_adapter = CustomEmbedAdapter(embed_fn=lambda texts: [[0.1] * 64 for _ in texts], dim=64)
    alt_memory = mock_memory.__class__(embed=alt_adapter, store=mock_memory._store)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        alt_memory.recall("user123", "food")


@pytest.mark.xfail(
    reason="Singleton metrics collector state bleed between tests — needs isolation fix"
)
def test_remember_many_tracks_per_memory_metrics_and_audit(real_db_memory) -> None:
    """Assert that remember_many populates per-memory dedup and audit metrics.

    Uses a real SQLite-backed Memory with audit trail enabled. The batch
    contains two unique contents and one duplicate. Verifies that:
    - remember_total increments once per item (3 total)
    - remember_many_total increments once for the batch
    - embed_total only counts non-duplicate stores (2)
    - embed_bytes_total sums non-duplicate content lengths
    - duplicates_detected counts the duplicate (1)
    - total_memories gauge reflects actual stored count (2)
    - audit trail has 3 entries (one per item)
    """
    mem = real_db_memory

    # Reset metrics to a clean baseline
    if mem._metrics is not None:
        mem._metrics.reset()

    # Enable audit trail (requires real SQLite store)
    mem.enable_audit_trail(retention_days=30, auto_purge=False)

    # Batch: first "alpha", then "beta", then duplicate "alpha"
    contents = ["alpha content", "beta content", "alpha content"]
    memory_ids = mem.remember_many("user123", contents)

    # All 3 items return a memory_id; duplicate resolves to the first
    assert len(memory_ids) == 3
    assert memory_ids[0] == memory_ids[2]  # duplicate resolved
    assert memory_ids[0] != memory_ids[1]

    metrics = mem.get_metrics()
    assert metrics is not None

    # remember_total: incremented once per item via _track_operation("remember")
    assert metrics["operations"]["remember"] == 3
    # remember_many_total: incremented once by the batch wrapper
    assert metrics["operations"]["remember_many"] == 1
    # embed_total: only non-duplicate items trigger embed_total.inc(1)
    assert metrics["embeddings"]["total"] == 2
    # embed_bytes_total: sum of lengths of non-duplicate contents
    expected_bytes = len("alpha content") + len("beta content")
    assert metrics["embeddings"]["bytes_approx"] == expected_bytes
    # duplicates_detected: one duplicate detected in the batch
    assert metrics["quality"]["duplicates_detected"] == 1
    # total_memories: only 2 unique memories stored
    assert metrics["memory_usage"]["total_memories"] == 2

    # Audit trail: 3 per-memory entries + 1 batch-level remember_many entry
    audit_stats = mem._audit_trail.get_stats()
    assert audit_stats["total_entries"] == 4
    assert audit_stats["unique_users"] == 1

    entries = mem._audit_trail.query(user_id="user123")
    assert len(entries) == 4

    # Defensively verify details is a dict on every entry
    for entry in entries:
        assert isinstance(entry.details, dict)

    # 3 "remember" entries (one per item) and 1 "remember_many" batch entry
    remember_entries = [e for e in entries if e.operation == "remember"]
    assert len(remember_entries) == 3
    batch_entries = [e for e in entries if e.operation == "remember_many"]
    assert len(batch_entries) == 1

    # Exactly one "remember" entry should be marked as a duplicate
    duplicate_entries = [e for e in remember_entries if e.details.get("duplicate") is True]
    assert len(duplicate_entries) == 1

    # The other 2 "remember" entries should NOT have the duplicate flag
    non_duplicate_entries = [e for e in remember_entries if e.details.get("duplicate") is not True]
    assert len(non_duplicate_entries) == 2


@pytest.mark.asyncio
async def test_recall_stream_returns_same_as_recall(mock_memory) -> None:
    """Verify streaming recall returns the same memories in the same order as batch recall."""
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")
    mock_memory.remember("user123", "I love python programming")

    batch_results = mock_memory.recall("user123", "user preferences", top_k=3)

    stream_results: list = []
    async for memory in mock_memory.recall_stream("user123", "user preferences", top_k=3):
        stream_results.append(memory)

    assert len(stream_results) == len(batch_results)
    for s, b in zip(stream_results, batch_results, strict=True):
        assert s.memory_id == b.memory_id
        assert s.content == b.content
        assert abs(s.score - b.score) < 0.001


@pytest.mark.asyncio
async def test_recall_stream_top_k(mock_memory) -> None:
    """Verify streaming recall respects top_k limit."""
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.remember("user123", "I live in Mumbai")
    mock_memory.remember("user123", "I love python programming")

    count = 0
    async for _ in mock_memory.recall_stream("user123", "user preferences", top_k=2):
        count += 1
    assert count == 2


@pytest.mark.asyncio
async def test_recall_stream_empty_user(mock_memory) -> None:
    """Verify streaming recall returns nothing for empty user."""
    count = 0
    async for _ in mock_memory.recall_stream("newuser", "any query"):
        count += 1
    assert count == 0


@pytest.mark.asyncio
async def test_arecall_stream_param(mock_memory) -> None:
    """Verify arecall(..., stream=True) returns an async generator."""
    mock_memory.remember("user123", "I am vegetarian")
    result = await mock_memory.arecall("user123", "food preferences", stream=True)
    # Should be an async generator object
    assert hasattr(result, "__aiter__")
    assert hasattr(result, "__anext__")


@pytest.mark.asyncio
async def test_arecall_no_stream_param(mock_memory) -> None:
    """Verify arecall(..., stream=False) returns a normal list."""
    mock_memory.remember("user123", "I am vegetarian")
    result = await mock_memory.arecall("user123", "food preferences", stream=False)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_recall_stream_validation(mock_memory) -> None:
    """Verify streaming recall validates inputs."""
    with pytest.raises(ValueError, match="user_id cannot be empty"):
        async for _ in mock_memory.recall_stream("", "test"):
            pass
    with pytest.raises(ValueError, match="query cannot be empty"):
        async for _ in mock_memory.recall_stream("user123", ""):
            pass
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        async for _ in mock_memory.recall_stream("user123", "test", top_k=0):
            pass


def test_remember_tracks_conflict_metric_and_audit(tmp_path) -> None:
    """Assert that remember() increments conflicts_detected and logs conflict details."""
    import math

    from kemi import Memory
    from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

    class ConflictEmbedAdapter:
        """Returns embeddings that produce a conflict between two phrases."""

        def __init__(self) -> None:
            self._dim = 64

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_single(t) for t in texts]

        def embed_single(self, text: str) -> list[float]:
            if text == "I like running":
                return [1.0, 0.0] * 32
            if text == "I hate running":
                rad = 50 * math.pi / 180
                return [math.cos(rad), math.sin(rad)] * 32
            return [0.0] * 64

        def dimension(self) -> int:
            return self._dim

    db_path = str(tmp_path / "conflict_audit.db")
    store = SQLiteStorageAdapter(db_path=db_path)
    mem = Memory(embed=ConflictEmbedAdapter(), store=store)

    mem.enable_audit_trail(auto_purge=False)
    if mem._metrics is not None:
        mem._metrics.reset()

    # Seed an existing memory (no conflict here)
    mem.remember("user123", "I like running")

    # Reset metrics so we only measure the conflict call
    if mem._metrics is not None:
        mem._metrics.reset()

    # This should trigger a conflict: similar embedding but not a duplicate
    mem.remember("user123", "I hate running")

    metrics = mem.get_metrics()
    assert metrics is not None
    assert metrics["quality"]["conflicts_detected"] == 1

    entries = mem._audit_trail.query(user_id="user123", operation="remember")
    conflict_entries = [e for e in entries if e.details.get("conflict") is True]
    assert len(conflict_entries) == 1
    assert conflict_entries[0].details.get("conflict_with") is not None
    assert conflict_entries[0].details.get("memory_id") is not None
