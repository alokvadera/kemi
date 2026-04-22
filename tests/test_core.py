import pytest

from datetime import datetime, timezone

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
