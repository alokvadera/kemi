from datetime import datetime, timedelta, timezone

import pytest

from kemi.exceptions import ValidationError
from kemi.memory.model import LifecycleState, MemorySource
from tests._helpers.factories import make_memory


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
    from kemi.memory.model import LifecycleState

    result = mock_memory.recall("user123", "food", lifecycle_filter=[LifecycleState.ACTIVE])
    assert isinstance(result, list)


def test_recall_updates_lifecycle(mock_memory) -> None:
    mock_memory.remember("user123", "I am vegetarian")
    mock_memory.recall("user123", "food")
    from kemi.memory.model import LifecycleState

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
    from kemi.memory.model import MemorySource

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
    with pytest.raises(ValidationError, match="importance must be a number"):
        mock_memory.remember("user123", "test", importance="high")


def test_prune_by_age_only(mock_memory) -> None:
    """prune with only max_age_days deletes old memories."""

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

    deleted = mock_memory.prune("user1", max_age_days=30.0)
    assert deleted == 1
    assert mock_memory._store.get("old-1") is None


def test_prune_by_importance_only(mock_memory) -> None:
    """prune with only min_importance deletes low-importance memories."""
    mem = make_memory(
        memory_id="low-1",
        user_id="user1",
        content="low importance",
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(mem)

    deleted = mock_memory.prune("user1", min_importance=0.1)
    assert deleted == 1
    assert mock_memory._store.get("low-1") is None


def test_prune_by_age_and_importance_and_logic(mock_memory) -> None:
    """prune with both filters uses AND logic."""

    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=100)

    # Old but high importance — should NOT be deleted
    old_high = make_memory(
        memory_id="old-high",
        user_id="user1",
        content="old but important",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.9,
        lifecycle_state=LifecycleState.DECAYING,
    )
    # Young but low importance — should NOT be deleted
    young_low = make_memory(
        memory_id="young-low",
        user_id="user1",
        content="young but unimportant",
        created_at=now,
        last_accessed_at=now,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )
    # Old AND low importance — SHOULD be deleted
    old_low = make_memory(
        memory_id="old-low",
        user_id="user1",
        content="old and unimportant",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )

    mock_memory._store.store(old_high)
    mock_memory._store.store(young_low)
    mock_memory._store.store(old_low)

    deleted = mock_memory.prune("user1", max_age_days=30.0, min_importance=0.1)
    assert deleted == 1
    assert mock_memory._store.get("old-high") is not None
    assert mock_memory._store.get("young-low") is not None
    assert mock_memory._store.get("old-low") is None


def test_prune_no_filters_deletes_nothing(mock_memory) -> None:
    """prune with no filters should not delete anything."""
    mem = make_memory(
        memory_id="mem-1",
        user_id="user1",
        content="some memory",
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(mem)

    deleted = mock_memory.prune("user1")
    assert deleted == 0
    assert mock_memory._store.get("mem-1") is not None


def test_prune_boundary_exact_age_not_deleted(mock_memory) -> None:
    """Memory exactly max_age_days old should NOT be deleted (strict >)."""
    from unittest.mock import patch

    frozen_now = datetime.now(timezone.utc)
    # created_at set so age is exactly 30 days when prune's clock is frozen
    exact_time = frozen_now - timedelta(days=30)
    mem = make_memory(
        memory_id="boundary-age",
        user_id="user1",
        content="exactly 30 days old",
        created_at=exact_time,
        last_accessed_at=exact_time,
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(mem)

    with patch("kemi.operations._io.datetime") as mock_dt:
        mock_dt.now.return_value = frozen_now
        deleted = mock_memory.prune("user1", max_age_days=30.0)

    assert deleted == 0
    assert mock_memory._store.get("boundary-age") is not None


def test_prune_boundary_exact_importance_not_deleted(mock_memory) -> None:
    """Memory with importance exactly at min_importance should NOT be deleted (strict <)."""
    mem = make_memory(
        memory_id="boundary-imp",
        user_id="user1",
        content="exactly at threshold",
        importance=0.1,
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(mem)

    deleted = mock_memory.prune("user1", min_importance=0.1)
    assert deleted == 0
    assert mock_memory._store.get("boundary-imp") is not None


def test_prune_default_only_decaying(mock_memory) -> None:
    """By default prune only considers DECAYING memories; ACTIVE is skipped."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    # ACTIVE, old, low importance — should be SKIPPED by default
    active_old = make_memory(
        memory_id="active-old",
        user_id="user1",
        content="active old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.ACTIVE,
    )
    # DECAYING, old, low importance — should be DELETED by default
    decaying_old = make_memory(
        memory_id="decaying-old",
        user_id="user1",
        content="decaying old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(active_old)
    mock_memory._store.store(decaying_old)

    deleted = mock_memory.prune("user1", max_age_days=30.0, min_importance=0.1)
    assert deleted == 1
    assert mock_memory._store.get("active-old") is not None
    assert mock_memory._store.get("decaying-old") is None


def test_prune_override_active_included(mock_memory) -> None:
    """Passing lifecycle_states=[ACTIVE, DECAYING] prunes matching ACTIVE memories too."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    active_old = make_memory(
        memory_id="active-old",
        user_id="user1",
        content="active old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.ACTIVE,
    )
    decaying_old = make_memory(
        memory_id="decaying-old",
        user_id="user1",
        content="decaying old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(active_old)
    mock_memory._store.store(decaying_old)

    deleted = mock_memory.prune(
        "user1",
        max_age_days=30.0,
        min_importance=0.1,
        lifecycle_states=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
    )
    assert deleted == 2
    assert mock_memory._store.get("active-old") is None
    assert mock_memory._store.get("decaying-old") is None


def test_prune_override_active_only(mock_memory) -> None:
    """Passing lifecycle_states=[ACTIVE] prunes only ACTIVE memories."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    active_old = make_memory(
        memory_id="active-old",
        user_id="user1",
        content="active old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.ACTIVE,
    )
    decaying_old = make_memory(
        memory_id="decaying-old",
        user_id="user1",
        content="decaying old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )
    mock_memory._store.store(active_old)
    mock_memory._store.store(decaying_old)

    deleted = mock_memory.prune(
        "user1",
        max_age_days=30.0,
        min_importance=0.1,
        lifecycle_states=[LifecycleState.ACTIVE],
    )
    assert deleted == 1
    assert mock_memory._store.get("active-old") is None
    assert mock_memory._store.get("decaying-old") is not None


def test_prune_mixed_lifecycle_and_filters(mock_memory) -> None:
    """Both ACTIVE and DECAYING are considered, but filters still use AND logic."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    now = datetime.now(timezone.utc)
    # ACTIVE, old, high importance — passes age but not importance
    active_old_high = make_memory(
        memory_id="active-old-high",
        user_id="user1",
        content="active old important",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.9,
        lifecycle_state=LifecycleState.ACTIVE,
    )
    # DECAYING, young, low importance — passes importance but not age
    decaying_young_low = make_memory(
        memory_id="decaying-young-low",
        user_id="user1",
        content="decaying young unimportant",
        created_at=now,
        last_accessed_at=now,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
    )
    # ACTIVE, old, low importance — passes both filters
    active_old_low = make_memory(
        memory_id="active-old-low",
        user_id="user1",
        content="active old unimportant",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.ACTIVE,
    )
    mock_memory._store.store(active_old_high)
    mock_memory._store.store(decaying_young_low)
    mock_memory._store.store(active_old_low)

    deleted = mock_memory.prune(
        "user1",
        max_age_days=30.0,
        min_importance=0.1,
        lifecycle_states=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
    )
    assert deleted == 1
    assert mock_memory._store.get("active-old-high") is not None
    assert mock_memory._store.get("decaying-young-low") is not None
    assert mock_memory._store.get("active-old-low") is None


def test_prune_default_namespace_isolation(mock_memory) -> None:
    """prune with default namespace only affects the 'default' namespace."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    default_ns = make_memory(
        memory_id="default-ns",
        user_id="user1",
        content="default namespace old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
        namespace="default",
    )
    other_ns = make_memory(
        memory_id="other-ns",
        user_id="user1",
        content="other namespace old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
        namespace="alpha",
    )
    mock_memory._store.store(default_ns)
    mock_memory._store.store(other_ns)

    deleted = mock_memory.prune("user1", max_age_days=30.0, min_importance=0.1)
    assert deleted == 1
    assert mock_memory._store.get("default-ns") is None
    assert mock_memory._store.get("other-ns") is not None


def test_prune_explicit_namespace_isolation(mock_memory) -> None:
    """prune with an explicit namespace only affects that namespace."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    default_ns = make_memory(
        memory_id="default-ns",
        user_id="user1",
        content="default namespace old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
        namespace="default",
    )
    alpha_ns = make_memory(
        memory_id="alpha-ns",
        user_id="user1",
        content="alpha namespace old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
        namespace="alpha",
    )
    beta_ns = make_memory(
        memory_id="beta-ns",
        user_id="user1",
        content="beta namespace old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
        namespace="beta",
    )
    mock_memory._store.store(default_ns)
    mock_memory._store.store(alpha_ns)
    mock_memory._store.store(beta_ns)

    deleted = mock_memory.prune(
        "user1", max_age_days=30.0, min_importance=0.1, namespace="alpha"
    )
    assert deleted == 1
    assert mock_memory._store.get("default-ns") is not None
    assert mock_memory._store.get("alpha-ns") is None
    assert mock_memory._store.get("beta-ns") is not None


def test_prune_nonexistent_namespace_returns_zero(mock_memory) -> None:
    """prune against a namespace with no memories returns 0 safely."""
    old_time = datetime.now(timezone.utc) - timedelta(days=100)
    default_ns = make_memory(
        memory_id="default-ns",
        user_id="user1",
        content="default namespace old memory",
        created_at=old_time,
        last_accessed_at=old_time,
        importance=0.05,
        lifecycle_state=LifecycleState.DECAYING,
        namespace="default",
    )
    mock_memory._store.store(default_ns)

    deleted = mock_memory.prune(
        "user1", max_age_days=30.0, min_importance=0.1, namespace="nonexistent"
    )
    assert deleted == 0
    assert mock_memory._store.get("default-ns") is not None


def test_forget_by_id_isolation(mock_memory) -> None:
    """forget by memory_id only deletes the requested memory; others survive."""
    default_mem = make_memory(
        memory_id="default-mem",
        user_id="user1",
        content="default ns memory",
        namespace="default",
    )
    alpha_mem = make_memory(
        memory_id="alpha-mem",
        user_id="user1",
        content="alpha ns memory",
        namespace="alpha",
    )
    mock_memory._store.store(default_mem)
    mock_memory._store.store(alpha_mem)

    deleted = mock_memory.forget("user1", "default-mem")
    assert deleted == 1
    assert mock_memory._store.get("default-mem") is None
    assert mock_memory._store.get("alpha-mem") is not None


def test_forget_by_user_deletes_all_namespaces(mock_memory) -> None:
    """forget by user_id (no memory_id) deletes across ALL namespaces."""
    default_mem = make_memory(
        memory_id="default-mem",
        user_id="user1",
        content="default ns memory",
        namespace="default",
    )
    alpha_mem = make_memory(
        memory_id="alpha-mem",
        user_id="user1",
        content="alpha ns memory",
        namespace="alpha",
    )
    mock_memory._store.store(default_mem)
    mock_memory._store.store(alpha_mem)

    deleted = mock_memory.forget("user1")
    assert deleted == 2
    assert mock_memory._store.get("default-mem") is None
    assert mock_memory._store.get("alpha-mem") is None


def test_update_preserves_namespace(mock_memory) -> None:
    """update() must not alter the memory's namespace."""
    mem = make_memory(
        memory_id="ns-mem",
        user_id="user1",
        content="original content",
        namespace="alpha",
    )
    mock_memory._store.store(mem)

    mock_memory.update("ns-mem", content="updated content")

    updated = mock_memory._store.get("ns-mem")
    assert updated is not None
    assert updated.namespace == "alpha"
    assert updated.content == "updated content"


def test_context_block_namespace_isolation(mock_memory) -> None:
    """context_block only returns memories from the requested namespace."""
    mock_memory.remember("user1", "I love pizza", namespace="default")
    mock_memory.remember("user1", "I love sushi", namespace="alpha")

    result = mock_memory.context_block("user1", "food", namespace="alpha")
    assert "sushi" in result
    assert "pizza" not in result


def test_recall_top_k_less_than_1(mock_memory) -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        mock_memory.recall("user123", "test", top_k=0)


def test_recall_with_dimension_mismatch(mock_memory) -> None:
    from kemi.adapters.embedding.custom import CustomEmbedAdapter

    mock_memory._store.store(
        make_memory(
            memory_id="id1",
            user_id="user123",
            content="I am vegetarian",
            embedding=[0.1] * 32,
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            source=MemorySource.USER_STATED,
            lifecycle_state=LifecycleState.ACTIVE,
            metadata={},
        )
    )

    alt_adapter = CustomEmbedAdapter(embed_fn=lambda texts: [[0.1] * 64 for _ in texts], dim=64)
    alt_memory = mock_memory.__class__(embed=alt_adapter, store=mock_memory._store)

    with pytest.raises(ValidationError, match="Embedding dimension mismatch"):
        alt_memory.recall("user123", "food")


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
