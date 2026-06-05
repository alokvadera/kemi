"""Tests for new high-priority features.

Covers: batch operations (recall_many, update_many, forget_many),
metadata filtering, event hooks, query cache, and multi-tenant fields.
"""

from typing import Any

import pytest

# ── Batch operations ─────────────────────────────────────────────────


def test_recall_many_returns_results_per_user(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0  # disable dedup for distinct memories
    mock_memory.remember("alice", "I love hiking in the Alps")
    mock_memory.remember("bob", "I prefer beach vacations")

    results = mock_memory.recall_many(
        ["alice", "bob"],
        ["mountains", "beach"],
    )

    assert "alice" in results
    assert "bob" in results
    assert len(results["alice"]) > 0
    assert len(results["bob"]) > 0


def test_recall_many_metadata_filter(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0  # disable dedup for distinct memories
    mock_memory.remember("alice", "I love hiking", metadata={"topic": "outdoor"})
    mock_memory.remember("alice", "I enjoy coding", metadata={"topic": "tech"})

    results = mock_memory.recall_many(
        ["alice"],
        ["hobbies"],
        metadata_filter={"topic": "outdoor"},
    )
    assert len(results["alice"]) == 1
    assert results["alice"][0].content == "I love hiking"


def test_update_many_updates_all(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    id1 = mock_memory.remember("user1", "content one")
    id2 = mock_memory.remember("user1", "content two")

    updated = mock_memory.update_many([id1, id2], importance=0.9)
    assert len(updated) == 2

    mem1 = mock_memory._store.get(id1)
    mem2 = mock_memory._store.get(id2)
    assert mem1 is not None and mem1.importance == 0.9
    assert mem2 is not None and mem2.importance == 0.9


def test_forget_many_deletes_multiple(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    id1 = mock_memory.remember("user1", "content one")
    id2 = mock_memory.remember("user1", "content two")
    id3 = mock_memory.remember("user1", "content three")

    deleted = mock_memory.forget_many([id1, id2])
    assert deleted == 2

    assert mock_memory._store.get(id1) is None
    assert mock_memory._store.get(id2) is None
    assert mock_memory._store.get(id3) is not None


# ── Metadata filter on recall ────────────────────────────────────────


def test_recall_metadata_filter_excludes_non_matching(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.remember("user1", "I love pizza", metadata={"cuisine": "italian"})
    mock_memory.remember("user1", "I love sushi", metadata={"cuisine": "japanese"})

    results = mock_memory.recall("user1", "food", metadata_filter={"cuisine": "italian"})
    assert len(results) == 1
    assert results[0].content == "I love pizza"


def test_recall_metadata_filter_no_match_returns_empty(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.remember("user1", "I love pizza", metadata={"cuisine": "italian"})

    results = mock_memory.recall("user1", "food", metadata_filter={"cuisine": "chinese"})
    assert results == []


# ── Event hooks ──────────────────────────────────────────────────────


def test_event_hooks_fire_pre_and_post(mock_memory) -> None:
    events = []

    def hook(phase: str, operation: str, **kwargs: Any) -> None:
        events.append((phase, operation))

    mock_memory.add_event_hook("pre", lambda op, **kw: hook("pre", op, **kw))
    mock_memory.add_event_hook("post", lambda op, **kw: hook("post", op, **kw))

    mock_memory.remember("user1", "test content")
    assert ("pre", "remember") in events
    assert ("post", "remember") in events

    mock_memory.recall("user1", "test")
    assert ("pre", "recall") in events
    assert ("post", "recall") in events


def test_remove_event_hook(mock_memory) -> None:
    calls = []

    def hook(op: str, **kw: Any) -> None:
        calls.append(op)

    mock_memory.add_event_hook("pre", hook)
    mock_memory.remember("user1", "a")
    assert len(calls) == 1

    removed = mock_memory.remove_event_hook("pre", hook)
    assert removed is True
    mock_memory.remember("user1", "b")
    assert len(calls) == 1  # no new call


# ── Query cache ──────────────────────────────────────────────────────


def test_query_cache_returns_same_results_on_repeat(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.enable_query_cache(max_size=10)
    mock_memory.remember("user1", "I love hiking in the mountains")

    r1 = mock_memory.recall("user1", "hiking")
    r2 = mock_memory.recall("user1", "hiking")

    assert len(r1) == len(r2)
    assert r1[0].content == r2[0].content


def test_query_cache_can_be_disabled(mock_memory) -> None:
    mock_memory.enable_query_cache(max_size=10)
    mock_memory.disable_query_cache()
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.remember("user1", "content")

    # Should not crash
    results = mock_memory.recall("user1", "content")
    assert len(results) == 1


# ── Multi-tenant fields (agent_id, run_id, app_id) ───────────────────


def test_remember_with_agent_run_app_ids(mock_memory) -> None:
    mid = mock_memory.remember(
        "user1",
        "agent task result",
        agent_id="agent-42",
        run_id="run-99",
        app_id="app-prod",
    )
    mem = mock_memory._store.get(mid)
    assert mem is not None
    assert mem.agent_id == "agent-42"
    assert mem.run_id == "run-99"
    assert mem.app_id == "app-prod"


def test_recall_respects_metadata_filter_with_multiple_conditions(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.remember("user1", "A", metadata={"x": "1", "y": "2"})
    mock_memory.remember("user1", "B", metadata={"x": "1", "y": "3"})

    results = mock_memory.recall("user1", "content", metadata_filter={"x": "1", "y": "2"})
    assert len(results) == 1
    assert results[0].content == "A"


# ── Update with metadata merge ───────────────────────────────────────


def test_update_merges_metadata(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mid = mock_memory.remember("user1", "original", metadata={"a": "1"})
    mock_memory.update(mid, metadata={"b": "2"})

    mem = mock_memory._store.get(mid)
    assert mem.metadata["a"] == "1"
    assert mem.metadata["b"] == "2"


# ── Async batch operations ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_arecall_many_concurrent(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.remember("alice", "I love hiking in the Alps")
    mock_memory.remember("bob", "I prefer beach vacations")

    results = await mock_memory.arecall_many(
        ["alice", "bob"],
        ["mountains", "beach"],
    )

    assert "alice" in results
    assert "bob" in results
    assert len(results["alice"]) > 0
    assert len(results["bob"]) > 0


@pytest.mark.asyncio
async def test_aupdate_many_concurrent(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    id1 = mock_memory.remember("user1", "content one")
    id2 = mock_memory.remember("user1", "content two")

    updated = await mock_memory.aupdate_many([id1, id2], importance=0.95)
    assert len(updated) == 2

    mem1 = mock_memory._store.get(id1)
    mem2 = mock_memory._store.get(id2)
    assert mem1 is not None and mem1.importance == 0.95
    assert mem2 is not None and mem2.importance == 0.95


@pytest.mark.asyncio
async def test_aforget_many_concurrent(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    id1 = mock_memory.remember("user1", "content one")
    id2 = mock_memory.remember("user1", "content two")
    id3 = mock_memory.remember("user1", "content three")

    deleted = await mock_memory.aforget_many([id1, id2])
    assert deleted == 2

    assert mock_memory._store.get(id1) is None
    assert mock_memory._store.get(id2) is None
    assert mock_memory._store.get(id3) is not None


# ── Event hooks with raise_on_error ──────────────────────────────────


def test_event_hooks_raise_on_error_aborts_operation(mock_memory) -> None:
    def abort_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("aborted")

    mock_memory.add_event_hook("pre", abort_hook)

    with pytest.raises(RuntimeError, match="aborted"):
        mock_memory._run_hooks("pre", "remember", raise_on_error=True)


def test_config_hooks_raise_on_error_false_swallows_errors(mock_memory) -> None:
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=False)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)

    def bad_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("should be swallowed")

    mem.add_event_hook("pre", bad_hook)
    # Should NOT raise — hook error is swallowed because config says so
    mid = mem.remember("user1", "test content with swallow")
    assert mid is not None


def test_config_hooks_raise_on_error_true_aborts_operation(mock_memory) -> None:
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)

    def bad_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("abort")

    mem.add_event_hook("pre", bad_hook)
    with pytest.raises(RuntimeError, match="abort"):
        mem.remember("user1", "test content with abort")


def test_post_hook_remember_raise_on_error_true_raises_after_store(mock_memory) -> None:
    """Post-hook failure with hooks_raise_on_error=True raises after side effects."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("post abort")

    mem.add_event_hook("post", bad_post_hook)

    with pytest.raises(RuntimeError, match="post abort"):
        mem.remember("user1", "post hook test")

    # The memory was stored before the post-hook failed
    all_mems = mem._store.get_all_by_user("user1")
    assert len(all_mems) == 1
    assert all_mems[0].content == "post hook test"


def test_post_hook_remember_raise_on_error_false_swallows(mock_memory) -> None:
    """Post-hook failure with hooks_raise_on_error=False is swallowed."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=False)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("should be swallowed")

    mem.add_event_hook("post", bad_post_hook)

    # Should NOT raise — post-hook error is swallowed
    mid = mem.remember("user1", "post hook swallow")
    assert mid is not None


def test_post_hook_recall_raise_on_error_true_raises_after_results(mock_memory) -> None:
    """Post-hook failure on recall with hooks_raise_on_error=True raises after results are built."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    mem.enable_query_cache(max_size=10)
    mem.remember("user1", "recall post hook test")

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("post recall abort")

    mem.add_event_hook("post", bad_post_hook)

    with pytest.raises(RuntimeError, match="post recall abort"):
        mem.recall("user1", "recall")

    # Query cache was populated before the post-hook failed — next identical
    # recall should return cached results without re-running the post-hook.
    cached = mem.recall("user1", "recall")
    assert len(cached) == 1
    assert cached[0].content == "recall post hook test"


def test_post_hook_recall_raise_on_error_false_swallows(mock_memory) -> None:
    """Post-hook failure on recall with hooks_raise_on_error=False is swallowed."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=False)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    mem.remember("user1", "recall post hook swallow")

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("should be swallowed")

    mem.add_event_hook("post", bad_post_hook)

    # Should NOT raise
    results = mem.recall("user1", "recall")
    assert len(results) == 1
    assert results[0].content == "recall post hook swallow"


def test_post_hook_update_raise_on_error_true_raises_after_update(mock_memory) -> None:
    """Post-hook failure on update with hooks_raise_on_error=True raises after store is updated."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    mid = mem.remember("user1", "original update post hook")

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("post update abort")

    mem.add_event_hook("post", bad_post_hook)

    with pytest.raises(RuntimeError, match="post update abort"):
        mem.update(mid, content="updated content")

    # The update happened before the post-hook failed
    updated = mem._store.get(mid)
    assert updated is not None
    assert updated.content == "updated content"


def test_post_hook_update_raise_on_error_false_swallows(mock_memory) -> None:
    """Post-hook failure on update with hooks_raise_on_error=False is swallowed."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=False)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    mid = mem.remember("user1", "original update swallow")

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("should be swallowed")

    mem.add_event_hook("post", bad_post_hook)

    # Should NOT raise
    mem.update(mid, content="updated after swallow")
    updated = mem._store.get(mid)
    assert updated is not None
    assert updated.content == "updated after swallow"


def test_post_hook_forget_raise_on_error_true_raises_after_delete(mock_memory) -> None:
    """Post-hook failure on forget with hooks_raise_on_error=True raises after deletion."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    mid = mem.remember("user1", "to be deleted")

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("post forget abort")

    mem.add_event_hook("post", bad_post_hook)

    with pytest.raises(RuntimeError, match="post forget abort"):
        mem.forget("user1", memory_id=mid)

    # Memory was deleted before the post-hook failed
    assert mem._store.get(mid) is None


def test_post_hook_forget_raise_on_error_false_swallows(mock_memory) -> None:
    """Post-hook failure on forget with hooks_raise_on_error=False is swallowed."""
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=False)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    mid = mem.remember("user1", "to be deleted swallow")

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("should be swallowed")

    mem.add_event_hook("post", bad_post_hook)

    # Should NOT raise
    deleted = mem.forget("user1", memory_id=mid)
    assert deleted == 1
    assert mem._store.get(mid) is None


# ── Query cache shallow-copy safety ──────────────────────────────────


def test_query_cache_mutation_does_not_corrupt_cache(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    mock_memory.enable_query_cache(max_size=10)
    mock_memory.remember("user1", "I love hiking")

    r1 = mock_memory.recall("user1", "hiking")
    # Mutate the returned list (should not corrupt cache)
    if r1:
        r1[0].metadata["extra"] = "mutation"

    r2 = mock_memory.recall("user1", "hiking")
    # r2 should still be from cache and not have the mutation
    if r2:
        assert "extra" not in r2[0].metadata


# ── remember_many event hooks ──────────────────────────────────────


def test_remember_many_fires_pre_and_post_hooks(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    events = []

    def hook(phase: str, operation: str, **kwargs: Any) -> None:
        events.append((phase, operation, kwargs.get("content"), kwargs.get("memory_id")))

    mock_memory.add_event_hook("pre", lambda op, **kw: hook("pre", op, **kw))
    mock_memory.add_event_hook("post", lambda op, **kw: hook("post", op, **kw))

    ids = mock_memory.remember_many("user1", ["alpha", "beta"])
    assert len(ids) == 2

    pre_events = [e for e in events if e[0] == "pre" and e[1] == "remember"]
    post_events = [e for e in events if e[0] == "post" and e[1] == "remember"]
    assert len(pre_events) == 2
    assert len(post_events) == 2
    assert pre_events[0][2] == "alpha"
    assert pre_events[1][2] == "beta"
    assert post_events[0][3] == ids[0]
    assert post_events[1][3] == ids[1]


def test_remember_many_pre_hook_abort_stops_batch(mock_memory) -> None:
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    calls = []

    def aborting_pre_hook(operation: str, **kwargs: Any) -> None:
        calls.append(kwargs.get("content"))
        if kwargs.get("content") == "beta":
            raise RuntimeError("abort at beta")

    mem.add_event_hook("pre", aborting_pre_hook)

    with pytest.raises(RuntimeError, match="abort at beta"):
        mem.remember_many("user1", ["alpha", "beta", "gamma"])

    # alpha was stored before the abort, beta and gamma were not
    all_mems = mem._store.get_all_by_user("user1")
    assert len(all_mems) == 1
    assert all_mems[0].content == "alpha"


def test_remember_many_post_hook_abort_after_store(mock_memory) -> None:
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=True)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0
    calls = []

    def aborting_post_hook(operation: str, **kwargs: Any) -> None:
        calls.append(kwargs.get("memory_id"))
        if kwargs.get("memory_id") == calls[0]:
            raise RuntimeError("abort after first")

    mem.add_event_hook("post", aborting_post_hook)

    with pytest.raises(RuntimeError, match="abort after first"):
        mem.remember_many("user1", ["alpha", "beta"])

    # First item was stored before the post-hook failed; second was not processed
    all_mems = mem._store.get_all_by_user("user1")
    assert len(all_mems) == 1
    assert all_mems[0].content == "alpha"
    assert not any(m.content == "beta" for m in all_mems)


def test_remember_many_post_hook_swallowed_when_config_false(mock_memory) -> None:
    from kemi import Memory
    from kemi.models import MemoryConfig

    config = MemoryConfig(hooks_raise_on_error=False)
    mem = Memory(embed=mock_memory._embed, store=mock_memory._store, config=config)
    mem._config.dedup_threshold = 1.0

    def bad_post_hook(operation: str, **kwargs: Any) -> None:
        raise RuntimeError("should be swallowed")

    mem.add_event_hook("post", bad_post_hook)

    # Should NOT raise — all items store normally despite failing post-hooks
    ids = mem.remember_many("user1", ["one", "two"])
    assert len(ids) == 2
    all_mems = mem._store.get_all_by_user("user1")
    assert len(all_mems) == 2


@pytest.mark.asyncio
async def test_aremember_many_inherits_hooks(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    events = []

    def hook(phase: str, operation: str, **kwargs: Any) -> None:
        events.append((phase, operation))

    mock_memory.add_event_hook("pre", lambda op, **kw: hook("pre", op, **kw))
    mock_memory.add_event_hook("post", lambda op, **kw: hook("post", op, **kw))

    ids = await mock_memory.aremember_many("user1", ["async one", "async two"])
    assert len(ids) == 2

    pre_events = [e for e in events if e[0] == "pre" and e[1] == "remember"]
    post_events = [e for e in events if e[0] == "post" and e[1] == "remember"]
    assert len(pre_events) == 2
    assert len(post_events) == 2


def test_metadata_filter_with_large_fetch_multiplier(mock_memory) -> None:
    mock_memory._config.dedup_threshold = 1.0
    # Create 20 memories, only 5 match the metadata filter
    for i in range(15):
        mock_memory.remember("user1", f"unrelated {i}", metadata={"topic": "other"})
    for i in range(5):
        mock_memory.remember("user1", f"target {i}", metadata={"topic": "target"})

    # Without the large fetch multiplier, metadata_filter might return <5 results
    results = mock_memory.recall("user1", "target", top_k=5, metadata_filter={"topic": "target"})
    assert len(results) == 5
    assert all("target" in r.content for r in results)
