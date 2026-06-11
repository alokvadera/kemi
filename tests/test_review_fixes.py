"""Tests for the async/sync parameter consistency, pre-validation, and
SQL aggregate fixes from the MemoryService review.

Covers:
- Phase 1a/1b: ttl_seconds propagation through aremember / aremember_many
- Phase 1c:    astats mirrors stats() parameter list
- Phase 1d:    remember_many pre-validates BEFORE batch embed
- Phase 2a:    arecall return type annotation
- Phase 2b:    upgrade() actually migrates (returns the new version)
- Phase 2d:    stats() uses count_aggregates (SQL pushdown path)
- Phase 2f:    enable_audit_trail degrades gracefully on non-SQLite
- Phase 3e:    memory_to_dict is the canonical implementation
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import pytest

from kemi import MemoryService
from kemi.adapters.base import StorageAdapter
from kemi.adapters.embedding.custom import CustomEmbedAdapter
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.memory.model import (
    LifecycleState,
    MemoryConfig,
    MemoryObject,
    MemorySource,
    memory_to_dict,
)
from tests._helpers.factories import make_memory

# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────


def _embed_32(texts: list[str]) -> list[list[float]]:
    # Text-content-driven unique 32-d vectors with low cosine sim. We
    # encode each character into a distinct dimension so similar content
    # ("alpha" vs "beta") still produces different vectors.
    out: list[list[float]] = []
    for t in texts:
        vec = [0.0] * 32
        for ch in t:
            vec[ord(ch) % 32] += 0.05
        # Normalize-ish to keep magnitudes reasonable
        norm = max(1.0, sum(x * x for x in vec) ** 0.5)
        vec = [v / norm for v in vec]
        out.append(vec)
    return out


def _in_memory_sqlite_store() -> SQLiteStorageAdapter:
    # Use a temp file (not ":memory:") because :memory: creates a
    # connection-scoped DB that disappears between calls.

    tmp = tempfile.NamedTemporaryFile(
        suffix=".db", prefix="kemi-review-", delete=False
    )
    tmp.close()
    return SQLiteStorageAdapter(db_path=tmp.name)


def _service_with_sqlite(
    *, dedup_threshold: float = 0.99
) -> tuple[MemoryService, SQLiteStorageAdapter]:
    store = _in_memory_sqlite_store()
    embed = CustomEmbedAdapter(embed_fn=_embed_32, dim=32)
    config = MemoryConfig(dedup_threshold=dedup_threshold)
    svc = MemoryService(embed=embed, store=store, config=config)
    return svc, store


# ────────────────────────────────────────────────────────────────────────
# Phase 1a / 1b: ttl_seconds on async remember paths
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aremember_accepts_ttl_seconds() -> None:
    """aremember must accept and forward ttl_seconds to remember()."""
    store = _in_memory_sqlite_store()
    embed = CustomEmbedAdapter(embed_fn=_embed_32, dim=32)
    svc = MemoryService(embed=embed, store=store)

    memory_id = await svc.aremember(
        "user-ttl", "I will expire", importance=0.7, ttl_seconds=60
    )
    assert memory_id

    # Confirm the memory was stored with the correct expiry.
    mem = store.get(memory_id)
    assert mem is not None
    assert mem.expires_at is not None
    assert mem.content == "I will expire"


@pytest.mark.asyncio
async def test_aremember_many_accepts_ttl_seconds() -> None:
    """aremember_many must accept and forward ttl_seconds to remember_many()."""
    store = _in_memory_sqlite_store()
    embed = CustomEmbedAdapter(embed_fn=_embed_32, dim=32)
    svc = MemoryService(embed=embed, store=store)

    ids = await svc.aremember_many(
        "user-ttl-many", ["alpha", "beta"], importance=0.5, ttl_seconds=120
    )
    assert len(ids) == 2
    for mid in ids:
        mem = store.get(mid)
        assert mem is not None
        assert mem.expires_at is not None


@pytest.mark.asyncio
async def test_aremember_ttl_seconds_rejects_invalid() -> None:
    """ttl_seconds validation must work in the async path too."""
    store = _in_memory_sqlite_store()
    embed = CustomEmbedAdapter(embed_fn=_embed_32, dim=32)
    svc = MemoryService(embed=embed, store=store)
    with pytest.raises(ValueError, match="ttl_seconds must be a positive integer"):
        await svc.aremember("u", "x", ttl_seconds=0)


# ────────────────────────────────────────────────────────────────────────
# Phase 1c: astats mirrors stats() parameter list
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_astats_accepts_lifecycle_filter() -> None:
    svc, _store = _service_with_sqlite()
    svc.remember("u", "alive", importance=0.5)
    result = await svc.astats("u", lifecycle_filter=[LifecycleState.ACTIVE])
    assert result["total"] == 1
    assert result["by_lifecycle"]["active"] == 1


@pytest.mark.asyncio
async def test_astats_accepts_session_id() -> None:
    svc, _store = _service_with_sqlite()
    svc.remember("u", "first session memory one", session_id="s1")
    svc.remember("u", "second session memory two", session_id="s2")
    result = await svc.astats("u", session_id="s1")
    assert result["total"] == 1


# ────────────────────────────────────────────────────────────────────────
# Phase 1d: remember_many pre-validates BEFORE batch embed
# ────────────────────────────────────────────────────────────────────────


def test_remember_many_rejects_empty_before_embed() -> None:
    """An empty string in the middle of a batch must raise before the
    embedding call. The mock embed records the number of times it was
    called; it should be 0 when pre-validation catches the issue.
    """
    embed_calls: list[list[str]] = []

    def _recorder(texts: list[str]) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[0.1] * 32 for _ in texts]

    store = _in_memory_sqlite_store()
    embed = CustomEmbedAdapter(embed_fn=_recorder, dim=32)
    svc = MemoryService(embed=embed, store=store)

    with pytest.raises(ValueError, match="content at index 1"):
        svc.remember_many("u", ["valid", "  ", "also-valid"], importance=0.5)

    # Pre-validation prevents the embed call from happening.
    assert embed_calls == [], (
        f"expected no embed calls, got {len(embed_calls)} "
        f"with content {embed_calls}"
    )

    # And nothing was stored.
    assert store.count("u") == 0


# ────────────────────────────────────────────────────────────────────────
# Phase 2a: arecall return type
# ────────────────────────────────────────────────────────────────────────


def test_arecall_return_type_annotation() -> None:
    """arecall's return annotation must include the AsyncGenerator case."""
    sig = inspect.signature(MemoryService.arecall)
    ret = sig.return_annotation
    assert ret is not inspect.Signature.empty
    # The string form is fine (PEP 563 deferred eval).
    assert "AsyncGenerator" in str(ret) or "AsyncGenerator" in repr(ret)
    assert "MemoryObject" in str(ret) or "MemoryObject" in repr(ret)


# ────────────────────────────────────────────────────────────────────────
# Phase 2b: upgrade() actually migrates
# ────────────────────────────────────────────────────────────────────────


def test_upgrade_returns_post_upgrade_version() -> None:
    """upgrade() must return the schema version after the upgrade,
    not be a no-op (1->1)."""
    svc, _store = _service_with_sqlite()
    new_version = svc.upgrade()
    # SQLite adapter is at CURRENT_VERSION=8, so upgrade() returns 8.
    assert new_version == 8
    assert new_version > 1


def test_upgrade_schema_signature_accepts_none() -> None:
    """upgrade_schema should default both args to None (caller picks target)."""
    sig = inspect.signature(StorageAdapter.upgrade_schema)
    assert sig.parameters["from_version"].default is None
    assert sig.parameters["to_version"].default is None


# ────────────────────────────────────────────────────────────────────────
# Phase 2d: count_aggregates (SQL pushdown)
# ────────────────────────────────────────────────────────────────────────


def test_count_aggregates_returns_expected_shape() -> None:
    store = _in_memory_sqlite_store()
    svc = MemoryService(
        embed=CustomEmbedAdapter(embed_fn=_embed_32, dim=32), store=store
    )
    svc.remember("u", "alpha", importance=0.8, tags=["t1", "t2"])
    svc.remember("u", "beta", importance=0.4, tags=["t1"])

    agg = store.count_aggregates("u")
    assert agg["total"] == 2
    assert agg["by_lifecycle"]["active"] == 2
    assert agg["total_with_tags"] == 2
    assert agg["tag_counts"] == {"t1": 2, "t2": 1}
    # avg_importance_numerator = 0.8 + 0.4 = 1.2
    assert agg["avg_importance_numerator"] == pytest.approx(1.2)


def test_count_aggregates_with_lifecycle_filter() -> None:
    store = _in_memory_sqlite_store()
    svc = MemoryService(
        embed=CustomEmbedAdapter(embed_fn=_embed_32, dim=32), store=store
    )
    svc.remember("u", "x")
    # No decay call needed; just confirm filter works.
    agg = store.count_aggregates("u", lifecycle_filter=[LifecycleState.ACTIVE])
    assert agg["total"] == 1
    agg2 = store.count_aggregates("u", lifecycle_filter=[LifecycleState.ARCHIVED])
    assert agg2["total"] == 0


def test_stats_uses_aggregates(tmp_path: Path) -> None:
    """stats() result must equal the old behavior even though the path
    changed (O(1) per state instead of O(N) per memory)."""
    store = _in_memory_sqlite_store()
    svc = MemoryService(
        embed=CustomEmbedAdapter(embed_fn=_embed_32, dim=32), store=store
    )
    svc.remember("u", "alpha", importance=0.7, tags=["t1"])
    svc.remember("u", "beta", importance=0.3, tags=["t1", "t2"])
    svc.remember("u", "no-tags", importance=0.1, tags=[])

    result = svc.stats("u")
    assert result["total"] == 3
    assert result["by_lifecycle"]["active"] == 3
    assert result["total_with_tags"] == 2
    assert result["total_without_tags"] == 1
    assert result["tag_counts"] == {"t1": 2, "t2": 1}
    # avg = (0.7 + 0.3 + 0.1) / 3 = 0.3666…
    assert result["avg_importance"] == pytest.approx(1.1 / 3.0)


# ────────────────────────────────────────────────────────────────────────
# Phase 2f: enable_audit_trail degrades gracefully on non-SQLite
# ────────────────────────────────────────────────────────────────────────


def test_enable_audit_trail_skips_non_sqlite(caplog) -> None:
    """On a non-SQLite store, enable_audit_trail should warn and skip
    rather than raise AttributeError."""
    from kemi.operations import _ops_metrics

    class FakeNonSQLiteStore(StorageAdapter):
        def __init__(self) -> None:
            self._audit_attempted = False

        def store(self, memory: MemoryObject) -> None:
            pass

        def search(self, *args, **kwargs) -> list[MemoryObject]:
            return []

        def get(self, memory_id: str) -> MemoryObject | None:
            return None

        def update(self, memory: MemoryObject) -> None:
            pass

        def delete_by_user(self, user_id: str) -> int:
            return 0

        def delete_by_id(self, memory_id: str) -> bool:
            return False

        def get_all_by_user(self, *args, **kwargs) -> list[MemoryObject]:
            return []

        def count(self, user_id: str) -> int:
            return 0

        def get_all(self, *args, **kwargs) -> list[MemoryObject]:
            return []

        def get_all_users(self) -> list[str]:
            return []

        def upgrade_schema(self, *args, **kwargs) -> int:
            return 1

        def get_by_tag(self, *args, **kwargs) -> list[MemoryObject]:
            return []

        def search_by_content(self, *args, **kwargs) -> list[MemoryObject]:
            return []

    store = FakeNonSQLiteStore()
    embed = CustomEmbedAdapter(embed_fn=_embed_32, dim=32)
    svc = MemoryService(embed=embed, store=store)

    # Should not raise.
    _ops_metrics.enable_audit_trail(svc)

    # And the registry should NOT have an audit sink added.
    assert svc._plugins.audit_sinks == []


# ────────────────────────────────────────────────────────────────────────
# Phase 3e: memory_to_dict is the canonical implementation
# ────────────────────────────────────────────────────────────────────────


def test_memory_to_dict_canonical_keys() -> None:
    mem = make_memory(
        memory_id="m1",
        user_id="u",
        content="hi",
        embedding=[0.1, 0.2],
        confidence=0.9,
    )
    d = memory_to_dict(mem)
    assert d["memory_id"] == "m1"
    assert d["content"] == "hi"
    assert d["importance"] == 0.5
    assert d["source"] == "user_stated"
    # "embedding" is intentionally NOT in the canonical dict — it is
    # reconstructed from the storage adapter on recall.
    assert "embedding" not in d
    # All expected keys present
    expected_keys = {
        "memory_id", "content", "importance", "confidence",
        "lifecycle_state", "memory_type", "source", "tags",
        "namespace", "session_id", "version", "created_at",
        "last_accessed_at", "metadata", "agent_id", "run_id", "app_id",
    }
    assert set(d.keys()) == expected_keys


def test_io_and_ingestion_agree() -> None:
    """Both pipelines' private _memory_to_dict should produce the same dict."""
    from kemi.operations._io import _memory_to_dict as io_to_dict
    from kemi.pipeline.ingestion import _memory_to_dict as ingest_to_dict

    mem = make_memory(
        memory_id="m1",
        user_id="u",
        content="x",
        embedding=[0.0] * 8,
        importance=0.4,
        source=MemorySource.SYSTEM_GENERATED,
    )
    assert io_to_dict(mem) == ingest_to_dict(mem)
    assert io_to_dict(mem) == memory_to_dict(mem)
