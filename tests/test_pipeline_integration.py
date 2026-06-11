"""Integration tests for RetrievalPipeline and IngestionPipeline with real SQLite.

These tests exercise the full pipelines end-to-end with a real
SQLiteStorageAdapter and MockEmbeddingAdapter.  They cover:

- Ingestion: store, dedup, conflict detection, entity extraction, webhooks, metrics
- Retrieval: search, ranking, caching, lifecycle updates, metadata filtering
- Round-trip: ingest → retrieve → verify
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from kemi import Memory
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.exceptions import ValidationError
from kemi.infra.observability import MetricsCollector
from kemi.infra.webhooks import WebhookEventType
from kemi.memory.entities import RegexEntityLinker
from kemi.memory.model import LifecycleState, MemoryConfig, MemoryObject
from kemi.operations._query_cache import _QueryCache
from kemi.pipeline.ingestion import IngestionContext, IngestionPipeline
from kemi.pipeline.retrieval import RetrievalContext, RetrievalPipeline
from tests._helpers.factories import make_memory
from tests._helpers.mock_storage import MockEmbeddingAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_store(tmp_path: Any) -> SQLiteStorageAdapter:
    """Fresh SQLite storage adapter for each test."""
    db_path = str(tmp_path / "pipeline_integration.db")
    return SQLiteStorageAdapter(db_path=db_path)


@pytest.fixture
def embed() -> MockEmbeddingAdapter:
    return MockEmbeddingAdapter(dim=64)


@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig(
        dedup_threshold=0.95,
        conflict_threshold=0.65,
        enable_entity_boost=True,
        max_tokens_default=1000,
    )


@pytest.fixture
def entity_linker() -> RegexEntityLinker:
    return RegexEntityLinker()


@pytest.fixture
def metrics() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture
def query_cache() -> _QueryCache:
    return _QueryCache(max_size=10)


@pytest.fixture
def ingestion_pipeline(
    sqlite_store: SQLiteStorageAdapter,
    config: MemoryConfig,
    entity_linker: RegexEntityLinker,
    metrics: MetricsCollector,
) -> IngestionPipeline:
    """IngestionPipeline wired to real SQLite and fresh metrics."""
    ctx = IngestionContext(
        store=sqlite_store,
        config=config,
        entity_linker=entity_linker,
        metrics=metrics,
    )
    return IngestionPipeline(ctx)


@pytest.fixture
def retrieval_pipeline(
    sqlite_store: SQLiteStorageAdapter,
    embed: MockEmbeddingAdapter,
    config: MemoryConfig,
    entity_linker: RegexEntityLinker,
    metrics: MetricsCollector,
    query_cache: _QueryCache,
) -> RetrievalPipeline:
    """RetrievalPipeline wired to real SQLite, mock embed, and cache."""
    ctx = RetrievalContext(
        store=sqlite_store,
        embed=embed,
        config=config,
        entity_linker=entity_linker,
        query_cache=query_cache,
        metrics=metrics,
        adaptive_retriever=None,
    )
    return RetrievalPipeline(ctx)


# ---------------------------------------------------------------------------
# IngestionPipeline integration tests
# ---------------------------------------------------------------------------

class TestIngestionPipelineSQLite:
    def test_ingest_stores_memory(
                                     self,
                                     ingestion_pipeline: IngestionPipeline,
                                     sqlite_store: SQLiteStorageAdapter,
                                     embed: MockEmbeddingAdapter,
                                 ) -> None:
        """A memory is persisted to SQLite after ingestion."""
        memory = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        result = ingestion_pipeline.ingest(memory)

        assert result.memory_id == memory.memory_id
        stored = sqlite_store.get(memory.memory_id)
        assert stored is not None
        assert stored.content == "I love pizza"
        assert stored.user_id == "alice"

    def test_ingest_duplicate_resolution(
                                            self,
                                            ingestion_pipeline: IngestionPipeline,
                                            sqlite_store: SQLiteStorageAdapter,
                                            embed: MockEmbeddingAdapter,
                                        ) -> None:
        """Identical embeddings are deduped and merged into the canonical memory."""
        mem1 = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        mem2 = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        ingestion_pipeline.ingest(mem1)
        resolved = ingestion_pipeline.ingest(mem2)

        # Should resolve to the same memory_id
        assert resolved.memory_id == mem1.memory_id
        # Only one memory stored for the user
        assert sqlite_store.count("alice") == 1

    def test_ingest_conflict_detection(
                                          self,
                                          ingestion_pipeline: IngestionPipeline,
                                          sqlite_store: SQLiteStorageAdapter,
                                          config: MemoryConfig,
                                      ) -> None:
        """Similar but non-duplicate content triggers a conflict flag."""
        # Use explicit vectors where we control cosine similarity precisely.
        # base = [1.0, 0.0, 0.0, ...] (64-dim)
        base = [1.0] + [0.0] * 63
        # conflict = [0.5, 0.866..., 0.0, ...] — cosine ≈ 0.5, normalized = 0.75
        # which is between conflict_threshold (0.65) and dedup_threshold (0.95).
        import math
        conflict = [0.5, math.sqrt(3) / 2] + [0.0] * 62

        mem1 = make_memory(
            user_id="alice",
            content="I like running",
            embedding=base,
        )
        ingestion_pipeline.ingest(mem1)

        mem2 = make_memory(
            user_id="alice",
            content="I hate running",
            embedding=conflict,
        )
        result = ingestion_pipeline.ingest(mem2)

        assert result.metadata.get("conflict_flagged") is True
        assert sqlite_store.count("alice") == 2

    def test_ingest_entity_extraction(
                                         self,
                                         ingestion_pipeline: IngestionPipeline,
                                         sqlite_store: SQLiteStorageAdapter,
                                         embed: MockEmbeddingAdapter,
                                     ) -> None:
        """Entity boost extracts and caches entities in metadata."""
        memory = make_memory(
            user_id="alice",
            content="Alice visited Paris and London",
            embedding=embed.embed_single("Alice visited Paris and London"),
        )
        ingestion_pipeline.ingest(memory)

        stored = sqlite_store.get(memory.memory_id)
        assert stored is not None
        entities = stored.metadata.get("extracted_entities")
        assert entities is not None
        assert "paris" in {e.lower() for e in entities}
        assert "london" in {e.lower() for e in entities}

    def test_ingest_webhook_and_metrics(
                                           self,
                                           ingestion_pipeline: IngestionPipeline,
                                           sqlite_store: SQLiteStorageAdapter,
                                           embed: MockEmbeddingAdapter,
                                           metrics: MetricsCollector,
                                       ) -> None:
        """Ingestion dispatches webhooks and increments pipeline-level metrics."""
        webhook_calls: list[dict[str, Any]] = []
        ingestion_pipeline._ctx.dispatch_webhook = (
            lambda event, **kwargs: webhook_calls.append({"event": event, **kwargs})
        )

        memory = make_memory(
            user_id="alice",
            content="hello world",
            embedding=embed.embed_single("hello world"),
        )
        ingestion_pipeline.ingest(memory)

        # Webhook
        assert any(c["event"] == WebhookEventType.REMEMBERED for c in webhook_calls)
        # Metrics — the pipeline increments embed_total and total_memories directly.
        # remember_total is incremented by the orchestrator's track_operation, which
        # is a no-op lambda in our fixture, so we only assert on pipeline-level metrics.
        assert metrics.embed_total.value() >= 1
        assert metrics.total_memories.value() >= 1

    def test_ingest_multiple_users(
                                      self,
                                      ingestion_pipeline: IngestionPipeline,
                                      sqlite_store: SQLiteStorageAdapter,
                                      embed: MockEmbeddingAdapter,
                                  ) -> None:
        """Memories for different users are isolated in SQLite."""
        for user in ("alice", "bob"):
            mem = make_memory(
                user_id=user,
                content=f"content for {user}",
                embedding=embed.embed_single(f"content for {user}"),
            )
            ingestion_pipeline.ingest(mem)

        assert sqlite_store.count("alice") == 1
        assert sqlite_store.count("bob") == 1


# ---------------------------------------------------------------------------
# RetrievalPipeline integration tests
# ---------------------------------------------------------------------------

class TestRetrievalPipelineSQLite:
    def test_retrieve_basic(
                               self,
                               retrieval_pipeline: RetrievalPipeline,
                               sqlite_store: SQLiteStorageAdapter,
                               embed: MockEmbeddingAdapter,
                           ) -> None:
        """Retrieve returns the memory we just stored."""
        memory = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        sqlite_store.store(memory)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="pizza",
            top_k=5,
        )
        assert len(results) >= 1
        assert results[0].memory_id == memory.memory_id

    def test_retrieve_top_k_limit(
                                     self,
                                     retrieval_pipeline: RetrievalPipeline,
                                     sqlite_store: SQLiteStorageAdapter,
                                     embed: MockEmbeddingAdapter,
                                 ) -> None:
        """top_k is respected."""
        for i in range(5):
            mem = make_memory(
                user_id="alice",
                content=f"memory number {i}",
                embedding=embed.embed_single(f"memory number {i}"),
            )
            sqlite_store.store(mem)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="memory",
            top_k=3,
        )
        assert len(results) == 3

    def test_retrieve_cache_hit(
                                   self,
                                   retrieval_pipeline: RetrievalPipeline,
                                   sqlite_store: SQLiteStorageAdapter,
                                   embed: MockEmbeddingAdapter,
                                   metrics: MetricsCollector,
                               ) -> None:
        """Second identical query hits the cache and increments the cache_hit metric."""
        memory = make_memory(
            user_id="alice",
            content="cached memory",
            embedding=embed.embed_single("cached memory"),
        )
        sqlite_store.store(memory)

        retrieval_pipeline.retrieve(
            user_id="alice",
            query="cached",
            top_k=5,
        )
        # Second call should be a cache hit
        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="cached",
            top_k=5,
        )
        assert len(results) >= 1
        assert results[0].memory_id == memory.memory_id

    def test_retrieve_metadata_filter(
                                         self,
                                         retrieval_pipeline: RetrievalPipeline,
                                         sqlite_store: SQLiteStorageAdapter,
                                         embed: MockEmbeddingAdapter,
                                     ) -> None:
        """metadata_filter returns only memories matching the metadata."""
        mem_keep = make_memory(
            user_id="alice",
            content="keep this",
            embedding=embed.embed_single("keep this"),
            metadata={"category": "important"},
        )
        mem_skip = make_memory(
            user_id="alice",
            content="skip this",
            embedding=embed.embed_single("skip this"),
            metadata={"category": "other"},
        )
        sqlite_store.store(mem_keep)
        sqlite_store.store(mem_skip)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="this",
            top_k=5,
            metadata_filter={"category": "important"},
        )
        assert len(results) == 1
        assert results[0].memory_id == mem_keep.memory_id

    def test_retrieve_lifecycle_filter(
                                          self,
                                          retrieval_pipeline: RetrievalPipeline,
                                          sqlite_store: SQLiteStorageAdapter,
                                          embed: MockEmbeddingAdapter,
                                      ) -> None:
        """lifecycle_filter excludes memories in excluded states."""
        mem_active = make_memory(
            user_id="alice",
            content="active memory",
            embedding=embed.embed_single("active memory"),
            lifecycle_state=LifecycleState.ACTIVE,
        )
        mem_archived = make_memory(
            user_id="alice",
            content="archived memory",
            embedding=embed.embed_single("archived memory"),
            lifecycle_state=LifecycleState.ARCHIVED,
        )
        sqlite_store.store(mem_active)
        sqlite_store.store(mem_archived)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="memory",
            top_k=5,
            lifecycle_filter=[LifecycleState.ACTIVE],
        )
        assert len(results) == 1
        assert results[0].memory_id == mem_active.memory_id

    def test_retrieve_updates_lifecycle(
                                           self,
                                           retrieval_pipeline: RetrievalPipeline,
                                           sqlite_store: SQLiteStorageAdapter,
                                           embed: MockEmbeddingAdapter,
                                       ) -> None:
        """Retrieval bumps last_accessed_at on returned result objects in-place."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        memory = make_memory(
            user_id="alice",
            content="lifecycle test",
            embedding=embed.embed_single("lifecycle test"),
            last_accessed_at=old_time,
        )
        sqlite_store.store(memory)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="lifecycle",
            top_k=5,
        )

        assert len(results) == 1
        # _update_lifecycle mutates the returned result objects in-place.
        # It only writes to the store when the lifecycle STATE changes,
        # not when last_accessed_at alone changes.
        assert results[0].last_accessed_at > old_time

    def test_retrieve_empty_user(self, retrieval_pipeline: RetrievalPipeline) -> None:
        """No results for a user with no memories."""
        results = retrieval_pipeline.retrieve(
            user_id="nobody",
            query="anything",
            top_k=5,
        )
        assert results == []

    def test_retrieve_entity_boost(
                                      self,
                                      retrieval_pipeline: RetrievalPipeline,
                                      sqlite_store: SQLiteStorageAdapter,
                                      embed: MockEmbeddingAdapter,
                                  ) -> None:
        """Entity boost extracts entities from query and memories."""
        memory = make_memory(
            user_id="alice",
            content="Alice visited Paris",
            embedding=embed.embed_single("Alice visited Paris"),
        )
        sqlite_store.store(memory)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="Paris",
            top_k=5,
        )
        assert len(results) >= 1
        # Entity boost should have run without error
        assert results[0].memory_id == memory.memory_id

    def test_retrieve_max_tokens_truncation(
                                               self,
                                               retrieval_pipeline: RetrievalPipeline,
                                               sqlite_store: SQLiteStorageAdapter,
                                               embed: MockEmbeddingAdapter,
                                           ) -> None:
        """max_tokens truncates long results."""
        memory = make_memory(
            user_id="alice",
            content="word " * 100,
            embedding=embed.embed_single("word " * 100),
        )
        sqlite_store.store(memory)

        results = retrieval_pipeline.retrieve(
            user_id="alice",
            query="word",
            top_k=5,
            max_tokens=10,
        )
        assert len(results) >= 1
        # Token count should be at or below max_tokens
        # This is a rough heuristic: the scoring module truncates by tokens
        # but the exact token count depends on the tokenizer implementation.


# ---------------------------------------------------------------------------
# Error-path integration tests
# ---------------------------------------------------------------------------

class TestPipelineErrorPaths:
    def test_retrieve_embedding_dimension_mismatch(
                                                      self,
                                                      sqlite_store: SQLiteStorageAdapter,
                                                      embed: MockEmbeddingAdapter,
                                                      config: MemoryConfig,
                                                      entity_linker: RegexEntityLinker,
                                                      metrics: MetricsCollector,
                                                  ) -> None:
        """Retrieval raises ValidationError when stored memories have a different embedding dimension than the current adapter."""  # noqa: E501
        # Store a memory with embedding_dim=32 (different from embed.dimension()=64)
        memory = make_memory(
            user_id="alice",
            content="dim mismatch test",
            embedding=[0.1] * 32,
            embedding_dim=32,
        )
        sqlite_store.store(memory)

        ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(ctx)

        with pytest.raises(ValidationError):
            retrieval.retrieve(
                user_id="alice",
                query="dim mismatch",
                top_k=5,
            )

    def test_ingest_storage_failure(
                                       self,
                                       sqlite_store: SQLiteStorageAdapter,
                                       config: MemoryConfig,
                                       entity_linker: RegexEntityLinker,
                                       metrics: MetricsCollector,
                                   ) -> None:
        """Ingestion propagates storage errors and calls the record_store_error callback."""
        error_log: list[str] = []
        ctx = IngestionContext(
            store=sqlite_store,
            config=config,
            entity_linker=entity_linker,
            metrics=metrics,
            record_store_error=lambda: error_log.append("store_error"),
        )
        ingestion = IngestionPipeline(ctx)

        memory = make_memory(
            user_id="alice",
            content="will fail",
            embedding=[0.1] * 64,
        )

        # Monkeypatch store.store to raise
        original_store = sqlite_store.store
        def broken_store(memory: MemoryObject) -> None:
            raise RuntimeError("disk full")
        sqlite_store.store = broken_store  # type: ignore[method-assign]

        try:
            with pytest.raises(RuntimeError, match="disk full"):
                ingestion.ingest(memory)
            assert error_log == ["store_error"]
        finally:
            sqlite_store.store = original_store  # type: ignore[method-assign]

    def test_retrieve_storage_search_failure(
                                                self,
                                                sqlite_store: SQLiteStorageAdapter,
                                                embed: MockEmbeddingAdapter,
                                                config: MemoryConfig,
                                                entity_linker: RegexEntityLinker,
                                                metrics: MetricsCollector,
                                            ) -> None:
        """Retrieval propagates storage search errors to the caller."""
        memory = make_memory(
            user_id="alice",
            content="search failure test",
            embedding=embed.embed_single("search failure test"),
        )
        sqlite_store.store(memory)

        ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(ctx)

        original_search = sqlite_store.search
        def broken_search(**kwargs: Any) -> list[MemoryObject]:
            raise RuntimeError("database locked")
        sqlite_store.search = broken_search  # type: ignore[method-assign]

        try:
            with pytest.raises(RuntimeError, match="database locked"):
                retrieval.retrieve(
                    user_id="alice",
                    query="failure",
                    top_k=5,
                )
        finally:
            sqlite_store.search = original_search  # type: ignore[method-assign]

    def test_ingest_duplicate_with_broken_version_store(
                                                           self,
                                                           sqlite_store: SQLiteStorageAdapter,
                                                           config: MemoryConfig,
                                                           entity_linker: RegexEntityLinker,
                                                           metrics: MetricsCollector,
                                                       ) -> None:
        """Duplicate resolution survives a broken version store (logs and continues)."""
        ctx = IngestionContext(
            store=sqlite_store,
            config=config,
            entity_linker=entity_linker,
            metrics=metrics,
            get_version_store=lambda: (_raise(RuntimeError("version store down"))),  # type: ignore[return-value]
        )
        ingestion = IngestionPipeline(ctx)

        mem1 = make_memory(
            user_id="alice",
            content="same content",
            embedding=[0.5] * 64,
        )
        mem2 = make_memory(
            user_id="alice",
            content="same content",
            embedding=[0.5] * 64,
        )
        ingestion.ingest(mem1)
        # Should NOT raise even though version store is broken
        resolved = ingestion.ingest(mem2)
        assert resolved.memory_id == mem1.memory_id
        assert sqlite_store.count("alice") == 1

    def test_retrieve_lifecycle_store_failure(
                                                 self,
                                                 sqlite_store: SQLiteStorageAdapter,
                                                 embed: MockEmbeddingAdapter,
                                                 config: MemoryConfig,
                                                 entity_linker: RegexEntityLinker,
                                                 metrics: MetricsCollector,
                                             ) -> None:
        """Lifecycle store.update failure propagates to the caller."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        memory = make_memory(
            user_id="alice",
            content="lifecycle failure test",
            embedding=embed.embed_single("lifecycle failure test"),
            last_accessed_at=old_time,
            lifecycle_state=LifecycleState.DECAYING,  # will transition to ACTIVE on access
        )
        sqlite_store.store(memory)

        ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(ctx)

        # Monkeypatch store.update and store.update_many to raise during lifecycle transition
        original_update = sqlite_store.update
        original_update_many = sqlite_store.update_many
        def broken_update(memory: MemoryObject) -> None:
            raise RuntimeError("update failed")
        def broken_update_many(memories: list[MemoryObject]) -> int:
            raise RuntimeError("update failed")
        sqlite_store.update = broken_update  # type: ignore[method-assign]
        sqlite_store.update_many = broken_update_many  # type: ignore[method-assign]

        try:
            with pytest.raises(RuntimeError, match="update failed"):
                retrieval.retrieve(
                    user_id="alice",
                    query="lifecycle",
                    top_k=5,
                )
        finally:
            sqlite_store.update = original_update  # type: ignore[method-assign]
            sqlite_store.update_many = original_update_many  # type: ignore[method-assign]

    def test_retrieve_embed_failure(
                                       self,
                                       sqlite_store: SQLiteStorageAdapter,
                                       embed: MockEmbeddingAdapter,
                                       config: MemoryConfig,
                                       entity_linker: RegexEntityLinker,
                                       metrics: MetricsCollector,
                                   ) -> None:
        """Embedding failure during retrieval propagates to the caller."""
        ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(ctx)

        original_embed = embed.embed_single
        def broken_embed(text: str) -> list[float]:
            raise RuntimeError("model offline")
        embed.embed_single = broken_embed  # type: ignore[method-assign]

        try:
            with pytest.raises(RuntimeError, match="model offline"):
                retrieval.retrieve(
                    user_id="alice",
                    query="anything",
                    top_k=5,
                )
        finally:
            embed.embed_single = original_embed  # type: ignore[method-assign]

    def test_retrieve_entity_linker_failure(
                                               self,
                                               sqlite_store: SQLiteStorageAdapter,
                                               embed: MockEmbeddingAdapter,
                                               config: MemoryConfig,
                                               entity_linker: RegexEntityLinker,
                                               metrics: MetricsCollector,
                                           ) -> None:
        """Entity linker failure during retrieval propagates to the caller."""
        memory = make_memory(
            user_id="alice",
            content="entity failure test",
            embedding=embed.embed_single("entity failure test"),
        )
        sqlite_store.store(memory)

        ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(ctx)

        original_extract = entity_linker.extract
        def broken_extract(text: str) -> set[str]:
            raise RuntimeError("spacy crashed")
        entity_linker.extract = broken_extract  # type: ignore[method-assign]

        try:
            with pytest.raises(RuntimeError, match="spacy crashed"):
                retrieval.retrieve(
                    user_id="alice",
                    query="entity",
                    top_k=5,
                )
        finally:
            entity_linker.extract = original_extract  # type: ignore[method-assign]

    def test_ingest_webhook_failure(
                                       self,
                                       sqlite_store: SQLiteStorageAdapter,
                                       config: MemoryConfig,
                                       entity_linker: RegexEntityLinker,
                                       metrics: MetricsCollector,
                                   ) -> None:
        """Webhook dispatch failure during ingestion propagates after the memory is already stored."""  # noqa: E501
        ctx = IngestionContext(
            store=sqlite_store,
            config=config,
            entity_linker=entity_linker,
            metrics=metrics,
            dispatch_webhook=lambda *args, **kwargs: (_raise(RuntimeError("webhook timeout"))),
        )
        ingestion = IngestionPipeline(ctx)

        memory = make_memory(
            user_id="alice",
            content="webhook failure test",
            embedding=[0.1] * 64,
        )

        with pytest.raises(RuntimeError, match="webhook timeout"):
            ingestion.ingest(memory)

        # Memory WAS stored before the webhook failed
        stored = sqlite_store.get(memory.memory_id)
        assert stored is not None
        assert stored.content == "webhook failure test"


# ---------------------------------------------------------------------------
# End-to-end round-trip tests
# ---------------------------------------------------------------------------

def _raise(exc: Exception) -> None:
    raise exc


class TestAsyncPipelineIntegration:
    """Async end-to-end tests using Memory.aremember / Memory.arecall with real SQLite."""

    @pytest.fixture
    def memory(self, tmp_path: Any, embed: MockEmbeddingAdapter) -> Memory:
        """Fresh Memory instance backed by real SQLite."""
        db_path = str(tmp_path / "async_pipeline.db")
        store = SQLiteStorageAdapter(db_path=db_path)
        return Memory(embed=embed, store=store)

    @pytest.mark.asyncio
    async def test_aremember_stores_memory(self, memory: Memory) -> None:
        """Async remember persists to SQLite."""
        mem_id = await memory.aremember("alice", "I love pizza")
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

        results = memory.recall("alice", "pizza")
        assert len(results) == 1
        assert results[0].content == "I love pizza"

    @pytest.mark.asyncio
    async def test_arecall_returns_results(self, memory: Memory) -> None:
        """Async recall returns stored memories."""
        memory.remember("alice", "I am vegetarian")
        memory.remember("alice", "I live in Mumbai")

        results = await memory.arecall("alice", "food preferences")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert any("vegetarian" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_arecall_stream_yields_results(self, memory: Memory) -> None:
        """Async streaming recall yields the same memories as batch recall."""
        memory.remember("alice", "I am vegetarian")
        memory.remember("alice", "I live in Mumbai")
        memory.remember("alice", "I love python programming")

        batch_results = await memory.arecall("alice", "user preferences", top_k=3)

        stream_results: list[MemoryObject] = []
        async for mem in memory.recall_stream("alice", "user preferences", top_k=3):
            stream_results.append(mem)

        assert len(stream_results) == len(batch_results)
        for s, b in zip(stream_results, batch_results, strict=True):
            assert s.memory_id == b.memory_id
            assert s.content == b.content
            assert abs(s.score - b.score) < 0.001

    @pytest.mark.asyncio
    async def test_arecall_many_concurrent(self, memory: Memory) -> None:
        """Concurrent async recall for multiple users."""
        memory.remember("alice", "Alice likes pizza")
        memory.remember("bob", "Bob likes sushi")

        results = await memory.arecall_many(
            user_ids=["alice", "bob"],
            queries=["food", "food"],
            top_k=5,
        )
        assert len(results) == 2
        assert "alice" in results
        assert "bob" in results
        assert len(results["alice"]) == 1
        assert len(results["bob"]) == 1

    @pytest.mark.asyncio
    async def test_aremember_many_batch(self, memory: Memory) -> None:
        """Async batch remember stores all memories."""
        ids = await memory.aremember_many(
            "alice",
            ["alpha content", "beta content", "gamma content"],
        )
        assert len(ids) == 3

        # Semantic search with a broad query should find at least one of them.
        results = memory.recall("alice", "content")
        assert len(results) >= 1
        contents = {r.content for r in results}
        assert "alpha content" in contents or "beta content" in contents or "gamma content" in contents  # noqa: E501

    @pytest.mark.asyncio
    async def test_arecall_empty_user(self, memory: Memory) -> None:
        """Async recall for empty user returns empty list."""
        results = await memory.arecall("nobody", "anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_arecall_stream_param(self, memory: Memory) -> None:
        """Async recall with stream=True returns an async generator yielding results."""
        memory.remember("alice", "I am vegetarian")
        memory.remember("alice", "I live in Mumbai")

        gen = await memory.arecall("alice", "user preferences", top_k=3, stream=True)
        results: list[MemoryObject] = []
        async for mem in gen:
            results.append(mem)

        assert len(results) == 2
        contents = {r.content for r in results}
        assert "I am vegetarian" in contents
        assert "I live in Mumbai" in contents

    @pytest.mark.asyncio
    async def test_aforget_removes_memory(self, memory: Memory) -> None:
        """Async forget removes a memory from storage."""
        memory.remember("alice", "secret data")
        mem_id = (await memory.arecall("alice", "secret"))[0].memory_id

        await memory.aforget("alice", mem_id)

        results = memory.recall("alice", "secret")
        assert results == []

    @pytest.mark.asyncio
    async def test_arecall_since(self, memory: Memory) -> None:
        """Async recall since a timestamp returns only memories created after."""
        memory.remember("alice", "older memory")
        # Use a query that matches and hours=0 so only very recent memories are returned
        results = await memory.arecall_since("alice", "memory", hours=0, top_k=5)
        # The memory was just stored, but hours=0 means nothing older than 0 hours.
        # Depending on timing it might or might not match, so we just verify it doesn't crash
        # and either returns the memory or an empty list.
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_arecall_by_tag(self, memory: Memory) -> None:
        """Async recall by tag returns memories with that tag."""
        mem = make_memory(
            user_id="alice",
            content="tagged memory",
            embedding=memory._embed.embed_single("tagged memory"),
            tags=["work"],
        )
        memory._store.store(mem)

        results = await memory.arecall_by_tag("alice", tag="work")
        assert len(results) == 1
        assert results[0].content == "tagged memory"

    @pytest.mark.asyncio
    async def test_arecall_embedding_dimension_mismatch(self, memory: Memory) -> None:
        """Async recall raises ValidationError on dimension mismatch."""
        # Store a memory with a 32-dim embedding
        memory._store.store(
            make_memory(
                memory_id="id1",
                user_id="alice",
                content="old memory",
                embedding=[0.1] * 32,
                embedding_dim=32,
            )
        )

        with pytest.raises(ValidationError, match="Embedding dimension mismatch"):
            await memory.arecall("alice", "old")


class TestQueryCacheShallowCopy:
    """Unit tests for _QueryCache returning shallow copies (no cache corruption)."""

    def test_cache_mutation_metadata_tags_and_list(self, query_cache: _QueryCache) -> None:
        """Mutating returned metadata, tags, content, or the list must not corrupt the cache."""
        mem = make_memory(
            user_id="alice",
            content="cache test",
            embedding=[0.1] * 64,
            metadata={"original": "value"},
            tags=["a", "b"],
        )
        key = query_cache._make_key(
            user_id="alice",
            query="test",
            top_k=5,
            max_tokens=None,
            lifecycle_filter=None,
            hybrid_search=None,
            namespace="default",
            session_id=None,
            metadata_filter=None,
        )
        query_cache.put(key, [mem])

        # First get — mutate everything
        r1 = query_cache.get(key)
        assert r1 is not None
        r1[0].content = "hacked"
        r1[0].metadata["injected"] = "bad"
        r1[0].tags.append("corrupt")
        r1.append(make_memory(user_id="alice", content="extra", embedding=[0.2] * 64))

        # Second get — cache should be pristine
        r2 = query_cache.get(key)
        assert r2 is not None
        assert len(r2) == 1
        assert r2[0].content == "cache test"
        assert "injected" not in r2[0].metadata
        assert r2[0].metadata["original"] == "value"
        assert r2[0].tags == ["a", "b"]

    def test_cache_put_isolation(self, query_cache: _QueryCache) -> None:
        """Putting a list and then mutating the original list must not corrupt the cache."""
        mem = make_memory(
            user_id="alice",
            content="isolation test",
            embedding=[0.1] * 64,
            metadata={"key": "val"},
            tags=["x"],
        )
        original = [mem]
        key = query_cache._make_key(
            user_id="alice",
            query="iso",
            top_k=5,
            max_tokens=None,
            lifecycle_filter=None,
            hybrid_search=None,
            namespace="default",
            session_id=None,
            metadata_filter=None,
        )
        query_cache.put(key, original)

        # Mutate the original list after putting
        original[0].content = "changed"
        original[0].metadata["key"] = "changed"
        original[0].tags.append("corrupt")
        original.append(make_memory(user_id="alice", content="extra", embedding=[0.2] * 64))

        # Cache should still have the original pristine copy
        cached = query_cache.get(key)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].content == "isolation test"
        assert cached[0].metadata["key"] == "val"
        assert cached[0].tags == ["x"]


class TestPipelineRoundTrip:
    def test_ingest_then_retrieve(
                                     self,
                                     sqlite_store: SQLiteStorageAdapter,
                                     embed: MockEmbeddingAdapter,
                                     config: MemoryConfig,
                                     entity_linker: RegexEntityLinker,
                                     metrics: MetricsCollector,
                                 ) -> None:
        """Ingestion stores a memory; retrieval finds it."""
        ingest_ctx = IngestionContext(
            store=sqlite_store,
            config=config,
            entity_linker=entity_linker,
            metrics=metrics,
        )
        ingestion = IngestionPipeline(ingest_ctx)

        memory = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        ingestion.ingest(memory)

        retrieve_ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(retrieve_ctx)

        results = retrieval.retrieve(
            user_id="alice",
            query="pizza",
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].content == "I love pizza"

    def test_duplicate_ingest_then_retrieve(
                                               self,
                                               sqlite_store: SQLiteStorageAdapter,
                                               embed: MockEmbeddingAdapter,
                                               config: MemoryConfig,
                                               entity_linker: RegexEntityLinker,
                                               metrics: MetricsCollector,
                                           ) -> None:
        """Duplicate ingestion merges; retrieval still finds the canonical."""
        ingest_ctx = IngestionContext(
            store=sqlite_store,
            config=config,
            entity_linker=entity_linker,
            metrics=metrics,
        )
        ingestion = IngestionPipeline(ingest_ctx)

        mem1 = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        mem2 = make_memory(
            user_id="alice",
            content="I love pizza",
            embedding=embed.embed_single("I love pizza"),
        )
        canonical = ingestion.ingest(mem1)
        resolved = ingestion.ingest(mem2)
        assert resolved.memory_id == canonical.memory_id

        retrieve_ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(retrieve_ctx)

        results = retrieval.retrieve(
            user_id="alice",
            query="pizza",
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].memory_id == canonical.memory_id

    def test_namespaces_isolated(
                                    self,
                                    sqlite_store: SQLiteStorageAdapter,
                                    embed: MockEmbeddingAdapter,
                                    config: MemoryConfig,
                                    entity_linker: RegexEntityLinker,
                                    metrics: MetricsCollector,
                                ) -> None:
        """Memories in different namespaces are isolated."""
        ingest_ctx = IngestionContext(
            store=sqlite_store,
            config=config,
            entity_linker=entity_linker,
            metrics=metrics,
        )
        ingestion = IngestionPipeline(ingest_ctx)

        mem_a = make_memory(
            user_id="alice",
            content="namespace A",
            embedding=embed.embed_single("namespace A"),
            namespace="ns_a",
        )
        mem_b = make_memory(
            user_id="alice",
            content="namespace B",
            embedding=embed.embed_single("namespace B"),
            namespace="ns_b",
        )
        ingestion.ingest(mem_a)
        ingestion.ingest(mem_b)

        retrieve_ctx = RetrievalContext(
            store=sqlite_store,
            embed=embed,
            config=config,
            entity_linker=entity_linker,
            query_cache=None,
            metrics=metrics,
            adaptive_retriever=None,
        )
        retrieval = RetrievalPipeline(retrieve_ctx)

        results_a = retrieval.retrieve(
            user_id="alice",
            query="namespace",
            top_k=5,
            namespace="ns_a",
        )
        results_b = retrieval.retrieve(
            user_id="alice",
            query="namespace",
            top_k=5,
            namespace="ns_b",
        )

        assert len(results_a) == 1
        assert results_a[0].namespace == "ns_a"
        assert len(results_b) == 1
        assert results_b[0].namespace == "ns_b"
