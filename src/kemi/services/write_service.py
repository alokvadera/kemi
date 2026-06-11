"""Write-side facade: remember, update, forget, migrate, feedback.

Methods on this facade mutate the memory store. Cross-facade composition
(e.g. ``run_maintenance`` which calls prune + consolidate + backfill) is
handled by the public ``MemoryService`` shim, not here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from kemi.exceptions import ValidationError
from kemi.memory import sanitize
from kemi.memory.model import MemorySource, MemoryType

if TYPE_CHECKING:
    from kemi.adapters.base import EmbeddingAdapter
    from kemi.memory.core import _MemoryCore

logger = logging.getLogger(__name__)


class MemoryWriteService:
    """Write-path methods: remember/update/forget/migrate/feedback."""

    def __init__(self, core: _MemoryCore) -> None:
        self._core = core

    def remember(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        metadata: dict[str, Any] | None = None,
        sanitize_input: bool = False,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """Store a new memory."""
        self._core.validate_remember_inputs(user_id, content, importance, ttl_seconds)

        self._core._run_hooks(
            "pre",
            "remember",
            user_id=user_id,
            content=content,
            namespace=namespace,
        )

        with self._core._latency_tracker("remember"):
            if sanitize_input:
                content = sanitize.sanitize(content, strict=self._core._config.sanitize)

            try:
                embedding = self._core._embed.embed_single(content)
            except Exception:
                self._core._record_embed_error()
                raise

            memory = self._core.build_memory_object(
                user_id=user_id,
                content=content,
                embedding=embedding,
                importance=importance,
                source=source,
                metadata=metadata,
                tags=tags,
                namespace=namespace,
                session_id=session_id,
                memory_type=memory_type,
                confidence=confidence,
                agent_id=agent_id,
                run_id=run_id,
                app_id=app_id,
                ttl_seconds=ttl_seconds,
            )

            from kemi.pipeline.ingestion import IngestionPipeline

            stored = IngestionPipeline(self._core.build_ingestion_context()).ingest(memory)

        self._core._run_hooks(
            "post",
            "remember",
            user_id=user_id,
            memory_id=stored.memory_id,
            namespace=namespace,
        )
        return stored.memory_id

    def remember_many(
        self,
        user_id: str,
        contents: list[str],
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> list[str]:
        """Store multiple memories at once."""
        if not contents:
            return []

        for i, content in enumerate(contents):
            if not content or not content.strip():
                raise ValidationError(
                    f"content at index {i} cannot be empty — there is nothing to remember"
                )

        with self._core._latency_tracker("remember_many"):
            embeddings = self._core._embed.embed(contents)

            memory_ids: list[str] = []
            audit_batch: list[dict[str, Any]] | None = (
                [] if self._core._plugins.audit_sinks else None
            )
            for i, content in enumerate(contents):
                self._core._run_hooks(
                    "pre", "remember", user_id=user_id, content=content, namespace=namespace
                )
                memory_id = self._remember_with_embedding(
                    user_id=user_id,
                    content=content,
                    embedding=embeddings[i],
                    importance=importance,
                    source=source,
                    tags=tags,
                    namespace=namespace,
                    session_id=session_id,
                    memory_type=memory_type,
                    confidence=confidence,
                    audit_batch=audit_batch,
                    agent_id=agent_id,
                    run_id=run_id,
                    app_id=app_id,
                    ttl_seconds=ttl_seconds,
                )
                self._core._run_hooks(
                    "post", "remember", user_id=user_id, memory_id=memory_id, namespace=namespace
                )
                memory_ids.append(memory_id)

            if self._core._metrics is not None:
                self._core._metrics.total_memories.set(self._core._store.count(user_id))

            if audit_batch is not None:
                self._core._track_operation(
                    "remember_many",
                    user_id,
                    {"count": len(memory_ids)},
                    namespace=namespace,
                    audit_batch=audit_batch,
                )
                for sink in self._core._plugins.audit_sinks:
                    try:
                        sink.log_batch(audit_batch)
                    except Exception:
                        logger.warning(
                            "Audit log batch failed for remember_many", exc_info=True
                        )
            return memory_ids

    def _remember_with_embedding(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        importance: float,
        source: MemorySource,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        audit_batch: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """Internal: store a memory with a pre-computed embedding."""
        from kemi.pipeline.ingestion import IngestionPipeline

        memory = self._core.build_memory_object(
            user_id=user_id,
            content=content,
            embedding=embedding,
            importance=importance,
            source=source,
            metadata=metadata,
            tags=tags,
            namespace=namespace,
            session_id=session_id,
            memory_type=memory_type,
            confidence=confidence,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            ttl_seconds=ttl_seconds,
        )
        stored = IngestionPipeline(self._core.build_ingestion_context()).ingest(
            memory, audit_batch=audit_batch
        )
        return stored.memory_id

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Update an existing memory."""
        from kemi.operations import _io

        return _io.update(
            self._core.build_io_runtime(),
            memory_id,
            content,
            importance,
            confidence,
            memory_type,
            metadata,
            tags,
        )

    def update_many(
        self,
        memory_ids: list[str],
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Update multiple memories at once."""
        from kemi.operations import _io

        return _io.update_many(
            self._core.build_io_runtime(),
            memory_ids,
            content,
            importance,
            confidence,
            memory_type,
            metadata,
        )

    def forget(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> int:
        """Delete a memory."""
        from kemi.operations import _io

        return _io.forget(self._core.build_io_runtime(), user_id, memory_id)

    def forget_many(
        self,
        memory_ids: list[str],
    ) -> int:
        """Delete multiple memories by ID at once."""
        from kemi.operations import _io

        return _io.forget_many(self._core.build_io_runtime(), memory_ids)

    def feedback(
        self,
        user_id: str,
        memory_id: str,
        helpful: bool = True,
        namespace: str = "default",
    ) -> None:
        """Record user feedback on a recalled memory."""
        from kemi.operations import _io

        _io.feedback(self._core.build_io_runtime(), user_id, memory_id, helpful, namespace)

    def backfill_entities(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Backfill ``extracted_entities`` for memories that don't have them yet."""
        from kemi.operations import _io

        return _io.backfill_entities(self._core.build_io_runtime(), user_id, namespace)

    def extract_entities(self, memory_id: str) -> list[dict[str, Any]]:
        """Extract named entities from a memory's content."""
        from kemi.operations import _io

        return _io.extract_entities(self._core.build_io_runtime(), memory_id)

    def migrate(
        self,
        user_id: str,
        new_embed_fn: EmbeddingAdapter,
        batch_size: int = 100,
    ) -> int:
        """Re-embed all ACTIVE/DECAYING memories for a user with a new embedder."""
        from kemi.operations import _io

        return _io.migrate(self._core.build_io_runtime(), user_id, new_embed_fn, batch_size)

    async def aremember(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        metadata: dict[str, Any] | None = None,
        sanitize_input: bool = False,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.remember,
            user_id,
            content,
            importance,
            source,
            metadata,
            sanitize_input,
            tags,
            namespace,
            session_id,
            memory_type,
            confidence,
            agent_id,
            run_id,
            app_id,
            ttl_seconds,
        )

    async def aremember_many(
        self,
        user_id: str,
        contents: list[str],
        importance: float = 0.5,
        source: MemorySource = MemorySource.USER_STATED,
        tags: list[str] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 1.0,
        agent_id: str | None = None,
        run_id: str | None = None,
        app_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.remember_many,
            user_id,
            contents,
            importance,
            source,
            tags,
            namespace,
            session_id,
            memory_type,
            confidence,
            agent_id,
            run_id,
            app_id,
            ttl_seconds,
        )

    async def aupdate(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from kemi.operations import _io

        return await _io.aupdate(
            self._core.build_io_runtime(),
            memory_id,
            content,
            importance,
            confidence,
            memory_type,
            metadata,
        )

    async def aupdate_many(
        self,
        memory_ids: list[str],
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        memory_type: MemoryType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        from kemi.operations import _io

        return await _io.aupdate_many(
            self._core.build_io_runtime(),
            memory_ids,
            content,
            importance,
            confidence,
            memory_type,
            metadata,
        )

    async def aforget(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> int:
        from kemi.operations import _io

        return await _io.aforget(self._core.build_io_runtime(), user_id, memory_id)

    async def aforget_many(
        self,
        memory_ids: list[str],
    ) -> int:
        from kemi.operations import _io

        return await _io.aforget_many(self._core.build_io_runtime(), memory_ids)

    async def abackfill_entities(
        self,
        user_id: str | None = None,
        namespace: str | None = None,
    ) -> int:
        from kemi.operations import _io

        return await _io.abackfill_entities(self._core.build_io_runtime(), user_id, namespace)
