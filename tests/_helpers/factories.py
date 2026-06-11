"""Factory functions for tests.

Replaces inline ``MemoryObject(...)`` constructors and ``Memory(...)``
wrappers that appear ~300 times across the test suite. Using these
factories:

- Eliminates drift (e.g. some tests set ``MemorySource.SYSTEM_GENERATED``
  while others forget it).
- Makes tests ~10 lines shorter each.
- Lets new test code opt into the canonical defaults without having to
  know which fields the dataclass considers required.

Use ``make_memory()`` to build a single ``MemoryObject``; use
``make_mock_memory()`` to build a ``Memory`` wired to the in-memory
adapters; use ``make_real_db_memory(tmp_path)`` for the same but with a
real SQLite file under ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from kemi import Memory
from kemi.memory.model import (
    LifecycleState,
    MemoryObject,
    MemorySource,
    MemoryType,
)
from tests._helpers.mock_storage import MockEmbeddingAdapter, MockStorageAdapter


def make_memory(
    *,
    memory_id: str | None = None,
    user_id: str = "user1",
    content: str = "test content",
    embedding: list[float] | None = None,
    embedding_dim: int | None = None,
    score: float = 0.0,
    source: MemorySource = MemorySource.USER_STATED,
    importance: float = 0.5,
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    memory_type: MemoryType = MemoryType.EPISODIC,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    namespace: str = "default",
    session_id: str | None = None,
    confidence: float = 1.0,
    agent_id: str | None = None,
    run_id: str | None = None,
    app_id: str | None = None,
    expires_at: datetime | None = None,
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    **overrides: Any,
) -> MemoryObject:
    """Build a ``MemoryObject`` with sensible test defaults.

    All 11 default fields are filled in; tests only need to override what
    they care about. ``memory_id`` defaults to a random UUID4 hex string
    so test order doesn't cause collisions.
    """
    if memory_id is None:
        memory_id = uuid4().hex
    if metadata is None:
        metadata = {}
    if tags is None:
        tags = []
    now = datetime.now(timezone.utc)
    if created_at is None:
        created_at = now
    if last_accessed_at is None:
        last_accessed_at = now

    if embedding is not None and embedding_dim is None:
        embedding_dim = len(embedding)

    return MemoryObject(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=embedding,
        score=score,
        created_at=created_at,
        last_accessed_at=last_accessed_at,
        source=source,
        importance=importance,
        lifecycle_state=lifecycle_state,
        memory_type=memory_type,
        metadata=metadata,
        tags=tags,
        namespace=namespace,
        session_id=session_id,
        confidence=confidence,
        agent_id=agent_id,
        run_id=run_id,
        app_id=app_id,
        expires_at=expires_at,
        embedding_dim=embedding_dim,
        **overrides,
    )


def make_mock_memory(
    *,
    embed: MockEmbeddingAdapter | None = None,
    store: MockStorageAdapter | None = None,
) -> Memory:
    """Build a ``Memory`` wired to fresh in-memory adapters.

    Use this when a test needs a ``Memory`` instance but doesn't need a
    real database on disk.
    """
    return Memory(
        embed=embed if embed is not None else MockEmbeddingAdapter(),
        store=store if store is not None else MockStorageAdapter(),
    )


def make_real_db_memory(
    tmp_path: Any,
    *,
    embed: MockEmbeddingAdapter | None = None,
) -> Memory:
    """Build a ``Memory`` backed by a real temporary SQLite file.

    ``tmp_path`` is a pytest ``tmp_path`` fixture (a ``pathlib.Path``).
    The database file is created at ``tmp_path / "test_kemi.db"`` and
    lives only for the duration of the test.
    """
    from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

    db_path = str(tmp_path / "test_kemi.db")
    adapter = SQLiteStorageAdapter(db_path=db_path)
    return Memory(
        embed=embed if embed is not None else MockEmbeddingAdapter(),
        store=adapter,
    )
