import hashlib

import pytest

from kemi import Memory
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.models import LifecycleState, MemoryObject


class MockEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self) -> None:
        self._dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._deterministic_vector(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._deterministic_vector(text)

    def dimension(self) -> int:
        return self._dim

    def _deterministic_vector(self, text: str) -> list[float]:
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        vector = [b / 255.0 for b in expanded[: self._dim]]
        return vector


class MockStorageAdapter(StorageAdapter):
    def __init__(self) -> None:
        self._store: dict[str, MemoryObject] = {}

    def store(self, memory: MemoryObject) -> None:
        self._store[memory.memory_id] = memory

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        from kemi.scoring import cosine_similarity

        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        candidates = [
            m
            for m in self._store.values()
            if m.user_id == user_id
            and m.lifecycle_state in lifecycle_filter
            and m.namespace == namespace
            and (session_id is None or m.session_id in (session_id, None))
        ]

        for m in candidates:
            if m.embedding:
                m.score = cosine_similarity(m.embedding, query_embedding)

        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]

    def get(self, memory_id: str) -> MemoryObject | None:
        return self._store.get(memory_id)

    def update(self, memory: MemoryObject) -> None:
        self._store[memory.memory_id] = memory

    def delete_by_user(self, user_id: str) -> int:
        to_delete = [mid for mid, m in self._store.items() if m.user_id == user_id]
        for mid in to_delete:
            del self._store[mid]
        return len(to_delete)

    def delete_by_id(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        result = [
            m
            for m in self._store.values()
            if m.user_id == user_id
            and m.namespace == namespace
            and (session_id is None or m.session_id in (session_id, None))
        ]
        if lifecycle_filter is not None:
            result = [m for m in result if m.lifecycle_state in lifecycle_filter]
        if offset is not None:
            result = result[offset:]
        if limit is not None:
            result = result[:limit]
        return result

    def count(self, user_id: str) -> int:
        return sum(1 for m in self._store.values() if m.user_id == user_id)

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        result = list(self._store.values())
        if offset is not None:
            result = result[offset:]
        if limit is not None:
            result = result[:limit]
        return result

    def get_all_users(self) -> list[str]:
        return list(set(m.user_id for m in self._store.values()))

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        pass

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        return [
            m
            for m in self._store.values()
            if m.user_id == user_id
            and m.lifecycle_state in lifecycle_filter
            and m.namespace == namespace
            and tag in m.tags
        ]

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        # Simple substring match for mock implementation
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        candidates = [
            m
            for m in self._store.values()
            if m.user_id == user_id
            and m.lifecycle_state in lifecycle_filter
            and m.namespace == namespace
            and query.lower() in m.content.lower()
            and (session_id is None or m.session_id in (session_id, None))
        ]

        return candidates[:top_k]


@pytest.fixture
def mock_embedding() -> type:
    return MockEmbeddingAdapter


@pytest.fixture
def mock_storage() -> type:
    return MockStorageAdapter


@pytest.fixture
def mock_memory(mock_embedding: type, mock_storage: type) -> Memory:
    return Memory(embed=mock_embedding(), store=mock_storage())


@pytest.fixture
def real_db_memory(tmp_path, mock_embedding: type) -> Memory:
    """Create a Memory instance backed by a real temporary SQLite database.

    This exercises the full CLI-to-storage pipeline including schema creation,
    SQL INSERT/SELECT operations, and storage adapter implementation.
    Uses the mock embedding adapter so no external embedding service is needed.
    """
    from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

    db_path = str(tmp_path / "test_kemi.db")
    adapter = SQLiteStorageAdapter(db_path=db_path)
    return Memory(embed=mock_embedding(), store=adapter)
