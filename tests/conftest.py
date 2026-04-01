import hashlib
from typing import Any, Dict, List, Optional

import pytest

from kemi import Memory
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemorySource


class MockEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self) -> None:
        self._dim = 64

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._deterministic_vector(t) for t in texts]

    def embed_single(self, text: str) -> List[float]:
        return self._deterministic_vector(text)

    def dimension(self) -> int:
        return self._dim

    def _deterministic_vector(self, text: str) -> List[float]:
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        vector = [b / 255.0 for b in expanded[: self._dim]]
        return vector


class MockStorageAdapter(StorageAdapter):
    def __init__(self) -> None:
        self._store: Dict[str, MemoryObject] = {}

    def store(self, memory: MemoryObject) -> None:
        self._store[memory.memory_id] = memory

    def search(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        from kemi.scoring import cosine_similarity

        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        candidates = [
            m
            for m in self._store.values()
            if m.user_id == user_id and m.lifecycle_state in lifecycle_filter
        ]

        for m in candidates:
            if m.embedding:
                m.score = cosine_similarity(m.embedding, query_embedding)

        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]

    def get(self, memory_id: str) -> Optional[MemoryObject]:
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
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        result = [m for m in self._store.values() if m.user_id == user_id]
        if lifecycle_filter is not None:
            result = [m for m in result if m.lifecycle_state in lifecycle_filter]
        return result

    def count(self, user_id: str) -> int:
        return sum(1 for m in self._store.values() if m.user_id == user_id)

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        pass


@pytest.fixture
def MockEmbedding() -> type:
    return MockEmbeddingAdapter


@pytest.fixture
def MockStorage() -> type:
    return MockStorageAdapter


@pytest.fixture
def mock_memory(MockEmbedding: type, MockStorage: type) -> Memory:
    return Memory(embed=MockEmbedding(), store=MockStorage())
