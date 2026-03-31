import json
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from kemi.adapters.base import StorageAdapter
from kemi.models import LifecycleState, MemoryObject
from kemi import scoring


class JSONStorageAdapter(StorageAdapter):
    """JSON file storage adapter.

    Thread safety: NOT guaranteed. Do not use from multiple threads.
    Embedding stored as list of floats in JSON for readability.
    """

    def __init__(self, path: str = "kemi.json"):
        self._path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, "r") as f:
                return json.load(f)
        return {"memories": {}, "schema_version": 1}

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _row_to_memory(self, data: dict) -> MemoryObject:
        return MemoryObject(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            score=0.0,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            source=LifecycleState(data["source"]),
            importance=data["importance"],
            lifecycle_state=LifecycleState(data["lifecycle_state"]),
            metadata=data.get("metadata", {}),
            embedding_dim=data.get("embedding_dim"),
        )

    def store(self, memory: MemoryObject) -> None:
        self._data["memories"][memory.memory_id] = {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "content": memory.content,
            "embedding": memory.embedding,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata": memory.metadata,
            "embedding_dim": memory.embedding_dim,
        }
        self._save()

    def search(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = {s.value for s in lifecycle_filter}

        memories = []
        for mem_data in self._data["memories"].values():
            if mem_data["user_id"] != user_id:
                continue
            if mem_data["lifecycle_state"] not in states:
                continue

            memory = self._row_to_memory(mem_data)
            if memory.embedding:
                similarity = scoring.cosine_similarity(
                    memory.embedding, query_embedding
                )
                memory.score = (similarity + 1.0) / 2.0
                memories.append(memory)

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:top_k]

    def get(self, memory_id: str) -> Optional[MemoryObject]:
        mem_data = self._data["memories"].get(memory_id)
        if mem_data:
            return self._row_to_memory(mem_data)
        return None

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def delete_by_user(self, user_id: str) -> int:
        to_delete = [
            mid for mid, m in self._data["memories"].items() if m["user_id"] == user_id
        ]
        for mid in to_delete:
            del self._data["memories"][mid]
        if to_delete:
            self._save()
        return len(to_delete)

    def delete_by_id(self, memory_id: str) -> bool:
        if memory_id in self._data["memories"]:
            del self._data["memories"][memory_id]
            self._save()
            return True
        return False

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = {s.value for s in lifecycle_filter}

        return [
            self._row_to_memory(m)
            for m in self._data["memories"].values()
            if m["user_id"] == user_id and m["lifecycle_state"] in states
        ]

    def count(self, user_id: str) -> int:
        return sum(
            1 for m in self._data["memories"].values() if m["user_id"] == user_id
        )

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        self._data["schema_version"] = to_version
        self._save()
