from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kemi.adapters.base import StorageAdapter
from kemi.memory import scoring
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType

if TYPE_CHECKING:
    from kemi.infra.encryption import EncryptionConfig


class JSONStorageAdapter(StorageAdapter):
    """JSON file storage adapter.

    Thread safety: NOT guaranteed. Do not use from multiple threads.
    Embedding stored as list of floats in JSON for readability.

    Encryption: When encryption config is provided, uses Fernet field-level
    encryption for content and metadata fields. Pass ``encryption=`` to
    ``__init__`` or set ``KEMI_ENCRYPTION_KEY`` / ``KEMI_ENCRYPTION_ENABLED``
    environment variables.
    """

    def __init__(self, path: str = "kemi.json", encryption: EncryptionConfig | None = None):
        self._path = Path(path)
        self._data = self._load()
        # Lazy import to avoid circular dependency
        from kemi.infra.encryption import EncryptionConfig, FieldEncryptor

        if encryption is None:
            try:
                env_config = EncryptionConfig.from_env()
                self._field_encryptor = FieldEncryptor(env_config) if env_config.enabled else None
            except Exception:
                self._field_encryptor = None
        else:
            self._field_encryptor = FieldEncryptor(encryption) if encryption.enabled else None

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return {"memories": {}, "schema_version": 1}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _row_to_memory(self, data: dict[str, Any]) -> MemoryObject:
        # Decrypt fields if encryption is enabled
        content_val: Any = data.get("content", "")
        metadata_val: Any = data.get("metadata", {})
        user_id_val: Any = data.get("user_id", "")

        if self._field_encryptor is not None:
            if self._field_encryptor._is_encrypted(content_val):
                content_val = self._field_encryptor.decrypt_field("content", content_val)
            if self._field_encryptor._is_encrypted(metadata_val):
                metadata_val = self._field_encryptor.decrypt_field("metadata", metadata_val)
            if self._field_encryptor._is_encrypted(user_id_val):
                user_id_val = self._field_encryptor.decrypt_field("user_id", user_id_val)

        expires_at_raw = data.get("expires_at")
        expires_at = (
            datetime.fromisoformat(expires_at_raw) if expires_at_raw else None
        )
        return MemoryObject(
            memory_id=data["memory_id"],
            user_id=user_id_val,
            content=str(content_val),
            embedding=data.get("embedding"),
            score=0.0,
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            source=MemorySource(data["source"]),
            importance=data["importance"],
            lifecycle_state=LifecycleState(data["lifecycle_state"]),
            metadata=metadata_val if isinstance(metadata_val, dict) else {},
            embedding_dim=data.get("embedding_dim"),
            tags=data.get("tags", []),
            confidence=data.get("confidence", 1.0),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            session_id=data.get("session_id"),
            namespace=data.get("namespace", "default"),
            version=data.get("version", 1),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            app_id=data.get("app_id"),
            expires_at=expires_at,
        )

    def store(self, memory: MemoryObject) -> None:
        content_val: Any = memory.content
        metadata_val: Any = memory.metadata
        user_id_val: Any = memory.user_id

        if self._field_encryptor is not None:
            content_val = self._field_encryptor.encrypt_field("content", memory.content)
            metadata_val = self._field_encryptor.encrypt_field("metadata", memory.metadata)
            if self._field_encryptor._encrypt_user_id:
                user_id_val = self._field_encryptor.encrypt_field("user_id", memory.user_id)

        self._data["memories"][memory.memory_id] = {
            "memory_id": memory.memory_id,
            "user_id": user_id_val,
            "content": content_val,
            "embedding": memory.embedding,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata": metadata_val,
            "embedding_dim": memory.embedding_dim,
            "tags": memory.tags,
            "confidence": memory.confidence,
            "memory_type": memory.memory_type.value,
            "session_id": memory.session_id,
            "namespace": memory.namespace,
            "version": memory.version,
            "agent_id": memory.agent_id,
            "run_id": memory.run_id,
            "app_id": memory.app_id,
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        }
        self._save()

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = {s.value for s in lifecycle_filter}

        memories = []
        for mem_data in self._data["memories"].values():
            if mem_data["user_id"] != user_id:
                continue
            if mem_data["lifecycle_state"] not in states:
                continue
            if mem_data.get("namespace", "default") != namespace:
                continue
            if session_id is not None and mem_data.get("session_id") not in (session_id, None):
                continue

            memory = self._row_to_memory(mem_data)
            if memory.embedding is not None:
                similarity = scoring.cosine_similarity(memory.embedding, query_embedding)
                memory.score = (similarity + 1.0) / 2.0
                memories.append(memory)

        memories.sort(key=lambda m: m.score, reverse=True)
        return memories[:top_k]

    def get(self, memory_id: str) -> MemoryObject | None:
        mem_data = self._data["memories"].get(memory_id)
        if mem_data:
            return self._row_to_memory(mem_data)
        return None

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def delete_by_user(self, user_id: str) -> int:
        to_delete = [mid for mid, m in self._data["memories"].items() if m["user_id"] == user_id]
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
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = {s.value for s in lifecycle_filter}

        results = [
            self._row_to_memory(m)
            for m in self._data["memories"].values()
            if m["user_id"] == user_id
            and m["lifecycle_state"] in states
            and m.get("namespace", "default") == namespace
            and (session_id is None or m.get("session_id") in (session_id, None))
        ]
        # Apply pagination
        if offset is not None:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def count(self, user_id: str) -> int:
        return sum(1 for m in self._data["memories"].values() if m["user_id"] == user_id)

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        results = [self._row_to_memory(m) for m in self._data["memories"].values()]
        if offset is not None:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def get_all_users(self) -> list[str]:
        users = set(m["user_id"] for m in self._data["memories"].values())
        return list(users)

    def upgrade_schema(
        self, from_version: int | None = None, to_version: int | None = None
    ) -> int:
        to_v = to_version if to_version is not None else 1
        self._data["schema_version"] = to_v
        self._save()
        return to_v

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = {s.value for s in lifecycle_filter}

        return [
            self._row_to_memory(m)
            for m in self._data["memories"].values()
            if m["user_id"] == user_id
            and m["lifecycle_state"] in states
            and m.get("namespace", "default") == namespace
            and tag in m.get("tags", [])
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
        """Search for memories using keyword matching (no embeddings required)."""
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        states = {s.value for s in lifecycle_filter}
        query_lower = query.lower()

        candidates = [
            self._row_to_memory(m)
            for m in self._data["memories"].values()
            if m["user_id"] == user_id
            and m["lifecycle_state"] in states
            and m.get("namespace", "default") == namespace
            and (session_id is None or m.get("session_id") in (session_id, None))
            and query_lower in m["content"].lower()
        ]

        # Simple scoring: longer matches rank higher
        for mem in candidates:
            mem.score = len(query) / max(len(mem.content), 1)

        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]
