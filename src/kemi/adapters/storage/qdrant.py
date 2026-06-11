"""Qdrant storage adapter for kemi.

Mirrors the SQLiteStorageAdapter API.  Uses Qdrant for vector storage
and search with cosine similarity via the ``qdrant-client`` package.

Install: pip install qdrant-client

Usage::

    from kemi import Memory
    from kemi.adapters.storage.qdrant import QdrantStorageAdapter

    # Connect to a running Qdrant server
    adapter = QdrantStorageAdapter(url="http://localhost:6333")
    memory = Memory(store=adapter)

    # Or use in-memory mode (useful for testing)
    adapter = QdrantStorageAdapter(location=":memory:")
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from kemi.adapters.base import StorageAdapter
from kemi.exceptions import ConfigurationError
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.http import models as qmodels

    _qdrant_available = True
except ImportError:  # pragma: no cover
    _qdrant_available = False

_QDRANT_ERR = (
    "qdrant-client>=1.9.0 is required for QdrantStorageAdapter. "
    "Install with: pip install qdrant-client"
)


class QdrantStorageAdapter(StorageAdapter):
    """Qdrant vector database storage adapter.

    Stores memories as Qdrant points with the embedding as the vector
    and all other fields as JSON payload.  Uses cosine distance for
    similarity search.

    Parameters
    ----------
    location : str, optional
        Qdrant server location.  Pass ``\":memory:\"`` for an in-memory
        instance (useful for testing).  Defaults to the ``QDRANT_URL``
        env var, falling back to ``http://localhost:6333``.
    url : str, optional
        Alias for *location* (provided for backward compatibility).
        If both are provided, *location* takes precedence.
    collection_name : str
        Name of the Qdrant collection to use (default ``kemi``).
    embedding_dim : int
        Dimension of embedding vectors (default 384, matching fastembed).
        Must match the dimension produced by the embedding adapter in use.
        **Cannot be changed after collection creation** (Qdrant requires a
        fixed dimension at collection definition time).
    prefer_grpc : bool
        Whether to prefer gRPC over HTTP (default False).
    api_key : str, optional
        Qdrant Cloud API key.  Defaults to the ``QDRANT_API_KEY`` env var.
    """

    def __init__(
        self,
        location: str | None = None,
        url: str | None = None,
        collection_name: str = "kemi",
        embedding_dim: int = 384,
        prefer_grpc: bool = False,
        api_key: str | None = None,
    ) -> None:
        if not _qdrant_available:
            raise ConfigurationError(_QDRANT_ERR)

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        resolved_location = (
            location or url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        self._client = _QdrantClient(
            location=resolved_location,
            prefer_grpc=prefer_grpc,
            api_key=api_key or os.environ.get("QDRANT_API_KEY"),
        )
        self._ensure_collection()

    # ── Collection management ──────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = self._client.get_collections().collections
        if any(c.name == self._collection_name for c in collections):
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(
                size=self._embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d)",
            self._collection_name,
            self._embedding_dim,
        )

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "QdrantStorageAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Point <-> MemoryObject conversion ──────────────────────────────

    @staticmethod
    def _memory_to_payload(memory: MemoryObject) -> dict[str, Any]:
        return {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "content": memory.content,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata": json.dumps(memory.metadata),
            "embedding_dim": memory.embedding_dim,
            "tags": json.dumps(memory.tags),
            "confidence": memory.confidence,
            "memory_type": memory.memory_type.value,
            "session_id": memory.session_id,
            "namespace": memory.namespace,
            "version": memory.version,
            "agent_id": memory.agent_id,
            "run_id": memory.run_id,
            "app_id": memory.app_id,
        }

    def _point_to_memory(self, point: Any) -> MemoryObject:
        """Convert a Qdrant point (ScoredPoint or Record) to MemoryObject."""
        p = point.payload or {}
        # Extract the original memory_id from payload (not the UUID point ID)
        extracted_memory_id = p.get("memory_id", str(point.id))

        embedding = (
            list(point.vector or [])
            if hasattr(point, "vector") and point.vector
            else None
        )

        tags_raw = p.get("tags", "[]")
        if isinstance(tags_raw, str):
            tags = json.loads(tags_raw)
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        metadata_raw = p.get("metadata", "{}")
        if isinstance(metadata_raw, str):
            metadata = json.loads(metadata_raw)
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            metadata = {}

        return MemoryObject(
            memory_id=extracted_memory_id,
            user_id=p["user_id"],
            content=p["content"],
            embedding=embedding,
            score=float(getattr(point, "score", 0.0)),
            created_at=datetime.fromisoformat(p["created_at"]),
            last_accessed_at=datetime.fromisoformat(p["last_accessed_at"]),
            source=MemorySource(p["source"]),
            importance=float(p["importance"]),
            lifecycle_state=LifecycleState(p["lifecycle_state"]),
            metadata=metadata,
            embedding_dim=p.get("embedding_dim"),
            tags=tags,
            confidence=float(p.get("confidence", 1.0)),
            memory_type=MemoryType(p.get("memory_type", "episodic")),
            session_id=p.get("session_id"),
            namespace=p.get("namespace", "default"),
            version=int(p.get("version", 1)),
            agent_id=p.get("agent_id"),
            run_id=p.get("run_id"),
            app_id=p.get("app_id"),
        )

    # ── Filter helpers ────────────────────────────────────────────────

    def _build_filter(
        self,
        user_id: str | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str | None = None,
        session_id: str | None = None,
    ) -> qmodels.Filter | None:
        """Build a Qdrant filter from kemi query parameters."""
        must_conditions: list[
            qmodels.FieldCondition | qmodels.IsEmptyCondition
        ] = []

        if user_id is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=user_id),
                )
            )

        if lifecycle_filter is not None:
            states = [s.value for s in lifecycle_filter]
            must_conditions.append(
                qmodels.FieldCondition(
                    key="lifecycle_state",
                    match=qmodels.MatchAny(any=states),
                )
            )

        if namespace is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="namespace",
                    match=qmodels.MatchValue(value=namespace),
                )
            )

        if session_id is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="session_id",
                    match=qmodels.MatchValue(value=session_id),
                )
            )

        if not must_conditions:
            return None

        return qmodels.Filter(must=must_conditions)

    # ── CRUD ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_uuid(point_id: str) -> str:
        """Convert a string ID to a UUID for Qdrant.

        Qdrant's local (in-memory) mode requires UUID or integer point IDs.
        We use UUIDv5 with a fixed namespace to produce deterministic UUIDs.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

    def store(self, memory: MemoryObject) -> None:
        payload = self._memory_to_payload(memory)
        point = qmodels.PointStruct(
            id=self._to_uuid(memory.memory_id),
            vector=memory.embedding or [],
            payload=payload,
        )
        self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
        )

    def store_many(self, memories: list[MemoryObject]) -> int:
        if not memories:
            return 0
        points = []
        for memory in memories:
            payload = self._memory_to_payload(memory)
            points.append(
                qmodels.PointStruct(
                    id=self._to_uuid(memory.memory_id),
                    vector=memory.embedding or [],
                    payload=payload,
                )
            )
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )
        return len(points)

    def get(self, memory_id: str) -> MemoryObject | None:
        try:
            points = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[self._to_uuid(memory_id)],
                with_payload=True,
                with_vectors=True,
            )
        except Exception:  # pragma: no cover
            return None

        if not points:
            return None
        return self._point_to_memory(points[0])

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def delete_by_user(self, user_id: str) -> int:
        count_filter = self._build_filter(user_id=user_id)
        count_result = self._client.count(
            collection_name=self._collection_name,
            count_filter=count_filter,
        )
        count = count_result.count

        qfilter = self._build_filter(user_id=user_id)
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.FilterSelector(filter=qfilter),
        )
        return count

    def delete_by_id(self, memory_id: str) -> bool:
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=qmodels.PointIdsList(
                    points=[self._to_uuid(memory_id)],
                ),
            )
            return True
        except Exception:  # pragma: no cover
            return False

    # ── Search / retrieval ─────────────────────────────────────────────

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

        qfilter = self._build_filter(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        result = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )

        return [self._point_to_memory(hit) for hit in result.points]

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Keyword-based search via Qdrant payload text index.

        Uses Qdrant's ``MatchText`` payload filter to find memories
        whose content contains the query terms.
        """
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        qfilter = self._build_filter(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        should_conditions: list[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="content",
                match=qmodels.MatchText(text=query),
            )
        ]

        if qfilter is not None:
            combined_filter = qmodels.Filter(
                must=qfilter.must,
                should=should_conditions,
            )
        else:
            combined_filter = qmodels.Filter(should=should_conditions)

        result = self._client.query_points(
            collection_name=self._collection_name,
            query=[0.0] * self._embedding_dim,
            query_filter=combined_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [self._point_to_memory(hit) for hit in result.points]

    # ── Bulk retrieval ─────────────────────────────────────────────────

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

        qfilter = self._build_filter(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        scroll_limit = limit or 100
        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=qfilter,
            limit=scroll_limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        return [self._point_to_memory(r) for r in records]

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        scroll_limit = limit or 100
        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=scroll_limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        return [self._point_to_memory(r) for r in records]

    def count(self, user_id: str) -> int:
        qfilter = self._build_filter(user_id=user_id)
        result = self._client.count(
            collection_name=self._collection_name,
            count_filter=qfilter,
        )
        return result.count  # type: ignore[no-any-return]

    def get_all_users(self) -> list[str]:
        """Get all unique user IDs by scrolling and collecting distinct values."""
        seen: set[str] = set()
        current_offset: int | None = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=100,
                offset=current_offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                uid = r.payload.get("user_id")
                if uid:
                    seen.add(uid)
            if next_offset is None or next_offset == current_offset:
                break
            current_offset = next_offset
        return sorted(seen)

    # ── Tags ────────────────────────────────────────────────────────────

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        qfilter = self._build_filter(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
        )

        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=qfilter,
            limit=1000,
            with_payload=True,
            with_vectors=True,
        )

        results = []
        for r in records:
            mem = self._point_to_memory(r)
            if tag in mem.tags:
                results.append(mem)

        return results

    # ── Schema ──────────────────────────────────────────────────────────

    def upgrade_schema(
        self, from_version: int | None = None, to_version: int | None = None
    ) -> int:
        from_v = from_version if from_version is not None else 1
        to_v = to_version if to_version is not None else 1
        logger.info(
            "Qdrant schema upgrade from v%d to v%d is a no-op (schema managed by Qdrant)",
            from_v,
            to_v,
        )
        return to_v
