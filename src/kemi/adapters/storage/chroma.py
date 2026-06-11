"""Chroma storage adapter for kemi.

Mirrors the SQLiteStorageAdapter API.  Uses Chroma for vector storage
and search with cosine similarity via the ``chromadb`` package.

Install: pip install chromadb

Usage::

    from kemi import Memory
    from kemi.adapters.storage.chroma import ChromaStorageAdapter

    adapter = ChromaStorageAdapter()
    memory = Memory(store=adapter)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from kemi.adapters.base import StorageAdapter
from kemi.exceptions import ConfigurationError
from kemi.memory.model import LifecycleState, MemoryObject, MemorySource, MemoryType

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction, Embeddings

    _chroma_available = True
except ImportError:  # pragma: no cover
    chromadb = None
    _chroma_available = False

_CHROMA_ERR = (
    "chromadb>=0.4.0 is required for ChromaStorageAdapter. "
    "Install with: pip install chromadb"
)


if _chroma_available:
    class _NoOpEmbeddingFunction(EmbeddingFunction):
        """No-op embedding function — we always pass embeddings ourselves."""

        def __call__(self, input: list[str]) -> Embeddings:
            return []  # pragma: no cover
else:
    _NoOpEmbeddingFunction = None  # type: ignore[misc,assignment]


class ChromaStorageAdapter(StorageAdapter):
    """Chroma vector database storage adapter.

    Stores memories as Chroma documents with embeddings and metadata.
    Uses cosine distance for similarity search (Chroma default).

    Parameters
    ----------
    path : str, optional
        Path to the Chroma persistent directory.  Defaults to the
        ``CHROMA_PATH`` env var, falling back to ``~/.kemi/chroma``.
    collection_name : str
        Name of the Chroma collection to use (default ``kemi``).
    """

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = "kemi",
    ) -> None:
        if not _chroma_available:
            raise ConfigurationError(_CHROMA_ERR)

        self._collection_name = collection_name
        chroma_path = path or os.environ.get(
            "CHROMA_PATH", os.path.expanduser("~/.kemi/chroma")
        )
        # Chroma's PersistentClient creates the directory itself if needed.
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_NoOpEmbeddingFunction(),
        )

    # ── Connection management ──────────────────────────────────────────

    def close(self) -> None:
        """Chroma's PersistentClient doesn't require explicit close, but
        we provide this for API consistency with other adapters."""
        pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "ChromaStorageAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── MemoryObject <-> Chroma format conversion ──────────────────────

    def _memory_to_metadata(self, memory: MemoryObject) -> dict[str, Any]:
        """Convert MemoryObject fields to a flat metadata dict for Chroma.

        Chroma metadata only supports string, int, float, and bool values.
        Nested types (lists, dicts) are JSON-serialized.
        """
        return {
            "memory_id": memory.memory_id,
            "user_id": memory.user_id,
            "content": memory.content,
            "created_at": memory.created_at.isoformat(),
            "last_accessed_at": memory.last_accessed_at.isoformat(),
            "source": memory.source.value,
            "importance": memory.importance,
            "lifecycle_state": memory.lifecycle_state.value,
            "metadata_json": json.dumps(memory.metadata),
            "embedding_dim": memory.embedding_dim or 0,
            "tags_json": json.dumps(memory.tags),
            "confidence": memory.confidence,
            "memory_type": memory.memory_type.value,
            "session_id": memory.session_id or "",
            "namespace": memory.namespace,
            "version": memory.version,
            "agent_id": memory.agent_id or "",
            "run_id": memory.run_id or "",
            "app_id": memory.app_id or "",
        }

    @staticmethod
    def _safe_get_embeddings(
        results: dict[str, Any], index: int, sub_index: int | None = None
    ) -> list[float] | None:
        """Safely extract an embedding from Chroma results, handling numpy arrays."""
        embeddings = results.get("embeddings")
        if embeddings is None or not isinstance(embeddings, (list, tuple)):
            return None
        if len(embeddings) <= 0:
            return None
        if sub_index is not None:
            # results["embeddings"][sub_index][index]
            if len(embeddings) <= sub_index:
                return None
            inner = embeddings[sub_index]
            if inner is None or not isinstance(inner, (list, tuple)):
                return None
            if len(inner) <= index:
                return None
            raw = inner[index]
        else:
            # results["embeddings"][index]
            if len(embeddings) <= index:
                return None
            raw = embeddings[index]

        if raw is None:
            return None
        return [float(v) for v in raw]

    @staticmethod
    def _safe_get_metadatas(
        results: dict[str, Any], index: int, sub_index: int | None = None
    ) -> dict[str, Any]:
        """Safely extract metadata from Chroma results."""
        metadatas = results.get("metadatas")
        if metadatas is None or not isinstance(metadatas, (list, tuple)):
            return {}
        if len(metadatas) <= 0:
            return {}
        if sub_index is not None:
            if len(metadatas) <= sub_index:
                return {}
            inner = metadatas[sub_index]
            if inner is None or not isinstance(inner, (list, tuple)):
                return {}
            if len(inner) <= index:
                return {}
            raw = inner[index]
        else:
            if len(metadatas) <= index:
                return {}
            raw = metadatas[index]

        if raw is None or not isinstance(raw, dict):
            return {}
        return raw

    @staticmethod
    def _safe_get_distances(
        results: dict[str, Any], index: int, sub_index: int | None = None
    ) -> float:
        """Safely extract a distance from Chroma results."""
        distances = results.get("distances")
        if distances is None or not isinstance(distances, (list, tuple)):
            return 0.0
        if len(distances) <= 0:
            return 0.0
        if sub_index is not None:
            if len(distances) <= sub_index:
                return 0.0
            inner = distances[sub_index]
            if inner is None or not isinstance(inner, (list, tuple)):
                return 0.0
            if len(inner) <= index:
                return 0.0
            raw = inner[index]
        else:
            if len(distances) <= index:
                return 0.0
            raw = distances[index]

        if raw is None:
            return 0.0
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0

    def _metadata_to_memory(
        self,
        mem_id: str,
        metadata: dict[str, Any],
        embedding: list[float] | None = None,
        score: float = 0.0,
    ) -> MemoryObject:
        """Convert Chroma metadata back to a MemoryObject."""

        def _safe_list(raw: Any, default: list[Any]) -> list[Any]:
            if raw is None:
                return default
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, list):
                return raw
            return default

        def _safe_dict(raw: Any, default: dict[str, Any]) -> dict[str, Any]:
            if raw is None:
                return default
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
            return default

        def _safe_str(raw: Any, default: str = "") -> str:
            if raw is None:
                return default
            if isinstance(raw, str):
                return raw
            # Handle potential numpy string types
            return str(raw)

        tags = _safe_list(metadata.get("tags_json", "[]"), [])
        mem_metadata = _safe_dict(metadata.get("metadata_json", "{}"), {})
        created = datetime.fromisoformat(metadata["created_at"])
        last = datetime.fromisoformat(metadata["last_accessed_at"])

        return MemoryObject(
            memory_id=mem_id,
            user_id=metadata["user_id"],
            content=metadata["content"],
            embedding=embedding,
            score=score,
            created_at=created,
            last_accessed_at=last,
            source=MemorySource(metadata["source"]),
            importance=float(metadata.get("importance", 0.5)),
            lifecycle_state=LifecycleState(metadata["lifecycle_state"]),
            metadata=mem_metadata,
            embedding_dim=metadata.get("embedding_dim") or None,
            tags=tags,
            confidence=float(metadata.get("confidence", 1.0)),
            memory_type=MemoryType(metadata.get("memory_type", "episodic")),
            session_id=_safe_str(metadata.get("session_id")) or None,
            namespace=metadata.get("namespace", "default"),
            version=int(metadata.get("version", 1)),
            agent_id=_safe_str(metadata.get("agent_id")) or None,
            run_id=_safe_str(metadata.get("run_id")) or None,
            app_id=_safe_str(metadata.get("app_id")) or None,
        )

    # ── Filter helper ───────────────────────────────────────────────────

    @staticmethod
    def _build_where(
        user_id: str | None = None,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a Chroma ``where`` dict from kemi query parameters."""
        conditions: list[dict[str, Any]] = []

        if user_id is not None:
            conditions.append({"user_id": user_id})

        if lifecycle_filter is not None:
            states = [s.value for s in lifecycle_filter]
            if len(states) == 1:
                conditions.append({"lifecycle_state": states[0]})
            else:
                conditions.append({"lifecycle_state": {"$in": states}})

        if namespace is not None:
            conditions.append({"namespace": namespace})

        if session_id is not None:
            conditions.append({"session_id": session_id})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    # ── CRUD ────────────────────────────────────────────────────────────

    def store(self, memory: MemoryObject) -> None:
        metadata = self._memory_to_metadata(memory)
        # Convert embedding to Python list if it's a numpy array
        emb = memory.embedding
        if emb is not None:
            emb = [float(v) for v in emb]
        self._collection.upsert(
            ids=[memory.memory_id],
            embeddings=[emb] if emb is not None else None,
            metadatas=[metadata],
            documents=[memory.content],
        )

    def store_many(self, memories: list[MemoryObject]) -> int:
        if not memories:
            return 0

        ids = [m.memory_id for m in memories]
        embeddings: list[list[float]] = [
            [float(v) for v in m.embedding] if m.embedding else []
            for m in memories
        ]
        metadatas = [self._memory_to_metadata(m) for m in memories]
        documents = [m.content for m in memories]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(memories)

    def get(self, memory_id: str) -> MemoryObject | None:
        try:
            results = self._collection.get(
                ids=[memory_id],
                include=["embeddings", "metadatas", "documents"],
            )
        except Exception:  # pragma: no cover
            return None

        if not results or not results["ids"]:
            return None

        emb = self._safe_get_embeddings(results, 0)
        metadata = self._safe_get_metadatas(results, 0)

        return self._metadata_to_memory(
            mem_id=results["ids"][0],
            metadata=metadata,
            embedding=emb,
        )

    def update(self, memory: MemoryObject) -> None:
        self.store(memory)

    def delete_by_user(self, user_id: str) -> int:
        where = self._build_where(user_id=user_id)
        results = self._collection.get(where=where, include=[])
        if not results or not results["ids"]:
            return 0

        ids = results["ids"]
        total = len(ids)

        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            try:
                self._collection.delete(ids=batch)
            except Exception:  # pragma: no cover
                pass

        return total

    def delete_by_id(self, memory_id: str) -> bool:
        # Check existence first — Chroma's delete() doesn't error on missing IDs
        existing = self.get(memory_id)
        if existing is None:
            return False
        try:
            self._collection.delete(ids=[memory_id])
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

        where = self._build_where(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        # Convert query_embedding to Python list in case it's numpy
        qe = [float(v) for v in query_embedding]

        results = self._collection.query(
            query_embeddings=[qe],
            n_results=top_k,
            where=where,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        memories = []
        for i, mem_id in enumerate(results["ids"][0]):
            metadata = self._safe_get_metadatas(results, i, sub_index=0)
            emb = self._safe_get_embeddings(results, i, sub_index=0)
            distance = self._safe_get_distances(results, i, sub_index=0)
            score = 1.0 / (1.0 + distance)
            memories.append(
                self._metadata_to_memory(
                    mem_id=mem_id,
                    metadata=metadata,
                    embedding=emb,
                    score=score,
                )
            )

        return memories

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        """Keyword-based search via Chroma ``where`` filter on content.

        Chroma doesn't have a built-in BM25 / text search, so we fetch
        matching memories by filter and score by simple keyword overlap
        in Python.
        """
        if lifecycle_filter is None:
            lifecycle_filter = [LifecycleState.ACTIVE, LifecycleState.DECAYING]

        where = self._build_where(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        results = self._collection.get(
            where=where,
            include=["embeddings", "metadatas", "documents"],
        )

        if not results or not results["ids"]:
            return []

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored: list[tuple[float, str, dict[str, Any], list[float] | None]] = []
        for i, mem_id in enumerate(results["ids"]):
            doc = results["documents"][i] if results.get("documents") else ""
            metadata = self._safe_get_metadatas(results, i)
            emb = self._safe_get_embeddings(results, i)

            if query_lower in doc.lower():
                doc_terms = set(doc.lower().split())
                overlap = len(query_terms & doc_terms)
                score_val = overlap / max(len(query_terms), 1)
                scored.append((score_val, mem_id, metadata, emb))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return [
            self._metadata_to_memory(
                mem_id=item[1],
                metadata=item[2],
                embedding=item[3],
                score=item[0],
            )
            for item in top
        ]

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

        where = self._build_where(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
            session_id=session_id,
        )

        results = self._collection.get(
            where=where,
            limit=limit,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )

        if not results or not results["ids"]:
            return []

        memories = []
        for i, mem_id in enumerate(results["ids"]):
            metadata = self._safe_get_metadatas(results, i)
            emb = self._safe_get_embeddings(results, i)
            memories.append(
                self._metadata_to_memory(
                    mem_id=mem_id,
                    metadata=metadata,
                    embedding=emb,
                )
            )

        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        results = self._collection.get(
            limit=limit,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )

        if not results or not results["ids"]:
            return []

        memories = []
        for i, mem_id in enumerate(results["ids"]):
            metadata = self._safe_get_metadatas(results, i)
            emb = self._safe_get_embeddings(results, i)
            memories.append(
                self._metadata_to_memory(
                    mem_id=mem_id,
                    metadata=metadata,
                    embedding=emb,
                )
            )

        return memories

    def count(self, user_id: str) -> int:
        where = self._build_where(user_id=user_id)
        results = self._collection.get(where=where, include=[])
        if not results or not results["ids"]:
            return 0
        return len(results["ids"])

    def get_all_users(self) -> list[str]:
        results = self._collection.get(include=["metadatas"])
        if not results or not results["ids"]:
            return []

        seen: set[str] = set()
        for meta in results.get("metadatas", []):
            if meta and "user_id" in meta:
                seen.add(meta["user_id"])
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

        where = self._build_where(
            user_id=user_id,
            lifecycle_filter=lifecycle_filter,
            namespace=namespace,
        )

        results = self._collection.get(
            where=where,
            include=["embeddings", "metadatas", "documents"],
        )

        if not results or not results["ids"]:
            return []

        tagged = []
        for i, mem_id in enumerate(results["ids"]):
            metadata = self._safe_get_metadatas(results, i)
            emb = self._safe_get_embeddings(results, i)

            tags_raw = metadata.get("tags_json", "[]")
            if isinstance(tags_raw, str):
                tags = json.loads(tags_raw)
            elif isinstance(tags_raw, list):
                tags = tags_raw
            else:
                tags = []

            if tag in tags:
                tagged.append(
                    self._metadata_to_memory(
                        mem_id=mem_id,
                        metadata=metadata,
                        embedding=emb,
                    )
                )

        return tagged

    # ── Schema ──────────────────────────────────────────────────────────

    def upgrade_schema(
        self, from_version: int | None = None, to_version: int | None = None
    ) -> int:
        from_v = from_version if from_version is not None else 1
        to_v = to_version if to_version is not None else 1
        logger.info(
            "Chroma schema upgrade from v%d to v%d is a no-op (schema managed by Chroma)",
            from_v,
            to_v,
        )
        return to_v
