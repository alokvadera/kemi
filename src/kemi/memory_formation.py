"""MemoryFormation – turn conversations into structured memories.

Extracts atomic facts and events from chat histories using a pluggable
LLM extractor, deduplicates them against existing memories, and returns
:class:`MemoryObject` instances ready for persistence.

Example::

    from kemi.core import Memory
    from kemi.memory_formation import remember_from_conversation, OpenAIMemoryExtractor

    mem = Memory()
    conversation = [
        {"role": "user", "content": "I love hiking in the Alps."},
        {"role": "assistant", "content": "That sounds amazing!"},
        {"role": "user", "content": "My favourite trail is the Tour du Mont Blanc."},
    ]
    ids = remember_from_conversation(
        mem, conversation, user_id="alice", extractor=OpenAIMemoryExtractor()
    )
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kemi import dedup
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.models import (
    LifecycleState,
    MemoryConfig,
    MemoryObject,
    MemorySource,
    MemoryType,
)

if TYPE_CHECKING:
    from kemi.core import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMMemoryExtractor(Protocol):
    """Protocol for conversation-to-memory extractors.

    Any class implementing ``extract`` can be plugged into
    :func:`extract_memories` or :func:`remember_from_conversation`.
    """

    def extract(
        self,
        conversation: list[dict[str, Any]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> list[CandidateMemory]:
        """Extract candidate memories from a conversation.

        Args:
            conversation: List of messages. Each item should contain at
                least ``role`` (``"user"`` | ``"assistant"`` | ``"system"``)
                and ``content`` (``str``). An optional ``timestamp`` may be
                included.
            user_id: User the conversation belongs to.
            session_id: Optional session identifier.

        Returns:
            List of candidate memories ready for deduplication and storage.
        """
        ...


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidateMemory:
    """A memory candidate produced by an extractor before embedding and storage."""

    content: str
    importance: float = 0.5
    memory_type: MemoryType = MemoryType.EPISODIC
    tags: list[str] = field(default_factory=lambda: [])
    metadata: dict[str, Any] = field(default_factory=lambda: {})


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------

class RegexMemoryExtractor:
    """Simple regex/heuristic extractor for tests and local use.

    Requires no external LLM.  Matches common patterns such as preferences,
    goals, and personal facts.
    """

    _PATTERNS: list[tuple[str, list[str], MemoryType, float]] = [
        # (regex, tags, memory_type, importance)
        (r"I like\s+(.+?)[.!?]", ["preference"], MemoryType.SEMANTIC, 0.6),
        (r"My name is\s+(.+?)[.!?]", ["identity"], MemoryType.SEMANTIC, 0.8),
        (r"I am\s+(.+?)[.!?]", ["identity"], MemoryType.SEMANTIC, 0.7),
        (r"I want to\s+(.+?)[.!?]", ["goal"], MemoryType.EPISODIC, 0.7),
        (r"I need to\s+(.+?)[.!?]", ["goal"], MemoryType.EPISODIC, 0.7),
        (r"I prefer\s+(.+?)[.!?]", ["preference"], MemoryType.SEMANTIC, 0.6),
        (r"I live in\s+(.+?)[.!?]", ["location"], MemoryType.SEMANTIC, 0.7),
        (r"I work at\s+(.+?)[.!?]", ["work"], MemoryType.SEMANTIC, 0.7),
        (r"Remember that\s+(.+?)[.!?]", ["reminder"], MemoryType.EPISODIC, 0.8),
        (r"Don't forget\s+(.+?)[.!?]", ["reminder"], MemoryType.EPISODIC, 0.8),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(p, re.IGNORECASE), tags, mtype, imp)
            for p, tags, mtype, imp in self._PATTERNS
        ]

    def extract(
        self,
        conversation: list[dict[str, Any]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> list[CandidateMemory]:
        candidates: list[CandidateMemory] = []
        seen: set[str] = set()

        for msg in conversation:
            content = msg.get("content", "")
            if not content:
                continue

            for pattern, tags, mtype, imp in self._compiled:
                for match in pattern.finditer(content):
                    fact = match.group(1).strip()
                    if not fact or fact.lower() in seen:
                        continue
                    seen.add(fact.lower())

                    meta: dict[str, Any] = {}
                    if "timestamp" in msg:
                        meta["extracted_from_timestamp"] = msg["timestamp"]
                    if session_id:
                        meta["session_id"] = session_id

                    candidates.append(
                        CandidateMemory(
                            content=fact,
                            importance=imp,
                            memory_type=mtype,
                            tags=list(tags),
                            metadata=meta,
                        )
                    )

        return candidates


class OpenAIMemoryExtractor:
    """OpenAI-powered memory extractor.

    Uses the Chat Completions API with a structured system prompt to turn
    a conversation into atomic memory candidates.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAIMemoryExtractor requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def extract(
        self,
        conversation: list[dict[str, Any]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> list[CandidateMemory]:
        system_prompt = (
            "You are a memory extraction assistant. "
            "Given a conversation, extract atomic facts and events that would be "
            "useful to remember about the user. "
            "For each memory, provide: content (short sentence), importance (0.0-1.0), "
            "type ('episodic' or 'semantic'), tags (list of strings), and metadata (dict). "
            "Return ONLY a JSON array of objects. Do not include markdown formatting."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.3,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or "[]"
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].strip("json").strip()
            parsed: list[dict[str, Any]] = json.loads(raw)  # type: ignore[assignment]
        except Exception:
            logger.exception("OpenAI memory extraction failed")
            return []

        candidates: list[CandidateMemory] = []
        for item in parsed:
            if not isinstance(item, dict):  # type: ignore[reportUnnecessaryIsinstance]
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            mtype_str = str(item.get("type", "episodic"))
            mtype = MemoryType.EPISODIC if mtype_str == "episodic" else MemoryType.SEMANTIC

            meta: dict[str, Any] = dict(item.get("metadata", {}))
            if session_id:
                meta["session_id"] = session_id

            candidates.append(
                CandidateMemory(
                    content=content,
                    importance=float(item.get("importance", 0.5)),
                    memory_type=mtype,
                    tags=list(item.get("tags", [])),
                    metadata=meta,
                )
            )

        return candidates


class StaticMemoryExtractor:
    """Extractor that returns a fixed list of candidates.

    Useful for deterministic testing or as a no-op placeholder.
    """

    def __init__(self, candidates: list[CandidateMemory]) -> None:
        self._candidates = list(candidates)

    def extract(
        self,
        conversation: list[dict[str, Any]],
        *,
        user_id: str,
        session_id: str | None = None,
    ) -> list[CandidateMemory]:
        return list(self._candidates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_memories(
    conversation: list[dict[str, Any]],
    *,
    user_id: str,
    session_id: str | None = None,
    extractor: LLMMemoryExtractor | None = None,
    embed: EmbeddingAdapter | None = None,
    store: StorageAdapter | None = None,
    config: MemoryConfig | None = None,
    dedup_threshold: float = 0.85,
    namespace: str = "default",
) -> list[MemoryObject]:
    """Extract and deduplicate memories from a conversation.

    Steps:

    1. Runs the extractor over the conversation to get candidate memories.
    2. Batch-embeds all candidate contents.
    3. Creates :class:`MemoryObject` instances.
    4. Deduplicates against *existing* memories in ``store`` and against
       each other using the configured threshold.
    5. Returns the accepted memories (unsorted).

    Args:
        conversation: List of messages. Each item should contain at
            least ``role`` and ``content``. An optional ``timestamp`` may
            be included.
        user_id: User the conversation belongs to.
        session_id: Optional session identifier.
        extractor: Pluggable extractor.  Defaults to
            :class:`RegexMemoryExtractor` if not provided.
        embed: Embedding adapter (e.g. ``memory._embed``).  Required if
            candidates are to be embedded.
        store: Storage adapter (e.g. ``memory._store``).  Used to fetch
            existing memories for deduplication.
        config: Optional :class:`MemoryConfig` for threshold defaults.
        dedup_threshold: Cosine-similarity threshold above which a
            candidate is considered a duplicate (default 0.85).
        namespace: Memory namespace for extracted objects and for the
            dedup query against existing memories.

    Returns:
        List of :class:`MemoryObject` instances ready to persist.
    """
    if not conversation:
        return []

    if embed is None:
        raise ValueError(
            "An embedding adapter is required. Pass embed=memory._embed "
            "or initialise a Memory instance first."
        )

    if extractor is None:
        extractor = RegexMemoryExtractor()

    candidates = extractor.extract(conversation, user_id=user_id, session_id=session_id)
    if not candidates:
        return []

    # Resolve threshold
    threshold = dedup_threshold
    if config is not None:
        threshold = config.dedup_threshold

    contents = [c.content for c in candidates]
    embeddings = embed.embed(contents)

    now = datetime.now(timezone.utc)
    memory_objects: list[MemoryObject] = []
    for i, cand in enumerate(candidates):
        memory_objects.append(
            MemoryObject(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                content=cand.content,
                embedding=embeddings[i],
                embedding_dim=len(embeddings[i]),
                importance=max(0.0, min(1.0, cand.importance)),
                memory_type=cand.memory_type,
                tags=list(cand.tags),
                metadata={
                    **cand.metadata,
                    "source": "memory_formation",
                    "extracted_at": now.isoformat(),
                },
                session_id=session_id,
                namespace=namespace,
                source=MemorySource.AGENT_INFERRED,
                created_at=now,
                last_accessed_at=now,
            )
        )

    # Deduplicate against existing memories
    existing: list[MemoryObject] = []
    if store is not None:
        existing = store.get_all_by_user(
            user_id,
            lifecycle_filter=[
                LifecycleState.ACTIVE,
                LifecycleState.DECAYING,
                LifecycleState.ARCHIVED,
            ],
            namespace=namespace,
        )

    accepted: list[MemoryObject] = []
    for mem in memory_objects:
        # Check against existing memories
        dups = dedup.find_duplicates(mem, existing, threshold)
        if dups:
            logger.debug(
                "Skipping duplicate of existing memory %s: %s",
                dups[0].memory_id,
                mem.content[:50],
            )
            continue

        # Check against already-accepted candidates
        dups = dedup.find_duplicates(mem, accepted, threshold)
        if dups:
            logger.debug(
                "Skipping intra-conversation duplicate: %s", mem.content[:50]
            )
            continue

        accepted.append(mem)

    return accepted


def remember_from_conversation(
    memory: "Memory",
    conversation: list[dict[str, Any]],
    *,
    user_id: str,
    session_id: str | None = None,
    extractor: LLMMemoryExtractor | None = None,
    dedup_threshold: float | None = None,
    namespace: str = "default",
) -> list[str]:
    """Extract memories from a conversation and persist them.

    This is a convenience wrapper around :func:`extract_memories` that
    handles batch embedding, deduplication, and storage via the internal
    ``_remember_with_embedding`` path for efficiency.

    Args:
        memory: A :class:`kemi.core.Memory` instance.
        conversation: List of messages with ``role``, ``content``, and
            optional ``timestamp``.
        user_id: User the conversation belongs to.
        session_id: Optional session identifier.
        extractor: Pluggable extractor (defaults to regex).
        dedup_threshold: Deduplication threshold.
        namespace: Memory namespace for extracted objects and for the
            dedup query against existing memories.

    Returns:
        List of persisted memory IDs.
    """
    if not hasattr(memory, "_embed") or not hasattr(memory, "_store"):
        raise TypeError(
            f"memory must be a kemi.core.Memory instance, got {type(memory).__name__}"
        )

    threshold = dedup_threshold
    if threshold is None:
        threshold = memory._config.dedup_threshold

    extracted = extract_memories(
        conversation,
        user_id=user_id,
        session_id=session_id,
        extractor=extractor,
        embed=memory._embed,
        store=memory._store,
        config=memory._config,
        dedup_threshold=threshold,
        namespace=namespace,
    )

    memory_ids: list[str] = []
    for mem in extracted:
        mid = memory._remember_with_embedding(
            user_id=user_id,
            content=mem.content,
            embedding=mem.embedding or [],
            importance=mem.importance,
            source=mem.source,
            metadata=mem.metadata,
            tags=mem.tags,
            namespace=namespace,
            session_id=mem.session_id,
            memory_type=mem.memory_type,
            confidence=mem.confidence,
        )
        memory_ids.append(mid)

    return memory_ids
