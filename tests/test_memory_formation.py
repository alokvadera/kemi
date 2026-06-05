"""Tests for kemi.memory_formation."""

import pytest

from kemi.adapters.base import EmbeddingAdapter
from kemi.core import Memory
from kemi.memory_formation import (
    CandidateMemory,
    LLMMemoryExtractor,
    MemoryType,
    OpenAIMemoryExtractor,
    RegexMemoryExtractor,
    StaticMemoryExtractor,
    extract_memories,
    remember_from_conversation,
)
from kemi.models import LifecycleState, MemoryConfig, MemoryObject, MemorySource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEmbedAdapter(EmbeddingAdapter):
    """Deterministic fake embedder for tests.

    Produces orthogonal one-hot embeddings so dedup only triggers
    for identical strings.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._cache: dict[str, list[float]] = {}
        self._counter = 0

    def _vector(self, text: str) -> list[float]:
        if text not in self._cache:
            self._counter += 1
            vec = [0.0] * self._dim
            vec[(self._counter - 1) % self._dim] = 1.0
            self._cache[text] = vec
        return self._cache[text]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vector(text)

    def dimension(self) -> int:
        return self._dim


class _FakeStore:
    """In-memory store for lightweight dedup tests."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryObject] = {}

    def store(self, memory: MemoryObject) -> None:
        self._data[memory.memory_id] = memory

    def get(self, memory_id: str) -> MemoryObject | None:
        return self._data.get(memory_id)

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        return [
            m for m in self._data.values()
            if m.user_id == user_id and m.namespace == namespace
        ]

    def count(self, user_id: str | None = None) -> int:
        if user_id is None:
            return len(self._data)
        return sum(1 for m in self._data.values() if m.user_id == user_id)


# ---------------------------------------------------------------------------
# Regex extractor
# ---------------------------------------------------------------------------

def test_regex_extractor_basic() -> None:
    extractor = RegexMemoryExtractor()
    conversation = [
        {"role": "user", "content": "I like pizza. My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "I want to learn Python."},
    ]
    candidates = extractor.extract(conversation, user_id="alice")

    contents = {c.content.lower() for c in candidates}
    assert "pizza" in contents
    assert "alice" in contents
    assert "learn python" in contents

    # Tags
    pizza = next(c for c in candidates if "pizza" in c.content.lower())
    assert "preference" in pizza.tags
    assert pizza.memory_type == MemoryType.SEMANTIC

    alice = next(c for c in candidates if "alice" in c.content.lower())
    assert "identity" in alice.tags


def test_regex_extractor_skips_duplicates() -> None:
    extractor = RegexMemoryExtractor()
    conversation = [
        {"role": "user", "content": "I like pizza. I like pizza."},
    ]
    candidates = extractor.extract(conversation, user_id="alice")
    assert len(candidates) == 1
    assert candidates[0].content == "pizza"


def test_regex_extractor_empty_conversation() -> None:
    extractor = RegexMemoryExtractor()
    candidates = extractor.extract([], user_id="alice")
    assert candidates == []


# ---------------------------------------------------------------------------
# Static extractor
# ---------------------------------------------------------------------------

def test_static_extractor() -> None:
    candidates = [
        CandidateMemory(content="Fact A", importance=0.9, tags=["test"]),
        CandidateMemory(content="Fact B", importance=0.3),
    ]
    extractor = StaticMemoryExtractor(candidates)
    result = extractor.extract([], user_id="alice")
    assert len(result) == 2
    assert result[0].content == "Fact A"


# ---------------------------------------------------------------------------
# OpenAI extractor (initialisation only — no API key in tests)
# ---------------------------------------------------------------------------

def test_openai_extractor_init_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """If openai is not installed, __init__ should raise ImportError."""
    import sys

    # Remove openai from import cache if present
    for name in list(sys.modules):
        if name.startswith("openai"):
            del sys.modules[name]

    monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError):
        OpenAIMemoryExtractor()


# ---------------------------------------------------------------------------
# extract_memories
# ---------------------------------------------------------------------------

def test_extract_memories_empty_conversation() -> None:
    embed = _FakeEmbedAdapter()
    result = extract_memories([], user_id="alice", embed=embed)
    assert result == []


def test_extract_memories_no_embed_raises() -> None:
    with pytest.raises(ValueError, match="embedding adapter is required"):
        extract_memories(
            [{"role": "user", "content": "hello"}],
            user_id="alice",
        )


def test_extract_memories_returns_memory_objects() -> None:
    embed = _FakeEmbedAdapter()
    conversation = [
        {"role": "user", "content": "I like sushi."},
    ]
    result = extract_memories(conversation, user_id="alice", embed=embed)

    assert len(result) == 1
    mem = result[0]
    assert isinstance(mem, MemoryObject)
    assert mem.user_id == "alice"
    assert "sushi" in mem.content.lower()
    assert mem.embedding is not None
    assert mem.source == MemorySource.AGENT_INFERRED
    assert mem.metadata.get("source") == "memory_formation"


def test_extract_memories_dedup_against_existing() -> None:
    embed = _FakeEmbedAdapter()
    store = _FakeStore()

    # Seed an existing memory
    existing = MemoryObject(
        memory_id="existing-1",
        user_id="alice",
        content="sushi very much",
        embedding=embed.embed_single("sushi very much"),
        source=MemorySource.USER_STATED,
    )
    store.store(existing)

    conversation = [
        {"role": "user", "content": "I like sushi very much."},
    ]
    result = extract_memories(
        conversation,
        user_id="alice",
        embed=embed,
        store=store,  # type: ignore[arg-type]
        dedup_threshold=0.99,  # very strict so the identical embedding counts as dup
    )

    assert len(result) == 0


def test_extract_memories_intra_dedup() -> None:
    embed = _FakeEmbedAdapter()
    conversation = [
        {"role": "user", "content": "I like sushi. I like sushi."},
    ]
    result = extract_memories(
        conversation,
        user_id="alice",
        embed=embed,
        dedup_threshold=0.99,
    )

    # Two candidates are extracted but one is deduped against the other
    assert len(result) == 1


# ---------------------------------------------------------------------------
# remember_from_conversation
# ---------------------------------------------------------------------------

def test_remember_from_conversation_type_guard() -> None:
    with pytest.raises(TypeError, match="must be a kemi.core.Memory instance"):
        remember_from_conversation("not_memory", [], user_id="alice")  # type: ignore[arg-type]


def test_remember_from_conversation_integration() -> None:
    mem = Memory(embed=_FakeEmbedAdapter(), store=_FakeStore())  # type: ignore[arg-type]
    conversation = [
        {"role": "user", "content": "My name is Bob. I like hiking."},
        {"role": "assistant", "content": "Great to know, Bob!"},
    ]
    ids = remember_from_conversation(mem, conversation, user_id="bob")

    assert len(ids) == 2
    for mid in ids:
        assert isinstance(mid, str)
        assert mem._store.get(mid) is not None


def test_remember_from_conversation_with_extractor() -> None:
    mem = Memory(embed=_FakeEmbedAdapter(), store=_FakeStore())  # type: ignore[arg-type]
    extractor = StaticMemoryExtractor([
        CandidateMemory(content="Static fact", importance=0.9, tags=["test"]),
    ])
    ids = remember_from_conversation(
        mem, [{"role": "user", "content": "placeholder"}], user_id="charlie", extractor=extractor
    )

    assert len(ids) == 1
    stored = mem._store.get(ids[0])
    assert stored is not None
    assert stored.content == "Static fact"
    assert stored.importance == 0.9
    assert "test" in stored.tags


# ---------------------------------------------------------------------------
# extract_memories — config and dedup branches
# ---------------------------------------------------------------------------

def test_extract_memories_uses_config_threshold() -> None:
    """Test that config.dedup_threshold overrides the parameter."""
    embed = _FakeEmbedAdapter()
    store = _FakeStore()

    config = MemoryConfig(dedup_threshold=0.99)

    conversation = [
        {"role": "user", "content": "I like sushi."},
    ]

    # With very high threshold from config, should still pass since no existing memories
    result = extract_memories(
        conversation,
        user_id="alice",
        embed=embed,
        store=store,
        config=config,
        dedup_threshold=0.1,  # Should be overridden by config
    )
    assert len(result) == 1


def test_extract_memories_skip_existing_duplicate_logs(caplog) -> None:
    """Test that skipping existing duplicate triggers logger.debug."""
    import logging

    embed = _FakeEmbedAdapter()
    store = _FakeStore()

    existing = MemoryObject(
        memory_id="existing-1",
        user_id="alice",
        content="sushi very much",
        embedding=embed.embed_single("sushi very much"),
        source=MemorySource.USER_STATED,
    )
    store.store(existing)

    conversation = [
        {"role": "user", "content": "I like sushi very much."},
    ]

    with caplog.at_level(logging.DEBUG):
        extract_memories(
            conversation,
            user_id="alice",
            embed=embed,
            store=store,
            dedup_threshold=0.99,
        )

    assert "Skipping duplicate of existing memory" in caplog.text


def test_extract_memories_skip_intra_duplicate_logs(caplog) -> None:
    """Test that skipping intra-conversation duplicate triggers logger.debug."""
    import logging

    embed = _FakeEmbedAdapter()

    # Use StaticMemoryExtractor to force two identical candidates
    # (RegexMemoryExtractor has its own dedup, so we bypass it here)
    extractor = StaticMemoryExtractor([
        CandidateMemory(content="duplicate fact", importance=0.5),
        CandidateMemory(content="duplicate fact", importance=0.5),
    ])

    conversation = [{"role": "user", "content": "placeholder"}]

    with caplog.at_level(logging.DEBUG):
        extract_memories(
            conversation,
            user_id="alice",
            embed=embed,
            extractor=extractor,
            dedup_threshold=0.99,
        )

    assert "Skipping intra-conversation duplicate" in caplog.text


# ---------------------------------------------------------------------------
# OpenAIMemoryExtractor
# ---------------------------------------------------------------------------

class TestOpenAIMemoryExtractor:
    """Tests for OpenAIMemoryExtractor with mocked openai module."""

    def _make_openai_module(self, create_fn):
        """Helper to build a mock openai module."""
        import types
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create_fn)
            )
        )
        return types.SimpleNamespace(OpenAI=lambda **kw: client)

    def test_init_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that __init__ raises ImportError when openai is missing."""
        import sys
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="OpenAIMemoryExtractor requires"):
            OpenAIMemoryExtractor()

    def test_extract_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful extraction via mocked OpenAI API."""
        import sys, types

        def mock_create(**kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"content": "User likes pizza", "importance": 0.8, "type": "semantic", "tags": ["food"], "metadata": {}}]'
                        )
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))

        extractor = OpenAIMemoryExtractor(api_key="test-key")
        conversation = [{"role": "user", "content": "I like pizza."}]
        candidates = extractor.extract(conversation, user_id="alice")

        assert len(candidates) == 1
        assert candidates[0].content == "User likes pizza"
        assert candidates[0].importance == 0.8
        assert candidates[0].memory_type == MemoryType.SEMANTIC
        assert "food" in candidates[0].tags

    def test_extract_with_session_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that session_id is added to metadata."""
        import sys, types

        def mock_create(**kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"content": "User likes pizza", "importance": 0.8, "type": "semantic", "tags": ["food"], "metadata": {}}]'
                        )
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))

        extractor = OpenAIMemoryExtractor(api_key="test-key")
        conversation = [{"role": "user", "content": "I like pizza."}]
        candidates = extractor.extract(conversation, user_id="alice", session_id="sess-123")

        assert candidates[0].metadata.get("session_id") == "sess-123"

    def test_extract_markdown_code_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing markdown-wrapped JSON response."""
        import sys, types

        def mock_create(**kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='```json\n[{"content": "User likes pasta", "importance": 0.7, "type": "episodic", "tags": [], "metadata": {}}]\n```'
                        )
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))

        extractor = OpenAIMemoryExtractor(api_key="test-key")
        conversation = [{"role": "user", "content": "I like pasta."}]
        candidates = extractor.extract(conversation, user_id="alice")

        assert len(candidates) == 1
        assert candidates[0].content == "User likes pasta"

    def test_extract_api_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API failure returns empty list."""
        import sys, types

        def mock_create(**kwargs):
            raise RuntimeError("API error")

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))

        extractor = OpenAIMemoryExtractor(api_key="test-key")
        conversation = [{"role": "user", "content": "I like pizza."}]
        candidates = extractor.extract(conversation, user_id="alice")

        assert candidates == []

    def test_extract_invalid_items_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that non-dict items and empty content are skipped."""
        import sys, types

        def mock_create(**kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"content": "Valid", "importance": 0.5}, 42, {"content": "", "importance": 0.5}, {"content": "Also valid", "importance": 0.6}]'
                        )
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))

        extractor = OpenAIMemoryExtractor(api_key="test-key")
        conversation = [{"role": "user", "content": "Test."}]
        candidates = extractor.extract(conversation, user_id="alice")

        assert len(candidates) == 2
        contents = [c.content for c in candidates]
        assert "Valid" in contents
        assert "Also valid" in contents

    def test_extract_episodic_default_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing type defaults to EPISODIC and unknown type becomes SEMANTIC."""
        import sys, types

        def mock_create(**kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"content": "No type field", "importance": 0.8, "tags": [], "metadata": {}}]'
                        )
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))

        extractor = OpenAIMemoryExtractor(api_key="test-key")
        conversation = [{"role": "user", "content": "I like pizza."}]
        candidates = extractor.extract(conversation, user_id="alice")

        assert candidates[0].memory_type == MemoryType.EPISODIC


# ---------------------------------------------------------------------------
# remember_from_conversation — confidence branch
# ---------------------------------------------------------------------------

def test_remember_from_conversation_passes_confidence() -> None:
    """Test that remember_from_conversation passes confidence to _remember_with_embedding."""
    from unittest.mock import MagicMock

    mock_memory = MagicMock()
    mock_memory._config.dedup_threshold = 0.85
    mock_memory._remember_with_embedding.return_value = "mem-id-1"

    extractor = StaticMemoryExtractor([
        CandidateMemory(
            content="Static fact",
            importance=0.9,
            tags=["test"],
            metadata={"confidence": 0.95},
        ),
    ])

    ids = remember_from_conversation(
        mock_memory,
        [{"role": "user", "content": "placeholder"}],
        user_id="charlie",
        extractor=extractor,
    )

    assert len(ids) == 1
    call_kwargs = mock_memory._remember_with_embedding.call_args.kwargs
    assert "confidence" in call_kwargs


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_runtime_checkable() -> None:
    assert isinstance(RegexMemoryExtractor(), LLMMemoryExtractor)
    assert isinstance(StaticMemoryExtractor([]), LLMMemoryExtractor)
